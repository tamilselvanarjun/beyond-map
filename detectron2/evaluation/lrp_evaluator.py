# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
#from detectron2.evaluation.fast_eval_api import COCOeval_opt as COCOeval
from pycocotools.cocoeval import COCOeval
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table

from .evaluator import DatasetEvaluator


class LRPEvaluator(DatasetEvaluator):
    """
    Evaluate LRP for segmentation
    """
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
        """
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.info(
                f"'{dataset_name}' is not registered by `register_coco_instances`."
                " Therefore trying to convert it to COCO format ..."
            )

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        self._kpt_oks_sigmas = cfg.TEST.KEYPOINT_OKS_SIGMAS
        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

    def reset(self):
        self._predictions = []

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("bbox",)
        if cfg.MODEL.MASK_ON:
            tasks = tasks + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            tasks = tasks + ("keypoints",)
        return tasks

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[LRPEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "instances" in predictions[0]:
            self._eval_predictions(set(self._tasks), predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, tasks, predictions):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api, coco_results, task, kpt_oks_sigmas=self._kpt_oks_sigmas
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )
            if task != 'segm':
                continue
            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results['lrp-' + task] = res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "segm": ['LRP', 'LRP-loc', 'LRP-FP', 'LRP-FN'],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = dict()
        # calculate TP, FP, FN, q(gi, dgi), final results for each class
        tpcalc = dict()
        fpcalc = dict()
        fncalc = dict()
        qcalc = dict()  
        # per class result (lrp, lrploc, lrpfp, lrpfn)
        lrp_class, lrp_loc_class, lrp_fp_class, lrp_fn_class = [dict() for _ in range(4)]

        # dict to keep area keys intact
        area_to_key = [(tuple(area), st) for area, st in zip(coco_eval.params.areaRng,  \
                                ['all', 'small', 'medium', 'large'])]
        area_to_key = dict(area_to_key)

        # get all classes
        all_cats = dict()
        # Calculate precision recall for each class, area, and iou threshold
        for evalImg in coco_eval.evalImgs:
            # Skip if maxDets is not the best value, or if evalImg is None (there was no detection)
            if evalImg is None:
                continue
            if evalImg['maxDet'] != coco_eval.params.maxDets[-1]:
                continue
            # Extract detected and ground truths
            # int, [a1, a2], [TxD], [TxD], [G], [G]
            # int, [dontcare], [D], [D], [G], [G]   (after taking dt[0] and dtIg[0])
            cate_class, area, dt, dtIg, gtIds, gtIg = [evalImg[x] for x in ['category_id', 'aRng', 'dtMatches', 'dtIgnore', 'gtIds', 'gtIgnore']]
            all_cats[cate_class] = 1
            # get area key
            areakey = area_to_key[tuple(area)]
            # ignore if not all areas are considered
            if areakey != 'all':
                continue
            # get image id, and iou matrix
            ious = evalImg['ious']
            dt = dt[0]
            dtIg = dtIg[0]
            # [G]
            gtm = evalImg['gtMatches'][0]
            # Total number of predictions
            num_gt = len(gtIds) - np.sum(gtIg)
            # total number of detections
            #num_dt = dt.shape - sum(dtIg)
            # There are some detections, see if there are matches and they are not ignored
            # Compute total true positives and false positives for the image (for each threshold)
            tp = np.logical_and(                dt , np.logical_not(dtIg) ).sum() * 1.0
            fp = np.logical_and( np.logical_not(dt), np.logical_not(dtIg) ).sum() * 1.0
            fn1 = np.logical_and( np.logical_not(gtm), np.logical_not(gtIg)).sum() * 1.0
            fn = num_gt - tp
            assert np.abs(fn1 - fn) < 1e-5, 'false negatives do not match, fn = {}, fn1 = {}'.format(fn, fn1)
            # add to these values
            tpcalc[cate_class] = tpcalc.get(cate_class, 0) + tp
            fpcalc[cate_class] = fpcalc.get(cate_class, 0) + fp
            fncalc[cate_class] = fncalc.get(cate_class, 0) + fn
            # get pairs of tps
            tpd_id, tpg_id = [], []
            for dind, gtid in enumerate(dt):   # dt contains a set of ids of the corresponding gt
                if dtIg[dind]:
                    continue
                # find where gtid is
                gtind = np.where([x==gtid for x in gtIds])[0]
                if len(gtind) == 0:
                    continue
                gtind = gtind[0]
                if gtIg[gtind]:
                    continue
                tpd_id.append(dind)
                tpg_id.append(gtind)
            # append
            assert np.abs(len(tpd_id) - tp) < 1e-5, 'number of true positives doesnt match with matching, tpds={}, tp={}'.format(len(tpd_id), tp)
            if tp > 0:
                neg_iou = (1-ious[tpd_id, tpg_id]).sum()
                qcalc[cate_class] = qcalc.get(cate_class, 0) + neg_iou

        # per class result (lrp, lrploc, lrpfp, lrpfn)
        for catid in all_cats.keys():
            tp = tpcalc.get(catid, 0)
            fp = fpcalc.get(catid, 0)
            fn = fncalc.get(catid, 0)
            Z = tp + fp + fn
            # skip if theres nothing for this class
            if Z <= 0:
                continue
            q = qcalc.get(catid, 0)   # this 'q' is actually sum_ (1 - q)
            # calc metrics
            _lrp = 1.0/Z  * (2 * q + fp + fn)
            _lrp_loc = 1/tp * q
            _lrp_fp = fp / (tp + fp)
            _lrp_fn = fn / (tp + fn)
            # append to class
            lrp_class[catid] = _lrp
            if not np.isnan(_lrp_loc):
                lrp_loc_class[catid] = _lrp_loc
            if not np.isnan(_lrp_fp):
                lrp_fp_class[catid] = _lrp_fp
            if not np.isnan(_lrp_fn):
                lrp_fn_class[catid] = _lrp_fn
            
        # get final values
        results['lrp'] = np.around(100 * np.mean(list(lrp_class.values())), 4)
        results['lrp_loc'] = np.around(100 * np.mean(list(lrp_loc_class.values())), 4)
        results['lrp_fp'] = np.around(100 * np.mean(list(lrp_fp_class.values())), 4)
        results['lrp_fn'] = np.around(100 * np.mean(list(lrp_fn_class.values())), 4)

        self._logger.info(
            "Evaluation results for {}: \n".format('LRP-' + iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        return results


def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results.append(result)
    return results


def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, kpt_oks_sigmas=None):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)
        # When evaluating mask AP, if the results contain bbox, cocoapi will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in coco_results:
            c.pop("bbox", None)

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)

    if iou_type == "keypoints":
        # Use the COCO default keypoint OKS sigmas unless overrides are specified
        if kpt_oks_sigmas:
            assert hasattr(coco_eval.params, "kpt_oks_sigmas"), "pycocotools is too old!"
            coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)
        # COCOAPI requires every detection and every gt to have keypoints, so
        # we just take the first entry from both
        num_keypoints_dt = len(coco_results[0]["keypoints"]) // 3
        num_keypoints_gt = len(next(iter(coco_gt.anns.values()))["keypoints"]) // 3
        num_keypoints_oks = len(coco_eval.params.kpt_oks_sigmas)
        assert num_keypoints_oks == num_keypoints_dt == num_keypoints_gt, (
            f"[LRPEvaluator] Prediction contain {num_keypoints_dt} keypoints. "
            f"Ground truth contains {num_keypoints_gt} keypoints. "
            f"The length of cfg.TEST.KEYPOINT_OKS_SIGMAS is {num_keypoints_oks}. "
            "They have to agree with each other. For meaning of OKS, please refer to "
            "http://cocodataset.org/#keypoints-eval."
        )

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval
