import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict, defaultdict
from pycocotools import coco
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
from .coco_evaluation import instances_to_coco_json 
from pycocotools import mask as maskUtils
from .f_boundary import db_eval_boundary
from .evaluator import DatasetEvaluator


class TPMQScoreEvaluator(DatasetEvaluator):
    ''' 
    This evaluator computes the F1-score at the boundary of true positive detected images with the set of ground truths
    '''
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
        self._tasks = ('segm',)
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
        self._do_evaluation = "annotations" in self._coco_api.dataset


    def reset(self):
        self._predictions = []

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

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)


    def evaluate(self):
        ## Gather up all predictions
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

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

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results['tpmq-' + task] = res


    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Use the COCOeval matches to derive tp, fp, fn => precision, recall for each image/category
        Next, for each (class, IoU threshold, object size), aggregate the F1-score (average them).

        Derive the desired score numbers from COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        # Final metric
        metrics = ['TPMQ(IoU=0.5)']

        gt_masks = coco_eval._gts
        dt_masks = coco_eval._dts
        # Return nans if there are no predictions!
        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}
        
        iou = defaultdict(list)
        # Calculate precision recall for each class, area, and iou threshold
        for evalImg in coco_eval.evalImgs:
            # Skip if maxDets is not the best value, or if evalImg is None (there was no detection)
            if evalImg is None:
                continue
            if evalImg['maxDet'] != coco_eval.params.maxDets[-1]:
                continue
            imgId = evalImg['image_id']
            catId = evalImg['category_id']
            gt_mask = gt_masks[imgId,catId]
            dt_mask = dt_masks[imgId,catId]
            cate_class, area, dtmatches, gtmatches, dtIg, gtIds, gtIg = [evalImg[x] for x in ['category_id', 'aRng', 'dtMatches', 'gtMatches','dtIgnore', 'gtIds', 'gtIgnore']]
            area = tuple(area)
            if len(gtIds) == 0: # no matches at any iou thresh
                continue
            iou_perimg = 0
            cnt = 0
            for dtid,dtm in enumerate(dtmatches[0]): # 0th index is iou thr 0.5
                if dtIg[0][dtid]:
                    continue
                for gtm in gt_mask:
                    if gtm['id'] == dtm: 
                        gt_bin_mask = maskUtils.decode(gtm['segmentation'])
                        dt_bin_mask = maskUtils.decode(dt_mask[dtid]['segmentation'])
                        iou_calc = db_eval_boundary(dt_bin_mask, gt_bin_mask, bound_th=1.0)
                        iou_perimg += iou_calc
                        cnt +=1
            if cnt==0:
                pass
            else:
                iou[area].append(iou_perimg/cnt)

        area_to_key = [(tuple(area), st) for area, st in zip(coco_eval.params.areaRng,  \
                                ['all', 'small', 'medium', 'large'])]
        area_to_key = dict(area_to_key)

        # Calculate individual f1-scores
        results = {
            'small': [],   # small, med, large f1 at all thresholds
            'medium': [],
            'large':[],
            'all': []
        }
        for key, val in iou.items():
            areakey = key
            results[area_to_key[areakey]].append(val)

        # Average these quantities
        for k, v in results.items():
            results[k] = np.around(np.mean(v), 4)

        # Final dict of results
        results = {f'TPMQ_{k}': v for k, v in results.items()}
        self._logger.info(
            "Evaluation results for {}: \n".format('TPMQ-score') + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")
        return results

def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, kpt_oks_sigmas=None):
    """
    Evaluate the coco results using COCOEval API.
    We will use the `evalImgs` datastructure to calculate precision/recall per image
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
    else:
        raise ValueError(f"iou_type {iou_type} not supported")

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)

    coco_eval.evaluate()

    return coco_eval
