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
from detectron2.evaluation.coco_evaluation import instances_to_coco_json 

from detectron2.evaluation.evaluator import DatasetEvaluator


class ConnectivenessEvaluator(DatasetEvaluator):
    ''' 
    This evaluator computes the Duplicate Confusion of detected images with the set of itself

    This is to capture the amount of overlap between predictions by the model, used
    as an indicator of how confused it is
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
        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
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
        # Gather up all the predictions, then we will intercept the matched images from the 
        # `COCOEval` class 
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
            self._results['cc-' + task] = res


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
        metrics = ["cc_all", "cc_0.5", "cc_0.75"]
        
        # Return nans if there are no predictions!
        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: 0. for metric in metrics}
            
        # get ious with dt
        import copy
        results_coco = copy.deepcopy(coco_eval.cocoDt)
        self_coco_eval = COCOeval(results_coco, results_coco)
        self_coco_eval.evaluate()
        
        dt_iou_submat = defaultdict(list)
        for evalImg in self_coco_eval.evalImgs:
            # Skip if maxDets is not the best value, or if evalImg is None (there was no detection), or if area is not all
            if evalImg is None:
                continue
            if evalImg['maxDet'] != self_coco_eval.params.maxDets[-1]:
                continue
            if evalImg['aRng'][0] != 0 or evalImg['aRng'][1] != 10000000000.:
                continue
            cate_class, area, dt, dtIg, dtScores, dtIds, gtIds, gtIg, ious = [evalImg[x] for x in ['category_id', 'aRng', 'dtMatches', 'dtIgnore', 'dtScores', 'dtIds', 'gtIds', 'gtIgnore', 'ious']]
            num_gt = len(gtIds) - np.sum(gtIg)
            if num_gt == 0:
                continue
            dt_iou_submat[evalImg['image_id']].append({'category_id': cate_class, 'dtIds': dtIds, 'gtIds': gtIds, 'dtScores': dtScores, 'ious': ious})
                
        counting_confusion = []
        from tqdm import tqdm
        for img_idx in tqdm(dt_iou_submat):
            counting_confusion.append([])
            scores = [dis['dtScores'] for dis in dt_iou_submat[img_idx]]
            self_ious = [dis['ious'] for dis in dt_iou_submat[img_idx]]
            
            for iou_thresh in list(np.linspace(0.05, 0.95, 10)) + [0.5]:
                class_confusions = []
                for class_idx in range(len(scores)):
                    connections = scores[class_idx] * (self_ious[class_idx] > iou_thresh)
                    connections = np.minimum(connections, connections.T)
                    
                    old_connections = np.zeros_like(connections)
                    while np.abs(connections - old_connections).sum() > 0.1:
                        old_connections = connections
                        stacked_conns = np.tile(connections, [len(connections), 1, 1])
                        connections = np.minimum(stacked_conns, stacked_conns.T).max(1)
                    
                    confusions = [_calc_counting_confusion(scores[class_idx], connections, ct) for ct in np.linspace(0.05, 0.95, 10)]
                    class_confusions.append(confusions)
                class_confusions = np.array(class_confusions).sum(0)
                class_confusions[:, 1] = np.maximum(class_confusions[:, 1], np.ones(len(class_confusions)))
                counting_confusion[-1].append((class_confusions[:, 0] / class_confusions[:, 1]).mean())
        
        counting_confusion = np.array(counting_confusion)
        F = 1000
        cc_50 = np.around(F * counting_confusion[:, -1].mean(), 2)
        cc_75 = np.around(F * counting_confusion[:, 7].mean(), 2)
        cc_all = np.around(F * counting_confusion[:, :-1].mean(), 2)
        results = {'cc_all': cc_all, 'cc_0.5': cc_50, 'cc_0.75': cc_75}
        
        self._logger.info(
            "Evaluation results for {}: \n".format('Counting Confusion') + create_small_table(results)
        )
        print(results)
        
        return results
    
def _calc_counting_confusion(scores, connectivity, conf_thresh):
    valid_preds = np.arange(len(scores))[scores > conf_thresh]
    error = np.zeros((len(valid_preds), len(valid_preds)))
    n_valid = len(valid_preds)
    for i in range(n_valid):
        for j in range(n_valid):
            if i == j:
                continue
            x, y = valid_preds[i], valid_preds[j]
            error[i, j] = scores[y] / scores[x] * connectivity[x, y]
    return error.sum(), n_valid
         

def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, kpt_oks_sigmas=None):
    """
    Evaluate the coco results using COCOEval API.
    We will use the `evalImgs` datastructure to calculate precision/recall per image
    """
    assert len(coco_results) > 0

    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)
        for c in coco_results:
            c.pop("bbox", None)
    else:
        raise ValueError(f"iou_type {iou_type} not supported")

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)

    coco_eval.evaluate()
    return coco_eval
