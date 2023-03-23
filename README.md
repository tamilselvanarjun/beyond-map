# Beyond mAP: Towards better evaluation of instance segmentation

This is the repository containing the source code for the _CVPR 2023_ [paper](https://arxiv.org/pdf/2207.01614.pdf) ✨. 

## Installation instructions
Go to `cocoapi/PythonAPI` directory and install pycocotools as follows:
```
python setup.py build_ext --inplace
```

Next, copy the files from `detectron2/evaluation` into your detectron2 installation at `detectron2/detectron2/evaluation/`. Your detectron2 directory should look like this:
```
detectron2
|-- README.md
|-- configs
|   `-- ...
|-- datasets
|   `-- ...
|-- demo
|   `-- ...
|-- detectron2
|   |-- evaluation
|   |   |-- __init__.py
|   |   |-- connectiveness_evaluator.py
|   |   |-- f1score_evaluator.py
|   |   |-- f_boundary.py
|   |   |-- lrp_evaluator.py
|   |   |-- namingerror_evaluator.py
|   |   |-- tpmqscore_evaluator.py
|   |   `-- <other existing evaluator files>
|   `-- ...
`-- ...
```

Next, replace the following line in `build_evaluator` function in your `train_net.py` script:

```
if evaluator_type in ["coco", "coco_panoptic_seg"]:
    evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
```

to
```
if evaluator_type in ["coco", "coco_panoptic_seg"]:
    evaluator_list.append(F1ScoreEvaluator(dataset_name, cfg, True, output_folder))
      evaluator_list.append(NamingErrorEvaluator(dataset_name, cfg, True, output_folder))
      evaluator_list.append(ConnectivenessEvaluator(dataset_name, cfg, True, output_folder))
      evaluator_list.append(LRPEvaluator(dataset_name, cfg, True, output_folder))
      evaluator_list.append(TPMQScoreEvaluator(dataset_name, cfg, True, output_folder))
      evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
```

Finally, run your code! 

## Bibtex

If you find our work useful for your research, please cite:
```
@article{jena2023beyond,
  author    = {Jena, Rohit and Zhornyak, Lukas and Doiphode, Nehal and Chaudhari, Pratik and Buch, Vivek and Gee, James and Shi, Jianbo},
  title     = {Beyond mAP: Towards better evaluation of instance segmentation},
  journal   = {CVPR},
  year      = {2023},
}
```

## To-do
I'm listing out the to-do items that I feel are important, feel free to convey your suggestions or feedback through the Issue Tracker, or email me directly. 

- [ ] Add Cython modules for Naming Error
- [ ] Currently, all modules use their own COCOEval, resulting in redundant evaluation. Idea is to collate all the evaluators (except `ConnectivenessEvaluator` and `NamingErrorEvaluator`) to use the same COCOEval object.
- [ ] Add mmdet implementations of all evaluators.
