# Fruit Detector (standalone)

Detectron2 Mask R-CNN training / evaluation / inference pipeline, extracted from
[aoc_fruit_detector](https://github.com/LCAS/aoc_fruit_detector) with all ROS2
dependencies removed. Use this as a starting point to train your own instance
segmentation model (e.g. ripe / unripe / flower) on a custom COCO-format dataset.

## Setup

```bash
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Dataset

Organize your data as COCO instance-segmentation annotations:

```
data/
  train/images/...       data/train/annotations/instances.json
  val/images/...          data/val/annotations/instances.json
  test/images/...         data/test/annotations/instances.json
```

Each `instances.json` needs a `categories` list, e.g.:

```json
"categories": [
  {"id": 1, "name": "unripe", "supercategory": "fruit"},
  {"id": 2, "name": "ripe", "supercategory": "fruit"},
  {"id": 3, "name": "flower", "supercategory": "fruit"}
]
```

Set `training.number_of_classes` in `config/params.yaml` to match (3 in this
example).

## Configuration

Edit `config/params.yaml`:
- `training.number_of_classes` - number of classes in your dataset
- `training.epochs` - SOLVER.MAX_ITER
- `training.learning_rate`, `training.optimizer` (`SGD` or `ADAM`)
- `files.config_file` - Detectron2 model zoo config (architecture/backbone)
- `files.pretrained_model_file` - optional checkpoint to fine-tune from

Any path starting with `./` is resolved relative to this project's root.

## Train + evaluate

```bash
python train.py
```

This registers the train/val/test COCO datasets with Detectron2, writes
dataset/metadata catalog pickle files to `data/dataset_catalogs/` (required
by `predict.py`), trains the model (checkpoints/logs go to
`directories.training_output_dir`), and runs COCOEvaluator on the test set.

Useful flags:
- `--resume` - resume from the last checkpoint in the output dir
- `--skip-training` - only run evaluation (e.g. after training elsewhere)

After training, point `files.model_file` at the checkpoint you want to use
for inference (e.g. `model_final.pth` in the training output dir).

## Inference

```bash
python predict.py
```

Runs the model over every image in `directories.test_image_dir` (matching the
`rgb` filename pattern), saving overlay images to
`directories.prediction_output_dir/predicted_images/` and per-image COCO-format
prediction JSON to `directories.prediction_json_dir`.

`predict.py` requires `data/dataset_catalogs/*.pkl`, which `train.py` generates
during dataset registration - run `train.py` (even with `--skip-training`) at
least once before `predict.py`.

## Layout

```
config/params.yaml          configuration
train.py / predict.py        entry points
fruit_detector/
  detectron_trainer/          DetectronTrainer + AOCTrainer (training loop, augmentations)
  detectron_predictor/        DetectronPredictor, visualizer, COCO JSON writer
  learner_trainer/             abstract base class
  learner_predictor/           abstract base class
  utils/                       config loading, optional asset download helper
data/                          datasets, catalogs, training/prediction output
model/                         trained model weights
```
