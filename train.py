#!/usr/bin/env python3
"""
Train and evaluate a Detectron2 Mask R-CNN model on a custom COCO-format dataset.

Usage:
    python train.py [--config CONFIG_PATH] [--resume] [--skip-training]

Dataset layout expected (see config/params.yaml):
    data/train/images/        + data/train/annotations/instances.json
    data/val/images/          + data/val/annotations/instances.json
    data/test/images/         + data/test/annotations/instances.json

Each annotations file must be a COCO-format JSON with a "categories" list
matching training.number_of_classes in the config (e.g. ripe, unripe, flower).
"""
import argparse
import logging
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fruit_detector'))

from utils.utils import load_config
from detectron_trainer.detectron_trainer import DetectronTrainer


def main():
    parser = argparse.ArgumentParser(description='Train/evaluate the fruit detection model')
    parser.add_argument('--config', default=None, help='Path to params.yaml (defaults to ./config/params.yaml)')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')
    parser.add_argument('--skip-training', action='store_true', help='Skip training and only run evaluation')
    args = parser.parse_args()

    config_data = load_config(args.config)

    try:
        trainer = DetectronTrainer(config_data)
        aoc_trainer = trainer.train_model(resumeType=args.resume, skipTraining=args.skip_training)
        trainer.evaluate_model(aoc_trainer.model)
    except Exception as e:
        logging.error(e)
        print(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
