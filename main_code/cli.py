"""
File: main_code/cli.py
Author: Ali Reza (ARO) Omrani
Email: omrani.alireza95@gmail.com
Date: 24 June 2025

Description
-----------
This file contains the command-line interface (CLI) for the CV Checker application.
It provides functionality to parse command-line arguments for configuring the application,
including the learning mode and model type to be used for audio classification tasks.

The main function in this file is `arg_parsing()`, which is responsible for:
1. Getting dataset and annotation directories.
2. Setting the batch size, number of epochs, learning rate, and number of classes.
3. Specifying the pretrained model directory.


Arguments:
---------
- `--logs_dir`: Directory to save logs.
- `--dataset_dir`: Directory to read the dataset from (for both train and test).
- `--annotation_dir`: Location of the input CSV file containing annotations.
- `--batch_size`: Size of the batches for training.
- `--epoch_number`: Number of epochs for training.
- `--lr_rate`: Learning rate for the optimizer.
- `--num_classes`: Number of classes in the dataset.
- `--pretrained_dir`: Directory of the pretrained model weights.
- `--target_sample_rate`: Target sample rate for audio files (for test).
- `--best_weight_dir`: Directory of the best model weights for test.
- `--threshold`: Threshold for classification (for test).


Note:
----
This script is designed to be run from the command line and will parse the provided arguments to configure the application accordingly.
"""

import argparse
from typing import Literal, Optional


def arg_parsing(
    timestamp: Optional[str] = None, mode: Literal["train", "test"] = "train"
) -> argparse.Namespace:
    """
    Parse command-line arguments for the CV Checker application.

    Parameters:
        - timestamp (Optional[str]): A timestamp for logging purposes, used in the logs directory.
        - mode (Literal["train", "test"]): The mode of operation, either 'train' or 'test'.

    returns:
        - argparse.Namespace: Parsed command-line arguments containing various configurations
        such as dataset directory, logs directory, annotation file location, batch size,
        number of epochs, learning rate, number of classes, pretrained model directory,
        and threshold for classification.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        default=f"./dataset/{mode}/",
        help="The address to read data from",
    )
    if mode == "train":
        parser.add_argument(
            "--logs_dir",
            default=f"runs/BP_{timestamp}",
            help="The address that logs will be saved",
        )
        parser.add_argument(
            "--annotation_dir",
            default="./Annotation_file.csv",
            help="The location of input CSV file",
        )
        parser.add_argument("--batch_size", default=128)
        parser.add_argument("--epoch_number", default=25)
        parser.add_argument("--num_classes", default=2)
        parser.add_argument(
            "--pretrained_dir",
            default="./weights/pretrained weight/pretrained weight",
            help="The location of the pretrained weight file",
        )
    elif mode == "test":
        parser.add_argument(
            "--best_weight_dir",
            default="./weights/best weight/best_weight",
            help="The location of the Best weight file",
        )
        parser.add_argument(
            "--threshold",
            default=0.5,
            type=float,
            help="The threshold for the classifier",
        )
    return parser.parse_args()
