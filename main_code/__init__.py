"""
File: main_code/__init__.py
Author: Ali Reza (ARO) Omrani
Email: ali.omrani@example.com
Date: 1 July 2025

Description:
-----------
This module initializes the main code package for the Vocal Stereotypy Measurement project. It imports necessary functions and classes
from various modules within the package, making them available for use in other parts of the project. This file serves as the entry point
for the main code package, allowing for easy access to functionalities such as audio dataset handling, model training and evaluation,
and command-line argument parsing.


Note:
----
This file is part of the Vocal Stereotypy Measurement project and is intended to be used as a module.
"""

from cli import arg_parsing
from pretrained_models import Pretrained_Models
from utils import (
    AudioDataset,
    create_data_loader,
    get_session_indices,
    load_checkpoint,
    mix_down_if_necessary,
    resample_if_necessary,
    save_checkpoint,
    set_seed_and_get_device,
    train_valid_separation,
)

__all__ = [
    "AudioDataset",
    "Pretrained_Models",
    "arg_parsing",
    "create_data_loader",
    "get_session_indices",
    "load_checkpoint",
    "mix_down_if_necessary",
    "resample_if_necessary",
    "save_checkpoint",
    "set_seed_and_get_device",
    "train_valid_separation",
]
