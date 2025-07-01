"""
File: main_code/utils/__init__.py
Author: Ali Reza (Aro) Omrani
email: omrani.alireza95@gmail.com
Date: 1st July 2025

Description:
-----------
This module initializes the utilities for the Vocal Stereotypy Measurement project. It imports
various utility functions and classes related to audio processing, data loading, and model
checkpointing.


Note:
This module is designed to be imported in other parts of the project, providing access to
essential functions and classes for audio dataset handling, data loading, and model management.
"""

from audio import (
    AudioDataset,
    mix_down_if_necessary,
    resample_if_necessary,
)

from data_utils import (
    create_data_loader,
    get_session_indices,
    train_valid_separation,
)

from general import (
    load_checkpoint,
    save_checkpoint,
    set_seed_and_get_device,
)

__all__ = [
    "AudioDataset",
    "create_data_loader",
    "get_session_indices",
    "load_checkpoint",
    "mix_down_if_necessary",
    "resample_if_necessary",
    "save_checkpoint",
    "set_seed_and_get_device",
    "train_valid_separation",
]
