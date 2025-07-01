"""
File: main_code/constants.py
Author: Ali Reza (ARO) Omrani
Email: ali.omrani@example.com
Date: 1 July 2025

Description:
-----------
This module defines constants used throughout the Vocal Stereotypy Measurement project. These constants include paths for dataset
directories, model configurations, and parameters for audio processing. These constants are used to ensure consistency and reproducibility
across different parts of the project, such as training, testing, and data processing.

Note:
----
This file is part of the Vocal Stereotypy Measurement project and is intended to be used as a module.
"""

# Constants for the project
SEED_NUMBER = 1  # Seed number for reproducibility
VALIDATION_BATCH_SIZE = 1024  # Batch size for validation set

# Constants for model and dataset
TIMM_MODEL_NAME = "xcit_tiny_12_p8_224"  # Pretrained model name from TIMM library
CLASS_NUMBER = 2  # Number of classes in the dataset
INPUT_SHAPE = (
    2,
    64,
    44,
)  # Input shape for the model, typically (channels, height, width)

# Constants for data processing
N_FFT = 1024  # Number of FFT components for spectrogram
HOP_LENGTH = 256  # Hop length for spectrogram
N_MELS = 128  # Number of Mel frequency bins for spectrogram
N_MFCC = 128  # Number of MFCC features to extract
MEL_SCALE = "htk"  # Mel scale type for MFCC extraction

# Constants for audio processing
TARGET_SAMPLE_RATE = 16000  # Target sample rate for audio files
