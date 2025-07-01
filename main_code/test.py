"""
File: main_code/test.py
Author: Ali Reza (Aro) Omrani
email: omrani.alireza95@gmail.com
Date: 9th July 2024

Description:
-----------
This file contains the main testing script for evaluating a pretrained model on audio data.
It loads audio files, processes them, and uses a pretrained model to classify segments of the audio as vocal stereotypy or not.
This script includes argument parsing for dataset directory, target sample rate, best weight directory, and classification threshold.

Functions:
---------
- arg_parsing: Parses command-line arguments for the script.
- init_info: Initializes the device and sets random seeds for reproducibility.
- test: Evaluates the model on the provided data.


Requirements:
------------
- numpy: For numerical operations.
- torch: For building and running neural networks.
- torchaudio: For audio processing and transformations.
"""

import os

import numpy as np
import torch
from torch import Tensor, nn
import torchaudio as ta
from torchaudio import transforms as tt

from main_code import (
    Pretrained_Models,
    arg_parsing,
    mix_down_if_necessary,
    resample_if_necessary,
    set_seed_and_get_device,
)
from constants import (
    CLASS_NUMBER,
    HOP_LENGTH,
    INPUT_SHAPE,
    MEL_SCALE,
    N_FFT,
    N_MELS,
    N_MFCC,
    SEED_NUMBER,
    TARGET_SAMPLE_RATE,
    TIMM_MODEL_NAME,
)


def test(model: nn.Module, data: Tensor, device: str) -> Tensor:
    """
    Evaluates the model on the provided data.

    Parameters:
    ----------
        - model (nn.Module): The pretrained model to be evaluated.
        - data (Tensor): The input data for the model, expected to be a tensor of shape
        (batch_size, channels, time).
        - device (str): The device on which the model and data are located, either "cpu" or "cuda".

    Returns:
    -------
        - pred (Tensor): The model's predictions, a tensor of shape (batch_size, 1) with sigmoid
        activation applied.
    """
    model.eval()

    with torch.no_grad():
        # Ensure the model is in evaluation mode and no gradients are computed
        logits = model(data.to(device))
        pred = torch.sigmoid(logits)

    return pred


def main() -> None:
    """
    Main function for testing the model on audio data. It initializes the device, parses command-line
    arguments, loads audio files, processes them, and uses a pretrained model to classify segments of
    the audio as vocal stereotypy or not. It prints the number and percentage of audio segments
    classified as vocal stereotypy.

    Parameters:
    ----------
        - None

    Returns:
    -------
        - None
    """
    # Dataset Info
    device = set_seed_and_get_device(
        seed=SEED_NUMBER
    )  # Initialize device and set random seed
    args = arg_parsing(mode="test")  # Parse command-line arguments
    files_list = os.listdir(args.dataset_dir)
    for file in files_list:
        audio_file, sample_rate = ta.load(os.path.join(args.dataset_dir, file))
        audio_file_seconds = []

        # Feature extractor
        mel_spectogram_transform = tt.MelSpectrogram(
            args.target_sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        mfcc_transform = tt.MFCC(
            args.target_sample_rate,
            n_mfcc=N_MFCC,
            melkwargs={
                "n_fft": N_FFT,
                "n_mels": N_MELS,
                "hop_length": HOP_LENGTH,
                "mel_scale": MEL_SCALE,
            },
        )
        # Mixing down the audio if it has multiple channels
        if audio_file.shape[0] > 1:
            audio_file = mix_down_if_necessary(audio_file)

        # Resampling the audio if necessary
        if sample_rate != TARGET_SAMPLE_RATE:
            audio_file, sample_rate = resample_if_necessary(
                audio_file, sample_rate, TARGET_SAMPLE_RATE
            )
        for i in range(int(audio_file.shape[1] / sample_rate)):
            if (i + 1) <= int(audio_file.shape[1] / sample_rate):
                sample = audio_file[:, i * sample_rate : (i + 1) * sample_rate]
                sample_mel_spectogram = mel_spectogram_transform(sample)
                sample__mfcc = mfcc_transform(sample)
                audio_file_seconds.append(
                    torch.concat([sample_mel_spectogram, sample__mfcc])
                )

        # Implementing the model
        model = Pretrained_Models(
            model_name=TIMM_MODEL_NAME, class_num=CLASS_NUMBER, input_size=INPUT_SHAPE
        ).get_model()
        model = model.to(device)

        model.head = nn.Linear(model.head.in_features, 1).to(device)
        model.load_state_dict(torch.load(args.best_weight_dir)["model_state_dict"])
        audio_file_seconds = torch.from_numpy(np.array(audio_file_seconds))
        results = np.array(test(model, audio_file_seconds, device).cpu())
        results = (results > args.threshold).astype(int)
        print(
            f"{np.count_nonzero(results==1)} / {len(results)} seconds of the total audio was predicted"
            f" as vocal stereotypy.\nIn another word, {np.count_nonzero(results==1) / len(results)*100 :.2f} "
            "percentage of the file has been recognized as vocal stereotypy"
        )


if __name__ == "__main__":
    main()
