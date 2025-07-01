"""
File: main_code/utils/audio.py
Author: Ali Reza (Aro) Omrani
email: omrani.alireza95@gmail.com
Date: 1st July 2025

Description:
-----------
This module defines a custom dataset class for loading audio data from a specified directory.
It includes methods for retrieving audio samples, their labels, and class counts.

Classes:
-------
- AudioDataset: A custom dataset class for loading audio data, including Mel spectrograms and MFCCs.
  - Methods:
    - __len__: Returns the number of samples in the dataset.
    - __getitem__: Returns a tuple containing the audio input and its corresponding label for a given index.
    - _get_class_num_: Returns the count of each class in the dataset.
    - _get_audio_sample_label_: Returns the label for the audio sample at a specified index.

Functions:
---------
- resample_if_necessary: Resamples the audio file to a target sample rate if it differs from the original sample rate.
- mix_down_if_necessary: Reduces the number of channels in the audio file to 1 by averaging across channels.

Requirements:
-----------
- numpy: For numerical operations.
- pandas: For handling annotations in DataFrame format.
- torch: For building and running neural networks.
- torchaudio: For audio processing and transformations.

Note:
----
This module is designed to be imported in other parts of the project, providing access to
essential functions and classes for audio dataset handling and processing.
"""

from ast import Tuple
import os
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchaudio import transforms as tt


class AudioDataset(Dataset):
    """
    A custom dataset class for loading audio data from a specified directory.

    Parameters:
    ----------
        - root (str): The root directory where the audio files are stored.
        - annotations (pd.DataFrame): A DataFrame containing the annotations for the audio files.
        - device (str): The device to use for computations, either "cpu" or "cuda". Default is "cpu".

    Methods:
    -------
        - __len__(): Returns the number of samples in the dataset.
        - __getitem__(index): Returns a tuple containing the audio input and its corresponding label for the given index.
        - _get_class_num_(): Returns the count of each class in the dataset.
        - _get_audio_sample_label_(index): Returns the label for the audio sample at the specified index.

    Returns:
    -------
        - input_ (torch.Tensor): A tensor containing the concatenated Mel spectrogram and MFCC features of the audio sample.
        - label (int): The label of the audio sample, where 0 represents one class and 1 represents another class.

    Usage:
    >>> dataset = AudioDataset(root="path/to/audio/files", annotations=pd.read_csv("path/to/annotations.csv"), device="cuda")

    Note:
    ----
        The audio files are expected to be stored in a specific directory structure, with Mel spectrograms and MFCCs saved as `.npy` files.
    """

    def __init__(
        self,
        root: str,
        annotations: pd.DataFrame,
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> None:
        super().__init__()
        """
        Initializes the AudioDataset with the root directory, annotations, and device.
        
        parameters:
        ----------
            - root (str): The root directory where the audio files are stored.
            - annotations (pd.DataFrame): A DataFrame containing the annotations for the audio files.
            - device (Literal["cpu", "cuda"]): The device to use for computations, either "cpu" or 
            "cuda". Default is "cpu".

        Returns:
        -------
            - None
        """
        self.device = device
        self.annotations = annotations
        self.root = root

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Parameters:
        ----------
            - None

        Returns:
        -------
            - int: The number of audio samples in the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Returns a tuple containing the audio input and its corresponding label for the given index.

        Parameters:
        ----------
            - index (int): The index of the audio sample to retrieve.

        Returns:
        -------
            - Tuple[torch.Tensor, int]: A tuple containing:
               - torch.Tensor: a tensor containing the concatenated Mel spectrogram and MFCC
               features of the audio sample,
               - int: the label of the audio sample (0 or 1).
        """
        required_columns = {"name", "time_span", "session", "label"}
        missing_columns = required_columns - set(self.annotations.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns in annotations DataFrame: {missing_columns}"
            )
        folder_name = self.annotations["name"].iloc[index]
        time_span = self.annotations["time_span"].iloc[index]
        session = self.annotations["session"].iloc[index]
        file_name = (
            f"{folder_name.strip()}_{time_span.strip()}_{session.strip()}.npy".replace(
                ":", "-"
            )
        )

        label = self._get_audio_sample_label_(index)

        try:
            mel_file_address = os.path.join(
                self.root, folder_name, "MelSpectogram", file_name
            )
            mel_spectogram_audio = torch.from_numpy(
                np.load(mel_file_address).astype(
                    np.float32
                )  # Convert to float32 for compatibility with PyTorch
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Mel spectrogram file not found: {mel_file_address}"
            )

        try:
            mfcc_file_address = os.path.join(self.root, folder_name, "MFCC", file_name)
            mfcc_audio = torch.from_numpy(
                np.load(mfcc_file_address).astype(
                    np.float32
                )  # Convert to float32 for compatibility with PyTorch
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"MFCC file not found: {mfcc_file_address}")
        assert mel_spectogram_audio.shape == mfcc_audio.shape, (
            f"Shape mismatch in file {file_name}: MelSpectogram shape {mel_spectogram_audio.shape},"
            f" MFCC shape {mfcc_audio.shape}. Shapes must match for concatenation."
            "Please ensure that both Mel spectrogram and MFCC files have the same shape, "
            "and the feature extraction methods are consistent"
        )
        input_ = torch.cat([mel_spectogram_audio, mfcc_audio], dim=0)

        return input_, label

    def _get_class_num_(self) -> np.ndarray:
        """
        Returns the count of each class in the dataset.

        Parameters:
        ----------
            - None
        Returns:
        -------
            - np.ndarray: An array containing the counts of each class, where the first element
            is the count of class 0 and the second element is the count of class 1.
        """
        labels = self.annotations["label"].astype(int)
        classes = np.array([(labels == 0).sum(), (labels == 1).sum()])

        return classes

    def _get_audio_sample_label_(self, index: int) -> int:
        """
        Returns the label for the audio sample at the specified index.

        Parameters:
        ----------
            - index (int): The index of the audio sample.

        Returns:
        -------
            - int: The label of the audio sample, where 0 represents one class and 1 represents
            another class.
        """

        return int(self.annotations["label"].loc[index])


def resample_if_necessary(
    audio_file: torch.Tensor, original_sample_rate: int, target_sample_rate: int
) -> Tuple[torch.Tensor, int]:
    """
    Resamples the audio file to the target sample rate if it differs from the original sample rate.

    Parameters:
    ----------
        - audio_file (torch.Tensor): The input audio file tensor.
        - original_sample_rate (int): The original sample rate of the audio file.
        - target_sample_rate (int): The desired sample rate to resample to.

    Returns:
    -------
        - Tuple[torch.Tensor, int]: A tuple containing:
            - The resampled audio file tensor.
            - The target sample rate.

    Usage:
    >>> audio_file, target_sample_rate = resample_if_necessary(audio_file, 16000, 8000)
    """
    resampler = tt.Resample(original_sample_rate, target_sample_rate)
    return resampler(audio_file), target_sample_rate


def mix_down_if_necessary(audio_file: torch.Tensor) -> torch.Tensor:
    """
    Reduces the number of channels in the audio file to 1 by averaging across channels.

    Parameters:
    ----------
        - audio_file (torch.Tensor): The input audio file tensor, expected to have shape
        (channels, time).
    Returns:
    -------
        - torch.Tensor: The audio file tensor with reduced channels, shape (1, time).

    Usage:
    -----
    >>> audio_file = torch.randn(2, 16000)  # Example stereo audio file

    Note:
    ----
        The number of channels in the audio file is checked before calling this function.
        If the channel number is greater than 1, this function will be called.
    """
    return torch.mean(audio_file, dim=0, keepdim=True)
