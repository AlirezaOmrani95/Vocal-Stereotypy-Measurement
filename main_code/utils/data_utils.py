"""
File: main_code/utils/data_utils.py
Author: Ali Reza (Aro) Omrani
email: omrani.alireza95@gmail.com
Date: 1st July 2025

Description:
-----------
This module provides utility functions for handling datasets, including separating training and
validation sets, creating data loaders, and retrieving session indices from a dataset.

Functions:
---------
- train_valid_separation: Splits the dataset into training and validation sets based on a specified
rate.
- create_data_loader: Creates a DataLoader for a given dataset with specified parameters.
- get_session_indices: Retrieves the start and end indices of each session in the dataset.

Requirements:
------------
- numpy: For numerical operations.
- pandas: For handling annotations in DataFrame format.
- torch: For building and running neural networks.

Note:
----
This module is designed to be imported in other parts of the project, providing access to essential
functions for dataset handling and processing.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def train_valid_separation(
    annotation_path: str, valid_rate: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into training and validation sets based on the provided annotation directory.

    Parameters:
    ----------
       - annotation_path (str): The path to the CSV file containing annotations.
       - valid_rate (float): The proportion of the dataset to be used for validation.

    Returns:
    -------
         - Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            - The first DataFrame contains training annotations.
            - The second DataFrame contains validation annotations.

    Usage:
    -------
    >>> train_annotations, valid_annotations = train_valid_separation("path/to/annotations.csv", 0.1)
    """

    def select_validation_indices(indices, rate) -> np.ndarray:
        """
        Selects a proportion of indices for validation based on the specified rate.

        Parameters:
        ----------
            - indices (list): A list of indices to select from.
            - rate (float): The proportion of indices to select for validation.

        Returns:
        -------
            - np.ndarray: An array of selected indices for validation.
        """
        num_samples = max(
            1, int(np.ceil(len(indices) * rate))
        )  # Ensure at least one sample is selected
        return np.random.choice(indices, num_samples, replace=False)

    valid_indices = []
    annotations = pd.read_csv(annotation_path)

    # Get the indices of samples for each class
    ones = annotations.index[annotations["label"] == 1].tolist()
    zeros = annotations.index[annotations["label"] == 0].tolist()

    assert len(ones) > 0, "The dataset must contain at least one sample of class 1."
    assert len(zeros) > 0, "The dataset must contain at least one sample of class 0."

    # Select a proportion of samples from each class for validation
    valid_indices.extend(select_validation_indices(ones, valid_rate))
    valid_indices.extend(select_validation_indices(zeros, valid_rate))

    # Get the remaining indices for training
    train_indices = np.setdiff1d(
        np.arange(len(annotations)), np.array(valid_indices)
    ).tolist()

    assert len(train_indices) > 0, (
        "The training set must contain at least one sample. "
        "Please ensure that the dataset is large enough for the train and validation split."
        " Or please change the valid_rate."
    )
    assert len(valid_indices) > 0, (
        "The validation set must contain at least one sample. "
        "Please ensure that the dataset is large enough for the train and validation split."
        " Or please change the valid_rate."
    )

    # Finalize the training and validation annotations
    train_annotations = annotations.loc[train_indices, :]
    valid_annotations = annotations.loc[valid_indices, :]

    return train_annotations, valid_annotations


def create_data_loader(
    dataset: Dataset,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """
    Creates a DataLoader for the given dataset.

    Parameters:
    ----------
        - dataset (Dataset): The dataset to create a DataLoader for.
        - batch_size (int): The number of samples per batch. Default is 128.
        - shuffle (bool): Whether to shuffle the data at every epoch. Default is True.
        - num_workers (int): How many subprocesses to use for data loading. Default is 0.
        - pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory
        before returning them. Default is False.
        - drop_last (bool): Set to True to drop the last incomplete batch. Default is False.

    Returns:
    -------
        - DataLoader: A DataLoader instance for the dataset.

    Usage:
    -----
    >>> dataset = MyDataset(...)
    >>> data_loader = create_data_loader(dataset, batch_size=32, num_workers=4, pin_memory=True)
    """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def get_session_indices(data: pd.DataFrame, session_list: List[str]) -> Dict[str, str]:
    """
    Retrieves the start and end indices of each session in the dataset.

    Parameters:
    ----------
         - data (pd.DataFrame): The DataFrame containing the dataset.
         - session_list (list): A list of unique session identifiers.

    Returns:
    -------
        - session_annotation (dict): A dictionary mapping each session identifier to its start
        and end indices in the DataFrame.


    Usage:
    -----
    >>> data = pd.read_csv("path/to/data.csv")
    >>> session_list = ["session1", "session2"]
    >>> session_indices = get_session_indices(data, session_list)
    """
    session_annotation = {}

    for session in session_list:
        row_indices = data.index[data["session"] == session].tolist()
        if not row_indices:
            raise ValueError(f"No data found for session: {session}")
        session_annotation[session] = str(row_indices[0]) + "-" + str(row_indices[-1])
    return session_annotation
