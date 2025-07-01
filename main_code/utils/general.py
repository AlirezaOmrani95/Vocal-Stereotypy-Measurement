"""
File: main_code/utils/general.py
Author: Ali Reza (Aro) Omrani
email: omrani.alireza95@gmail.com
Date: 1st July 2025

Description:
-----------
This module provides utility functions for saving and loading model checkpoints,
setting random seeds for reproducibility, and determining the device for model computation.

Functions:
---------
- save_checkpoint: Saves the model and optimizer state to a checkpoint file.
- load_checkpoint: Loads the model and optimizer state from a checkpoint file.
- set_seed_and_get_device: Sets seeds for reproducibility and returns the device string for model
computation.

Requirements:
------------
- numpy: For numerical operations.
- torch: For building and running neural networks.

Note:
----
This module is designed to be imported in other parts of the project, providing access to
essential functions for model management and reproducibility.
"""

import random as rnd
from typing import Literal, Tuple


import numpy as np
from torch import nn, optim
import torch


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    save_path: str,
    epoch: int,
    verbose: bool = False,
) -> None:
    """
    Saves the model and optimizer state to a checkpoint file.

    Parameters:
    ----------
         - model (nn.Module): The model to save.
         - optimizer (optim.Optimizer): The optimizer to save.
         - save_path (str): The path where the checkpoint will be saved.
         - epoch (int): The current epoch number.
         - verbose (bool): If True, prints a confirmation message after saving. Default is False.

    Returns:
    -------
        - None

    Usage:
    >>> save_checkpoint(model, optimizer, "checkpoint.pth", epoch=5, verbose=True)

    Note:
    ----
        If a file already exists at `save_path`, it will be overwritten.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        },
        save_path,
    )
    if verbose:
        print(f"Checkpoint saved to {save_path} at epoch {epoch}.")


def load_checkpoint(
    model: nn.Module, optimizer: optim.Optimizer, load_path: str
) -> Tuple[nn.Module, optim.Optimizer, int]:
    """
    Loads the model and optimizer state from a checkpoint file.

    Parameters:
    ----------
        - model (nn.Module): The model to load the state into.
        - optimizer (optim.Optimizer): The optimizer to load the state into.
        - load_path (str): The path from which the checkpoint will be loaded.

    Returns:
    -------
        - Tuple[nn.Module, optim.Optimizer, int]: A tuple containing:
            - The model with loaded state.
            - The optimizer with loaded state.
            - The epoch number from the checkpoint.

    Raises:
    ------
        - KeyError: If the checkpoint is missing required keys.
        - RuntimeError: If the checkpoint is incompatible with the model or optimizer.

    Usage:
    >>> model, optimizer, epoch = load_checkpoint(model, optimizer, "checkpoint.pth")

    Note:
    ----
        The optimizer must be constructed with the same parameters and on the same device as when the checkpoint was saved.
        Otherwise, loading the optimizer state may result in errors or unexpected behavior.
    """
    try:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint file not found: {load_path}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading checkpoint: {e}")

    return model, optimizer, epoch


def set_seed_and_get_device(seed_num: int) -> Literal["cpu", "cuda"]:
    """
    Sets seeds for PyTorch, NumPy, and Python's random module for reproducibility, and returns the device string for convenience.

    This function ensures reproducible results by setting the random seeds for PyTorch, NumPy, and Python's built-in `random` module.
    It also determines whether to use "cuda" or "cpu" for model computation and returns the device string.

    Parameters
    ----------
        - seed_num (int): The seed number to use for all random number generators.

    Returns
    -------
        - device (Literal["cpu", "cuda"]): The string "cpu" or "cuda" indicating the device for model computation.

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed_num)
    if device == "cuda":
        # Set the seed for all GPUs
        torch.cuda.manual_seed_all(seed_num)
    np.random.seed(seed_num)
    rnd.seed(seed_num)

    return device
