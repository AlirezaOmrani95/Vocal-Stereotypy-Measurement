"""
File: main_code/pretrained_models.py
Author: Ali Reza (Aro) Omrani
email: omrani.alireza95@gmail.com
Date: 9th July 2024

Description:
-----------
This file contains the `Pretrained_Models` class, which is designed to retrieve
pretrained models from the `timm` library.

Classes:
-----------
Pretrained_Models:
    A class to handle the retrieval of pretrained models with specified configurations.

Requirements:
------------
- timm: A library for accessing pretrained models.
- torch: A library for building and training neural networks.
"""

from typing import Tuple

import timm
import torch.nn as nn


class Pretrained_Models:
    """
    A class to handle the retrieval of pretrained models from the timm library.
    This class allows you to specify the model name, number of output classes,
    and input size for the model.

    Attributes:
    ----------
    model_name : str
        The name of the pretrained model to be used.
    class_num : int
        The number of output classes for the model.
    input_size : tuple(int, int, int)
        The input size of the model in the format (channels, height, width).

    Methods:
    -------
    get_model() -> nn.Module:
        Gets a pretrained model with the specified configuration.

    Example:
    --------
    >>> model = Pretrained_Models("xcit_tiny_24_p16_384", 1, (2, 64, 44)).get_model()
    >>> print(model.patch_embed.proj[2][0].weight[0, 0, 0, 0])
    Output: tensor value
    """

    def __init__(
        self, model_name: str, class_num: int, input_size: Tuple[int, int, int]
    ) -> None:
        """
        Initializes the Pretrained_Models class.

        parameters:
        ----------
        model_name : str
            The name of the pretrained model to be used.
        class_num : int
            The number of output classes for the model.
        input_size : tuple(int, int, int)
            The input size of the model in the format (channels, height, width).

        Returns:
        -------
            - None
        """
        self.model_name = model_name
        self.class_num = class_num
        self.input_size = input_size

    def get_model(self) -> nn.Module:
        """
        Get a pretrained model with the specified configuration.

        parameters:
        ----------
            - None

        Returns:
        -------
            - model : nn.Module
                A pretrained model with the specified configuration.
        """
        cfg_file = timm.get_pretrained_cfg(self.model_name)
        cfg_file.input_size = self.input_size
        model = timm.create_model(
            self.model_name, pretrained_cfg=cfg_file, pretrained=True
        )
        model.head = nn.Linear(model.head.in_features, self.class_num)

        return model
