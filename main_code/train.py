"""
File: main_code/train.py
Author: Ali Reza (ARO) Omrani
Email: omrani.alireza95@gmail.com
Date: 1 July 2025

Description:
-----------
This file contains the training and validation logic for the CV Checker application.
It includes functions to train the model for several epochs, validate it, and log the results using TensorBoard.
It also handles the setup of the model, data loaders, loss function, optimizer, and metrics.

This script is designed to be run from the command line and will parse the provided arguments to configure the application accordingly.

Arguments:
-----------
- `--logs_dir`: Directory to save logs.
- `--dataset_dir`: Directory to read the dataset from (for both train and test).
- `--annotation_dir`: Location of the input CSV file containing annotations.
- `--batch_size`: Size of the batches for training.
- `--epoch_number`: Number of epochs for training.
- `--lr_rate`: Learning rate for the optimizer.
- `--num_classes`: Number of classes in the dataset.
- `--pretrained_dir`: Directory of the pretrained model weights.
- `--target_sample_rate`: Target sample rate for audio files (for test).

Functions:
---------
- `train`: Main train function to train the model for several epochs.
- `train_single_epoch`: Trains the model for a single epoch and logs the results.
- `validation_single_epoch`: Validates the model for a single epoch and logs the results.

Requirements:
-----------
- numpy: A library for numerical operations.
- tensorboard: A library for logging and visualizing training metrics.
- torch: A library for building and training neural networks.
- torchaudio: A library for audio processing in PyTorch.
- torchmetrics: A library for computing metrics in PyTorch.
- tqdm: A library for progress bars.
"""

from ast import Dict
from datetime import datetime
import os
from typing import List

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryCohenKappa
from tqdm import tqdm

from constants import (
    CLASS_NUMBER,
    INPUT_SHAPE,
    SEED_NUMBER,
    TIMM_MODEL_NAME,
    VALIDATION_BATCH_SIZE,
)
from main_code import (
    AudioDataset as AD,
    Pretrained_Models,
    arg_parsing,
    create_data_loader,
    save_checkpoint,
    set_seed_and_get_device,
    train_valid_separation,
)

from states import (
    MODEL_WEIGHT_PATH,
    TIMESTEPS,
)


def train(
    model: nn.Module,
    train_data_loader: DataLoader,
    valid_data_loader: DataLoader,
    loss_fn: nn.Module,
    metrics: List[nn.Module],
    optimizer: optim.Optimizer,
    device: str,
    epochs: int,
    writer: SummaryWriter,
) -> None:
    """
    Train the model for several epochs. This function iterates over the training dataset, computes
    the loss and metrics, and logs the results using TensorBoard.

    Parameters:
    ----------
        - model (nn.Module): The model to be trained.
        - train_data_loader (DataLoader): DataLoader for the training dataset.
        - valid_data_loader (DataLoader): DataLoader for the validation dataset.
        - loss_fn (nn.Module): Loss function to compute the loss.
        - metrics (List[nn.Module]): List of metric functions to evaluate the model.
        - optimizer (optim.Optimizer): Optimizer for updating model parameters.
        - device (str): Device on which the model and data are located, either "cpu" or "cuda".
        - epochs (int): Number of training epochs.
        - writer (SummaryWriter): TensorBoard writer for logging.

    Returns:
    -------
        - None
    """
    best_kappa = 0
    for epoch in range(epochs):
        model.train(True)
        train_hisotry = train_single_epoch(
            model, train_data_loader, loss_fn, metrics, optimizer, device, epoch, writer
        )
        model.eval()
        valid_history = validation_single_epoch(
            model, valid_data_loader, loss_fn, metrics, device, epoch, writer
        )

        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": train_hisotry["loss"], "Validation": valid_history["loss"]},
            epoch + 1,
        )
        writer.add_scalars(
            "Training vs. Validation Accuracy",
            {"Training": train_hisotry["b_acc"], "Validation": valid_history["b_acc"]},
            epoch + 1,
        )
        writer.add_scalars(
            "Training vs. Validation F1_score",
            {"Training": train_hisotry["b_f1"], "Validation": valid_history["b_f1"]},
            epoch + 1,
        )
        writer.add_scalars(
            "Training vs. Validation Kappa_Cohen",
            {
                "Training": train_hisotry["b_kappa"],
                "Validation": valid_history["b_kappa"],
            },
            epoch + 1,
        )
        if valid_history["b_kappa"] > best_kappa:
            best_kappa = valid_history["b_kappa"]
            if not os.path.exists(MODEL_WEIGHT_PATH):
                os.mkdir(MODEL_WEIGHT_PATH)
            model_path = f"{MODEL_WEIGHT_PATH}/model_{TIMESTEPS}_{epoch}"
            save_checkpoint(model, optimizer, model_path, epoch)
        elif epoch == epochs - 1:
            model_path = f"{MODEL_WEIGHT_PATH}/model_{TIMESTEPS}_{epoch}"
            save_checkpoint(model, optimizer, model_path, epoch)


def train_single_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    metrics: List[nn.Module],
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    writer: SummaryWriter,
) -> Dict[str, float]:
    """
    Train the model for a single epoch. This function iterates over the training dataset, computes
    the loss and metrics, and logs the results using TensorBoard.

    Parameters:
    ----------
        - model (nn.Module): The model to be trained.
        - data_loader (DataLoader): DataLoader for the training dataset.
        - loss_fn (nn.Module): Loss function to compute the loss.
        - metrics (List[nn.Module]): List of metric functions to evaluate the model.
        - optimizer (optim.Optimizer): Optimizer for updating model parameters.
        - device (str): Device on which the model and data are located, either "cpu" or "cuda".
        - epoch (int): Current epoch number.
        - writer (SummaryWriter): TensorBoard writer for logging.

    Returns:
    -------
        - history (Dict[str, float]): A dictionary containing the average loss and metrics for the
        training epoch.
    """
    loss_lst = []
    b_acc_lst = []
    b_f1_lst = []
    b_kappa_lst = []
    with tqdm(data_loader, unit="batch") as t_data_loader:
        for counter, (input_, target_) in enumerate(t_data_loader):
            t_data_loader.set_description(f"Epoch_Train {epoch+1}")
            optimizer.zero_grad()

            input_, target_ = input_.to(device), torch.unsqueeze(target_, -1).to(device)

            pred = model(input_)

            loss = loss_fn(pred, target_.float())
            loss.backward()

            b_acc = metrics[0](torch.sigmoid(pred), target_)
            b_f1 = metrics[1](torch.sigmoid(pred), target_)
            b_kappa = metrics[2](torch.sigmoid(pred), target_)
            optimizer.step()

            loss_lst.append(loss.item())
            b_acc_lst.append(b_acc.item())
            b_f1_lst.append(b_f1.item())
            b_kappa_lst.append(b_kappa.item())

            t_data_loader.set_postfix(
                loss=np.mean(loss_lst),
                binary_accuracy=np.mean(b_acc_lst),
                binary_f1=np.mean(b_f1_lst),
                binary_kappa=np.mean(b_kappa_lst),
            )
            if counter % 50 == 0:
                tbx = epoch * len(data_loader) + counter + 1
                writer.add_scalar("Loss/train", np.mean(loss_lst), tbx)
                writer.add_scalar("Accuracy/train", np.mean(b_acc_lst), tbx)
                writer.add_scalar("F1_score/train", np.mean(b_f1_lst), tbx)
                writer.add_scalar("Kappa_Cohen/train", np.mean(b_kappa_lst), tbx)

    history = {
        "loss": np.mean(loss_lst),
        "b_acc": np.mean(b_acc_lst),
        "b_f1": np.mean(b_f1_lst),
        "b_kappa": np.mean(b_kappa_lst),
    }
    return history


def validation_single_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    metrics: List[nn.Module],
    device: str,
    epoch: int,
    writer: SummaryWriter,
) -> Dict[str, float]:
    """
    Validate the model for a single epoch. This function evaluates the model on the validation dataset,
    computes the loss and metrics, and logs the results using TensorBoard.

    Parameters:
    ----------
        - model (nn.Module): The model to be validated.
        - data_loader (DataLoader): DataLoader for the validation dataset.
        - loss_fn (nn.Module): Loss function to compute the loss.
        - metrics (List[nn.Module]): List of metric functions to evaluate the model.
        - device (str): Device on which the model and data are located, either "cpu" or "cuda".
        - epoch (int): Current epoch number.
        - writer (SummaryWriter): TensorBoard writer for logging.

    Returns:
    -------
        - history (Dict[str, float]): A dictionary containing the average loss and metrics for the
        validation epoch.
    """

    loss_lst = []
    b_acc_lst = []
    b_f1_lst = []
    b_kappa_lst = []
    with tqdm(data_loader, unit="batch") as t_data_loader:
        with torch.no_grad():
            for counter, (input_, target_) in enumerate(t_data_loader):
                t_data_loader.set_description(f"Epoch_Valid {epoch+1}")

                input_, target_ = input_.to(device), torch.unsqueeze(target_, -1).to(
                    device
                )

                pred = model(input_)

                loss = loss_fn(pred, target_.float())

                b_acc = metrics[0](torch.sigmoid(pred), target_)
                b_f1 = metrics[1](torch.sigmoid(pred), target_)
                b_kappa = metrics[2](torch.sigmoid(pred), target_)

                loss_lst.append(loss.item())
                b_acc_lst.append(b_acc.item())
                b_f1_lst.append(b_f1.item())
                b_kappa_lst.append(b_kappa.item())

                t_data_loader.set_postfix(
                    loss=np.mean(loss_lst),
                    binary_accuracy=np.mean(b_acc_lst),
                    binary_f1=np.mean(b_f1_lst),
                    binary_kappa=np.mean(b_kappa_lst),
                )

                if counter % 50 == 0:
                    tbx = epoch * len(data_loader) + counter + 1
                    writer.add_scalar("Loss/valid", np.mean(loss_lst), tbx)
                    writer.add_scalar("Accuracy/valid", np.mean(b_acc_lst), tbx)
                    writer.add_scalar("F1_score/valid", np.mean(b_f1_lst), tbx)
                    writer.add_scalar("Kappa_Cohen/valid", np.mean(b_kappa_lst), tbx)

    history = {
        "loss": np.mean(loss_lst),
        "b_acc": np.mean(b_acc_lst),
        "b_f1": np.mean(b_f1_lst),
        "b_kappa": np.mean(b_kappa_lst),
    }
    return history


def main() -> None:
    """
    Main function to train and validate the model. This function sets up the training environment,
    initializes the model, data loaders, loss function, optimizer, and metrics, and then calls the
    training function. It also handles logging and saving the model weights.

    Parameters:
    ----------
        - None

    Returns:
    -------
        - None
    """
    # General Info
    MODEL_WEIGHT_PATH = os.path.join(args.logs_dir, "model_weight")
    device = set_seed_and_get_device(
        seed=SEED_NUMBER
    )  # Initialize device and set random seed
    TIMESTEPS = datetime.now().strftime("%Y%m%d_%H%M%S")

    args = arg_parsing(
        timestamp=TIMESTEPS, mode="train"
    )  # Parse command-line arguments

    writer = SummaryWriter(args.logs_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Implementing the audio data loader
    train_annotations, valid_annotations = train_valid_separation(args.annotation_dir)

    train_dataset = AD(args.dataset_dir, train_annotations, device=device)
    valid_dataset = AD(args.dataset_dir, valid_annotations, device=device)

    train_data_loader = create_data_loader(train_dataset, args.batch_size)
    valid_data_loader = create_data_loader(valid_dataset, VALIDATION_BATCH_SIZE)

    # In case of having imbalance data, you can use weighted loss function.
    weights = torch.tensor(train_dataset._get_class_num_(), dtype=torch.float32)
    weights = weights.sum() / (weights * 2)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights[-1]).to(device)

    metrics = []
    metrics.append(BinaryAccuracy().to(device))
    metrics.append(BinaryF1Score().to(device))
    metrics.append(BinaryCohenKappa().to(device))

    # Implementing the model
    model = Pretrained_Models(
        model_name=TIMM_MODEL_NAME, class_num=CLASS_NUMBER, input_size=INPUT_SHAPE
    ).get_model()

    model = model.to(device)
    model.load_state_dict(torch.load(args.pretrained_dir)["model_state_dict"])
    for param in model.parameters():
        param.requires_grad == False

    model.head = nn.Linear(model.head.in_features, 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_rate)

    train(
        model=model,
        train_data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
        criterion=criterion,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        epochs=args.epoch_number,
        writer=writer,
    )
    writer.flush()


if __name__ == "__main__":
    main()
