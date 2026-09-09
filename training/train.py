"""
Training pipeline for Time-Series LSTM Autoencoder.
Includes:
- Training loop with forward, loss computation, backprop, optimizer step, gradient zeroing
- Validation tracking
- Early stopping based on validation loss
- Model checkpointing (saving best weights)
- History recording and model persistence
"""
import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.utils import get_device, set_seed


class EarlyStopping:
    """
    Early stopping monitor that stops training when validation loss stops improving.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def create_dataloader(
    X: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """
    Convert numpy array of sequences into a PyTorch DataLoader.

    Args:
        X (np.ndarray): Array of shape (N, seq_len, feature_dim).
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle mini-batches.

    Returns:
        DataLoader: PyTorch DataLoader yielding tensor batches.
    """
    tensor_x = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(tensor_x)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def get_optimizer(
    model: nn.Module,
    optimizer_name: str = "adam",
    lr: float = 0.001,
    weight_decay: float = 1e-5,
) -> torch.optim.Optimizer:
    """
    Factory function for optimizers.
    """
    name = optimizer_name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Choose from adam, rmsprop, sgd, adamw.")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Execute one complete training epoch:
    1. Forward Pass
    2. Compute Loss
    3. Backpropagation (loss.backward)
    4. Optimizer step (optimizer.step)
    5. Clear gradients (optimizer.zero_grad)

    Args:
        model (nn.Module): Neural network.
        dataloader (DataLoader): Training dataloader.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function (e.g. MSELoss).
        device (torch.device): CPU or CUDA device.

    Returns:
        float: Mean training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Ensure clean gradients before starting epoch
    optimizer.zero_grad()

    for batch in dataloader:
        x = batch[0].to(device)

        # 1. Forward Pass
        reconstruction = model(x)

        # 2. Calculate Loss (MSE between input sequence and reconstructed sequence)
        loss = criterion(reconstruction, x)

        # 3. Backpropagation (calculate gradients dL/dw)
        loss.backward()

        # Optional gradient clipping to prevent exploding gradients in recurrent networks
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        # 4. Optimizer step (update weights: w = w - lr * grad)
        optimizer.step()

        # 5. Clear gradients
        optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Compute mean reconstruction loss on validation/test dataloader without gradients.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            reconstruction = model(x)
            loss = criterion(reconstruction, x)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def save_checkpoint(
    model: nn.Module,
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model state dictionary and optional training metadata.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    model: nn.Module,
    filepath: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load saved weights into model.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    map_location = device if device is not None else get_device()
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        metadata = checkpoint.get("metadata", {})
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
        metadata = {}
    else:
        raise ValueError(f"Invalid checkpoint format at {filepath}")

    model.to(map_location)
    return model, metadata


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    X_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    optimizer_name: str = "adam",
    weight_decay: float = 1e-5,
    patience: int = 10,
    min_delta: float = 1e-4,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    progress_callback: Optional[Any] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Full training pipeline with early stopping, checkpointing, and history tracking.

    Args:
        model (nn.Module): The Autoencoder model to train.
        X_train (np.ndarray): Training sequences (N_train, seq_len, feat_dim).
        X_val (np.ndarray): Validation sequences (N_val, seq_len, feat_dim).
        epochs (int): Maximum training epochs.
        batch_size (int): Mini-batch size.
        learning_rate (float): Initial optimizer learning rate.
        optimizer_name (str): Optimizer ('adam', 'rmsprop', 'sgd', 'adamw').
        weight_decay (float): L2 regularization weight decay.
        patience (int): Early stopping patience epochs.
        min_delta (float): Minimum validation improvement to reset patience.
        checkpoint_path (Optional[Union[str, Path]]): Destination for best checkpoint.
        device (Optional[torch.device]): Compute device.
        verbose (bool): Whether to print progress each epoch.
        progress_callback (Optional[Callable]): Callback for UI progress bars (e.g. Streamlit).

    Returns:
        Tuple[nn.Module, Dict[str, Any]]: (best_model, training_history)
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    train_loader = create_dataloader(X_train, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(X_val, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = get_optimizer(model, optimizer_name, lr=learning_rate, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    history = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "epochs_trained": 0,
    }

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epochs_trained"] = epoch

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            history["best_val_loss"] = best_val_loss
            history["best_epoch"] = epoch
            best_weights = copy.deepcopy(model.state_dict())

            if checkpoint_path is not None:
                save_checkpoint(
                    model,
                    checkpoint_path,
                    metadata={
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "train_loss": train_loss,
                        "window_size": model.seq_len if hasattr(model, "seq_len") else None,
                        "hidden_dims": model.hidden_dims if hasattr(model, "hidden_dims") else None,
                        "latent_dim": model.latent_dim if hasattr(model, "latent_dim") else None,
                    },
                )

        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if progress_callback is not None:
            progress_callback(epoch, epochs, train_loss, val_loss)

        # Early stopping check
        if early_stopping(val_loss, model):
            if verbose:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch was {history['best_epoch']}.")
            break

    # Restore best weights to ensure returned model is optimal
    model.load_state_dict(best_weights)
    return model, history
