"""
Neural Network Models for Time-Series Anomaly Detection.
Contains:
1. LSTM Encoder-Decoder Autoencoder (primary architecture)
2. Dense Autoencoder (baseline for comparative evaluation)
"""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """
    LSTM Encoder for temporal sequence compression into a latent representation.

    Architecture:
        Input: (batch_size, seq_len, input_dim)
        Stacked LSTM layers with specified hidden dimensions.
        Linear projection from the final hidden state to the latent space.
        Output: (batch_size, latent_dim)
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        self.lstm_layers = nn.ModuleList()
        current_dim = input_dim

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for i, h_dim in enumerate(hidden_dims):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=current_dim,
                    hidden_size=h_dim,
                    batch_first=True,
                )
            )
            current_dim = h_dim

        # Project final LSTM hidden state to latent dimension
        self.fc_latent = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of LSTM Encoder.

        Args:
            x (torch.Tensor): Input sequences of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Latent representation of shape (batch_size, latent_dim).
        """
        out = x
        for i, lstm in enumerate(self.lstm_layers):
            out, (h_n, c_n) = lstm(out)
            if i < len(self.lstm_layers) - 1:
                out = self.dropout_layer(out)

        # Use the hidden state of the final time step or h_n[-1]
        last_hidden = h_n[-1]  # shape: (batch_size, hidden_dims[-1])
        latent = self.fc_latent(last_hidden)  # shape: (batch_size, latent_dim)
        return latent


class LSTMDecoder(nn.Module):
    """
    LSTM Decoder for reconstructing temporal sequences from latent representations.

    Architecture:
        Input: Latent vector (batch_size, latent_dim)
        Repeated across seq_len time steps: (batch_size, seq_len, latent_dim)
        Stacked LSTM layers with specified hidden dimensions (in reverse order).
        Linear projection to output_dim at each time step.
        Output: (batch_size, seq_len, output_dim)
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 1,
        seq_len: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64]

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.seq_len = seq_len

        self.lstm_layers = nn.ModuleList()
        current_dim = latent_dim

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for i, h_dim in enumerate(hidden_dims):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=current_dim,
                    hidden_size=h_dim,
                    batch_first=True,
                )
            )
            current_dim = h_dim

        self.fc_out = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, z: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass of LSTM Decoder.

        Args:
            z (torch.Tensor): Latent representation of shape (batch_size, latent_dim).
            seq_len (Optional[int]): Target sequence length to reconstruct. Defaults to self.seq_len.

        Returns:
            torch.Tensor: Reconstructed sequence of shape (batch_size, seq_len, output_dim).
        """
        batch_size = z.size(0)
        target_len = seq_len if seq_len is not None else self.seq_len
        # Repeat latent vector across sequence length (RepeatVector equivalent)
        # Shape: (batch_size, target_len, latent_dim)
        repeated = z.unsqueeze(1).repeat(1, target_len, 1)

        out = repeated
        for i, lstm in enumerate(self.lstm_layers):
            out, _ = lstm(out)
            if i < len(self.lstm_layers) - 1:
                out = self.dropout_layer(out)

        # Project each time step to output_dim
        reconstructed = self.fc_out(out)  # shape: (batch_size, target_len, output_dim)
        return reconstructed


class LSTMAutoencoder(nn.Module):
    """
    Complete LSTM Autoencoder combining Encoder and Decoder.
    Trains in an unsupervised/semi-supervised fashion on normal time series data
    to minimize reconstruction error MSE(x, x_hat).
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 16,
        seq_len: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.dropout = dropout

        # Decoder hidden dims are the reverse of encoder hidden dims
        decoder_hidden_dims = list(reversed(hidden_dims))

        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.decoder = LSTMDecoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=input_dim,
            seq_len=seq_len,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Input -> Encoder -> Latent -> Decoder -> Reconstructed Output.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, seq_len, input_dim)
                              or (batch_size, seq_len).

        Returns:
            torch.Tensor: Reconstructed tensor of matching shape.
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        z = self.encoder(x)
        x_hat = self.decoder(z, seq_len=x.size(1))
        return x_hat

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent representation for visualization or inspection."""
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        return self.encoder(x)

    def compute_reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute mean squared error loss for a batch of sequences.

        Args:
            x (torch.Tensor): Input batch.

        Returns:
            torch.Tensor: Scalar MSE loss.
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x_hat = self.forward(x)
        return nn.functional.mse_loss(x_hat, x)

    def compute_sequence_errors(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sequence reconstruction error MSE = mean((x - x_hat)^2).

        Args:
            x (torch.Tensor): Input batch (batch_size, seq_len, input_dim) or (batch_size, seq_len).

        Returns:
            torch.Tensor: 1D tensor of per-sequence error scores of shape (batch_size,).
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        with torch.no_grad():
            x_hat = self.forward(x)
            # MSE across time steps and features: (batch_size,)
            error = torch.mean((x - x_hat) ** 2, dim=(1, 2))
        return error


class DenseAutoencoder(nn.Module):
    """
    Baseline Dense (Fully Connected) Autoencoder without recurrent connections.
    Flattens the entire sequence into a single 1D vector.
    Used for Experiment 4 to demonstrate whether temporal recurrence (LSTM)
    provides significant advantages over static feedforward architectures.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        seq_len: int = 24,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.flattened_dim = input_dim * seq_len

        self.encoder = nn.Sequential(
            nn.Linear(self.flattened_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.flattened_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        flat_x = x.view(batch_size, -1)
        z = self.encoder(flat_x)
        recon_flat = self.decoder(z)
        return recon_flat.view(batch_size, self.seq_len, self.input_dim)

    def compute_sequence_errors(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x_hat = self.forward(x)
            return torch.mean((x - x_hat) ** 2, dim=(1, 2))
