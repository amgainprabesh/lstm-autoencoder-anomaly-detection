"""
Unit tests for neural network architectures (LSTM Autoencoder and Dense Autoencoder).
"""
import torch
import pytest

from models.lstm_autoencoder import (
    LSTMEncoder,
    LSTMDecoder,
    LSTMAutoencoder,
    DenseAutoencoder,
)


def test_lstm_encoder_shape():
    batch_size = 16
    seq_len = 24
    input_dim = 1
    latent_dim = 8

    encoder = LSTMEncoder(input_dim=input_dim, hidden_dims=[32, 16], latent_dim=latent_dim)
    x = torch.randn(batch_size, seq_len, input_dim)
    z = encoder(x)

    assert z.shape == (batch_size, latent_dim)


def test_lstm_decoder_shape():
    batch_size = 16
    seq_len = 24
    output_dim = 1
    latent_dim = 8

    decoder = LSTMDecoder(latent_dim=latent_dim, hidden_dims=[16, 32], output_dim=output_dim, seq_len=seq_len)
    z = torch.randn(batch_size, latent_dim)
    x_hat = decoder(z)

    assert x_hat.shape == (batch_size, seq_len, output_dim)


def test_lstm_autoencoder_forward_and_backward():
    batch_size = 8
    seq_len = 24
    input_dim = 1

    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dims=[32, 16],
        latent_dim=8,
        seq_len=seq_len,
    )
    x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
    x_hat = model(x)

    assert x_hat.shape == x.shape

    loss = torch.nn.functional.mse_loss(x_hat, x)
    loss.backward()

    # Verify gradients computed across all parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient is None for {name}"
        assert not torch.isnan(param.grad).any(), f"Gradient contains NaN for {name}"


def test_dense_autoencoder_shape():
    batch_size = 8
    seq_len = 24
    input_dim = 1

    dense_model = DenseAutoencoder(input_dim=input_dim, hidden_dim=32, latent_dim=8, seq_len=seq_len)
    x = torch.randn(batch_size, seq_len, input_dim)
    x_hat = dense_model(x)

    assert x_hat.shape == x.shape


def test_lstm_autoencoder_dynamic_sequence_length():
    # Model configured with seq_len=24, but receives seq_len=15
    model = LSTMAutoencoder(input_dim=1, hidden_dims=[32, 16], latent_dim=8, seq_len=24)
    x_short = torch.randn(4, 15, 1)
    recon_short = model(x_short)
    assert recon_short.shape == (4, 15, 1)

    # Reconstruct longer sequence
    x_long = torch.randn(4, 36, 1)
    recon_long = model(x_long)
    assert recon_long.shape == (4, 36, 1)


def test_lstm_autoencoder_multivariate():
    # Test multivariate feature support (e.g. 3 features)
    batch_size = 4
    seq_len = 20
    feat_dim = 3

    model = LSTMAutoencoder(input_dim=feat_dim, hidden_dims=[32, 16], latent_dim=8, seq_len=seq_len)
    x = torch.randn(batch_size, seq_len, feat_dim, requires_grad=True)
    recon = model(x)
    assert recon.shape == (batch_size, seq_len, feat_dim)

    loss = model.compute_reconstruction_loss(x)
    loss.backward()
    assert loss.item() > 0.0

    errors = model.compute_sequence_errors(x)
    assert errors.shape == (batch_size,)


def test_lstm_autoencoder_2d_tensor_support():
    model = LSTMAutoencoder(input_dim=1, hidden_dims=[32, 16], latent_dim=8, seq_len=24)
    x_2d = torch.randn(6, 24)
    recon = model(x_2d)
    assert recon.shape == (6, 24, 1)

    errors = model.compute_sequence_errors(x_2d)
    assert errors.shape == (6,)

