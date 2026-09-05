"""
Unit tests for evaluation, threshold selection, and metrics computation.
"""
import numpy as np
import pytest

from evaluation.evaluate import (
    calculate_metrics,
    calculate_threshold,
    detect_anomalies,
    StatisticalBaselineDetector,
)


def test_calculate_threshold_percentile():
    errors = np.linspace(0.01, 1.0, 100)
    th_95 = calculate_threshold(errors, method="percentile", percentile=95.0)
    assert np.isclose(th_95, np.percentile(errors, 95.0))


def test_calculate_threshold_std():
    errors = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    th_std = calculate_threshold(errors, method="std", k_std=3.0)
    assert np.isclose(th_std, 1.0)


def test_calculate_threshold_iqr():
    errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    th_iqr = calculate_threshold(errors, method="iqr")
    assert th_iqr > 8.0


def test_detect_anomalies():
    errors = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
    threshold = 0.5
    preds = detect_anomalies(errors, threshold)
    np.testing.assert_array_equal(preds, [0, 0, 1, 0, 1])


def test_calculate_metrics():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 0])

    m = calculate_metrics(y_true, y_pred)
    assert "accuracy" in m
    assert "precision" in m
    assert "recall" in m
    assert "f1" in m
    assert m["tp"] == 2
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["tn"] == 4


def test_statistical_baseline_detector():
    train_data = np.ones(50) * 10.0
    test_data = np.array([10.0, 10.0, 10.0, 100.0, 10.0])  # point 100 is anomaly

    detector = StatisticalBaselineDetector(window_size=5, k_std=2.0)
    detector.fit(train_data)
    preds, z_scores = detector.predict(test_data)

    assert preds[3] == 1  # spike detected
    assert preds[0] == 0


def test_statistical_baseline_predict_sequences():
    train_data = np.ones(50) * 10.0
    test_data = np.array([10.0, 10.0, 10.0, 100.0, 10.0, 10.0, 10.0])

    detector = StatisticalBaselineDetector(window_size=3, k_std=2.0)
    detector.fit(train_data)
    seq_preds, seq_scores = detector.predict_sequences(test_data, window_size=3)

    assert len(seq_preds) == len(test_data) - 3 + 1
    # Windows covering index 3 should be flagged as anomalous
    assert seq_preds[1] == 1  # window [1, 2, 3] includes point 3
    assert seq_preds[2] == 1  # window [2, 3, 4] includes point 3


def test_calculate_threshold_filters_normal_validation_samples():
    # Errors: normal errors [0.01..0.05], anomaly errors [0.90..0.99]
    normal_err = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    anom_err = np.array([0.90, 0.95, 0.99])
    all_err = np.concatenate([normal_err, anom_err])
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1])

    # Threshold with val_labels should compute percentile ONLY from normal samples
    th_filtered = calculate_threshold(all_err, method="percentile", percentile=95.0, val_labels=labels)
    th_unfiltered = calculate_threshold(all_err, method="percentile", percentile=95.0)

    assert th_filtered < 0.10  # isolated to normal errors
    assert th_unfiltered > 0.80  # corrupted by anomaly errors


def test_build_results_dataframe_contains_anomaly_score():
    from evaluation.evaluate import build_results_dataframe

    ts = np.array(["2024-01-01 00:00", "2024-01-01 01:00"])
    actual = np.array([100.0, 250.0])
    recon = np.array([102.0, 120.0])
    errors = np.array([0.001, 0.050])
    preds = np.array([0, 1])

    df_res = build_results_dataframe(
        timestamps=ts,
        actual_values=actual,
        reconstructed_values=recon,
        reconstruction_errors=errors,
        predictions=preds,
        threshold=0.010,
    )

    assert "anomaly_score" in df_res.columns
    assert "reconstruction_error" in df_res.columns
    assert "prediction" in df_res.columns
    assert df_res.iloc[0]["prediction_label"] == "Normal"
    assert df_res.iloc[1]["prediction_label"] == "Anomaly"


def test_compute_reconstruction_errors_2d_input():
    import torch
    from evaluation.evaluate import compute_reconstruction_errors, reconstruct_sequences
    from models.lstm_autoencoder import LSTMAutoencoder

    model = LSTMAutoencoder(input_dim=1, hidden_dims=[16, 8], latent_dim=4, seq_len=12)
    x_2d = np.random.randn(8, 12).astype(np.float32)

    errors = compute_reconstruction_errors(model, x_2d)
    assert errors.shape == (8,)

    recon = reconstruct_sequences(model, x_2d)
    assert recon.shape == (8, 12, 1)

