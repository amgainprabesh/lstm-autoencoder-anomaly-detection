"""
Evaluation and Anomaly Detection Module.
Calculates sequence reconstruction errors, principled threshold selection,
classification metrics (Precision, Recall, F1, Confusion Matrix),
and statistical baseline methods.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from utils.utils import get_device


def compute_reconstruction_errors(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int = 128,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Compute Mean Squared Error (MSE) reconstruction error for each sequence.
    error_i = (1 / (seq_len * feat_dim)) * sum((x_i - x_hat_i)^2)

    Args:
        model (nn.Module): Trained autoencoder model.
        X (np.ndarray): Array of sequences (N, seq_len, feat_dim) or (N, seq_len).
        batch_size (int): Evaluation batch size.
        device (Optional[torch.device]): Compute device.

    Returns:
        np.ndarray: 1D array of length N containing per-sequence reconstruction errors.
    """
    if len(X) == 0:
        return np.array([], dtype=np.float32)

    if X.ndim == 2:
        X = np.expand_dims(X, axis=-1)

    if device is None:
        device = get_device()

    model.eval()
    model.to(device)

    errors = []
    num_samples = len(X)

    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch = torch.tensor(X[start_idx:end_idx], dtype=torch.float32).to(device)
            recon = model(batch)
            # Compute MSE per sequence over (seq_len, feat_dim)
            batch_errors = torch.mean((batch - recon) ** 2, dim=(1, 2))
            errors.append(batch_errors.cpu().numpy())

    if len(errors) == 0:
        return np.array([], dtype=np.float32)

    return np.concatenate(errors, axis=0)


def reconstruct_sequences(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int = 128,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Generate reconstructed sequences from the autoencoder.

    Args:
        model (nn.Module): Trained autoencoder.
        X (np.ndarray): Input sequences of shape (N, seq_len, feat_dim) or (N, seq_len).
        batch_size (int): Batch size.
        device (Optional[torch.device]): Compute device.

    Returns:
        np.ndarray: Reconstructed sequences of shape (N, seq_len, feat_dim).
    """
    if len(X) == 0:
        return np.empty_like(X)

    if X.ndim == 2:
        X = np.expand_dims(X, axis=-1)

    if device is None:
        device = get_device()

    model.eval()
    model.to(device)

    reconstructions = []
    num_samples = len(X)

    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch = torch.tensor(X[start_idx:end_idx], dtype=torch.float32).to(device)
            recon = model(batch)
            reconstructions.append(recon.cpu().numpy())

    if len(reconstructions) == 0:
        return np.empty_like(X)

    return np.concatenate(reconstructions, axis=0)


def calculate_threshold(
    val_errors: np.ndarray,
    method: str = "percentile",
    percentile: float = 95.0,
    k_std: float = 3.0,
    val_labels: Optional[np.ndarray] = None,
) -> float:
    """
    Principled threshold selection based on normal/validation reconstruction error distribution.
    If ground truth validation labels are provided, computes threshold exclusively from normal sequences.

    Methods:
    1. 'percentile': Empirically sets threshold at the p-th percentile of validation errors.
       Example: 95th percentile marks top 5% highest error sequences as anomalous.
    2. 'std': Mean + k * standard_deviation (Gauss Chebyshev inequality approach).
    3. 'iqr': Tukey's Fences method: Q3 + 1.5 * (Q3 - Q1), robust to heavy tails.

    Args:
        val_errors (np.ndarray): 1D array of validation reconstruction errors.
        method (str): 'percentile', 'std', or 'iqr'.
        percentile (float): Percentile value in [0, 100].
        k_std (float): Standard deviation multiplier for 'std' method.
        val_labels (Optional[np.ndarray]): Ground truth labels for validation set (0=normal, 1=anomaly).

    Returns:
        float: Computed anomaly decision threshold.
    """
    if len(val_errors) == 0:
        raise ValueError("Cannot calculate threshold from empty validation error array.")

    # Filter to normal validation samples if labels exist per semi-supervised definition
    if val_labels is not None and len(val_labels) == len(val_errors):
        normal_mask = (val_labels == 0)
        if np.any(normal_mask):
            val_errors = val_errors[normal_mask]

    method = method.lower()
    if method == "percentile":
        if not (0 <= percentile <= 100):
            raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")
        threshold = float(np.percentile(val_errors, percentile))
    elif method == "std":
        mean_err = np.mean(val_errors)
        std_err = np.std(val_errors)
        threshold = float(mean_err + k_std * std_err)
    elif method == "iqr":
        q25, q75 = np.percentile(val_errors, [25, 75])
        iqr = q75 - q25
        threshold = float(q75 + 1.5 * iqr)
    else:
        raise ValueError(f"Unknown threshold method: {method}. Choose from 'percentile', 'std', 'iqr'.")

    return max(threshold, 1e-8)


def detect_anomalies(errors: np.ndarray, threshold: float) -> np.ndarray:
    """
    Classify sequences into normal (0) or anomaly (1) based on decision rule:
        error <= threshold -> 0 (Normal)
        error > threshold  -> 1 (Anomaly)

    Args:
        errors (np.ndarray): 1D array of reconstruction errors.
        threshold (float): Anomaly decision threshold.

    Returns:
        np.ndarray: Binary array (0 or 1).
    """
    return (errors > threshold).astype(np.int32)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics for anomaly detection:
    - Accuracy
    - Precision: TP / (TP + FP) [crucial for minimizing false alarms]
    - Recall: TP / (TP + FN) [crucial for capturing actual faults]
    - F1 Score: Harmonic mean of precision and recall
    - Specificity: TN / (TN + FP)
    - Confusion Matrix (TP, FP, TN, FN)

    Args:
        y_true (np.ndarray): Ground truth binary labels (0 = normal, 1 = anomaly).
        y_pred (np.ndarray): Predicted binary labels (0 = normal, 1 = anomaly).

    Returns:
        Dict[str, Any]: Evaluation metrics dictionary.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch between y_true ({len(y_true)}) and y_pred ({len(y_pred)})")

    acc = float(accuracy_score(y_true, y_pred))
    # zero_division=0 ensures we don't throw errors when no positives are predicted
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": specificity,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "confusion_matrix": cm.tolist(),
        "total_samples": len(y_true),
        "actual_anomalies": int(np.sum(y_true)),
        "predicted_anomalies": int(np.sum(y_pred)),
    }


def find_best_threshold(
    val_errors: np.ndarray,
    val_labels: np.ndarray,
    candidate_percentiles: Optional[List[float]] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Evaluate candidate thresholds on validation set and select the threshold
    that maximizes the F1 score.

    Args:
        val_errors (np.ndarray): Validation reconstruction errors.
        val_labels (np.ndarray): Validation ground-truth binary labels.
        candidate_percentiles (Optional[List[float]]): List of percentiles to test.

    Returns:
        Tuple[float, float, Dict[str, Any]]: (best_threshold, best_percentile, best_metrics)
    """
    if candidate_percentiles is None:
        candidate_percentiles = [80.0, 85.0, 90.0, 92.5, 95.0, 97.0, 97.5, 98.0, 99.0, 99.5]

    if len(val_labels) == 0 or np.sum(val_labels) == 0:
        default_th = calculate_threshold(val_errors, method="percentile", percentile=95.0)
        default_preds = detect_anomalies(val_errors, default_th)
        return default_th, 95.0, calculate_metrics(val_labels, default_preds)

    best_f1 = -1.0
    best_threshold = 0.0
    best_percentile = 95.0
    best_metrics = {}

    for p in candidate_percentiles:
        th = calculate_threshold(val_errors, method="percentile", percentile=p, val_labels=val_labels)
        preds = detect_anomalies(val_errors, th)
        metrics = calculate_metrics(val_labels, preds)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = th
            best_percentile = p
            best_metrics = metrics

    return best_threshold, best_percentile, best_metrics


def build_results_dataframe(
    timestamps: np.ndarray,
    actual_values: np.ndarray,
    reconstructed_values: np.ndarray,
    reconstruction_errors: np.ndarray,
    predictions: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    anomaly_scores: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Construct a tabular DataFrame containing detailed sequence anomaly results.
    Includes all required columns: Timestamp, Actual Value, Reconstructed Value,
    Reconstruction Error, Anomaly Score, and Prediction.

    Args:
        timestamps (np.ndarray): Array of timestamps corresponding to each sequence.
        actual_values (np.ndarray): True signal values at window endpoint.
        reconstructed_values (np.ndarray): Autoencoder reconstructed signal values.
        reconstruction_errors (np.ndarray): Per-sequence MSE errors.
        predictions (np.ndarray): Binary anomaly predictions (0 or 1).
        ground_truth (Optional[np.ndarray]): Optional ground truth labels.
        anomaly_scores (Optional[np.ndarray]): Optional explicit anomaly score values.
        threshold (Optional[float]): Optional threshold for computing relative anomaly scores.

    Returns:
        pd.DataFrame: Structured results dataframe.
    """
    if anomaly_scores is None:
        if threshold is not None and threshold > 0:
            anomaly_scores = reconstruction_errors / threshold
        else:
            anomaly_scores = reconstruction_errors

    df_results = pd.DataFrame(
        {
            "timestamp": timestamps,
            "actual_value": actual_values,
            "reconstructed_value": reconstructed_values,
            "reconstruction_error": np.round(reconstruction_errors, 6),
            "anomaly_score": np.round(anomaly_scores, 6),
            "prediction": predictions,
            "prediction_label": np.where(predictions == 1, "Anomaly", "Normal"),
        }
    )

    if ground_truth is not None and len(ground_truth) == len(predictions):
        df_results["ground_truth"] = ground_truth
        df_results["ground_truth_label"] = np.where(ground_truth == 1, "Anomaly", "Normal")

        # Tag diagnostic status
        def get_status(row):
            if row["ground_truth"] == 1 and row["prediction"] == 1:
                return "True Positive (TP)"
            elif row["ground_truth"] == 0 and row["prediction"] == 1:
                return "False Positive (FP)"
            elif row["ground_truth"] == 0 and row["prediction"] == 0:
                return "True Negative (TN)"
            else:
                return "False Negative (FN)"

        df_results["status"] = df_results.apply(get_status, axis=1)

    return df_results


class StatisticalBaselineDetector:
    """
    Classical Rolling Z-Score Anomaly Detector for comparative baseline (Experiment 4).
    Computes rolling mean and rolling standard deviation over a temporal window.
    Points deviating by more than k standard deviations are classified as anomalies:
        |x_t - mean_t| / std_t > k
    """

    def __init__(self, window_size: int = 24, k_std: float = 3.0):
        self.window_size = window_size
        self.k_std = k_std
        self.mean_val = 0.0
        self.std_val = 1.0

    def fit(self, train_values: np.ndarray) -> "StatisticalBaselineDetector":
        self.mean_val = float(np.mean(train_values))
        self.std_val = float(np.std(train_values)) if np.std(train_values) > 1e-8 else 1.0
        return self

    def predict(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict point-level anomalies using rolling Z-scores.
        Uses past historical values in the window (excluding current point) or fitted distribution
        to compute rolling mean and std, preventing the anomaly itself from inflating the variance.
        """
        series = pd.Series(values.squeeze())
        lagged = series.shift(1)
        rolling_mean = lagged.rolling(window=self.window_size, min_periods=1).mean().fillna(self.mean_val)
        rolling_std = lagged.rolling(window=self.window_size, min_periods=1).std().fillna(self.std_val)
        rolling_std = rolling_std.replace(0.0, self.std_val)

        z_scores = np.abs((series - rolling_mean) / rolling_std).values
        predictions = (z_scores > self.k_std).astype(np.int32)
        return predictions, z_scores

    def predict_sequences(self, values: np.ndarray, window_size: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """
        Produce sliding-window sequence predictions matching the sequence evaluation formulation.
        A sequence window is classified as an anomaly if any point within the window
        exceeds the rolling Z-score threshold.
        """
        point_preds, point_scores = self.predict(values)
        num_windows = len(values) - window_size + 1
        if num_windows <= 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        seq_preds = np.empty(num_windows, dtype=np.int32)
        seq_scores = np.empty(num_windows, dtype=np.float32)

        for i in range(num_windows):
            seq_preds[i] = int(np.any(point_preds[i : i + window_size] == 1))
            seq_scores[i] = float(np.max(point_scores[i : i + window_size]))

        return seq_preds, seq_scores
