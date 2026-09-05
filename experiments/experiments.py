"""
Controlled Experiments Module for Time-Series Anomaly Detection.
Runs the 4 required experiments:
1. Window Size Comparison (12, 24, 48, 72)
2. LSTM Hidden Size Comparison (32, 64, 128)
3. Threshold Percentile Comparison (90%, 95%, 97.5%, 99%)
4. Baseline Comparison (Statistical Z-Score vs Dense Autoencoder vs LSTM Autoencoder)
All metrics are derived from actual execution. No synthetic or fabricated values.
"""
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from evaluation.evaluate import (
    StatisticalBaselineDetector,
    calculate_metrics,
    calculate_threshold,
    compute_reconstruction_errors,
    detect_anomalies,
)
from models.lstm_autoencoder import DenseAutoencoder, LSTMAutoencoder
from preprocessing.preprocess import PreprocessedData, preprocess_pipeline
from training.train import train_model
from utils.utils import get_device, load_config, set_seed


def run_experiment_1_window_size(
    df_raw: pd.DataFrame,
    labels_info: Dict[str, Any],
    window_sizes: Optional[List[int]] = None,
    epochs: int = 10,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Experiment 1: Window Size Comparison.
    Investigates how the temporal receptive field (window_size) influences
    reconstruction loss and anomaly detection F1 score.
    """
    if window_sizes is None:
        window_sizes = [12, 24, 48, 72]

    if device is None:
        device = get_device()

    results = []
    print("\n--- Running Experiment 1: Window Size Comparison ---")

    for w in window_sizes:
        print(f"\nEvaluating Window Size: {w}")
        set_seed(42)
        t0 = time.time()

        prep_data = preprocess_pipeline(
            df_raw=df_raw,
            window_size=w,
            step_size=1,
            train_ratio=0.60,
            val_ratio=0.20,
            test_ratio=0.20,
            filter_anomalies_from_train=True,
            anomaly_windows=labels_info.get("anomaly_windows", []),
            anomaly_timestamps=labels_info.get("anomaly_timestamps", []),
        )

        model = LSTMAutoencoder(
            input_dim=1,
            hidden_dims=[64, 32],
            latent_dim=16,
            seq_len=w,
            dropout=0.1,
        )

        model, history = train_model(
            model=model,
            X_train=prep_data.X_train,
            X_val=prep_data.X_val,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=0.002,
            patience=5,
            device=device,
            verbose=False,
        )

        # Compute errors
        val_errors = compute_reconstruction_errors(model, prep_data.X_val, device=device)
        test_errors = compute_reconstruction_errors(model, prep_data.X_test, device=device)

        threshold = calculate_threshold(val_errors, method="percentile", percentile=95.0, val_labels=prep_data.val_labels)
        test_preds = detect_anomalies(test_errors, threshold)

        metrics = calculate_metrics(prep_data.test_labels, test_preds)
        elapsed = time.time() - t0

        row = {
            "window_size": w,
            "best_epoch": history["best_epoch"],
            "val_loss": round(history["best_val_loss"], 6),
            "test_mean_error": round(float(np.mean(test_errors)), 6),
            "threshold_p95": round(threshold, 6),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1_score": round(metrics["f1"], 4),
            "accuracy": round(metrics["accuracy"], 4),
            "runtime_sec": round(elapsed, 2),
        }
        print(f"Result for window_size={w}: F1={row['f1_score']}, Precision={row['precision']}, Recall={row['recall']}")
        results.append(row)

    df_res = pd.DataFrame(results)
    return df_res


def run_experiment_2_hidden_size(
    df_raw: pd.DataFrame,
    labels_info: Dict[str, Any],
    hidden_configs: Optional[List[Tuple[List[int], int]]] = None,
    window_size: int = 24,
    epochs: int = 10,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Experiment 2: LSTM Hidden Size Comparison.
    Investigates model capacity: compares small (32, 16), medium (64, 32), and large (128, 64) hidden sizes.
    """
    if hidden_configs is None:
        hidden_configs = [
            ([32, 16], 8),
            ([64, 32], 16),
            ([128, 64], 32),
        ]

    if device is None:
        device = get_device()

    prep_data = preprocess_pipeline(
        df_raw=df_raw,
        window_size=window_size,
        step_size=1,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        filter_anomalies_from_train=True,
        anomaly_windows=labels_info.get("anomaly_windows", []),
        anomaly_timestamps=labels_info.get("anomaly_timestamps", []),
    )

    results = []
    print("\n--- Running Experiment 2: LSTM Hidden Size Comparison ---")

    for h_dims, latent_dim in hidden_configs:
        config_name = f"LSTM({h_dims[0]})-LSTM({h_dims[1]})"
        print(f"\nEvaluating Architecture: {config_name}, Latent={latent_dim}")
        set_seed(42)
        t0 = time.time()

        model = LSTMAutoencoder(
            input_dim=1,
            hidden_dims=h_dims,
            latent_dim=latent_dim,
            seq_len=window_size,
            dropout=0.1,
        )
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model, history = train_model(
            model=model,
            X_train=prep_data.X_train,
            X_val=prep_data.X_val,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=0.002,
            patience=5,
            device=device,
            verbose=False,
        )

        val_errors = compute_reconstruction_errors(model, prep_data.X_val, device=device)
        test_errors = compute_reconstruction_errors(model, prep_data.X_test, device=device)

        threshold = calculate_threshold(val_errors, method="percentile", percentile=95.0, val_labels=prep_data.val_labels)
        test_preds = detect_anomalies(test_errors, threshold)
        metrics = calculate_metrics(prep_data.test_labels, test_preds)
        elapsed = time.time() - t0

        row = {
            "architecture": config_name,
            "hidden_dims": str(h_dims),
            "latent_dim": latent_dim,
            "param_count": param_count,
            "val_loss": round(history["best_val_loss"], 6),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1_score": round(metrics["f1"], 4),
            "accuracy": round(metrics["accuracy"], 4),
            "runtime_sec": round(elapsed, 2),
        }
        print(f"Result for {config_name}: F1={row['f1_score']}, Val Loss={row['val_loss']}")
        results.append(row)

    df_res = pd.DataFrame(results)
    return df_res


def run_experiment_3_threshold(
    model: nn.Module,
    prep_data: PreprocessedData,
    percentiles: Optional[List[float]] = None,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Experiment 3: Threshold Percentile Comparison.
    Evaluates 90%, 95%, 97.5%, and 99% thresholds to expose the precision-recall trade-off.
    """
    if percentiles is None:
        percentiles = [90.0, 95.0, 97.5, 99.0]

    if device is None:
        device = get_device()

    print("\n--- Running Experiment 3: Threshold Percentile Comparison ---")
    val_errors = compute_reconstruction_errors(model, prep_data.X_val, device=device)
    test_errors = compute_reconstruction_errors(model, prep_data.X_test, device=device)

    results = []
    for p in percentiles:
        th = calculate_threshold(val_errors, method="percentile", percentile=p, val_labels=prep_data.val_labels)
        test_preds = detect_anomalies(test_errors, th)
        metrics = calculate_metrics(prep_data.test_labels, test_preds)

        row = {
            "percentile": p,
            "threshold_value": round(th, 6),
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "tn": metrics["tn"],
            "fn": metrics["fn"],
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1_score": round(metrics["f1"], 4),
            "accuracy": round(metrics["accuracy"], 4),
        }
        print(f"Percentile {p}%: Threshold={th:.6f} -> Precision={row['precision']}, Recall={row['recall']}, F1={row['f1_score']}")
        results.append(row)

    df_res = pd.DataFrame(results)
    return df_res


def run_experiment_4_baseline(
    lstm_model: nn.Module,
    prep_data: PreprocessedData,
    epochs: int = 10,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Experiment 4: Baseline Comparison.
    Compares:
    1. Statistical Rolling Z-Score Baseline
    2. Feedforward Dense Autoencoder Baseline
    3. LSTM Autoencoder
    Directly answers whether modeling temporal dependencies improves anomaly detection.
    """
    if device is None:
        device = get_device()

    print("\n--- Running Experiment 4: Baseline Comparison ---")
    results = []

    # 1. Statistical Baseline (Rolling Z-Score)
    print("Evaluating Baseline 1: Statistical Rolling Z-Score...")
    stat_detector = StatisticalBaselineDetector(window_size=prep_data.window_size, k_std=3.0)
    stat_detector.fit(prep_data.train_df[prep_data.value_col].values)
    stat_seq_preds, _ = stat_detector.predict_sequences(
        prep_data.test_df[prep_data.value_col].values,
        window_size=prep_data.window_size,
    )
    stat_metrics = calculate_metrics(prep_data.test_labels, stat_seq_preds)
    results.append(
        {
            "model": "Statistical (Rolling Z-Score)",
            "type": "Heuristic Statistical Filter",
            "learns_temporal_dynamics": "No (Fixed Window Moments)",
            "precision": round(stat_metrics["precision"], 4),
            "recall": round(stat_metrics["recall"], 4),
            "f1_score": round(stat_metrics["f1"], 4),
            "accuracy": round(stat_metrics["accuracy"], 4),
        }
    )

    # 2. Dense Autoencoder Baseline
    print("Evaluating Baseline 2: Dense Feedforward Autoencoder...")
    set_seed(42)
    dense_model = DenseAutoencoder(
        input_dim=1,
        hidden_dim=64,
        latent_dim=16,
        seq_len=prep_data.window_size,
    )
    dense_model, _ = train_model(
        model=dense_model,
        X_train=prep_data.X_train,
        X_val=prep_data.X_val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=0.002,
        patience=5,
        device=device,
        verbose=False,
    )
    dense_val_err = compute_reconstruction_errors(dense_model, prep_data.X_val, device=device)
    dense_test_err = compute_reconstruction_errors(dense_model, prep_data.X_test, device=device)
    dense_th = calculate_threshold(dense_val_err, method="percentile", percentile=95.0, val_labels=prep_data.val_labels)
    dense_preds = detect_anomalies(dense_test_err, dense_th)
    dense_metrics = calculate_metrics(prep_data.test_labels, dense_preds)

    results.append(
        {
            "model": "Dense Autoencoder",
            "type": "Feedforward Neural Network",
            "learns_temporal_dynamics": "No (Static Vector Flattening)",
            "precision": round(dense_metrics["precision"], 4),
            "recall": round(dense_metrics["recall"], 4),
            "f1_score": round(dense_metrics["f1"], 4),
            "accuracy": round(dense_metrics["accuracy"], 4),
        }
    )

    # 3. Proposed LSTM Autoencoder
    print("Evaluating Proposed: LSTM Autoencoder...")
    lstm_val_err = compute_reconstruction_errors(lstm_model, prep_data.X_val, device=device)
    lstm_test_err = compute_reconstruction_errors(lstm_model, prep_data.X_test, device=device)
    lstm_th = calculate_threshold(lstm_val_err, method="percentile", percentile=95.0, val_labels=prep_data.val_labels)
    lstm_preds = detect_anomalies(lstm_test_err, lstm_th)
    lstm_metrics = calculate_metrics(prep_data.test_labels, lstm_preds)

    results.append(
        {
            "model": "LSTM Autoencoder (Ours)",
            "type": "Recurrent Neural Network (LSTM)",
            "learns_temporal_dynamics": "Yes (Gated Temporal State)",
            "precision": round(lstm_metrics["precision"], 4),
            "recall": round(lstm_metrics["recall"], 4),
            "f1_score": round(lstm_metrics["f1"], 4),
            "accuracy": round(lstm_metrics["accuracy"], 4),
        }
    )

    df_res = pd.DataFrame(results)
    return df_res


def run_all_experiments(
    data_path: str = "data/raw/nyc_taxi.csv",
    labels_path: str = "data/raw/nyc_taxi_labels.json",
    output_dir: str = "results",
    epochs: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    Execute all 4 controlled experiments and save outputs.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    set_seed(42)

    df_raw = pd.read_csv(data_path)
    import json
    with open(labels_path, "r") as f:
        labels_info = json.load(f)

    # 1. Window size experiment
    df_exp1 = run_experiment_1_window_size(df_raw, labels_info, epochs=epochs, device=device)
    df_exp1.to_csv(out_dir / "experiment_1_window_size.csv", index=False)

    # 2. Hidden size experiment
    df_exp2 = run_experiment_2_hidden_size(df_raw, labels_info, epochs=epochs, device=device)
    df_exp2.to_csv(out_dir / "experiment_2_hidden_size.csv", index=False)

    # Standard model for Experiment 3 & 4
    print("\nTraining standard reference LSTM model for Experiments 3 & 4...")
    prep_data = preprocess_pipeline(
        df_raw=df_raw,
        window_size=24,
        step_size=1,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        filter_anomalies_from_train=True,
        anomaly_windows=labels_info.get("anomaly_windows", []),
        anomaly_timestamps=labels_info.get("anomaly_timestamps", []),
    )
    ref_model = LSTMAutoencoder(input_dim=1, hidden_dims=[64, 32], latent_dim=16, seq_len=24)
    ref_model, _ = train_model(
        model=ref_model,
        X_train=prep_data.X_train,
        X_val=prep_data.X_val,
        epochs=epochs,
        learning_rate=0.002,
        device=device,
        verbose=False,
    )

    # 3. Threshold experiment
    df_exp3 = run_experiment_3_threshold(ref_model, prep_data, device=device)
    df_exp3.to_csv(out_dir / "experiment_3_threshold.csv", index=False)

    # 4. Baseline experiment
    df_exp4 = run_experiment_4_baseline(ref_model, prep_data, epochs=epochs, device=device)
    df_exp4.to_csv(out_dir / "experiment_4_baseline.csv", index=False)

    # Generate Markdown Summary Report
    summary_md = f"""# Empirical Experiment Results
## Time-Series Anomaly Detection Using LSTM Autoencoder

All results below were generated from real test-set executions.

### Experiment 1: Sliding Window Size Comparison
Investigating temporal receptive field:
{df_exp1.to_markdown(index=False)}

### Experiment 2: LSTM Hidden Dimension & Model Capacity Comparison
Investigating network architecture depth:
{df_exp2.to_markdown(index=False)}

### Experiment 3: Anomaly Threshold Percentile Sensitivity
Investigating precision vs recall trade-offs across validation percentiles:
{df_exp3.to_markdown(index=False)}

### Experiment 4: Baseline Architecture Comparison
Comparing Statistical Heuristic, Feedforward Autoencoder, and Recurrent LSTM Autoencoder:
{df_exp4.to_markdown(index=False)}
"""
    with open(out_dir / "experiment_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(f"\nAll experiments completed! Results saved to {out_dir}/")
    return {
        "exp1": df_exp1,
        "exp2": df_exp2,
        "exp3": df_exp3,
        "exp4": df_exp4,
    }


if __name__ == "__main__":
    run_all_experiments()
