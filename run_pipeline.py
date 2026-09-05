"""
End-to-End Pipeline Execution Script.
Orchestrates:
1. Loading configuration and setting random seeds
2. Preprocessing raw dataset and generating chronological sliding windows
3. Instantiating LSTM Autoencoder
4. Training with early stopping and best-model checkpointing
5. Computing reconstruction errors and setting principled anomaly threshold
6. Evaluating against test ground truth labels (Precision, Recall, F1)
7. Generating and saving all publication-quality static visualizations
8. Exporting anomaly predictions to results CSV
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.evaluate import (
    build_results_dataframe,
    calculate_metrics,
    calculate_threshold,
    compute_reconstruction_errors,
    detect_anomalies,
    find_best_threshold,
    reconstruct_sequences,
)
from models.lstm_autoencoder import LSTMAutoencoder
from preprocessing.preprocess import preprocess_pipeline
from training.train import load_checkpoint, train_model
from utils.utils import ensure_dirs, get_device, load_config, save_json, set_seed
from visualization.plots import (
    plot_confusion_matrix_matplotlib,
    plot_detected_anomalies_matplotlib,
    plot_loss_curves_matplotlib,
    plot_raw_time_series_matplotlib,
    plot_reconstruction_error_matplotlib,
    plot_reconstruction_matplotlib,
)


def run_pipeline(config_path: str = "config.yaml", force_train: bool = True):
    print("=" * 70)
    print("TIME-SERIES ANOMALY DETECTION USING LSTM AUTOENCODER")
    print("=" * 70)

    # 1. Configuration & Directories
    config = load_config(config_path)
    set_seed(config.get("project", {}).get("random_seed", 42))
    device = get_device()
    print(f"[*] Compute Device: {device}")

    res_dir = Path(config.get("paths", {}).get("results_dir", "results"))
    plot_dir = Path(config.get("paths", {}).get("plots_dir", "results/plots"))
    ckpt_dir = Path(config.get("training", {}).get("checkpoint_dir", "models_saved"))
    ensure_dirs([str(res_dir), str(plot_dir), str(ckpt_dir)])

    # 2. Data Loading & Preprocessing
    data_cfg = config["data"]
    raw_path = Path(data_cfg["raw_path"])
    labels_path = Path(data_cfg.get("labels_path", "data/raw/nyc_taxi_labels.json"))

    print(f"[*] Loading raw dataset from: {raw_path}")
    df_raw = pd.read_csv(raw_path)
    print(f"[*] Total observations: {len(df_raw):,d}")

    labels_info = {}
    if labels_path.exists():
        with open(labels_path, "r") as f:
            labels_info = json.load(f)

    window_size = data_cfg.get("window_size", 24)
    print(f"[*] Preprocessing with window_size={window_size}, chronological split "
          f"({data_cfg['train_ratio']*100:.0f}% Train, {data_cfg['val_ratio']*100:.0f}% Val, {data_cfg['test_ratio']*100:.0f}% Test)...")

    prep_data = preprocess_pipeline(
        df_raw=df_raw,
        timestamp_col=data_cfg.get("timestamp_col"),
        value_col=data_cfg.get("value_col"),
        window_size=window_size,
        step_size=data_cfg.get("step_size", 1),
        train_ratio=data_cfg.get("train_ratio", 0.60),
        val_ratio=data_cfg.get("val_ratio", 0.20),
        test_ratio=data_cfg.get("test_ratio", 0.20),
        filter_anomalies_from_train=True,
        anomaly_windows=labels_info.get("anomaly_windows", []),
        anomaly_timestamps=labels_info.get("anomaly_timestamps", []),
    )

    # Save cleaned processed dataset
    proc_path = Path(data_cfg.get("processed_path", "data/processed/nyc_taxi_processed.csv"))
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    df_combined_clean = pd.concat([prep_data.train_df, prep_data.val_df, prep_data.test_df])
    df_combined_clean.to_csv(proc_path, index=False)
    print(f"[*] Processed dataset saved to: {proc_path}")

    print(f"[*] Sequence shapes: Train={prep_data.X_train.shape}, Val={prep_data.X_val.shape}, Test={prep_data.X_test.shape}")

    # Plot 1: Raw Time Series
    plot_raw_time_series_matplotlib(
        df=pd.concat([prep_data.train_df, prep_data.val_df, prep_data.test_df]),
        timestamp_col=prep_data.timestamp_col,
        value_col=prep_data.value_col,
        anomaly_mask=(pd.concat([prep_data.train_df, prep_data.val_df, prep_data.test_df])["is_anomaly"] == 1),
        title=f"Raw Time Series: {raw_path.stem} with Ground Truth Anomalies",
        save_path=plot_dir / "01_raw_time_series.png",
    )

    # 3. Model Architecture
    model_cfg = config["model"]
    model = LSTMAutoencoder(
        input_dim=model_cfg.get("input_dim", 1),
        hidden_dims=model_cfg.get("hidden_dims", [64, 32]),
        latent_dim=model_cfg.get("latent_dim", 16),
        seq_len=window_size,
        dropout=model_cfg.get("dropout", 0.1),
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] LSTM Autoencoder instantiated. Trainable parameters: {total_params:,d}")

    # 4. Training
    train_cfg = config["training"]
    ckpt_path = ckpt_dir / train_cfg.get("best_model_name", "best_lstm_autoencoder.pt")

    if force_train or not ckpt_path.exists():
        print(f"[*] Starting training for up to {train_cfg['epochs']} epochs...")
        t_start = time.time()
        model, history = train_model(
            model=model,
            X_train=prep_data.X_train,
            X_val=prep_data.X_val,
            epochs=train_cfg.get("epochs", 40),
            batch_size=train_cfg.get("batch_size", 64),
            learning_rate=train_cfg.get("learning_rate", 0.001),
            optimizer_name=train_cfg.get("optimizer", "adam"),
            weight_decay=train_cfg.get("weight_decay", 1e-5),
            patience=train_cfg.get("early_stopping_patience", 10),
            min_delta=train_cfg.get("early_stopping_min_delta", 1e-4),
            checkpoint_path=ckpt_path,
            device=device,
            verbose=True,
        )
        print(f"[*] Training finished in {time.time() - t_start:.1f}s. Best Epoch: {history['best_epoch']}, Best Val Loss: {history['best_val_loss']:.6f}")

        # Plot 2: Training & Validation Loss
        plot_loss_curves_matplotlib(
            history=history,
            title="LSTM Autoencoder Training and Validation Loss Curve",
            save_path=plot_dir / "02_loss_curve.png",
        )
    else:
        print(f"[*] Loading existing checkpoint from: {ckpt_path}")
        model, _ = load_checkpoint(model, ckpt_path, device=device)

    # 5. Reconstruction & Error Computation
    print("[*] Computing reconstruction errors on Validation and Test sets...")
    val_errors = compute_reconstruction_errors(model, prep_data.X_val, device=device)
    test_errors = compute_reconstruction_errors(model, prep_data.X_test, device=device)
    test_reconstructed = reconstruct_sequences(model, prep_data.X_test, device=device)

    # 6. Threshold Selection
    anom_cfg = config.get("anomaly_detection", {})
    method = anom_cfg.get("threshold_method", "percentile")
    p = anom_cfg.get("percentile", 95.0)
    k_std = anom_cfg.get("k_std", 3.0)

    threshold = calculate_threshold(
        val_errors,
        method=method,
        percentile=p,
        k_std=k_std,
        val_labels=prep_data.val_labels,
    )
    print(f"[*] Anomaly decision threshold ({method}, p={p}): {threshold:.6f}")

    # If ground truth available, also find optimal threshold on validation set
    if prep_data.val_labels is not None and np.sum(prep_data.val_labels) > 0:
        best_th, best_p, best_val_m = find_best_threshold(val_errors, prep_data.val_labels)
        print(f"[*] Best validation threshold found: {best_th:.6f} ({best_p}th percentile, Val F1={best_val_m['f1']:.4f})")

    # 7. Detection on Test Set
    test_preds = detect_anomalies(test_errors, threshold)
    test_anom_count = int(np.sum(test_preds))
    print(f"[*] Detected {test_anom_count:,d} anomalies in Test set ({test_anom_count / len(test_preds) * 100:.2f}%)")

    # 8. Evaluation against ground-truth
    metrics = {}
    if prep_data.test_labels is not None:
        metrics = calculate_metrics(prep_data.test_labels, test_preds)
        print("\n" + "=" * 50)
        print("TEST SET EVALUATION METRICS:")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f} (TP / [TP + FP])")
        print(f"  Recall:      {metrics['recall']:.4f} (TP / [TP + FN])")
        print(f"  F1 Score:    {metrics['f1']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Confusion Matrix: TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}")
        print("=" * 50 + "\n")

        # Plot 3: Confusion Matrix
        plot_confusion_matrix_matplotlib(
            cm=metrics["confusion_matrix"],
            title="Confusion Matrix on Test Dataset",
            save_path=plot_dir / "03_confusion_matrix.png",
        )
        save_json(metrics, res_dir / "test_evaluation_metrics.json")

    # 9. Visualization Plots
    # Inverse transform values for realistic plotting
    last_step_actual_scaled = prep_data.X_test[:, -1, 0]
    last_step_recon_scaled = test_reconstructed[:, -1, 0]
    actual_unscaled = prep_data.scaler.inverse_transform(last_step_actual_scaled.reshape(-1, 1)).flatten()
    recon_unscaled = prep_data.scaler.inverse_transform(last_step_recon_scaled.reshape(-1, 1)).flatten()

    # Plot 4: Signal vs Reconstructed
    plot_reconstruction_matplotlib(
        timestamps=prep_data.test_timestamps,
        actual=last_step_actual_scaled,
        reconstructed=last_step_recon_scaled,
        num_points=350,
        title="Test Set: Original vs LSTM Autoencoder Reconstruction",
        save_path=plot_dir / "04_reconstruction_overlay.png",
    )

    # Plot 5: Reconstruction Error with Threshold
    plot_reconstruction_error_matplotlib(
        timestamps=prep_data.test_timestamps,
        errors=test_errors,
        threshold=threshold,
        title="Reconstruction Error (MSE) vs Anomaly Threshold",
        save_path=plot_dir / "05_reconstruction_error_threshold.png",
    )

    # Plot 6: Detected Anomalies
    plot_detected_anomalies_matplotlib(
        timestamps=prep_data.test_timestamps,
        values=actual_unscaled,
        predictions=test_preds,
        ground_truth=prep_data.test_labels,
        title="Detected Anomalies on Unseen Test Series",
        save_path=plot_dir / "06_detected_anomalies.png",
    )

    # 10. Save Detailed Results CSV
    df_results = build_results_dataframe(
        timestamps=prep_data.test_timestamps,
        actual_values=actual_unscaled,
        reconstructed_values=recon_unscaled,
        reconstruction_errors=test_errors,
        predictions=test_preds,
        ground_truth=prep_data.test_labels,
        threshold=threshold,
    )
    results_csv = res_dir / "detected_anomalies.csv"
    df_results.to_csv(results_csv, index=False)
    print(f"[*] Anomaly detection table exported to: {results_csv}")
    print(f"[*] All static figures saved in: {plot_dir}")
    print("[*] Pipeline completed successfully!")

    return prep_data, model, threshold, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete time-series anomaly detection pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--no-train", action="store_true", help="Skip training if model checkpoint exists")
    args = parser.parse_args()

    run_pipeline(config_path=args.config, force_train=not args.no_train)
