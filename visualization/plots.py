"""
Visualization module for Time-Series Anomaly Detection.
Provides Matplotlib plotting for saving static figures and Plotly plotting
for interactive Streamlit dashboard rendering.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Set clean matplotlib styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


def plot_raw_time_series_matplotlib(
    df: pd.DataFrame,
    timestamp_col: str,
    value_col: str,
    title: str = "Raw Time Series Data",
    anomaly_mask: Optional[pd.Series] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot raw time-series with optional anomaly highlighting using Matplotlib."""
    fig, ax = plt.subplots(figsize=(14, 5), dpi=dpi)
    ax.plot(df[timestamp_col], df[value_col], color="#1f77b4", linewidth=1.2, label="Signal Value")

    if anomaly_mask is not None and anomaly_mask.any():
        anom_points = df[anomaly_mask]
        ax.scatter(
            anom_points[timestamp_col],
            anom_points[value_col],
            color="#d62728",
            s=25,
            zorder=5,
            label="Ground Truth Anomaly",
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Timestamp", fontsize=11)
    ax.set_ylabel(value_col.capitalize(), fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)

    return fig


def plot_loss_curves_matplotlib(
    history: Dict[str, Any],
    title: str = "LSTM Autoencoder Training & Validation Loss",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot training and validation loss curves over epochs using Matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    epochs = range(1, len(train_loss) + 1)

    ax.plot(epochs, train_loss, color="#1f77b4", linewidth=2.0, label="Training Loss (MSE)")
    if val_loss:
        ax.plot(epochs, val_loss, color="#ff7f0e", linewidth=2.0, linestyle="--", label="Validation Loss (MSE)")

    best_epoch = history.get("best_epoch")
    if best_epoch and val_loss and best_epoch <= len(val_loss):
        ax.axvline(best_epoch, color="#2ca02c", linestyle=":", linewidth=1.8, label=f"Best Model (Epoch {best_epoch})")
        ax.scatter([best_epoch], [val_loss[best_epoch - 1]], color="#2ca02c", s=60, zorder=5)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Reconstruction Loss (MSE)", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.set_yscale("log")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)

    return fig


def plot_reconstruction_matplotlib(
    timestamps: np.ndarray,
    actual: np.ndarray,
    reconstructed: np.ndarray,
    num_points: int = 400,
    title: str = "Original vs Reconstructed Signal",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot original signal overlaid with autoencoder reconstruction."""
    fig, ax = plt.subplots(figsize=(14, 5), dpi=dpi)
    n = min(len(timestamps), num_points)

    ax.plot(timestamps[-n:], actual[-n:], color="#1f77b4", linewidth=1.5, label="Original Signal")
    ax.plot(
        timestamps[-n:],
        reconstructed[-n:],
        color="#ff7f0e",
        linewidth=1.5,
        linestyle="--",
        label="Reconstructed Signal",
    )

    ax.set_title(f"{title} (Last {n} Observations)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Timestamp", fontsize=11)
    ax.set_ylabel("Normalized Value", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)

    return fig


def plot_reconstruction_error_matplotlib(
    timestamps: np.ndarray,
    errors: np.ndarray,
    threshold: float,
    title: str = "Reconstruction Error and Anomaly Threshold",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot reconstruction error alongside the anomaly decision threshold."""
    fig, ax = plt.subplots(figsize=(14, 5), dpi=dpi)
    ax.plot(timestamps, errors, color="#7f7f7f", linewidth=1.0, alpha=0.8, label="Reconstruction Error (MSE)")
    ax.axhline(threshold, color="#d62728", linestyle="--", linewidth=2.0, label=f"Anomaly Threshold ({threshold:.5f})")

    # Highlight points above threshold
    anom_idx = np.where(errors > threshold)[0]
    if len(anom_idx) > 0:
        ax.scatter(timestamps[anom_idx], errors[anom_idx], color="#d62728", s=15, zorder=5, label="Detected Anomaly")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Timestamp", fontsize=11)
    ax.set_ylabel("Reconstruction Error (MSE)", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)

    return fig


def plot_detected_anomalies_matplotlib(
    timestamps: np.ndarray,
    values: np.ndarray,
    predictions: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    title: str = "Time Series with Detected Anomalies Highlighted",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Highlight detected anomaly regions against actual observations."""
    fig, ax = plt.subplots(figsize=(14, 5), dpi=dpi)
    ax.plot(timestamps, values, color="#1f77b4", linewidth=1.0, alpha=0.85, label="Actual Value")

    detected_idx = np.where(predictions == 1)[0]
    if len(detected_idx) > 0:
        ax.scatter(
            timestamps[detected_idx],
            values[detected_idx],
            color="#d62728",
            s=30,
            zorder=6,
            label=f"Predicted Anomaly ({len(detected_idx)} sequences)",
        )

    if ground_truth is not None:
        gt_idx = np.where(ground_truth == 1)[0]
        if len(gt_idx) > 0:
            ax.scatter(
                timestamps[gt_idx],
                values[gt_idx],
                edgecolors="#2ca02c",
                facecolors="none",
                s=60,
                linewidths=1.5,
                zorder=5,
                label="Ground Truth Anomaly",
            )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Timestamp", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)

    return fig


def plot_confusion_matrix_matplotlib(
    cm: List[List[int]],
    title: str = "Confusion Matrix",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot annotated confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
    cm_arr = np.array(cm)

    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    classes = ["Normal (0)", "Anomaly (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=10)

    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_arr[i, j]:,d}",
                horizontalalignment="center",
                color="white" if cm_arr[i, j] > thresh else "black",
                fontsize=13,
                fontweight="bold",
            )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)

    return fig


# -------------------------------------------------------------
# Interactive Plotly Visualizations (For Streamlit & Web GUI)
# -------------------------------------------------------------


def plot_interactive_raw(
    df: pd.DataFrame,
    timestamp_col: str,
    value_col: str,
    anomaly_col: Optional[str] = None,
) -> go.Figure:
    """Interactive Plotly time-series chart with zoom, hover, and range sliders."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[timestamp_col],
            y=df[value_col],
            mode="lines",
            name="Observed Value",
            line=dict(color="#1f77b4", width=1.5),
        )
    )

    if anomaly_col and anomaly_col in df.columns:
        anom_df = df[df[anomaly_col] == 1]
        if not anom_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=anom_df[timestamp_col],
                    y=anom_df[value_col],
                    mode="markers",
                    name="Ground Truth Anomaly",
                    marker=dict(color="#d62728", size=6, symbol="circle"),
                )
            )

    fig.update_layout(
        title="Interactive Time Series Signal",
        xaxis_title="Timestamp",
        yaxis_title=value_col,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_interactive_reconstruction(
    timestamps: np.ndarray,
    actual: np.ndarray,
    reconstructed: np.ndarray,
) -> go.Figure:
    """Interactive Plotly overlay of Original vs Reconstructed signals."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=actual,
            mode="lines",
            name="Original Signal",
            line=dict(color="#1f77b4", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=reconstructed,
            mode="lines",
            name="LSTM Reconstructed",
            line=dict(color="#ff7f0e", width=1.5, dash="dot"),
        )
    )
    fig.update_layout(
        title="Original vs LSTM Reconstructed Signal",
        xaxis_title="Timestamp",
        yaxis_title="Normalized Value",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_interactive_error_and_anomalies(
    timestamps: np.ndarray,
    actual_values: np.ndarray,
    errors: np.ndarray,
    threshold: float,
    predictions: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
) -> go.Figure:
    """Interactive two-panel Plotly figure: Top shows signal + anomalies, Bottom shows error + threshold."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Actual Signal with Detected Anomalies", "Reconstruction Error (MSE) & Decision Threshold"),
    )

    # Subplot 1: Signal & Anomaly Markers
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=actual_values,
            mode="lines",
            name="Signal",
            line=dict(color="#1f77b4", width=1.2),
        ),
        row=1,
        col=1,
    )

    # Ground Truth Anomaly Markers if available
    if ground_truth is not None:
        gt_indices = np.where(ground_truth == 1)[0]
        if len(gt_indices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=timestamps[gt_indices],
                    y=actual_values[gt_indices],
                    mode="markers",
                    name="Ground Truth Anomaly",
                    marker=dict(color="#2ca02c", size=8, symbol="circle-open", line=dict(width=2)),
                ),
                row=1,
                col=1,
            )

    anom_indices = np.where(predictions == 1)[0]
    if len(anom_indices) > 0:
        fig.add_trace(
            go.Scatter(
                x=timestamps[anom_indices],
                y=actual_values[anom_indices],
                mode="markers",
                name="Detected Anomaly",
                marker=dict(color="#d62728", size=6, symbol="x"),
            ),
            row=1,
            col=1,
        )

    # Subplot 2: Reconstruction Error and Threshold
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=errors,
            mode="lines",
            name="Reconstruction Error",
            line=dict(color="#6c757d", width=1.0),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[timestamps[0], timestamps[-1]],
            y=[threshold, threshold],
            mode="lines",
            name=f"Threshold ({threshold:.4f})",
            line=dict(color="#dc3545", width=2.0, dash="dash"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=650,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_interactive_confusion_matrix(cm: List[List[int]]) -> go.Figure:
    """Interactive annotated confusion matrix heatmap using Plotly."""
    cm_arr = np.array(cm)
    labels = ["Normal (0)", "Anomaly (1)"]

    annotations = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_arr[i, j]
            desc = ""
            if i == 0 and j == 0:
                desc = " (TN)"
            elif i == 0 and j == 1:
                desc = " (FP)"
            elif i == 1 and j == 0:
                desc = " (FN)"
            elif i == 1 and j == 1:
                desc = " (TP)"

            annotations.append(
                dict(
                    x=labels[j],
                    y=labels[i],
                    text=f"<b>{val:,d}</b>{desc}",
                    showarrow=False,
                    font=dict(color="white" if val > cm_arr.max() / 2 else "black", size=14),
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_arr,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=True,
        )
    )
    fig.update_layout(
        title="Confusion Matrix (Evaluation on Test Split)",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        annotations=annotations,
        template="plotly_white",
        width=450,
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig
