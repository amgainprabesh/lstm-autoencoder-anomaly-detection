"""
REST API Inference Service for the LSTM Autoencoder Anomaly Detector.

Exposes the trained model as an HTTP service — the third interface of the
project alongside the CLI pipeline (run_pipeline.py) and the Streamlit app
(app/app.py):

    GET  /                Service metadata
    GET  /health          Model / device / checkpoint status
    POST /api/v1/predict  Anomaly detection on a submitted univariate series

Run from the project root:

    uvicorn api.server:app --host 0.0.0.0 --port 8000

Example:

    curl -X POST http://localhost:8000/api/v1/predict \
      -H "Content-Type: application/json" \
      -d '{"values": [103.0, 101.5, 99.2, ...]}'
"""
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path so project modules import regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from evaluation.evaluate import (
    calculate_threshold,
    compute_reconstruction_errors,
    detect_anomalies,
)
from models.lstm_autoencoder import LSTMAutoencoder
from training.train import load_checkpoint
from utils.utils import get_device

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models_saved" / "best_lstm_autoencoder.pt"
DEFAULT_WINDOW_SIZE = 24

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Request body for the /api/v1/predict endpoint."""

    values: List[float] = Field(
        ...,
        min_length=1,
        description="Univariate time-series values (at least window_size points).",
    )
    window_size: Optional[int] = Field(
        None,
        ge=2,
        description="Sliding window length (default: checkpoint metadata or 24).",
    )
    threshold: Optional[float] = Field(
        None,
        gt=0.0,
        description="Explicit anomaly threshold. Defaults to percentile/statistical method on input scores.",
    )
    method: str = Field(
        "percentile",
        description="Threshold method when no explicit threshold: 'percentile', 'std', or 'iqr'.",
    )
    percentile: float = Field(
        95.0,
        ge=0.0,
        le=100.0,
        description="Percentile for threshold selection (default 95.0).",
    )
    k_std: float = Field(
        3.0,
        ge=0.0,
        description="Std-dev multiplier when method='std'.",
    )
    timestamps: Optional[List[str]] = Field(
        None,
        description="Optional timestamps aligned with `values`, echoed in the response.",
    )


class PredictResponse(BaseModel):
    """Response body of the /api/v1/predict endpoint."""

    model: str
    window_size: int
    num_windows: int
    threshold: float
    threshold_method: str
    scores: List[float]
    predictions: List[int]
    anomaly_count: int
    timestamps: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model():
    """Load the trained LSTM autoencoder from the default checkpoint."""
    if not DEFAULT_CHECKPOINT.exists():
        return None, {}

    device = get_device()
    metadata = {}
    try:
        checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)
        metadata = checkpoint.get("metadata", {})
        hidden_dims = metadata.get("hidden_dims", [64, 32])
        latent_dim = int(metadata.get("latent_dim", 16))
        seq_len = int(metadata.get("window_size") or DEFAULT_WINDOW_SIZE)
    except Exception:
        hidden_dims, latent_dim, seq_len = [64, 32], 16, DEFAULT_WINDOW_SIZE

    model = LSTMAutoencoder(
        input_dim=1,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        seq_len=seq_len,
    )
    try:
        model, _ = load_checkpoint(model, DEFAULT_CHECKPOINT, device=device)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to load checkpoint {DEFAULT_CHECKPOINT}: {exc}")

    return model, metadata


MODEL, MODEL_METADATA = _load_model()
DEVICE = get_device()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LSTM Autoencoder Anomaly Detection API",
    description="REST interface for the BSc CSIT LSTM Autoencoder time-series anomaly detector.",
    version="1.0.0",
)


@app.get("/")
def service_info() -> dict:
    """Service metadata and usage hints."""
    return {
        "service": "lstm-autoencoder-anomaly-detection",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /api/v1/predict",
        },
        "model_loaded": MODEL is not None,
    }


@app.get("/health")
def health() -> dict:
    """Model, device, and checkpoint status."""
    return {
        "status": "ok" if MODEL is not None else "degraded",
        "device": str(DEVICE),
        "model_loaded": MODEL is not None,
        "checkpoint": str(DEFAULT_CHECKPOINT),
        "checkpoint_metadata": MODEL_METADATA if MODEL_METADATA else None,
    }


def _build_windows(values: np.ndarray, window_size: int) -> np.ndarray:
    """Convert a flat 1-D series into overlapping windows (N, window_size, 1)."""
    if len(values) < window_size:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least {window_size} values, got {len(values)}.",
        )
    windows = np.lib.stride_tricks.sliding_window_view(values, window_size)
    return windows.reshape(-1, window_size, 1).astype(np.float32)


@app.post("/api/v1/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Score a submitted univariate series for anomalies.

    The series is MinMax-scaled on its own range (online normalization;
    approximates the training-split scaler for demonstration purposes),
    sliced into overlapping windows, and each window is reconstructed by the
    LSTM autoencoder. Windows whose reconstruction error exceeds the decision
    threshold are flagged anomalous.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="No trained checkpoint available. Run `python run_pipeline.py` first.",
        )

    values = np.asarray(request.values, dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise HTTPException(status_code=422, detail="`values` must be finite numbers.")

    window_size = request.window_size or int(MODEL_METADATA.get("window_size") or DEFAULT_WINDOW_SIZE)

    # Online MinMax normalization (0-1) on the submitted series
    vmin, vmax = float(values.min()), float(values.max())
    if vmax - vmin < 1e-12:
        raise HTTPException(status_code=422, detail="`values` must contain at least two distinct values.")
    scaled = (values - vmin) / (vmax - vmin)

    windows = _build_windows(scaled, window_size)
    scores = compute_reconstruction_errors(MODEL, windows, device=DEVICE)

    if request.threshold is not None:
        threshold = float(request.threshold)
    else:
        threshold = calculate_threshold(
            scores,
            method=request.method,
            percentile=request.percentile,
            k_std=request.k_std,
        )

    preds = detect_anomalies(scores, threshold).astype(int)
    return PredictResponse(
        model="LSTMAutoencoder",
        window_size=window_size,
        num_windows=len(windows),
        threshold=threshold,
        threshold_method=request.method if request.threshold is None else "explicit",
        scores=[float(s) for s in scores],
        predictions=[int(p) for p in preds],
        anomaly_count=int(preds.sum()),
        timestamps=request.timestamps,
    )


# ---------------------------------------------------------------------------
# Development server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)