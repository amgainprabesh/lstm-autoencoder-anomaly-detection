"""
Streamlit Web Application for Time-Series Anomaly Detection Using LSTM Autoencoders.
Provides an interactive dashboard for dataset upload, exploratory data analysis,
model configuration, real-time training/inference, threshold tuning, interactive
visualizations, tabular anomaly inspection, CSV export, and academic theory explanations.
"""
import io
import json
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from evaluation.evaluate import (
    build_results_dataframe,
    calculate_metrics,
    calculate_threshold,
    compute_reconstruction_errors,
    detect_anomalies,
    reconstruct_sequences,
)
from models.lstm_autoencoder import LSTMAutoencoder
from preprocessing.preprocess import (
    clean_data,
    detect_columns,
    load_data,
    preprocess_pipeline,
)
from training.train import load_checkpoint, train_model
from utils.utils import get_device, set_seed
from visualization.plots import (
    plot_interactive_confusion_matrix,
    plot_interactive_error_and_anomalies,
    plot_interactive_raw,
    plot_interactive_reconstruction,
)

# Page configuration
st.set_page_config(
    page_title="LSTM Autoencoder Anomaly Detection",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def get_default_dataset():
    """Load default NAB NYC Taxi dataset and labels."""
    data_path = Path("data/raw/nyc_taxi.csv")
    labels_path = Path("data/raw/nyc_taxi_labels.json")

    if not data_path.exists():
        return None, None

    df = pd.read_csv(data_path)
    labels = {}
    if labels_path.exists():
        with open(labels_path, "r") as f:
            labels = json.load(f)
    return df, labels


def find_matching_checkpoint(window_size: int) -> Optional[Path]:
    """Find a pre-trained checkpoint matching the requested window size."""
    candidates = [
        Path(f"models_saved/best_lstm_autoencoder_w{window_size}.pt"),
        Path("models_saved/best_lstm_autoencoder.pt"),
    ]
    for cp in candidates:
        if cp.exists():
            try:
                data = torch.load(cp, map_location="cpu", weights_only=False)
                meta = data.get("metadata", {})
                ckpt_w = meta.get("window_size")
                if ckpt_w is None or ckpt_w == window_size:
                    return cp
            except Exception:
                continue
    return None


def main():
    st.title("📈 Time-Series Anomaly Detection Using LSTM Autoencoder")
    st.caption("BSc CSIT 6th Semester Neural Networks Course Project | PyTorch Implementation")

    # Tabs
    tab_dashboard, tab_eda, tab_experiments, tab_academic = st.tabs(
        ["📊 Detection Dashboard", "🔍 Exploratory Data Analysis", "🧪 Empirical Experiments", "📚 Academic Theory"]
    )

    # -------------------------------------------------------------
    # Sidebar: Configuration
    # -------------------------------------------------------------
    st.sidebar.header("⚙️ Configuration & Data")

    # Data Source Selection
    data_source = st.sidebar.radio(
        "Data Source",
        ["NAB NYC Taxi Benchmark (Default)", "Upload Custom CSV"],
    )

    df_raw = None
    labels_info = {}
    timestamp_col = None
    value_col = None
    selected_label_col = None
    dataset_name = "NAB NYC Taxi"

    if data_source == "NAB NYC Taxi Benchmark (Default)":
        df_raw, labels_info = get_default_dataset()
        if df_raw is None:
            st.sidebar.error("Default dataset not found in data/raw/nyc_taxi.csv")
            return
        timestamp_col = "timestamp"
        value_col = "value"
        dataset_name = "NAB NYC Taxi (10.3k records)"
        st.sidebar.success(f"Loaded NAB NYC Taxi Dataset ({len(df_raw):,d} records)")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file is not None:
            try:
                df_raw = pd.read_csv(uploaded_file)
                if df_raw.empty:
                    st.sidebar.error("The uploaded CSV is empty.")
                    return
                dataset_name = uploaded_file.name
                st.sidebar.success(f"Uploaded: {dataset_name} ({len(df_raw):,d} rows, {len(df_raw.columns)} cols)")

                cols = list(df_raw.columns)
                try:
                    def_t, def_v = detect_columns(df_raw)
                    t_idx = cols.index(def_t) if def_t in cols else 0
                    v_idx = cols.index(def_v) if def_v in cols else (1 if len(cols) > 1 else 0)
                except Exception:
                    st.sidebar.info("Auto-detection note: Please confirm column mapping below.")
                    t_idx = 0
                    v_idx = 1 if len(cols) > 1 else 0

                timestamp_col = st.sidebar.selectbox("Timestamp Column", cols, index=t_idx)
                val_cols = [c for c in cols if c != timestamp_col]
                if not val_cols:
                    st.sidebar.error("At least one numerical column is required.")
                    return
                v_idx = min(v_idx, len(val_cols) - 1)
                value_col = st.sidebar.selectbox("Numerical Value Column", val_cols, index=v_idx)

                # Optional Ground Truth Column
                label_candidates = ["None"] + [c for c in cols if c not in (timestamp_col, value_col)]
                sel_lbl = st.sidebar.selectbox(
                    "Ground Truth Label Column (Optional)",
                    label_candidates,
                    index=0,
                    help="Select a binary anomaly column (0=Normal, 1=Anomaly) if available.",
                )
                selected_label_col = None if sel_lbl == "None" else sel_lbl
            except Exception as e:
                st.sidebar.error(f"Error parsing uploaded CSV: {str(e)}")
                return
        else:
            st.info("👈 Please select a data source or upload a CSV in the sidebar.")
            return

    # Hyperparameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Hyperparameters")
    window_size = st.sidebar.selectbox("Window Size (Sequence Length)", [12, 24, 48, 72], index=1)
    threshold_method = st.sidebar.selectbox("Threshold Method", ["Percentile", "Mean + k*Std", "IQR"])

    if threshold_method == "Percentile":
        percentile = st.sidebar.slider("Threshold Percentile", 80.0, 99.9, 95.0, step=0.5)
        k_std = 3.0
    elif threshold_method == "Mean + k*Std":
        k_std = st.sidebar.slider("k Standard Deviations", 1.0, 5.0, 3.0, step=0.25)
        percentile = 95.0
    else:
        percentile = 95.0
        k_std = 3.0

    st.sidebar.markdown("---")
    st.sidebar.subheader("Training Settings")
    epochs = st.sidebar.slider("Epochs", 5, 50, 20, step=5)
    batch_size = st.sidebar.selectbox("Batch Size", [32, 64, 128], index=1)
    lr = st.sidebar.select_slider("Learning Rate", [0.0005, 0.001, 0.002, 0.005], value=0.001)

    analyze_button = st.sidebar.button("🚀 Run Anomaly Detection Pipeline", type="primary")

    # Cache key to detect when heavy model training or data preprocessing must rerun
    cache_key = (
        dataset_name,
        timestamp_col,
        value_col,
        selected_label_col,
        window_size,
        epochs,
        batch_size,
        lr,
    )

    if "model_cache" not in st.session_state or st.session_state.get("model_cache_key") != cache_key or analyze_button:
        with st.spinner("Processing data, loading/training LSTM Autoencoder, and computing sequences..."):
            set_seed(42)
            device = get_device()

            # Preprocessing
            prep_data = preprocess_pipeline(
                df_raw=df_raw,
                timestamp_col=timestamp_col,
                value_col=value_col,
                label_col=selected_label_col,
                window_size=window_size,
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
                seq_len=window_size,
            )

            # Check for existing checkpoint
            matching_ckpt = find_matching_checkpoint(window_size)
            history = None

            # Only reuse pre-trained benchmark checkpoint if we are using the benchmark dataset and not forcing retrain
            can_reuse_ckpt = (
                data_source == "NAB NYC Taxi Benchmark (Default)"
                and matching_ckpt is not None
                and not analyze_button
            )

            if can_reuse_ckpt:
                model, _ = load_checkpoint(model, matching_ckpt, device=device)
            else:
                progress_bar = st.progress(0.0, text="Training LSTM Autoencoder...")

                def on_epoch_progress(current_epoch, total_epochs, tr_loss, v_loss):
                    progress_bar.progress(
                        current_epoch / total_epochs,
                        text=f"Training Epoch {current_epoch}/{total_epochs} | Train Loss: {tr_loss:.5f} | Val Loss: {v_loss:.5f}",
                    )

                ckpt_save_path = Path(f"models_saved/best_lstm_autoencoder_w{window_size}.pt")
                model, history = train_model(
                    model=model,
                    X_train=prep_data.X_train,
                    X_val=prep_data.X_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=lr,
                    patience=8,
                    checkpoint_path=ckpt_save_path,
                    device=device,
                    verbose=False,
                    progress_callback=on_epoch_progress,
                )
                progress_bar.empty()

            # Inference (done once per model run)
            val_errors = compute_reconstruction_errors(model, prep_data.X_val, device=device)
            test_errors = compute_reconstruction_errors(model, prep_data.X_test, device=device)
            test_reconstructed = reconstruct_sequences(model, prep_data.X_test, device=device)

            # Inverse scaling
            last_actual_scaled = prep_data.X_test[:, -1, 0]
            last_recon_scaled = test_reconstructed[:, -1, 0]
            actual_unscaled = prep_data.scaler.inverse_transform(last_actual_scaled.reshape(-1, 1)).flatten()
            recon_unscaled = prep_data.scaler.inverse_transform(last_recon_scaled.reshape(-1, 1)).flatten()

            st.session_state["model_cache"] = {
                "prep_data": prep_data,
                "model": model,
                "history": history,
                "val_errors": val_errors,
                "test_errors": test_errors,
                "test_reconstructed": test_reconstructed,
                "actual_unscaled": actual_unscaled,
                "recon_unscaled": recon_unscaled,
                "last_actual_scaled": last_actual_scaled,
                "last_recon_scaled": last_recon_scaled,
                "dataset_name": dataset_name,
            }
            st.session_state["model_cache_key"] = cache_key

    cache = st.session_state.get("model_cache")
    if cache is None:
        return

    # -------------------------------------------------------------
    # REACTIVE THRESHOLD CALCULATION (Instantaneous, No Retraining)
    # -------------------------------------------------------------
    prep_data = cache["prep_data"]
    method_key = "percentile" if threshold_method == "Percentile" else ("std" if threshold_method == "Mean + k*Std" else "iqr")

    threshold = calculate_threshold(
        cache["val_errors"],
        method=method_key,
        percentile=percentile,
        k_std=k_std,
        val_labels=prep_data.val_labels,
    )

    test_preds = detect_anomalies(cache["test_errors"], threshold)

    metrics = None
    if prep_data.test_labels is not None and len(prep_data.test_labels) == len(test_preds):
        metrics = calculate_metrics(prep_data.test_labels, test_preds)

    results_df = build_results_dataframe(
        timestamps=prep_data.test_timestamps,
        actual_values=cache["actual_unscaled"],
        reconstructed_values=cache["recon_unscaled"],
        reconstruction_errors=cache["test_errors"],
        predictions=test_preds,
        ground_truth=prep_data.test_labels,
        threshold=threshold,
    )

    # -------------------------------------------------------------
    # TAB 1: DETECTION DASHBOARD
    # -------------------------------------------------------------
    with tab_dashboard:
        st.subheader("📌 Key Summary Indicators")

        total_obs = len(df_raw)
        total_seq = len(test_preds)
        anom_count = int(np.sum(test_preds))
        anom_pct = (anom_count / total_seq * 100) if total_seq > 0 else 0.0

        # Row 1: Dataset & Sequence Counts
        r1_col1, r1_col2, r1_col3, r1_col4 = st.columns(4)
        r1_col1.metric("Dataset", cache["dataset_name"])
        r1_col2.metric("Total Observations", f"{total_obs:,d}")
        r1_col3.metric("Test Sequences", f"{total_seq:,d}")
        r1_col4.metric("Window Size", f"{window_size} steps")

        # Row 2: Detection & Threshold Metrics
        r2_col1, r2_col2, r2_col3, r2_col4 = st.columns(4)
        r2_col1.metric("Detected Anomalies", f"{anom_count:,d}")
        r2_col2.metric("Anomaly Rate", f"{anom_pct:.2f}%")
        r2_col3.metric("Decision Threshold", f"{threshold:.5f}")

        if metrics:
            r2_col4.metric("Test F1 Score", f"{metrics['f1']:.4f}")
        else:
            r2_col4.metric("Ground Truth", "N/A (Unlabeled)")

        # Row 3: Evaluation metrics breakdown if ground truth exists
        if metrics:
            st.markdown("##### Ground-Truth Evaluation Metrics (Test Split)")
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            m_col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            m_col2.metric("Precision", f"{metrics['precision']:.4f}", help="TP / (TP + FP)")
            m_col3.metric("Recall", f"{metrics['recall']:.4f}", help="TP / (TP + FN)")
            m_col4.metric("Specificity", f"{metrics['specificity']:.4f}", help="TN / (TN + FP)")
            m_col5.metric("Confusion Matrix", f"TP:{metrics['tp']} | FP:{metrics['fp']} | FN:{metrics['fn']}")

        st.markdown("---")

        # Interactive Anomaly Detection Plot
        st.subheader("📉 Interactive Anomaly Detection & Error Analysis")
        st.caption("Pan, zoom, and hover over observations. Ground-truth windows (green circles) and detected anomalies (red crosses) are highlighted.")

        fig_anom = plot_interactive_error_and_anomalies(
            timestamps=prep_data.test_timestamps,
            actual_values=cache["actual_unscaled"],
            errors=cache["test_errors"],
            threshold=threshold,
            predictions=test_preds,
            ground_truth=prep_data.test_labels,
        )
        st.plotly_chart(fig_anom, use_container_width=True)

        # Split columns for Signal Reconstruction overlay and Confusion Matrix
        col_rec, col_cm = st.columns([1.5, 1] if metrics else [1, 0.001])
        with col_rec:
            st.subheader("🔄 Signal vs LSTM Reconstruction Overlay")
            st.caption("The Autoencoder recreates normal temporal dynamics; deviations produce reconstruction error spikes.")
            fig_recon = plot_interactive_reconstruction(
                timestamps=prep_data.test_timestamps[-300:],
                actual=cache["last_actual_scaled"][-300:],
                reconstructed=cache["last_recon_scaled"][-300:],
            )
            st.plotly_chart(fig_recon, use_container_width=True)

        if metrics:
            with col_cm:
                st.subheader("🎯 Confusion Matrix Heatmap")
                st.caption("Annotated True vs Predicted label breakdown.")
                fig_cm = plot_interactive_confusion_matrix(metrics["confusion_matrix"])
                st.plotly_chart(fig_cm, use_container_width=True)

        # Tabular Results & Export
        st.subheader("📋 Detected Anomaly Details & Export")
        view_filter = st.radio(
            "Filter Table:",
            ["Show Anomalies Only", "Show All Sequences"],
            horizontal=True,
        )

        display_df = results_df
        if view_filter == "Show Anomalies Only":
            display_df = display_df[display_df["prediction"] == 1]

        st.dataframe(
            display_df,
            use_container_width=True,
            height=300,
        )

        # CSV Download Button
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Complete Results as CSV",
            data=csv_buffer.getvalue(),
            file_name="detected_anomalies_results.csv",
            mime="text/csv",
        )

    # -------------------------------------------------------------
    # TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------
    with tab_eda:
        st.subheader("🔍 Exploratory Data Analysis")

        eda_col1, eda_col2 = st.columns([2, 1])
        with eda_col1:
            st.markdown("##### Raw Time Series Signal")
            fig_raw = plot_interactive_raw(
                df=df_raw,
                timestamp_col=timestamp_col,
                value_col=value_col,
                anomaly_col="is_anomaly" if "is_anomaly" in df_raw.columns else None,
            )
            st.plotly_chart(fig_raw, use_container_width=True)

        with eda_col2:
            st.markdown("##### Summary Statistics")
            st.write(df_raw[value_col].describe())

            st.markdown("##### Dataset Information")
            st.write({
                "Dataset Name": cache["dataset_name"],
                "Total Rows": len(df_raw),
                "Timestamp Column": timestamp_col,
                "Value Column": value_col,
                "Missing Values": int(df_raw[value_col].isna().sum()),
                "Min Timestamp": str(df_raw[timestamp_col].min()),
                "Max Timestamp": str(df_raw[timestamp_col].max()),
            })

    # -------------------------------------------------------------
    # TAB 3: EMPIRICAL EXPERIMENTS
    # -------------------------------------------------------------
    with tab_experiments:
        st.subheader("🧪 Controlled Scientific Experiments")
        st.markdown(
            "All empirical results below were generated on the benchmark dataset "
            "to answer core research questions regarding temporal modeling."
        )

        exp_tab1, exp_tab2, exp_tab3, exp_tab4 = st.tabs(
            ["Experiment 1: Window Size", "Experiment 2: Hidden Size", "Experiment 3: Threshold Percentile", "Experiment 4: Baseline Comparison"]
        )

        with exp_tab1:
            st.markdown("#### Experiment 1: Impact of Sliding Window Size")
            st.markdown(
                "Evaluates receptive fields: **12 (6 hrs)**, **24 (12 hrs)**, **48 (24 hrs)**, and **72 (36 hrs)**."
            )
            exp1_file = Path("results/experiment_1_window_size.csv")
            if exp1_file.exists():
                df_e1 = pd.read_csv(exp1_file)
                st.dataframe(df_e1, use_container_width=True)
                st.line_chart(df_e1.set_index("window_size")[["precision", "recall", "f1_score"]])
            else:
                st.info("Run `python -m experiments.experiments` or `run_pipeline.py` to generate offline experiment tables.")

        with exp_tab2:
            st.markdown("#### Experiment 2: LSTM Hidden Dimension & Model Capacity")
            st.markdown("Compares shallow, medium, and deep LSTM encoder-decoder architectures.")
            exp2_file = Path("results/experiment_2_hidden_size.csv")
            if exp2_file.exists():
                df_e2 = pd.read_csv(exp2_file)
                st.dataframe(df_e2, use_container_width=True)
            else:
                st.info("Run `python -m experiments.experiments` to generate this table.")

        with exp_tab3:
            st.markdown("#### Experiment 3: Threshold Percentile Sensitivity")
            st.markdown("Exposes the fundamental Precision-Recall trade-off across 90%, 95%, 97.5%, and 99% percentiles.")
            exp3_file = Path("results/experiment_3_threshold.csv")
            if exp3_file.exists():
                df_e3 = pd.read_csv(exp3_file)
                st.dataframe(df_e3, use_container_width=True)
                st.line_chart(df_e3.set_index("percentile")[["precision", "recall", "f1_score"]])
            else:
                st.info("Run `python -m experiments.experiments` to generate this table.")

        with exp_tab4:
            st.markdown("#### Experiment 4: Baseline Comparison")
            st.markdown("Proves whether recurrent temporal modeling outperforms statistical filtering and static feedforward nets.")
            exp4_file = Path("results/experiment_4_baseline.csv")
            if exp4_file.exists():
                df_e4 = pd.read_csv(exp4_file)
                st.dataframe(df_e4, use_container_width=True)
            else:
                st.info("Run `python -m experiments.experiments` to generate this table.")

    # -------------------------------------------------------------
    # TAB 4: ACADEMIC THEORY & VIVA PREPARATION
    # -------------------------------------------------------------
    with tab_academic:
        st.subheader("📚 Academic Theory & Neural Network Foundations")
        st.markdown("""
### 1. Artificial Neuron & Perceptron Foundation
An artificial neuron is the foundational processing unit of deep neural networks.
It computes an affine transformation followed by an activation function:
$$z = \\sum_{i=1}^{n} w_i x_i + b = W^T x + b$$
$$a = \\sigma(z)$$
where $W$ represents the trainable synaptic weights, $b$ is the scalar bias, and $\\sigma(\\cdot)$ is a non-linear activation.

---

### 2. Nonlinear Activation Functions
Without non-linear activation functions, stacking multiple neural network layers merely computes a composition of linear functions:
$$W_2(W_1 x + b_1) + b_2 = W' x + b'$$
Non-linearities allow the network to approximate arbitrary non-linear functions (Universal Approximation Theorem).
Common activations used in this system:
- **Tanh**: $\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}} \\in (-1, 1)$ (zero-centered, used in LSTM cell state updates)
- **Sigmoid**: $\\sigma(z) = \\frac{1}{1 + e^{-z}} \\in (0, 1)$ (gating mechanism for LSTM gates)
- **ReLU**: $\\text{ReLU}(z) = \\max(0, z)$ (sparsity and gradient flow in dense baselines)

---

### 3. Loss Function & Backpropagation
The network minimizes the **Mean Squared Error (MSE)** reconstruction loss:
$$\\mathcal{L}(X, \\hat{X}) = \\frac{1}{B \\cdot T \\cdot D} \\sum_{b=1}^{B} \\sum_{t=1}^{T} \\sum_{d=1}^{D} (x_{b,t,d} - \\hat{x}_{b,t,d})^2$$
Gradients are calculated via the chain rule during **Backpropagation Through Time (BPTT)**:
$$\\frac{\\partial \\mathcal{L}}{\\partial W} = \\sum_{t=1}^{T} \\frac{\\partial \\mathcal{L}}{\\partial h_t} \\frac{\\partial h_t}{\\partial W}$$
The optimizer (Adam) updates parameters via adaptive moment estimation:
$$m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t$$
$$v_t = \\beta_2 v_{t-1} + (1-\\beta_2) g_t^2$$
$$\\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t$$

---

### 4. Recurrent Neural Networks & Long Short-Term Memory (LSTM)
Standard feedforward networks treat inputs as independent and identically distributed (i.i.d.), ignoring sequential autocorrelation.
Vanilla RNNs suffer from **vanishing and exploding gradients** when backpropagating across long time horizons due to repeated matrix multiplications:
$$\\prod_{k=t+1}^{T} W_{hh}^T$$
**LSTM solves this via gated constant error carousels (Cell State $c_t$):**
1. **Forget Gate**: Decides what past information to discard:
   $$f_t = \\sigma(W_f [h_{t-1}, x_t] + b_f)$$
2. **Input Gate**: Decides which new values will be updated:
   $$i_t = \\sigma(W_i [h_{t-1}, x_t] + b_i)$$
3. **Candidate Memory Cell**: New candidate values:
   $$\\tilde{c}_t = \\tanh(W_c [h_{t-1}, x_t] + b_c)$$
4. **Cell State Update**: Linear vector addition prevents vanishing gradients:
   $$c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t$$
5. **Output Gate**: Determines next hidden state:
   $$o_t = \\sigma(W_o [h_{t-1}, x_t] + b_o)$$
   $$h_t = o_t \\odot \\tanh(c_t)$$

---

### 5. Autoencoder & Reconstruction Error Anomaly Detection
An Autoencoder is an unsupervised bottleneck architecture comprising:
1. **Encoder**: $z = f_\\theta(X)$ mapping high-dimensional input sequences $(T \\times D)$ into a low-dimensional latent vector $z \\in \\mathbb{R}^d$.
2. **Decoder**: $\\hat{X} = g_\\phi(z)$ reconstructing the sequence from the compressed latent code.

**Anomaly Detection Principle:**
- During training on predominantly **normal** time-series data, the network parameters $(\\theta, \\phi)$ adapt to minimize reconstruction error on normal temporal dynamics (seasonality, diurnal cycles, smooth transitions).
- When anomalous sequences appear (spikes, dropouts, blizzard shutdowns, marathon crowds), the encoder compresses them into latent points that decode into expected normal patterns.
- Consequently, the **Reconstruction Error** $\\text{MSE}(X, \\hat{X})$ spikes significantly on anomalous sequences:
$$\\text{Anomaly Score}(t) = \\frac{1}{T} \\sum_{i=1}^{T} (x_{t,i} - \\hat{x}_{t,i})^2$$
$$\\hat{y}(t) = \\begin{cases} 1 & \\text{if Score}(t) > \\tau \\\\ 0 & \\text{otherwise} \\end{cases}$$
where $\\tau$ is chosen via validation error distribution (e.g. 95th percentile).
""")


if __name__ == "__main__":
    main()
