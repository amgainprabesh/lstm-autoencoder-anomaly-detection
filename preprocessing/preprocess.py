"""
Preprocessing pipeline for Time-Series Anomaly Detection.
Includes data loading, auto-detection of columns, cleaning,
chronological splitting, Min-Max normalization, and sliding-window generation.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass
class PreprocessedData:
    """Container for processed dataset splits, sequences, scalers, and metadata."""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    train_timestamps: np.ndarray
    val_timestamps: np.ndarray
    test_timestamps: np.ndarray
    scaler: MinMaxScaler
    timestamp_col: str
    value_col: str
    window_size: int
    train_labels: Optional[np.ndarray] = None
    val_labels: Optional[np.ndarray] = None
    test_labels: Optional[np.ndarray] = None


def detect_columns(
    df: pd.DataFrame,
    timestamp_col: Optional[str] = None,
    value_col: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Detect timestamp and numerical value columns from DataFrame.
    If explicitly provided, validates their presence.

    Args:
        df (pd.DataFrame): Input dataframe.
        timestamp_col (Optional[str]): Explicit timestamp column name.
        value_col (Optional[str]): Explicit value column name.

    Returns:
        Tuple[str, str]: (timestamp_col, value_col)
    """
    if df.empty:
        raise ValueError("Cannot detect columns: provided DataFrame is empty.")

    columns = list(df.columns)

    # 1. Detect or validate timestamp column
    if timestamp_col is not None:
        if timestamp_col not in columns:
            raise KeyError(f"Specified timestamp column '{timestamp_col}' not found in DataFrame. Available columns: {columns}")
        detected_time = timestamp_col
    else:
        detected_time = None
        # Check datetime dtype first
        for col in columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                detected_time = col
                break
        # Check for common naming patterns
        if detected_time is None:
            time_patterns = ["timestamp", "time", "date", "datetime", "dt", "ts"]
            for col in columns:
                if col.lower().strip() in time_patterns:
                    detected_time = col
                    break
        # Try parsing strings as datetime
        if detected_time is None:
            for col in columns:
                if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                    try:
                        sample = df[col].dropna().head(10)
                        if len(sample) > 0:
                            pd.to_datetime(sample)
                            detected_time = col
                            break
                    except Exception:
                        continue

        if detected_time is None:
            raise ValueError(
                f"Could not automatically detect timestamp column from columns: {columns}. "
                f"Please specify timestamp_col explicitly."
            )

    # 2. Detect or validate numerical value column
    remaining_cols = [c for c in columns if c != detected_time]
    if not remaining_cols:
        raise ValueError("DataFrame contains only a timestamp column; at least one numerical column is required.")

    if value_col is not None:
        if value_col not in columns:
            raise KeyError(f"Specified value column '{value_col}' not found in DataFrame. Available columns: {columns}")
        if value_col == detected_time:
            raise ValueError(f"Value column cannot be identical to timestamp column: '{value_col}'.")
        detected_val = value_col
    else:
        detected_val = None
        # Check for numeric column
        for col in remaining_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                detected_val = col
                break
        # If still none, check if column can be converted to numeric
        if detected_val is None:
            for col in remaining_cols:
                try:
                    pd.to_numeric(df[col].dropna().head(10))
                    detected_val = col
                    break
                except Exception:
                    continue

        if detected_val is None:
            raise ValueError(
                f"Could not automatically detect numerical value column from remaining columns: {remaining_cols}. "
                f"Please specify value_col explicitly."
            )

    return detected_time, detected_val


def load_data(
    filepath_or_buffer: Any,
    timestamp_col: Optional[str] = None,
    value_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, str]:
    """
    Load time-series CSV data and detect timestamp and numerical columns.

    Args:
        filepath_or_buffer (Any): Path to CSV or file-like buffer.
        timestamp_col (Optional[str]): Explicit timestamp column name.
        value_col (Optional[str]): Explicit value column name.

    Returns:
        Tuple[pd.DataFrame, str, str]: (df, timestamp_col, value_col)
    """
    try:
        df = pd.read_csv(filepath_or_buffer)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {str(e)}") from e

    if df.empty:
        raise ValueError("Uploaded or loaded CSV dataset is completely empty.")

    detected_time, detected_val = detect_columns(df, timestamp_col, value_col)
    return df, detected_time, detected_val


def clean_data(
    df: pd.DataFrame,
    timestamp_col: str,
    value_col: str,
    label_col: Optional[str] = None,
    deduplicate_strategy: str = "mean",
) -> pd.DataFrame:
    """
    Clean raw time-series data:
    1. Parse timestamp column to pd.to_datetime.
    2. Ensure numerical values are float.
    3. Remove invalid (inf / -inf) numerical values.
    4. Sort chronologically.
    5. Deduplicate timestamps using the specified strategy ('mean' or 'first').
    6. Impute missing values via linear interpolation + forward/backward fill.
    7. Reset index.
    Preserves label_col if provided or detected.

    Args:
        df (pd.DataFrame): Raw dataframe.
        timestamp_col (str): Timestamp column name.
        value_col (str): Numerical value column name.
        label_col (Optional[str]): Ground-truth binary label column name.
        deduplicate_strategy (str): 'mean' or 'first' for handling identical timestamps.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    selected_cols = [timestamp_col, value_col]
    has_input_labels = bool(label_col and label_col in df.columns)
    if has_input_labels:
        selected_cols.append(label_col)

    df_clean = df[selected_cols].copy()

    # 1. Parse timestamps
    df_clean[timestamp_col] = pd.to_datetime(df_clean[timestamp_col], errors="coerce")
    missing_times = df_clean[timestamp_col].isna().sum()
    if missing_times > 0:
        # Drop rows with completely unparseable timestamps
        df_clean = df_clean.dropna(subset=[timestamp_col])

    if df_clean.empty:
        raise ValueError("All timestamps in the dataset were invalid or null.")

    # 2. Convert value column to numeric
    df_clean[value_col] = pd.to_numeric(df_clean[value_col], errors="coerce")

    # Replace inf and -inf with NaN
    df_clean[value_col] = df_clean[value_col].replace([np.inf, -np.inf], np.nan)

    # 3. Sort chronologically
    df_clean = df_clean.sort_values(by=timestamp_col).reset_index(drop=True)

    # 4. Handle duplicates
    if df_clean[timestamp_col].duplicated().any():
        if deduplicate_strategy == "mean":
            agg_dict = {value_col: "mean"}
            if has_input_labels:
                agg_dict[label_col] = "max"
            df_clean = (
                df_clean.groupby(timestamp_col, as_index=False)
                .agg(agg_dict)
                .sort_values(by=timestamp_col)
            )
        else:
            df_clean = df_clean.drop_duplicates(subset=[timestamp_col], keep="first")

    # 5. Handle missing values via linear interpolation and boundary padding
    if df_clean[value_col].isna().any():
        df_clean[value_col] = (
            df_clean[value_col]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )

    # If any NaNs remain (e.g. entire column was NaN), raise an error
    if df_clean[value_col].isna().any():
        raise ValueError(f"Value column '{value_col}' contains only NaN values; cannot proceed.")

    if has_input_labels and label_col != "is_anomaly":
        df_clean["is_anomaly"] = pd.to_numeric(df_clean[label_col], errors="coerce").fillna(0).astype(int)

    df_clean = df_clean.reset_index(drop=True)
    return df_clean


def tag_ground_truth_labels(
    df: pd.DataFrame,
    timestamp_col: str,
    anomaly_windows: Optional[List[List[str]]] = None,
    anomaly_timestamps: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Tag DataFrame rows with binary ground truth anomaly label (1 = anomaly, 0 = normal)
    based on NAB anomaly windows or exact timestamp lists.

    Args:
        df (pd.DataFrame): Cleaned dataframe with timestamp_col.
        timestamp_col (str): Timestamp column name.
        anomaly_windows (Optional[List[List[str]]]): Pairs of [start_time, end_time].
        anomaly_timestamps (Optional[List[str]]): Specific anomaly timestamps.

    Returns:
        pd.DataFrame: Dataframe with added 'is_anomaly' column (0 or 1).
    """
    df = df.copy()
    df["is_anomaly"] = 0

    if anomaly_windows:
        for win in anomaly_windows:
            start_ts = pd.to_datetime(win[0])
            end_ts = pd.to_datetime(win[1])
            mask = (df[timestamp_col] >= start_ts) & (df[timestamp_col] <= end_ts)
            df.loc[mask, "is_anomaly"] = 1

    if anomaly_timestamps:
        ts_set = {pd.to_datetime(t) for t in anomaly_timestamps}
        mask = df[timestamp_col].isin(ts_set)
        df.loc[mask, "is_anomaly"] = 1

    return df


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform strictly chronological splitting of the time series into Train, Validation, and Test.

    WHY CHRONOLOGICAL SPLITTING IS ESSENTIAL:
    In time-series modeling, data points possess strong temporal autocorrelation.
    Random shuffling causes 'temporal data leakage' where the model trains on future data points
    to predict or reconstruct past ones. Chronological splitting preserves the natural arrow of time:
    Earlier data -> Training (learning baseline normal dynamics)
    Middle data  -> Validation (tuning threshold & early stopping)
    Later data   -> Testing (evaluating out-of-sample anomaly detection)

    Args:
        df (pd.DataFrame): Cleaned chronological dataframe.
        train_ratio (float): Ratio of earlier observations for training.
        val_ratio (float): Ratio of middle observations for validation.
        test_ratio (float): Ratio of later observations for testing.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-3):
        raise ValueError(
            f"Split ratios must sum to 1.0. Given: {train_ratio} + {val_ratio} + {test_ratio} = {total_ratio}"
        )

    n = len(df)
    if n < 10:
        raise ValueError(f"Dataset has only {n} samples, which is too small for chronological splitting.")

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError(
            f"Dataset length {n} is too small to yield non-empty splits with ratios "
            f"({train_ratio}, {val_ratio}, {test_ratio})."
        )

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df


def create_sequences(
    values: np.ndarray,
    window_size: int = 24,
    step_size: int = 1,
    timestamps: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generate sliding-window sequences from a 1D or 2D time series array.
    For an input array of length N, returns sequences of shape (num_windows, window_size, feature_dim).

    Example with window_size=3, step=1:
    values = [10, 20, 30, 40, 50]
    Sequence 0: [10, 20, 30], target timestamp = timestamps[2]
    Sequence 1: [20, 30, 40], target timestamp = timestamps[3]
    Sequence 2: [30, 40, 50], target timestamp = timestamps[4]

    Args:
        values (np.ndarray): 1D array of shape (N,) or 2D array of shape (N, feature_dim).
        window_size (int): Length of each sequence window.
        step_size (int): Stride between consecutive windows (default 1).
        timestamps (Optional[np.ndarray]): Array of timestamps matching values.
        labels (Optional[np.ndarray]): Array of binary anomaly labels matching values.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
            - sequences: Shape (num_windows, window_size, feature_dim)
            - seq_timestamps: Timestamp corresponding to the end of each window.
            - seq_labels: 1 if any point in the window is anomalous, else 0 (if labels provided).
    """
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    n_samples, n_features = values.shape

    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    if n_samples < window_size:
        raise ValueError(
            f"Input data length ({n_samples}) is shorter than sequence window_size ({window_size}). "
            f"Cannot create sliding sequences."
        )

    num_windows = (n_samples - window_size) // step_size + 1

    sequences = np.empty((num_windows, window_size, n_features), dtype=np.float32)
    seq_timestamps = [] if timestamps is not None else None
    seq_labels = [] if labels is not None else None

    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        sequences[i] = values[start_idx:end_idx]

        if timestamps is not None:
            # Associate the window with its concluding timestamp
            seq_timestamps.append(timestamps[end_idx - 1])

        if labels is not None:
            # If any observation within the sliding window is an anomaly, tag sequence as anomalous
            window_labels = labels[start_idx:end_idx]
            is_window_anom = int(np.any(window_labels == 1))
            seq_labels.append(is_window_anom)

    seq_timestamps_arr = np.array(seq_timestamps) if seq_timestamps is not None else None
    seq_labels_arr = np.array(seq_labels, dtype=np.int32) if seq_labels is not None else None

    return sequences, seq_timestamps_arr, seq_labels_arr


def preprocess_pipeline(
    df_raw: pd.DataFrame,
    timestamp_col: Optional[str] = None,
    value_col: Optional[str] = None,
    label_col: Optional[str] = None,
    window_size: int = 24,
    step_size: int = 1,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    filter_anomalies_from_train: bool = True,
    anomaly_windows: Optional[List[List[str]]] = None,
    anomaly_timestamps: Optional[List[str]] = None,
) -> PreprocessedData:
    """
    Complete end-to-end preprocessing pipeline:
    1. Detect timestamp & value columns
    2. Clean dataset (parse dates, sort, deduplicate, interpolate)
    3. Tag ground-truth labels if available (from column or windows)
    4. Split chronologically (Train, Validation, Test)
    5. Fit MinMaxScaler on Train data ONLY (preventing temporal leakage)
    6. Transform Train, Validation, and Test
    7. Generate sliding windows for all three sets
    8. Filter out anomalous sequences from training set if requested and labels exist

    Args:
        df_raw (pd.DataFrame): Raw dataframe.
        timestamp_col (Optional[str]): Timestamp column name.
        value_col (Optional[str]): Value column name.
        label_col (Optional[str]): Ground truth anomaly label column name (if present).
        window_size (int): Sliding window size.
        step_size (int): Window step size.
        train_ratio (float): Ratio for training split.
        val_ratio (float): Ratio for validation split.
        test_ratio (float): Ratio for test split.
        filter_anomalies_from_train (bool): Train only on normal data.
        anomaly_windows (Optional[List[List[str]]]): Ground truth windows.
        anomaly_timestamps (Optional[List[str]]): Ground truth timestamps.

    Returns:
        PreprocessedData: Complete container with splits, sequences, and scaler.
    """
    t_col, v_col = detect_columns(df_raw, timestamp_col, value_col)

    # Auto-detect label column if not explicitly passed
    if label_col is None:
        for candidate in ["is_anomaly", "anomaly", "label", "ground_truth"]:
            if candidate in df_raw.columns and candidate not in [t_col, v_col]:
                label_col = candidate
                break

    df_clean = clean_data(df_raw, t_col, v_col, label_col=label_col)

    # Tag ground truth if provided via windows/timestamps
    has_window_labels = bool(anomaly_windows or anomaly_timestamps)
    has_column_labels = "is_anomaly" in df_clean.columns and df_clean["is_anomaly"].sum() > 0

    if has_window_labels:
        df_clean = tag_ground_truth_labels(df_clean, t_col, anomaly_windows, anomaly_timestamps)
        has_labels = True
    elif has_column_labels:
        has_labels = True
    else:
        df_clean["is_anomaly"] = 0
        has_labels = False

    train_df, val_df, test_df = chronological_split(
        df_clean, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
    )

    # Fit scaler on Train ONLY to prevent data leakage
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_vals_scaled = scaler.fit_transform(train_df[[v_col]].values)
    val_vals_scaled = scaler.transform(val_df[[v_col]].values)
    test_vals_scaled = scaler.transform(test_df[[v_col]].values)

    # Create sequences
    train_ts = train_df[t_col].values
    val_ts = val_df[t_col].values
    test_ts = test_df[t_col].values

    train_lbl = train_df["is_anomaly"].values if has_labels else None
    val_lbl = val_df["is_anomaly"].values if has_labels else None
    test_lbl = test_df["is_anomaly"].values if has_labels else None

    X_train, train_seq_ts, train_seq_lbl = create_sequences(
        train_vals_scaled, window_size, step_size, train_ts, train_lbl
    )
    X_val, val_seq_ts, val_seq_lbl = create_sequences(
        val_vals_scaled, window_size, step_size, val_ts, val_lbl
    )
    X_test, test_seq_ts, test_seq_lbl = create_sequences(
        test_vals_scaled, window_size, step_size, test_ts, test_lbl
    )

    # In semi-supervised anomaly detection, we train on normal behavior
    if filter_anomalies_from_train and train_seq_lbl is not None:
        normal_idx = np.where(train_seq_lbl == 0)[0]
        if len(normal_idx) > 0:
            X_train = X_train[normal_idx]
            train_seq_ts = train_seq_ts[normal_idx]
            train_seq_lbl = train_seq_lbl[normal_idx]

    return PreprocessedData(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        train_timestamps=train_seq_ts,
        val_timestamps=val_seq_ts,
        test_timestamps=test_seq_ts,
        scaler=scaler,
        timestamp_col=t_col,
        value_col=v_col,
        window_size=window_size,
        train_labels=train_seq_lbl,
        val_labels=val_seq_lbl,
        test_labels=test_seq_lbl,
    )
