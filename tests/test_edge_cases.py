"""
Edge case testing for robustness:
- Empty dataset
- Missing numerical column
- Missing timestamps / unparseable dates
- Entirely NaN or inf columns
- Too-short time series compared to window size
- Zero divisions in evaluation
"""
import io
import numpy as np
import pandas as pd
import pytest

from evaluation.evaluate import calculate_metrics, calculate_threshold
from preprocessing.preprocess import (
    clean_data,
    create_sequences,
    detect_columns,
    load_data,
)


def test_empty_dataset_raises_error():
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="empty"):
        detect_columns(empty_df)


def test_missing_numerical_column_raises_error():
    df_no_numeric = pd.DataFrame({"timestamp": ["2024-01-01", "2024-01-02"]})
    with pytest.raises(ValueError, match="numerical column is required"):
        detect_columns(df_no_numeric)


def test_missing_timestamp_raises_error():
    df_no_ts = pd.DataFrame({"val1": [1.0, 2.0], "val2": [3.0, 4.0]})
    with pytest.raises(ValueError, match="Could not automatically detect timestamp"):
        detect_columns(df_no_ts)


def test_all_nans_in_value_raises_error():
    df_all_nan = pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=5), "value": [np.nan] * 5}
    )
    with pytest.raises(ValueError, match="contains only NaN values"):
        clean_data(df_all_nan, "timestamp", "value")


def test_too_short_time_series_raises_error():
    # Only 5 observations but window size is 10
    short_values = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="shorter than sequence window_size"):
        create_sequences(short_values, window_size=10)


def test_metrics_handles_zero_positives_gracefully():
    # All normal observations, model predicts all normal (no positive predictions)
    y_true = np.zeros(20, dtype=int)
    y_pred = np.zeros(20, dtype=int)

    metrics = calculate_metrics(y_true, y_pred)
    # Precision, recall, f1 should be 0.0 without ZeroDivisionError
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["accuracy"] == 1.0


def test_invalid_csv_format_handling():
    invalid_csv_bytes = io.BytesIO(b"\x00\x01\x02\x03Not a csv file")
    with pytest.raises(ValueError):
        load_data(invalid_csv_bytes)


def test_threshold_empty_errors_raises_error():
    with pytest.raises(ValueError, match="empty"):
        calculate_threshold(np.array([]))
