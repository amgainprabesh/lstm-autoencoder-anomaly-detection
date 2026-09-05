"""
Unit tests for preprocessing and sliding window sequence generation.
"""
import numpy as np
import pandas as pd
import pytest

from preprocessing.preprocess import (
    clean_data,
    detect_columns,
    chronological_split,
    create_sequences,
    tag_ground_truth_labels,
    preprocess_pipeline,
)


@pytest.fixture
def sample_df():
    """Create a clean synthetic time-series DataFrame."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    values = np.sin(np.linspace(0, 10, 100)) * 50 + 100
    return pd.DataFrame({"timestamp": dates, "value": values})


def test_detect_columns_auto(sample_df):
    t_col, v_col = detect_columns(sample_df)
    assert t_col == "timestamp"
    assert v_col == "value"


def test_detect_columns_explicit(sample_df):
    t_col, v_col = detect_columns(sample_df, timestamp_col="timestamp", value_col="value")
    assert t_col == "timestamp"
    assert v_col == "value"


def test_clean_data_handles_nans_and_duplicates():
    dates = pd.date_range("2024-01-01", periods=10, freq="1h")
    df = pd.DataFrame(
        {
            "timestamp": list(dates) + [dates[0]],  # one duplicate
            "value": [10.0, np.nan, 30.0, np.inf, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 15.0],
        }
    )
    cleaned = clean_data(df, "timestamp", "value", deduplicate_strategy="mean")

    # Should have exactly 10 rows
    assert len(cleaned) == 10
    # No NaNs or infinities should exist
    assert not cleaned["value"].isna().any()
    assert not np.isinf(cleaned["value"]).any()
    # Timestamps must be sorted
    assert cleaned["timestamp"].is_monotonic_increasing


def test_chronological_split(sample_df):
    train_df, val_df, test_df = chronological_split(
        sample_df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
    )

    assert len(train_df) == 60
    assert len(val_df) == 20
    assert len(test_df) == 20

    # Ensure strictly chronological ordering (no leakage)
    assert train_df["timestamp"].max() < val_df["timestamp"].min()
    assert val_df["timestamp"].max() < test_df["timestamp"].min()


def test_create_sequences_shape():
    values = np.arange(100, dtype=np.float32).reshape(-1, 1)
    window_size = 24
    sequences, timestamps, labels = create_sequences(values, window_size=window_size, step_size=1)

    expected_count = 100 - 24 + 1
    assert sequences.shape == (expected_count, window_size, 1)
    # Check first sequence
    np.testing.assert_array_equal(sequences[0].flatten(), np.arange(24))
    # Check last sequence
    np.testing.assert_array_equal(sequences[-1].flatten(), np.arange(76, 100))


def test_tag_ground_truth_labels():
    dates = pd.date_range("2024-01-01 00:00", periods=5, freq="1h")
    df = pd.DataFrame({"timestamp": dates, "value": [1, 2, 3, 4, 5]})
    tagged = tag_ground_truth_labels(
        df,
        timestamp_col="timestamp",
        anomaly_windows=[["2024-01-01 01:00", "2024-01-01 02:00"]],
    )
    assert tagged["is_anomaly"].tolist() == [0, 1, 1, 0, 0]


def test_preprocess_pipeline(sample_df):
    prep = preprocess_pipeline(
        df_raw=sample_df,
        window_size=12,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )
    assert prep.X_train.shape[1] == 12
    assert prep.X_val.shape[1] == 12
    assert prep.X_test.shape[1] == 12
    # Ensure min-max scaling is bounded [0, 1] on train
    assert prep.X_train.min() >= -1e-6
    assert prep.X_train.max() <= 1.0 + 1e-6


def test_clean_data_preserves_label_col_with_duplicates():
    dates = pd.date_range("2024-01-01", periods=4, freq="1h")
    # Duplicate timestamp at index 0 and 4, one normal (0) and one anomaly (1)
    df = pd.DataFrame(
        {
            "timestamp": [dates[0], dates[1], dates[2], dates[3], dates[0]],
            "value": [10.0, 20.0, 30.0, 40.0, 12.0],
            "is_anomaly": [0, 0, 1, 0, 1],
        }
    )
    cleaned = clean_data(df, "timestamp", "value", label_col="is_anomaly", deduplicate_strategy="mean")
    assert len(cleaned) == 4
    assert "is_anomaly" in cleaned.columns
    # Duplicate for dates[0] had labels 0 and 1 -> max should be 1
    assert cleaned.loc[cleaned["timestamp"] == dates[0], "is_anomaly"].values[0] == 1


def test_preprocess_pipeline_with_custom_label_col(sample_df):
    sample_df = sample_df.copy()
    sample_df["my_label"] = 0
    sample_df.iloc[80:85, sample_df.columns.get_loc("my_label")] = 1  # anomalies in test set

    prep = preprocess_pipeline(
        df_raw=sample_df,
        label_col="my_label",
        window_size=12,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )

    assert prep.test_labels is not None
    assert np.sum(prep.test_labels) > 0


def test_create_sequences_multivariate():
    values = np.random.randn(50, 3).astype(np.float32)
    sequences, _, _ = create_sequences(values, window_size=10, step_size=2)
    expected_windows = (50 - 10) // 2 + 1
    assert sequences.shape == (expected_windows, 10, 3)

