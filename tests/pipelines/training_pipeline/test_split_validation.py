"""Unit tests for train/test integrity and distribution checks."""

import pandas as pd
import pytest

from pipelines.training_pipeline.split_validation import (
    SplitValidationConfig,
    TrainTestValidationError,
    TrainTestValidationReport,
    validate_train_test_split,
)
from pipelines.training_pipeline.train_pipeline import DataSplit

EXPECTED_FEATURES = ("age", "group")
NUMERIC_FEATURES = ("age",)
CATEGORICAL_FEATURES = ("group",)
EXPECTED_TEST_SIZE = 1 / 3


def _valid_split() -> DataSplit:
    """Return small, disjoint and representative partitions."""
    X_train = pd.DataFrame(
        {
            "age": [30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0],
            "group": ["A", "B", "A", "B", "A", "B", "A", "B"],
        },
        index=range(8),
    )
    X_test = pd.DataFrame(
        {
            "age": [38.0, 39.0, 40.0, 41.0],
            "group": ["A", "B", "A", "B"],
        },
        index=range(8, 12),
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], index=X_train.index)
    y_test = pd.Series([0, 1, 0, 1], index=X_test.index)
    return DataSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def _validate(
    split: DataSplit,
    *,
    numeric_ks_threshold: float = 0.20,
    categorical_tvd_threshold: float = 0.20,
) -> TrainTestValidationReport:
    """Run the validator with the compact test schema."""
    return validate_train_test_split(
        split,
        SplitValidationConfig(
            expected_features=EXPECTED_FEATURES,
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            expected_test_size=EXPECTED_TEST_SIZE,
            numeric_ks_threshold=numeric_ks_threshold,
            categorical_tvd_threshold=categorical_tvd_threshold,
        ),
    )


def test_validate_train_test_split_accepts_a_valid_partition() -> None:
    """A representative split should pass every critical and distribution check."""
    report = _validate(
        _valid_split(),
        numeric_ks_threshold=1.00,
        categorical_tvd_threshold=0.50,
    )

    assert report.status == "passed"
    assert report.warnings == ()
    assert all(
        bool(check["passed"])
        for check in report.critical_checks.values()
        if isinstance(check, dict)
    )


def test_validate_train_test_split_rejects_index_leakage() -> None:
    """Shared source indices should stop the training pipeline."""
    split = _valid_split()
    X_test = split.X_test.copy()
    y_test = split.y_test.copy()
    new_index = list(X_test.index)
    new_index[0] = split.X_train.index[0]
    X_test.index = new_index
    y_test.index = new_index
    invalid = DataSplit(split.X_train, X_test, split.y_train, y_test)

    with pytest.raises(TrainTestValidationError, match=r"índice.*compartidos"):
        _validate(invalid)


def test_validate_train_test_split_rejects_duplicate_samples() -> None:
    """An identical labeled sample in both partitions is data leakage."""
    split = _valid_split()
    X_test = split.X_test.copy()
    y_test = split.y_test.copy()
    X_test.iloc[0] = split.X_train.iloc[0]
    y_test.iloc[0] = split.y_train.iloc[0]
    invalid = DataSplit(split.X_train, X_test, split.y_train, y_test)

    with pytest.raises(TrainTestValidationError, match=r"muestra.*idénticas"):
        _validate(invalid)


def test_validate_train_test_split_rejects_new_test_categories() -> None:
    """Categories unseen during training should produce a controlled error."""
    split = _valid_split()
    X_test = split.X_test.copy()
    X_test.loc[X_test.index[0], "group"] = "C"
    invalid = DataSplit(split.X_train, X_test, split.y_train, split.y_test)

    with pytest.raises(TrainTestValidationError, match="categorías nuevas"):
        _validate(invalid)


def test_validate_train_test_split_rejects_new_test_labels() -> None:
    """A target label unavailable in train should produce a controlled error."""
    split = _valid_split()
    y_test = split.y_test.copy()
    y_test.iloc[0] = 2
    invalid = DataSplit(split.X_train, split.X_test, split.y_train, y_test)

    with pytest.raises(TrainTestValidationError, match="etiquetas nuevas"):
        _validate(invalid)


def test_validate_train_test_split_warns_about_distribution_drift() -> None:
    """Large distribution differences should be reported without implying leakage."""
    split = _valid_split()
    X_test = split.X_test.copy()
    X_test["age"] = [100.0, 101.0, 102.0, 103.0]
    X_test["group"] = "A"
    shifted = DataSplit(split.X_train, X_test, split.y_train, split.y_test)

    report = _validate(
        shifted,
        numeric_ks_threshold=0.20,
        categorical_tvd_threshold=0.20,
    )

    assert report.status == "passed_with_warnings"
    assert any("age" in warning for warning in report.warnings)
    assert any("group" in warning for warning in report.warnings)
