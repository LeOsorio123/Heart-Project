"""Tests for reproducible model validation and generalization diagnostics."""

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipelines.training_pipeline.model_validation import (
    ModelValidationConfig,
    analyze_generalization,
    run_cross_validation,
)

SYNTHETIC_ROWS = 80
SYNTHETIC_FEATURES = 6
CV_FOLDS = 4
EXPECTED_FOLD_SCORES = CV_FOLDS


def _classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create a deterministic balanced binary classification dataset."""
    features, target = make_classification(
        n_samples=SYNTHETIC_ROWS,
        n_features=SYNTHETIC_FEATURES,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )
    columns = [f"feature_{index}" for index in range(SYNTHETIC_FEATURES)]
    return pd.DataFrame(features, columns=columns), pd.Series(target, dtype="int8")


def _pipeline() -> Pipeline:
    """Build a lightweight classifier for validation unit tests."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=42)),
        ]
    )


def _metric_set(value: float) -> dict[str, object]:
    """Build a complete metric set with a chosen recall value."""
    return {
        "accuracy": value,
        "balanced_accuracy": value,
        "recall": value,
        "precision": value,
        "f1": value,
        "roc_auc": value,
    }


def _cross_validation(recall_mean: float) -> dict[str, object]:
    """Build a compact cross-validation summary for diagnosis tests."""
    return {
        "metrics": {
            metric: {
                "fold_scores": [recall_mean] * CV_FOLDS,
                "mean": recall_mean,
                "std": 0.02,
            }
            for metric in _metric_set(recall_mean)
        }
    }


def test_cross_validation_is_reproducible_and_reports_every_fold() -> None:
    """The same seed should generate identical stratified validation results."""
    features, target = _classification_data()
    config = ModelValidationConfig(cv_folds=CV_FOLDS, random_state=17)

    first = run_cross_validation(_pipeline(), features, target, config)
    second = run_cross_validation(_pipeline(), features, target, config)

    assert first == second
    assert first["strategy"] == "StratifiedKFold"
    assert first["uses_training_data_only"] is True
    metrics = first["metrics"]
    assert isinstance(metrics, dict)
    for summary in metrics.values():
        assert isinstance(summary, dict)
        assert len(summary["fold_scores"]) == EXPECTED_FOLD_SCORES
        assert 0.0 <= float(summary["mean"]) <= 1.0


@pytest.mark.parametrize(
    ("train_recall", "cv_recall", "test_recall", "expected_diagnosis"),
    [
        (0.84, 0.82, 0.80, "stable_generalization"),
        (0.95, 0.75, 0.76, "overfitting"),
        (0.70, 0.65, 0.60, "underfitting"),
        (0.82, 0.80, 0.62, "unstable_generalization"),
    ],
)
def test_generalization_diagnosis_is_explainable(
    train_recall: float,
    cv_recall: float,
    test_recall: float,
    expected_diagnosis: str,
) -> None:
    """Known train/CV/test patterns should map to transparent diagnoses."""
    report = analyze_generalization(
        _metric_set(train_recall),
        _cross_validation(cv_recall),
        _metric_set(test_recall),
        ModelValidationConfig(),
    )

    assert report.diagnosis == expected_diagnosis
    assert report.status == (
        "passed" if expected_diagnosis == "stable_generalization" else "passed_with_warnings"
    )
    assert report.recommendations


def test_cross_validation_rejects_too_many_folds_for_a_class() -> None:
    """Validation should fail clearly when a class cannot populate every fold."""
    features, target = _classification_data()
    target.iloc[:] = 0
    target.iloc[:2] = 1

    with pytest.raises(ValueError, match="tantos registros como folds"):
        run_cross_validation(
            _pipeline(),
            features,
            target,
            ModelValidationConfig(cv_folds=CV_FOLDS),
        )
