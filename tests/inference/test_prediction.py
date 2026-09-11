"""Tests for the local prediction service."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from heart_project.prediction import (
    FEATURE_COLUMNS,
    PredictivePipeline,
    build_patient_frame,
    load_pipeline,
    predict_patient,
)

VALID_PATIENT: dict[str, object] = {
    "age": 54,
    "sex": "Male",
    "chest_pain": "asymptomatic",
    "rest_bp": 130,
    "chol": 241,
    "fbs": False,
    "rest_ecg": "normal",
    "max_hr": 153,
    "exang": False,
    "old_peak": 0.8,
    "slope": 2,
    "ca": 0,
    "thal": "normal",
}


class PositivePipeline:
    """Small deterministic pipeline used to isolate serving logic."""

    feature_names_in_ = np.asarray(FEATURE_COLUMNS, dtype=object)
    classes_ = np.asarray([0, 1])

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return the positive class."""
        return np.ones(len(features), dtype=int)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return fixed class probabilities."""
        return np.tile(np.asarray([0.2, 0.8]), (len(features), 1))


def test_build_patient_frame_preserves_expected_schema() -> None:
    """The application should build the exact schema used for training."""
    frame = build_patient_frame(VALID_PATIENT)

    assert frame.columns.tolist() == list(FEATURE_COLUMNS)
    assert frame.shape == (1, len(FEATURE_COLUMNS))
    assert frame.loc[0, "age"] == VALID_PATIENT["age"]
    assert bool(frame.loc[0, "fbs"]) is False


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [("age", 15), ("sex", "Unknown"), ("old_peak", 9.0), ("fbs", 1)],
)
def test_build_patient_frame_rejects_invalid_values(field: str, invalid_value: object) -> None:
    """Values outside the documented domain should fail clearly."""
    patient = {**VALID_PATIENT, field: invalid_value}

    with pytest.raises((TypeError, ValueError)):
        build_patient_frame(patient)


def test_build_patient_frame_rejects_missing_fields() -> None:
    """Incomplete requests should not reach the model."""
    patient = VALID_PATIENT.copy()
    patient.pop("thal")

    with pytest.raises(ValueError, match="Faltantes"):
        build_patient_frame(patient)


def test_predict_patient_returns_positive_probability() -> None:
    """Serving logic should expose the positive-class probability."""
    pipeline: PredictivePipeline = PositivePipeline()

    result = predict_patient(pipeline, VALID_PATIENT)

    assert result.predicted_class == 1
    assert result.disease_probability == pytest.approx(0.8)
    assert result.has_disease_pattern is True


def test_persisted_pipeline_generates_a_prediction() -> None:
    """The repository artifact should load and accept an application payload."""
    project_dir = Path(__file__).resolve().parents[2]
    pipeline = load_pipeline(project_dir / "models" / "heart_disease_best_pipeline.joblib")

    result = predict_patient(pipeline, VALID_PATIENT)

    assert result.predicted_class in {0, 1}
    assert 0.0 <= result.disease_probability <= 1.0
