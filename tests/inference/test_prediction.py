"""Tests for the local prediction service."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from heart_project.prediction import (
    FEATURE_COLUMNS,
    PredictivePipeline,
    build_batch_frame,
    build_patient_frame,
    load_pipeline,
    predict_batch,
    predict_patient,
)

BATCH_ROWS = 2
OUTPUT_COLUMNS = 4

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

def test_build_batch_frame_accepts_csv_boolean_representations() -> None:
    """CSV values using 0/1 should be normalized before prediction."""
    second_patient = {**VALID_PATIENT, "age": 63, "fbs": 1, "exang": "1"}
    first_patient = {**VALID_PATIENT, "fbs": "0", "exang": False}
    uploaded_data = pd.DataFrame([first_patient, second_patient])

    frame = build_batch_frame(uploaded_data)

    assert frame.shape == (BATCH_ROWS, len(FEATURE_COLUMNS))
    assert frame.columns.tolist() == list(FEATURE_COLUMNS)
    assert frame["fbs"].tolist() == [False, True]
    assert frame["exang"].tolist() == [False, True]


def test_build_batch_frame_identifies_the_invalid_csv_row() -> None:
    """A malformed record should report its spreadsheet row number."""
    uploaded_data = pd.DataFrame(
        [
            VALID_PATIENT,
            {**VALID_PATIENT, "rest_ecg": "unknown"},
        ]
    )

    with pytest.raises(ValueError, match="Fila 3"):
        build_batch_frame(uploaded_data)


def test_build_batch_frame_rejects_an_incomplete_schema() -> None:
    """Uploaded files should contain exactly the documented predictor columns."""
    uploaded_data = pd.DataFrame([VALID_PATIENT]).drop(columns="thal")

    with pytest.raises(ValueError, match="Faltantes"):
        build_batch_frame(uploaded_data)


def test_predict_batch_returns_downloadable_results() -> None:
    """Batch serving should append a class, label, probability and threshold."""
    pipeline: PredictivePipeline = PositivePipeline()
    uploaded_data = pd.DataFrame([VALID_PATIENT, {**VALID_PATIENT, "age": 63}])

    results = predict_batch(pipeline, uploaded_data)

    assert results.shape == (BATCH_ROWS, len(FEATURE_COLUMNS) + OUTPUT_COLUMNS)
    assert results["predicted_class"].tolist() == [1, 1]
    assert results["disease_probability"].tolist() == pytest.approx([0.8, 0.8])
    assert results["classification_threshold"].tolist() == [0.5, 0.5]
    assert results["prediction_label"].str.contains("enfermedad cardiaca").all()
