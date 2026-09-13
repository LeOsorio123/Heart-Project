"""Tests for autonomous batch inference with synthetic data and a dummy model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from heart_project.prediction import FEATURE_COLUMNS
from heart_project.transformers import HeartFeatureEngineer
from pipelines.inference_pipeline.inference_pipeline import (
    DEFAULT_THRESHOLD,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    engineer_inference_features,
    generate_predictions,
    load_inference_data,
    load_inference_model,
    main,
    normalize_inference_data,
    run_inference_pipeline,
)
from pipelines.training_pipeline.train_pipeline import MODEL_FEATURES

SYNTHETIC_ROWS = 4
EXPECTED_POSITIVE_PROBABILITY = 0.75
HIGH_THRESHOLD = 0.80


def _input_data() -> pd.DataFrame:
    """Create valid raw observations in a deterministic order."""
    return pd.DataFrame(
        [
            {
                "age": 45,
                "sex": "Female",
                "chest_pain": "nontypical",
                "rest_bp": 120,
                "chol": 210,
                "fbs": 0,
                "rest_ecg": "normal",
                "max_hr": 165,
                "exang": 0,
                "old_peak": 0.2,
                "slope": 1,
                "ca": 0,
                "thal": "normal",
            },
            {
                "age": 61,
                "sex": "Male",
                "chest_pain": "asymptomatic",
                "rest_bp": 145,
                "chol": 280,
                "fbs": 1,
                "rest_ecg": "ST-T wave abnormality",
                "max_hr": 125,
                "exang": 1,
                "old_peak": 2.1,
                "slope": 2,
                "ca": 2,
                "thal": "reversable",
            },
            {
                "age": 52,
                "sex": "Female",
                "chest_pain": "nonanginal",
                "rest_bp": 132,
                "chol": 235,
                "fbs": False,
                "rest_ecg": "normal",
                "max_hr": 154,
                "exang": False,
                "old_peak": 0.8,
                "slope": 2,
                "ca": 1,
                "thal": "fixed",
            },
            {
                "age": 68,
                "sex": "Male",
                "chest_pain": "typical",
                "rest_bp": 150,
                "chol": 300,
                "fbs": True,
                "rest_ecg": "left ventricular hypertrophy",
                "max_hr": 110,
                "exang": True,
                "old_peak": 3.0,
                "slope": 3,
                "ca": 3,
                "thal": "reversable",
            },
        ],
        columns=FEATURE_COLUMNS,
    )


def _fit_dummy_model(data: pd.DataFrame) -> DummyClassifier:
    """Fit a serializable dummy classifier with the production feature schema."""
    normalized = normalize_inference_data(data)
    engineer = HeartFeatureEngineer()
    engineered = engineer.fit_transform(normalized).loc[:, MODEL_FEATURES]
    target = np.asarray([0, 1, 1, 1], dtype=int)
    return DummyClassifier(strategy="prior").fit(engineered, target)


def _write_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Persist synthetic model and input artifacts for integration tests."""
    data = _input_data()
    model_path = tmp_path / "models" / "dummy.joblib"
    input_path = tmp_path / "input" / "patients.csv"
    output_path = tmp_path / "output" / "predictions.csv"
    model_path.parent.mkdir(parents=True)
    input_path.parent.mkdir(parents=True)
    joblib.dump(_fit_dummy_model(data), model_path)
    data.to_csv(input_path, index=False)
    return model_path, input_path, output_path


def test_load_and_engineer_data_match_the_trained_schema(tmp_path: Path) -> None:
    """Raw CSV rows should become the exact features expected by the model."""
    model_path, input_path, _ = _write_artifacts(tmp_path)
    loaded = load_inference_data(input_path)
    model = joblib.load(model_path)

    engineered = engineer_inference_features(normalize_inference_data(loaded), model)

    assert tuple(loaded.columns) == FEATURE_COLUMNS
    assert tuple(engineered.columns) == MODEL_FEATURES
    assert len(engineered) == SYNTHETIC_ROWS


def test_pipeline_persists_traceable_predictions(tmp_path: Path) -> None:
    """The orchestrator should preserve rows and append understandable results."""
    model_path, input_path, output_path = _write_artifacts(tmp_path)

    result = run_inference_pipeline(model_path, input_path, output_path)
    persisted = pd.read_csv(output_path)

    assert result.input_rows == SYNTHETIC_ROWS
    assert result.positive_predictions == SYNTHETIC_ROWS
    assert result.negative_predictions == 0
    assert persisted.loc[:, list(FEATURE_COLUMNS)].equals(pd.read_csv(input_path))
    assert persisted["predicted_class"].tolist() == [1] * SYNTHETIC_ROWS
    assert set(persisted["prediction_label"]) == {POSITIVE_LABEL}
    assert persisted["disease_probability"].tolist() == pytest.approx(
        [EXPECTED_POSITIVE_PROBABILITY] * SYNTHETIC_ROWS
    )
    assert set(persisted["classification_threshold"]) == {DEFAULT_THRESHOLD}


def test_custom_threshold_changes_the_generated_class() -> None:
    """Classification should use the configured threshold, not a hidden default."""
    data = _input_data()
    model = _fit_dummy_model(data)
    engineered = engineer_inference_features(normalize_inference_data(data), model)

    predictions = generate_predictions(model, data, engineered, HIGH_THRESHOLD)

    assert predictions["predicted_class"].tolist() == [0] * SYNTHETIC_ROWS
    assert set(predictions["prediction_label"]) == {NEGATIVE_LABEL}


def test_category_whitespace_is_removed_before_validation() -> None:
    """Incidental spaces from a CSV should not invalidate a known category."""
    data = _input_data()
    data.loc[0, "rest_ecg"] = "left ventricular hypertrophy "

    normalized = normalize_inference_data(data)

    assert normalized.loc[0, "rest_ecg"] == "left ventricular hypertrophy"


@pytest.mark.parametrize(
    ("invalid_data", "message"),
    [
        (_input_data().drop(columns="age"), "Faltantes"),
        (_input_data().assign(disease=1), "adicionales"),
        (_input_data().assign(chest_pain="unknown"), "categorías inválidas"),
        (_input_data().assign(age="not-a-number"), "valores no numéricos"),
    ],
)
def test_invalid_input_does_not_create_an_output(
    tmp_path: Path,
    invalid_data: pd.DataFrame,
    message: str,
) -> None:
    """Schema and domain failures should happen before persistence."""
    valid_data = _input_data()
    model_path = tmp_path / "dummy.joblib"
    input_path = tmp_path / "invalid.csv"
    output_path = tmp_path / "predictions.csv"
    joblib.dump(_fit_dummy_model(valid_data), model_path)
    invalid_data.to_csv(input_path, index=False)

    with pytest.raises(ValueError, match=message):
        run_inference_pipeline(model_path, input_path, output_path)

    assert not output_path.exists()


def test_missing_artifacts_produce_actionable_errors(tmp_path: Path) -> None:
    """Missing model and input files should identify the absent artifact."""
    data = _input_data()
    model_path = tmp_path / "dummy.joblib"
    input_path = tmp_path / "patients.csv"
    joblib.dump(_fit_dummy_model(data), model_path)

    with pytest.raises(FileNotFoundError, match="modelo"):
        run_inference_pipeline(tmp_path / "missing.joblib", input_path, tmp_path / "out.csv")

    with pytest.raises(FileNotFoundError, match="archivo de inferencia"):
        run_inference_pipeline(model_path, input_path, tmp_path / "out.csv")


def test_incompatible_model_is_rejected_before_output(tmp_path: Path) -> None:
    """A serialized object without the prediction contract should fail clearly."""
    model_path = tmp_path / "invalid.joblib"
    joblib.dump({"not": "a model"}, model_path)

    with pytest.raises(TypeError, match="modelo de clasificación compatible"):
        load_inference_model(model_path)


def test_main_accepts_custom_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command-line entry point should run autonomously with explicit paths."""
    model_path, input_path, output_path = _write_artifacts(tmp_path)

    exit_code = main(
        [
            "--model",
            str(model_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--threshold",
            str(HIGH_THRESHOLD),
        ]
    )

    assert exit_code == 0
    assert output_path.is_file()
    captured = capsys.readouterr().out
    assert "Prediction Pipeline completado correctamente" in captured
    assert "no constituye un diagnóstico clínico" in captured
