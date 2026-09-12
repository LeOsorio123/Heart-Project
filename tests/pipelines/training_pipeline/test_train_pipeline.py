"""Unit tests for the autonomous heart training pipeline."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from pipelines.training_pipeline.train_pipeline import (
    EXPECTED_COLUMNS,
    MODEL_FEATURES,
    RANDOM_STATE,
    TARGET,
    build_training_pipeline,
    evaluate_classifier,
    load_feature_data,
    main,
    persist_training_artifacts,
    run_training_pipeline,
    split_training_data,
)

SYNTHETIC_ROWS = 40
EXPECTED_TRAIN_ROWS = 32
EXPECTED_TEST_ROWS = 8


def _feature_data(rows: int = SYNTHETIC_ROWS) -> pd.DataFrame:
    """Create a balanced feature table with the production schema."""
    records: list[dict[str, object]] = []
    chest_pain_values = ("typical", "asymptomatic", "nonanginal", "nontypical")
    ecg_values = ("normal", "ST-T wave abnormality", "left ventricular hypertrophy")
    thal_values = ("normal", "fixed", "reversable")
    for index in range(rows):
        target = index % 2
        age = 35 + index
        old_peak = float(index % 5) / 2
        slope = (index % 3) + 1
        max_hr = 180 - index
        chest_pain = chest_pain_values[index % len(chest_pain_values)]
        exang = float(target)
        records.append(
            {
                "age": float(age),
                "sex": "Male" if index % 2 else "Female",
                "chest_pain": chest_pain,
                "rest_bp": float(110 + index),
                "chol": float(180 + index * 2),
                "fbs": float(index % 2),
                "rest_ecg": ecg_values[index % len(ecg_values)],
                "max_hr": float(max_hr),
                "exang": exang,
                "old_peak": old_peak,
                "slope": float(slope),
                "ca": float(index % 4),
                "thal": thal_values[index % len(thal_values)],
                "age_squared": float(age**2),
                "old_peak_slope_interaction": old_peak * slope,
                "max_hr_old_peak_interaction": max_hr * old_peak,
                "chest_pain_exang": f"{chest_pain}__{'yes' if exang else 'no'}",
                TARGET: target,
            }
        )
    return pd.DataFrame(records, columns=EXPECTED_COLUMNS)


def _write_feature_file(path: Path) -> None:
    """Persist a synthetic feature table for integration tests."""
    _feature_data().to_parquet(path, index=False)


def test_load_feature_data_reads_the_expected_schema(tmp_path: Path) -> None:
    """The loader should return the complete validated training schema."""
    input_path = tmp_path / "features.parquet"
    _write_feature_file(input_path)

    loaded = load_feature_data(input_path)

    assert loaded.shape == (SYNTHETIC_ROWS, len(EXPECTED_COLUMNS))
    assert tuple(loaded.columns) == EXPECTED_COLUMNS


def test_load_feature_data_rejects_missing_file(tmp_path: Path) -> None:
    """A missing feature table should produce an actionable error."""
    with pytest.raises(FileNotFoundError, match="Feature Pipeline"):
        load_feature_data(tmp_path / "missing.parquet")


def test_load_feature_data_rejects_an_invalid_schema(tmp_path: Path) -> None:
    """Training should stop when a required feature is unavailable."""
    input_path = tmp_path / "invalid.parquet"
    _feature_data().drop(columns="age_squared").to_parquet(input_path, index=False)

    with pytest.raises(ValueError, match="Faltantes"):
        load_feature_data(input_path)


def test_split_and_training_are_reproducible() -> None:
    """The split should be stratified and the selected model should fit successfully."""
    split = split_training_data(_feature_data(), random_state=RANDOM_STATE)
    pipeline = build_training_pipeline(random_state=RANDOM_STATE)
    pipeline.fit(split.X_train, split.y_train)

    assert len(split.X_train) == EXPECTED_TRAIN_ROWS
    assert len(split.X_test) == EXPECTED_TEST_ROWS
    assert split.y_train.mean() == pytest.approx(split.y_test.mean())
    assert tuple(pipeline.feature_names_in_) == MODEL_FEATURES
    assert set(pipeline.predict(split.X_test)).issubset({0, 1})


def test_evaluate_classifier_generates_complete_metrics() -> None:
    """Evaluation should include the priority metric and a complete confusion matrix."""
    split = split_training_data(_feature_data())
    pipeline = build_training_pipeline()
    pipeline.fit(split.X_train, split.y_train)

    metrics = evaluate_classifier(pipeline, split.X_test, split.y_test)
    matrix = metrics["confusion_matrix"]

    assert metrics["primary_metric"] == "recall"
    for metric in (
        "accuracy",
        "recall",
        "specificity",
        "precision",
        "f1",
        "balanced_accuracy",
        "roc_auc",
    ):
        assert 0.0 <= float(metrics[metric]) <= 1.0
    assert isinstance(matrix, dict)
    assert sum(int(value) for value in matrix.values()) == EXPECTED_TEST_ROWS


def test_persist_training_artifacts_writes_reloadable_outputs(tmp_path: Path) -> None:
    """The fitted pipeline and JSON report should survive a storage round trip."""
    split = split_training_data(_feature_data())
    pipeline = build_training_pipeline()
    pipeline.fit(split.X_train, split.y_train)
    metrics = evaluate_classifier(pipeline, split.X_test, split.y_test)
    model_path = tmp_path / "models" / "pipeline.joblib"
    metrics_path = tmp_path / "output" / "metrics.json"
    report: dict[str, object] = {"metrics": metrics}

    persist_training_artifacts(pipeline, report, model_path, metrics_path)
    reloaded = joblib.load(model_path)
    reloaded_report = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert np.array_equal(
        pipeline.predict(split.X_test),
        reloaded.predict(split.X_test),
    )
    assert reloaded_report == report


def test_run_training_pipeline_generates_model_and_metrics(tmp_path: Path) -> None:
    """The orchestrator should produce every required training artifact."""
    input_path = tmp_path / "features.parquet"
    model_path = tmp_path / "models" / "pipeline.joblib"
    metrics_path = tmp_path / "output" / "metrics.json"
    _write_feature_file(input_path)

    result = run_training_pipeline(input_path, model_path, metrics_path)

    assert result.input_rows == SYNTHETIC_ROWS
    assert result.train_rows == EXPECTED_TRAIN_ROWS
    assert result.test_rows == EXPECTED_TEST_ROWS
    assert result.model_path.is_file()
    assert result.metrics_path.is_file()
    assert result.metrics["primary_metric"] == "recall"
    assert result.split_validation["status"] in {"passed", "passed_with_warnings"}
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert report["split_validation"] == result.split_validation


def test_main_accepts_custom_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command-line entry point should run autonomously with explicit paths."""
    input_path = tmp_path / "features.parquet"
    model_path = tmp_path / "pipeline.joblib"
    metrics_path = tmp_path / "metrics.json"
    _write_feature_file(input_path)

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--model-output",
            str(model_path),
            "--metrics-output",
            str(metrics_path),
        ]
    )

    assert exit_code == 0
    assert model_path.is_file()
    assert metrics_path.is_file()
    assert "Training Pipeline completado correctamente" in capsys.readouterr().out
