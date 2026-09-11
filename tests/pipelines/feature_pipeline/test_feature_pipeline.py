"""Unit tests for the autonomous heart feature pipeline."""

from pathlib import Path

import pandas as pd
import pytest

from pipelines.feature_pipeline.feature_pipeline import (
    EXPECTED_COLUMNS,
    HeartFeatureEngineer,
    build_feature_table,
    load_raw_data,
    main,
    normalize_raw_schema,
    run_feature_pipeline,
)


def _raw_rows() -> list[dict[str, object]]:
    """Return representative raw rows, including a duplicate and an unlabeled row."""
    first = {
        "age": "63",
        "sex": "Male",
        "chest_pain": "typical",
        "rest_bp": "145",
        "chol": "233",
        "fbs": "1",
        "rest_ecg": "left ventricular hypertrophy ",
        "max_hr": "150",
        "exang": "0",
        "old_peak": "2.3",
        "slope": "3",
        "ca": "0.0",
        "thal": "fixed",
        "disease": "0",
    }
    second = {
        "age": "67",
        "sex": "Male",
        "chest_pain": "asymptomatic",
        "rest_bp": "160",
        "chol": "286",
        "fbs": "0",
        "rest_ecg": "normal",
        "max_hr": "108",
        "exang": "1",
        "old_peak": "1.5",
        "slope": "2",
        "ca": "3.0",
        "thal": "normal",
        "disease": "1",
    }
    unlabeled = {**second, "age": "55", "disease": ""}
    return [first, first.copy(), second, unlabeled]


def _write_raw_csv(path: Path) -> None:
    """Create a small CSV source used by the tests."""
    pd.DataFrame(_raw_rows(), columns=EXPECTED_COLUMNS).to_csv(path, index=False)


def test_load_raw_data_trims_text_and_marks_empty_values(tmp_path: Path) -> None:
    """The loader should preserve raw semantics while normalizing whitespace and empties."""
    input_path = tmp_path / "heart.csv"
    _write_raw_csv(input_path)

    loaded = load_raw_data(input_path)

    assert loaded.loc[0, "rest_ecg"] == "left ventricular hypertrophy"
    assert pd.isna(loaded.loc[3, "disease"])


def test_load_raw_data_rejects_missing_file(tmp_path: Path) -> None:
    """A missing source should produce an actionable error."""
    with pytest.raises(FileNotFoundError, match="archivo RAW"):
        load_raw_data(tmp_path / "missing.csv")


def test_normalize_raw_schema_rejects_unexpected_columns() -> None:
    """The first structural check should fail before transformations run."""
    invalid = pd.DataFrame(_raw_rows()).drop(columns="thal")

    with pytest.raises(ValueError, match="Faltantes"):
        normalize_raw_schema(invalid)


def test_build_feature_table_removes_unusable_rows_and_derives_features() -> None:
    """Feature creation should be deterministic and retain only labeled unique rows."""
    raw_data = pd.DataFrame(_raw_rows(), columns=EXPECTED_COLUMNS).astype("string")
    normalized = normalize_raw_schema(raw_data.apply(lambda column: column.str.strip()))

    features = build_feature_table(normalized)

    assert len(features) == 2
    assert features["disease"].tolist() == [0, 1]
    assert features.loc[0, "age_squared"] == pytest.approx(63**2)
    assert features.loc[1, "chest_pain_exang"] == "asymptomatic__yes"
    assert set(HeartFeatureEngineer.derived_features).issubset(features.columns)


def test_build_feature_table_rejects_data_without_valid_target() -> None:
    """An unlabeled source cannot produce a supervised feature table."""
    rows = [{**_raw_rows()[0], "disease": ""}]
    raw_data = pd.DataFrame(rows, columns=EXPECTED_COLUMNS).astype("string")
    normalized = normalize_raw_schema(raw_data.replace("", pd.NA))

    with pytest.raises(ValueError, match="No hay registros"):
        build_feature_table(normalized)


def test_run_feature_pipeline_persists_a_reusable_parquet(tmp_path: Path) -> None:
    """The orchestrator should write and verify the expected feature table."""
    input_path = tmp_path / "heart.csv"
    output_path = tmp_path / "features" / "heart_features.parquet"
    _write_raw_csv(input_path)

    result = run_feature_pipeline(input_path, output_path)
    persisted = pd.read_parquet(output_path)

    assert result.input_rows == 4
    assert result.duplicate_rows_removed == 1
    assert result.unlabeled_rows_removed == 1
    assert result.output_rows == 2
    assert result.output_columns == len(EXPECTED_COLUMNS) + len(
        HeartFeatureEngineer.derived_features
    )
    assert persisted.shape == (result.output_rows, result.output_columns)
    assert not persisted["disease"].isna().any()


def test_main_accepts_custom_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command-line entry point should run independently with explicit paths."""
    input_path = tmp_path / "heart.csv"
    output_path = tmp_path / "features.parquet"
    _write_raw_csv(input_path)

    exit_code = main(["--input", str(input_path), "--output", str(output_path)])

    assert exit_code == 0
    assert output_path.is_file()
    assert "Feature Pipeline completado correctamente" in capsys.readouterr().out
