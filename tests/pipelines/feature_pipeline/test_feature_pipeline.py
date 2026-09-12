"""Unit tests for the autonomous heart feature pipeline."""

from pathlib import Path

import pandas as pd
import pytest

from pipelines.feature_pipeline.feature_pipeline import (
    EXPECTED_COLUMNS,
    MAX_MISSING_PERCENTAGE,
    DataValidationError,
    HeartFeatureEngineer,
    build_feature_table,
    load_raw_data,
    main,
    normalize_raw_schema,
    run_feature_pipeline,
    validate_feature_table,
    validate_normalized_data,
)

EXPECTED_RAW_ROWS = 12
EXPECTED_DUPLICATE_ROWS = 9
EXPECTED_FEATURE_ROWS = 2


def _raw_rows() -> list[dict[str, object]]:
    """Return representative raw rows, including a duplicate and an unlabeled row."""
    first: dict[str, object] = {
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
    second: dict[str, object] = {
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
    unlabeled: dict[str, object] = {**second, "age": "55", "disease": ""}
    return [first, first.copy(), second, unlabeled]


def _write_raw_csv(path: Path) -> None:
    """Create a small CSV source used by the tests."""
    base_rows = _raw_rows()
    rows = [base_rows[0].copy() for _ in range(10)]
    rows.extend((base_rows[2], base_rows[3]))
    pd.DataFrame(rows, columns=EXPECTED_COLUMNS).to_csv(path, index=False)


def test_load_raw_data_trims_text_and_marks_empty_values(tmp_path: Path) -> None:
    """The loader should preserve raw semantics while normalizing whitespace and empties."""
    input_path = tmp_path / "heart.csv"
    _write_raw_csv(input_path)

    loaded = load_raw_data(input_path)

    assert loaded.loc[0, "rest_ecg"] == "left ventricular hypertrophy"
    assert pd.isna(loaded.iloc[-1]["disease"])


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

    assert len(features) == EXPECTED_FEATURE_ROWS
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


def test_validate_normalized_data_accepts_the_documented_contract() -> None:
    """A normalized dataset that follows every rule should pass without errors."""
    valid_rows = _raw_rows()[:3]
    raw_data = pd.DataFrame(valid_rows, columns=EXPECTED_COLUMNS).astype("string")
    trimmed = raw_data.apply(lambda column: column.str.strip())
    normalized = normalize_raw_schema(trimmed.replace("", pd.NA))

    validate_normalized_data(normalized)


def test_validate_normalized_data_rejects_wrong_types_and_categories() -> None:
    """Type and category failures should identify the affected columns."""
    raw_data = pd.DataFrame(_raw_rows(), columns=EXPECTED_COLUMNS).astype("string")
    invalid = normalize_raw_schema(raw_data.replace("", pd.NA))
    invalid["age"] = invalid["age"].astype("string")
    invalid["sex"] = invalid["sex"].astype("object")
    invalid.loc[0, "sex"] = "Unknown"

    with pytest.raises(DataValidationError) as error:
        validate_normalized_data(invalid)

    assert "age: se esperaba un tipo entero" in str(error.value)
    assert "sex: se esperaba un tipo categórico" in str(error.value)
    assert "sex: 1 registro(s) fuera de las categorías" in str(error.value)


def test_validate_normalized_data_rejects_ranges_and_excessive_missingness() -> None:
    """Clinical ranges and maximum missing percentages should be enforced."""
    rows = [_raw_rows()[0].copy() for _ in range(20)]
    raw_data = pd.DataFrame(rows, columns=EXPECTED_COLUMNS).astype("string")
    normalized = normalize_raw_schema(raw_data)
    normalized.loc[0, "rest_bp"] = 400
    missing_rows = int(len(normalized) * MAX_MISSING_PERCENTAGE / 100) + 1
    normalized.loc[: missing_rows - 1, "chol"] = pd.NA

    with pytest.raises(DataValidationError) as error:
        validate_normalized_data(normalized)

    assert "rest_bp: 1 registro(s) fuera del rango" in str(error.value)
    assert "chol:" in str(error.value)
    assert "máximo permitido" in str(error.value)


def test_validate_feature_table_rejects_broken_integrity() -> None:
    """Derived attributes must remain consistent with the fields that generate them."""
    raw_data = pd.DataFrame(_raw_rows(), columns=EXPECTED_COLUMNS).astype("string")
    normalized = normalize_raw_schema(raw_data.replace("", pd.NA))
    features = build_feature_table(normalized)
    features.loc[0, "age_squared"] = -1

    with pytest.raises(DataValidationError, match="age_squared"):
        validate_feature_table(features)


def test_invalid_data_does_not_create_a_feature_file(tmp_path: Path) -> None:
    """The pipeline should stop before persistence when source validation fails."""
    input_path = tmp_path / "invalid_heart.csv"
    output_path = tmp_path / "features.parquet"
    rows = [_raw_rows()[0].copy() for _ in range(10)]
    rows[0]["age"] = "250"
    pd.DataFrame(rows, columns=EXPECTED_COLUMNS).to_csv(input_path, index=False)

    with pytest.raises(DataValidationError, match="age"):
        run_feature_pipeline(input_path, output_path)

    assert not output_path.exists()


def test_run_feature_pipeline_persists_a_reusable_parquet(tmp_path: Path) -> None:
    """The orchestrator should write and verify the expected feature table."""
    input_path = tmp_path / "heart.csv"
    output_path = tmp_path / "features" / "heart_features.parquet"
    _write_raw_csv(input_path)

    result = run_feature_pipeline(input_path, output_path)
    persisted = pd.read_parquet(output_path)

    assert result.input_rows == EXPECTED_RAW_ROWS
    assert result.duplicate_rows_removed == EXPECTED_DUPLICATE_ROWS
    assert result.unlabeled_rows_removed == 1
    assert result.output_rows == EXPECTED_FEATURE_ROWS
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
