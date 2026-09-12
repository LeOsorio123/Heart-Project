"""Transform the raw heart dataset into a reusable feature table."""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_float_dtype,
    is_integer_dtype,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from heart_project.prediction import FEATURE_COLUMNS  # noqa: E402
from heart_project.transformers import HeartFeatureEngineer  # noqa: E402

TARGET = "disease"
EXPECTED_COLUMNS = (*FEATURE_COLUMNS, TARGET)
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "01_raw" / "corazon.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "04_feature" / "heart_features.parquet"
MAX_MISSING_PERCENTAGE = 10.0

NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "age": (0, 120),
    "rest_bp": (50, 300),
    "chol": (50, 1000),
    "max_hr": (30, 250),
    "old_peak": (0, 10),
    "slope": (1, 3),
    "ca": (0, 3),
}

NOMINAL_CATEGORIES: dict[str, tuple[str, ...]] = {
    "sex": ("Male", "Female"),
    "chest_pain": ("typical", "asymptomatic", "nonanginal", "nontypical"),
    "rest_ecg": (
        "normal",
        "ST-T wave abnormality",
        "left ventricular hypertrophy",
    ),
    "thal": ("normal", "fixed", "reversable"),
}


class DataValidationError(ValueError):
    """Raised when source or feature data violates the documented contract."""


@dataclass(frozen=True)
class FeaturePipelineResult:
    """Execution summary returned by the feature pipeline."""

    input_rows: int
    duplicate_rows_removed: int
    unlabeled_rows_removed: int
    output_rows: int
    output_columns: int
    output_path: Path


def load_raw_data(input_path: Path) -> pd.DataFrame:
    """Read the immutable CSV as trimmed text with explicit missing values."""
    if not input_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo RAW: {input_path}. "
            "Ubique corazon.csv en data/01_raw o use --input."
        )

    raw_data = pd.read_csv(input_path, dtype="string", keep_default_na=False)
    trimmed_data = raw_data.apply(lambda column: column.str.strip())
    return trimmed_data.replace("", pd.NA)


def _check_columns(data: pd.DataFrame) -> None:
    """Fail clearly when the source does not expose the expected schema."""
    missing = sorted(set(EXPECTED_COLUMNS).difference(data.columns))
    unexpected = sorted(set(data.columns).difference(EXPECTED_COLUMNS))
    if missing or unexpected:
        raise DataValidationError(
            "El esquema RAW no coincide con el esperado. "
            f"Faltantes: {missing or 'ninguna'}; adicionales: {unexpected or 'ninguna'}."
        )


def normalize_raw_schema(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the structural type corrections established during exploration."""
    _check_columns(data)
    normalized = data.loc[:, EXPECTED_COLUMNS].copy()

    for column, categories in NOMINAL_CATEGORIES.items():
        normalized[column] = (
            normalized[column]
            .where(normalized[column].isin(categories), pd.NA)
            .astype(pd.CategoricalDtype(categories=categories))
        )

    for column in ("age", "rest_bp", "chol", "max_hr"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("Int16")

    normalized["old_peak"] = pd.to_numeric(normalized["old_peak"], errors="coerce").astype(
        "Float64"
    )

    for column, domain in {"slope": (1, 2, 3), "ca": (0, 1, 2, 3)}.items():
        parsed = pd.to_numeric(normalized[column], errors="coerce")
        normalized[column] = parsed.where(parsed.isin(domain), pd.NA).astype("Int8")

    for column in ("fbs", "exang", TARGET):
        parsed = pd.to_numeric(normalized[column], errors="coerce")
        binary_values = parsed.where(parsed.isin((0, 1)), pd.NA)
        normalized[column] = binary_values.map({0: False, 1: True}).astype("boolean")

    return normalized


def _raise_validation_errors(stage: str, errors: list[str]) -> None:
    """Raise one actionable exception containing every failed validation rule."""
    if errors:
        details = "\n- ".join(errors)
        raise DataValidationError(f"La validación de {stage} falló:\n- {details}")


def _validate_normalized_types(data: pd.DataFrame) -> list[str]:
    """Return errors for columns that do not use the expected nullable dtype."""
    errors: list[str] = []
    integer_columns = ("age", "rest_bp", "chol", "max_hr", "slope", "ca")
    errors.extend(
        f"{column}: se esperaba un tipo entero anulable"
        for column in integer_columns
        if not is_integer_dtype(data[column].dtype)
    )
    if not is_float_dtype(data["old_peak"].dtype):
        errors.append("old_peak: se esperaba un tipo decimal anulable")
    errors.extend(
        f"{column}: se esperaba un tipo booleano anulable"
        for column in ("fbs", "exang", TARGET)
        if not is_bool_dtype(data[column].dtype)
    )
    errors.extend(
        f"{column}: se esperaba un tipo categórico"
        for column in NOMINAL_CATEGORIES
        if not isinstance(data[column].dtype, pd.CategoricalDtype)
    )
    return errors


def _validate_missingness(data: pd.DataFrame) -> list[str]:
    """Return errors for columns exceeding the accepted missing-data threshold."""
    percentages = data.isna().mean().mul(100)
    return [
        f"{column}: {percentage:.2f}% de valores faltantes; "
        f"máximo permitido {MAX_MISSING_PERCENTAGE:.2f}%"
        for column, percentage in percentages.items()
        if percentage > MAX_MISSING_PERCENTAGE
    ]


def _validate_numeric_ranges(data: pd.DataFrame) -> list[str]:
    """Return errors for non-null numeric values outside documented ranges."""
    errors: list[str] = []
    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        numeric_values = pd.to_numeric(data[column], errors="coerce")
        invalid_count = int(
            (data[column].notna() & ~numeric_values.between(minimum, maximum)).sum()
        )
        if invalid_count:
            errors.append(
                f"{column}: {invalid_count} registro(s) fuera del rango [{minimum:g}, {maximum:g}]"
            )
    return errors


def _validate_domains(data: pd.DataFrame) -> list[str]:
    """Return errors for categorical and binary values outside their domains."""
    errors: list[str] = []
    for column, allowed_values in NOMINAL_CATEGORIES.items():
        invalid_count = int((data[column].notna() & ~data[column].isin(allowed_values)).sum())
        if invalid_count:
            errors.append(
                f"{column}: {invalid_count} registro(s) fuera de las categorías permitidas"
            )
    for column in ("fbs", "exang", TARGET):
        invalid_count = int((data[column].notna() & ~data[column].isin((False, True))).sum())
        if invalid_count:
            errors.append(f"{column}: {invalid_count} valor(es) fuera del dominio binario")
    return errors


def validate_normalized_data(data: pd.DataFrame) -> None:
    """Validate schema, types, missingness, domains and clinical ranges.

    Values that could not be repaired reliably during structural normalization are
    represented as missing. The pipeline accepts that behavior only while each
    column remains below the documented missing-data threshold.
    """
    _check_columns(data)
    errors = ["el dataset no contiene registros"] if data.empty else []
    errors.extend(_validate_normalized_types(data))
    errors.extend(_validate_domains(data))
    errors.extend(_validate_missingness(data))
    errors.extend(_validate_numeric_ranges(data))
    if not data[TARGET].notna().any():
        errors.append(f"{TARGET}: no existe ninguna etiqueta válida")
    _raise_validation_errors("los datos normalizados", errors)


def build_feature_table(normalized_data: pd.DataFrame) -> pd.DataFrame:
    """Remove unusable rows and create leakage-safe deterministic features."""
    _check_columns(normalized_data)
    unique_data = normalized_data.drop_duplicates().reset_index(drop=True)
    modeling_data = unique_data.loc[unique_data[TARGET].notna()].reset_index(drop=True)

    if modeling_data.empty:
        raise ValueError("No hay registros con una etiqueta válida para generar features.")

    predictors = modeling_data.loc[:, FEATURE_COLUMNS]
    feature_engineer = HeartFeatureEngineer()
    engineered = feature_engineer.fit_transform(predictors).reset_index(drop=True)
    engineered[TARGET] = modeling_data[TARGET].astype("Int8").reset_index(drop=True)
    return engineered


def _check_feature_columns(feature_table: pd.DataFrame) -> None:
    """Fail clearly when the generated table does not expose the expected schema."""
    expected_columns = (
        *FEATURE_COLUMNS,
        *HeartFeatureEngineer.derived_features,
        TARGET,
    )
    missing = sorted(set(expected_columns).difference(feature_table.columns))
    unexpected = sorted(set(feature_table.columns).difference(expected_columns))
    if missing or unexpected:
        _raise_validation_errors(
            "la tabla de features",
            [
                "el esquema de features no coincide con el esperado; "
                f"faltantes: {missing or 'ninguna'}; adicionales: {unexpected or 'ninguna'}"
            ],
        )


def _validate_feature_rows(feature_table: pd.DataFrame) -> list[str]:
    """Return errors for empty, duplicate, missing-target or invalid-target rows."""
    errors: list[str] = []
    if feature_table.empty:
        errors.append("la tabla de features no contiene registros")
    duplicate_count = int(feature_table.duplicated().sum())
    if duplicate_count:
        errors.append(f"se encontraron {duplicate_count} registro(s) duplicados")
    missing_targets = int(feature_table[TARGET].isna().sum())
    if missing_targets:
        errors.append(f"{TARGET}: se encontraron {missing_targets} etiquetas faltantes")
    invalid_targets = int(
        (feature_table[TARGET].notna() & ~feature_table[TARGET].isin((0, 1))).sum()
    )
    if invalid_targets:
        errors.append(f"{TARGET}: {invalid_targets} valor(es) fuera del dominio [0, 1]")
    return errors


def _validate_numeric_feature_integrity(feature_table: pd.DataFrame) -> list[str]:
    """Return errors for infinite or inconsistent numeric derived features."""
    errors: list[str] = []
    numeric_columns = (
        *HeartFeatureEngineer.numeric_input,
        *HeartFeatureEngineer.derived_features[:3],
    )
    for column in numeric_columns:
        values = pd.to_numeric(feature_table[column], errors="coerce").to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        infinite_count = int(np.isinf(values).sum())
        if infinite_count:
            errors.append(f"{column}: {infinite_count} valor(es) infinitos")

    expected_numeric_features = {
        "age_squared": feature_table["age"] ** 2,
        "old_peak_slope_interaction": feature_table["old_peak"] * feature_table["slope"],
        "max_hr_old_peak_interaction": feature_table["max_hr"] * feature_table["old_peak"],
    }
    for column, expected_values in expected_numeric_features.items():
        actual = pd.to_numeric(feature_table[column], errors="coerce").to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        expected = pd.to_numeric(expected_values, errors="coerce").to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        inconsistent_count = int((~np.isclose(actual, expected, equal_nan=True)).sum())
        if inconsistent_count:
            errors.append(
                f"{column}: {inconsistent_count} valor(es) inconsistentes con sus campos origen"
            )
    return errors


def _validate_categorical_feature_integrity(feature_table: pd.DataFrame) -> list[str]:
    """Return errors for an inconsistent chest-pain/angina interaction."""
    exang_labels = feature_table["exang"].map({0.0: "no", 1.0: "yes"})
    valid_interaction = feature_table["chest_pain"].notna() & exang_labels.notna()
    expected_interaction = pd.Series(
        np.nan,
        index=feature_table.index,
        dtype=object,
    )
    expected_interaction.loc[valid_interaction] = (
        feature_table.loc[valid_interaction, "chest_pain"].astype(str)
        + "__"
        + exang_labels.loc[valid_interaction]
    )
    actual_interaction = feature_table["chest_pain_exang"]
    interaction_matches = actual_interaction.eq(expected_interaction) | (
        actual_interaction.isna() & expected_interaction.isna()
    )
    inconsistent_interactions = int((~interaction_matches).sum())
    if inconsistent_interactions:
        return [
            "chest_pain_exang: "
            f"{inconsistent_interactions} valor(es) inconsistentes con sus campos origen"
        ]
    return []


def validate_feature_table(feature_table: pd.DataFrame) -> None:
    """Validate uniqueness, target integrity and deterministic derived features."""
    _check_feature_columns(feature_table)
    errors = _validate_feature_rows(feature_table)
    errors.extend(_validate_numeric_feature_integrity(feature_table))
    errors.extend(_validate_categorical_feature_integrity(feature_table))
    _raise_validation_errors("la tabla de features", errors)


def persist_feature_table(feature_table: pd.DataFrame, output_path: Path) -> None:
    """Persist the feature table as Parquet and verify its round trip."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_parquet(output_path, index=False, engine="pyarrow")

    reloaded = pd.read_parquet(output_path, engine="pyarrow")
    if reloaded.shape != feature_table.shape or reloaded.columns.tolist() != list(
        feature_table.columns
    ):
        raise RuntimeError("La verificación del archivo de features persistido falló.")


def run_feature_pipeline(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> FeaturePipelineResult:
    """Execute raw loading, structural preparation, feature creation and storage."""
    raw_data = load_raw_data(input_path)
    normalized_data = normalize_raw_schema(raw_data)
    validate_normalized_data(normalized_data)

    unique_data = normalized_data.drop_duplicates()
    duplicate_rows_removed = len(normalized_data) - len(unique_data)
    unlabeled_rows_removed = int(unique_data[TARGET].isna().sum())

    feature_table = build_feature_table(normalized_data)
    validate_feature_table(feature_table)
    persist_feature_table(feature_table, output_path)

    return FeaturePipelineResult(
        input_rows=len(raw_data),
        duplicate_rows_removed=duplicate_rows_removed,
        unlabeled_rows_removed=unlabeled_rows_removed,
        output_rows=len(feature_table),
        output_columns=feature_table.shape[1],
        output_path=output_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface parser."""
    parser = argparse.ArgumentParser(
        description="Genera la tabla de features del proyecto de enfermedad cardiaca."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Ruta del CSV RAW (por defecto: data/01_raw/corazon.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Ruta del Parquet de salida (por defecto: data/04_feature/heart_features.parquet).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the feature pipeline from the command line."""
    arguments = _build_parser().parse_args(argv)
    result = run_feature_pipeline(arguments.input, arguments.output)

    print("Feature Pipeline completado correctamente.")
    print(f"- Registros RAW: {result.input_rows:,}")
    print(f"- Duplicados eliminados: {result.duplicate_rows_removed:,}")
    print(f"- Registros sin target eliminados: {result.unlabeled_rows_removed:,}")
    print(f"- Tabla final: {result.output_rows:,} filas x {result.output_columns} columnas")
    print(f"- Archivo generado: {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
