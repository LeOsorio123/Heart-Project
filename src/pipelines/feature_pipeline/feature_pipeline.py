"""Transform the raw heart dataset into a reusable feature table."""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

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
        raise ValueError(
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

    normalized["old_peak"] = pd.to_numeric(
        normalized["old_peak"], errors="coerce"
    ).astype("Float64")

    for column, domain in {"slope": (1, 2, 3), "ca": (0, 1, 2, 3)}.items():
        parsed = pd.to_numeric(normalized[column], errors="coerce")
        normalized[column] = parsed.where(parsed.isin(domain), pd.NA).astype("Int8")

    for column in ("fbs", "exang", TARGET):
        parsed = pd.to_numeric(normalized[column], errors="coerce")
        binary_values = parsed.where(parsed.isin((0, 1)), pd.NA)
        normalized[column] = binary_values.map({0: False, 1: True}).astype("boolean")

    return normalized


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

    unique_data = normalized_data.drop_duplicates()
    duplicate_rows_removed = len(normalized_data) - len(unique_data)
    unlabeled_rows_removed = int(unique_data[TARGET].isna().sum())

    feature_table = build_feature_table(normalized_data)
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
