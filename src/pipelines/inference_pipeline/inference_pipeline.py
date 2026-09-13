"""Run reproducible batch inference with the trained heart-disease pipeline."""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from heart_project.prediction import (  # noqa: E402
    CATEGORY_VALUES,
    FEATURE_COLUMNS,
    NUMERIC_BOUNDS,
    PredictivePipeline,
    load_pipeline,
)
from heart_project.transformers import HeartFeatureEngineer  # noqa: E402
from pipelines.training_pipeline.train_pipeline import MODEL_FEATURES  # noqa: E402

POSITIVE_CLASS = 1
DEFAULT_THRESHOLD = 0.50
POSITIVE_LABEL = "Patrón compatible con enfermedad cardiaca"
NEGATIVE_LABEL = "No se identifica patrón compatible con enfermedad cardiaca"
BINARY_COLUMNS = ("fbs", "exang")

DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "06_models" / "heart_disease_training_pipeline.joblib"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "05_model_input" / "heart_prediction_input.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "07_model_output" / "heart_predictions.csv"


@dataclass(frozen=True)
class InferencePipelineResult:
    """Execution summary returned by the batch inference pipeline."""

    input_rows: int
    positive_predictions: int
    negative_predictions: int
    threshold: float
    output_path: Path


def _validate_input_schema(data: pd.DataFrame) -> None:
    """Enforce the raw predictor schema expected by the feature engineer."""
    missing = sorted(set(FEATURE_COLUMNS).difference(data.columns))
    unexpected = sorted(set(data.columns).difference(FEATURE_COLUMNS))
    if missing or unexpected:
        raise ValueError(
            "El esquema de inferencia no coincide con el esperado. "
            f"Faltantes: {missing or 'ninguna'}; adicionales: {unexpected or 'ninguna'}."
        )
    if data.empty:
        raise ValueError("El archivo de inferencia no contiene registros.")


def load_inference_data(input_path: Path) -> pd.DataFrame:
    """Load new observations from CSV without allowing a target column."""
    if not input_path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de inferencia: {input_path}")

    data = pd.read_csv(input_path)
    _validate_input_schema(data)
    return data.loc[:, FEATURE_COLUMNS].copy()


def load_inference_model(model_path: Path) -> PredictivePipeline:
    """Load the model and enforce the contract required for batch inference."""
    pipeline = load_pipeline(model_path)
    required_attributes = ("feature_names_in_", "classes_", "predict_proba")
    missing = [name for name in required_attributes if not hasattr(pipeline, name)]
    if missing:
        raise TypeError(
            "El archivo cargado no contiene un modelo de clasificación compatible. "
            f"Atributos faltantes: {missing}."
        )
    return pipeline


def _normalize_numeric_columns(data: pd.DataFrame) -> None:
    """Parse numeric values while distinguishing blanks from malformed values."""
    for column, (lower, upper) in NUMERIC_BOUNDS.items():
        original = data[column]
        parsed = pd.to_numeric(original, errors="coerce")
        malformed = original.notna() & parsed.isna()
        if malformed.any():
            rows = [int(index) for index in data.index[malformed]]
            raise ValueError(f"{column} contiene valores no numéricos en las filas: {rows}.")

        outside_range = parsed.notna() & ~parsed.between(lower, upper)
        if outside_range.any():
            rows = [int(index) for index in data.index[outside_range]]
            raise ValueError(
                f"{column} debe estar entre {lower:g} y {upper:g}; filas inválidas: {rows}."
            )
        data[column] = parsed.astype(float)


def _normalize_binary_columns(data: pd.DataFrame) -> None:
    """Normalize accepted boolean and 0/1 representations."""
    boolean_map: dict[object, bool] = {
        0: False,
        1: True,
        "0": False,
        "1": True,
        "false": False,
        "true": True,
        "False": False,
        "True": True,
    }
    for column in BINARY_COLUMNS:
        original = data[column]
        normalized = original.map(boolean_map)
        invalid = original.notna() & normalized.isna()
        if invalid.any():
            rows = [int(index) for index in data.index[invalid]]
            raise ValueError(f"{column} debe ser booleano o 0/1; filas inválidas: {rows}.")
        data[column] = normalized.astype("boolean")


def normalize_inference_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate domains and normalize values before feature engineering."""
    _validate_input_schema(data)
    normalized = data.loc[:, FEATURE_COLUMNS].copy()
    _normalize_numeric_columns(normalized)
    _normalize_binary_columns(normalized)

    for column, valid_values in CATEGORY_VALUES.items():
        values = normalized[column].astype("string").str.strip()
        invalid = values.notna() & ~values.isin(valid_values)
        if invalid.any():
            rows = [int(index) for index in normalized.index[invalid]]
            raise ValueError(
                f"{column} contiene categorías inválidas en las filas: {rows}. "
                f"Valores permitidos: {', '.join(valid_values)}."
            )
        normalized[column] = values

    return normalized


def engineer_inference_features(
    normalized_data: pd.DataFrame,
    pipeline: PredictivePipeline,
) -> pd.DataFrame:
    """Create deterministic attributes and verify the trained-model schema."""
    feature_engineer = HeartFeatureEngineer()
    engineered = feature_engineer.fit_transform(normalized_data)
    expected_by_model = tuple(str(column) for column in pipeline.feature_names_in_)
    if expected_by_model != MODEL_FEATURES:
        raise ValueError(
            "El esquema del modelo almacenado no coincide con las variables de entrenamiento."
        )
    missing = sorted(set(expected_by_model).difference(engineered.columns))
    if missing:
        raise ValueError(f"No fue posible generar las variables requeridas: {missing}.")
    return engineered.loc[:, expected_by_model]


def generate_predictions(
    pipeline: PredictivePipeline,
    original_data: pd.DataFrame,
    engineered_features: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Append understandable classes and positive-class probabilities."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("El umbral de clasificación debe estar entre 0 y 1.")

    probabilities = pipeline.predict_proba(engineered_features)
    positive_positions = np.flatnonzero(pipeline.classes_ == POSITIVE_CLASS)
    if len(positive_positions) != 1:
        raise ValueError("El modelo no contiene una única clase positiva identificada como 1.")
    if probabilities.shape != (len(original_data), len(pipeline.classes_)):
        raise ValueError("El modelo generó una matriz de probabilidades con dimensiones inválidas.")

    disease_probability = probabilities[:, int(positive_positions[0])]
    predicted_class = (disease_probability >= threshold).astype(int)
    prediction_label = np.where(
        predicted_class == POSITIVE_CLASS,
        POSITIVE_LABEL,
        NEGATIVE_LABEL,
    )

    predictions = original_data.reset_index(drop=True).copy()
    predictions["predicted_class"] = predicted_class
    predictions["prediction_label"] = prediction_label
    predictions["disease_probability"] = disease_probability
    predictions["classification_threshold"] = threshold
    return predictions


def persist_predictions(predictions: pd.DataFrame, output_path: Path) -> None:
    """Persist batch results only after all validation and inference succeed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)


def run_inference_pipeline(
    model_path: Path = DEFAULT_MODEL_PATH,
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> InferencePipelineResult:
    """Execute loading, validation, transformation, prediction and persistence."""
    pipeline = load_inference_model(model_path)
    original_data = load_inference_data(input_path)
    normalized_data = normalize_inference_data(original_data)
    engineered_features = engineer_inference_features(normalized_data, pipeline)
    predictions = generate_predictions(
        pipeline,
        original_data,
        engineered_features,
        threshold,
    )
    persist_predictions(predictions, output_path)

    positive_predictions = int(predictions["predicted_class"].sum())
    return InferencePipelineResult(
        input_rows=len(original_data),
        positive_predictions=positive_predictions,
        negative_predictions=len(original_data) - positive_predictions,
        threshold=threshold,
        output_path=output_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface parser."""
    parser = argparse.ArgumentParser(
        description="Genera predicciones batch de enfermedad cardiaca desde un archivo CSV."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the inference pipeline from the command line."""
    arguments = _build_parser().parse_args(argv)
    result = run_inference_pipeline(
        arguments.model,
        arguments.input,
        arguments.output,
        threshold=arguments.threshold,
    )
    print("Prediction Pipeline completado correctamente.")
    print(f"- Registros procesados: {result.input_rows:,}")
    print(f"- Predicciones con patrón compatible: {result.positive_predictions:,}")
    print(f"- Predicciones sin patrón identificado: {result.negative_predictions:,}")
    print(f"- Umbral utilizado: {result.threshold:.2f}")
    print(f"- Archivo generado: {result.output_path}")
    print("- Aviso: resultado académico; no constituye un diagnóstico clínico.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
