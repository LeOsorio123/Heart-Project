"""Utilities for validating inputs and serving heart disease predictions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import joblib
import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "age",
    "sex",
    "chest_pain",
    "rest_bp",
    "chol",
    "fbs",
    "rest_ecg",
    "max_hr",
    "exang",
    "old_peak",
    "slope",
    "ca",
    "thal",
)

CATEGORY_VALUES = {
    "sex": ("Male", "Female"),
    "chest_pain": ("typical", "asymptomatic", "nonanginal", "nontypical"),
    "rest_ecg": (
        "normal",
        "ST-T wave abnormality",
        "left ventricular hypertrophy",
    ),
    "thal": ("normal", "fixed", "reversable"),
}

NUMERIC_BOUNDS = {
    "age": (29.0, 77.0),
    "rest_bp": (94.0, 200.0),
    "chol": (126.0, 564.0),
    "max_hr": (71.0, 202.0),
    "old_peak": (0.0, 6.2),
    "slope": (1.0, 3.0),
    "ca": (0.0, 3.0),
}


class PredictivePipeline(Protocol):
    """Minimal interface required from the persisted scikit-learn pipeline."""

    feature_names_in_: np.ndarray
    classes_: np.ndarray

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict the class for each row."""

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for each row."""


@dataclass(frozen=True)
class PredictionResult:
    """Prediction returned by the local POC."""

    predicted_class: int
    disease_probability: float
    threshold: float

    @property
    def has_disease_pattern(self) -> bool:
        """Return whether the positive class was predicted."""
        return self.predicted_class == 1


def load_pipeline(model_path: Path) -> PredictivePipeline:
    """Load the persisted preprocessing and model pipeline."""
    if not model_path.is_file():
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
    return cast(PredictivePipeline, joblib.load(model_path))


def build_patient_frame(patient_data: dict[str, object]) -> pd.DataFrame:
    """Validate a patient payload and return one model-ready row."""
    missing = [column for column in FEATURE_COLUMNS if column not in patient_data]
    extra = [column for column in patient_data if column not in FEATURE_COLUMNS]
    if missing or extra:
        raise ValueError(
            "El registro no coincide con el esquema esperado. "
            f"Faltantes: {missing or 'ninguno'}; adicionales: {extra or 'ninguno'}."
        )

    validated = dict(patient_data)
    for column, valid_values in CATEGORY_VALUES.items():
        if validated[column] not in valid_values:
            raise ValueError(f"{column} debe ser uno de estos valores: {', '.join(valid_values)}.")

    for column, (lower, upper) in NUMERIC_BOUNDS.items():
        try:
            numeric_value = float(cast(str | int | float, validated[column]))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{column} debe ser numérico.") from error
        if not lower <= numeric_value <= upper:
            raise ValueError(f"{column} debe estar entre {lower:g} y {upper:g}.")

    for column in ("fbs", "exang"):
        if not isinstance(validated[column], bool):
            raise TypeError(f"{column} debe ser un valor booleano.")

    for column in ("age", "rest_bp", "chol", "max_hr", "slope", "ca"):
        validated[column] = int(float(cast(str | int | float, validated[column])))
    validated["old_peak"] = float(cast(str | int | float, validated["old_peak"]))

    return pd.DataFrame([validated], columns=FEATURE_COLUMNS)


def predict_patient(
    pipeline: PredictivePipeline,
    patient_data: dict[str, object],
    *,
    threshold: float = 0.5,
) -> PredictionResult:
    """Validate one record and generate its positive-class probability."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("El umbral debe estar entre 0 y 1.")

    features = build_patient_frame(patient_data)
    expected_columns = tuple(str(column) for column in pipeline.feature_names_in_)
    if expected_columns != FEATURE_COLUMNS:
        raise ValueError("El esquema del modelo no coincide con el esquema de la aplicación.")

    probabilities = pipeline.predict_proba(features)
    positive_positions = np.flatnonzero(pipeline.classes_ == 1)
    if len(positive_positions) != 1:
        raise ValueError("El modelo no contiene una única clase positiva identificada como 1.")

    disease_probability = float(probabilities[0, int(positive_positions[0])])
    predicted_class = int(disease_probability >= threshold)
    return PredictionResult(
        predicted_class=predicted_class,
        disease_probability=disease_probability,
        threshold=threshold,
    )


POSITIVE_LABEL = "Patrón compatible con enfermedad cardiaca"
NEGATIVE_LABEL = "No se identifica patrón compatible con enfermedad cardiaca"
BINARY_COLUMNS = ("fbs", "exang")


def _normalize_binary_value(value: object, column: str) -> bool:
    """Convert the boolean representations accepted in uploaded CSV files."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true"}:
            return True
        if normalized in {"0", "false"}:
            return False

    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric_value = float(value)
        if numeric_value in {0.0, 1.0}:
            return bool(int(numeric_value))

    raise ValueError(f"{column} debe ser booleano o usar 0/1.")


def build_batch_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Validate multiple uploaded records and return the model input frame."""
    missing = sorted(set(FEATURE_COLUMNS).difference(data.columns))
    unexpected = sorted(set(data.columns).difference(FEATURE_COLUMNS))
    if missing or unexpected:
        raise ValueError(
            "El archivo no coincide con el esquema esperado. "
            f"Faltantes: {missing or 'ninguna'}; adicionales: {unexpected or 'ninguna'}."
        )
    if data.empty:
        raise ValueError("El archivo no contiene registros para procesar.")

    validated_rows: list[pd.DataFrame] = []
    for row_number, (_, row) in enumerate(data.iterrows(), start=2):
        patient_data = {str(column): value for column, value in row.to_dict().items()}
        try:
            for column in BINARY_COLUMNS:
                patient_data[column] = _normalize_binary_value(patient_data[column], column)
            validated_rows.append(build_patient_frame(patient_data))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Fila {row_number}: {error}") from error

    return pd.concat(validated_rows, ignore_index=True).loc[:, FEATURE_COLUMNS]


def predict_batch(
    pipeline: PredictivePipeline,
    data: pd.DataFrame,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Validate a batch and append understandable prediction results."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("El umbral debe estar entre 0 y 1.")

    features = build_batch_frame(data)
    expected_columns = tuple(str(column) for column in pipeline.feature_names_in_)
    if expected_columns != FEATURE_COLUMNS:
        raise ValueError("El esquema del modelo no coincide con el esquema de la aplicación.")

    probabilities = pipeline.predict_proba(features)
    positive_positions = np.flatnonzero(pipeline.classes_ == 1)
    if len(positive_positions) != 1:
        raise ValueError("El modelo no contiene una única clase positiva identificada como 1.")
    if probabilities.shape != (len(features), len(pipeline.classes_)):
        raise ValueError("El modelo generó probabilidades con dimensiones inválidas.")

    disease_probability = probabilities[:, int(positive_positions[0])]
    predicted_class = (disease_probability >= threshold).astype(int)
    prediction_label = np.where(
        predicted_class == 1,
        POSITIVE_LABEL,
        NEGATIVE_LABEL,
    )

    results = features.copy()
    results["predicted_class"] = predicted_class
    results["prediction_label"] = prediction_label
    results["disease_probability"] = disease_probability
    results["classification_threshold"] = threshold
    return results
