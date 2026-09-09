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
