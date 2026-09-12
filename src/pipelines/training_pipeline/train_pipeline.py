"""Train and evaluate the selected heart-disease classifier autonomously."""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from heart_project.prediction import FEATURE_COLUMNS  # noqa: E402
from heart_project.transformers import HeartFeatureEngineer  # noqa: E402
from pipelines.training_pipeline.split_validation import (  # noqa: E402
    SplitValidationConfig,
    TrainTestValidationReport,
    validate_train_test_split,
)

TARGET = "disease"
POSITIVE_CLASS = 1
NEGATIVE_CLASS = 0
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 250
MIN_SAMPLES_LEAF = 3
MIN_CLASS_ROWS_FOR_SPLIT = 2
PARALLEL_JOBS = 1

NUMERIC_FEATURES = (
    "age",
    "rest_bp",
    "chol",
    "max_hr",
    "old_peak",
    "ca",
    "age_squared",
    "old_peak_slope_interaction",
    "max_hr_old_peak_interaction",
)
BINARY_FEATURES = ("fbs", "exang")
ORDINAL_FEATURES = ("slope",)
NOMINAL_FEATURES = (
    "sex",
    "chest_pain",
    "rest_ecg",
    "thal",
    "chest_pain_exang",
)
CATEGORICAL_SPLIT_FEATURES = (*BINARY_FEATURES, *ORDINAL_FEATURES, *NOMINAL_FEATURES)
MODEL_FEATURES = (
    *NUMERIC_FEATURES,
    *BINARY_FEATURES,
    *ORDINAL_FEATURES,
    *NOMINAL_FEATURES,
)
EXPECTED_COLUMNS = (*FEATURE_COLUMNS, *HeartFeatureEngineer.derived_features, TARGET)

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "04_feature" / "heart_features.parquet"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "06_models" / "heart_disease_training_pipeline.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "data" / "07_model_output" / "training_metrics.json"


@dataclass(frozen=True)
class DataSplit:
    """Reproducible train/test partitions used by the training pipeline."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class TrainingPipelineResult:
    """Execution summary returned by the training pipeline."""

    input_rows: int
    train_rows: int
    test_rows: int
    metrics: dict[str, object]
    split_validation: dict[str, object]
    model_path: Path
    metrics_path: Path


def load_feature_data(input_path: Path) -> pd.DataFrame:
    """Load the validated feature table and enforce its training schema."""
    if not input_path.is_file():
        raise FileNotFoundError(
            f"No se encontró la tabla de features: {input_path}. "
            "Ejecute primero el Feature Pipeline o use --input."
        )

    feature_data = pd.read_parquet(input_path, engine="pyarrow")
    missing = sorted(set(EXPECTED_COLUMNS).difference(feature_data.columns))
    unexpected = sorted(set(feature_data.columns).difference(EXPECTED_COLUMNS))
    if missing or unexpected:
        raise ValueError(
            "El esquema de entrenamiento no coincide con el esperado. "
            f"Faltantes: {missing or 'ninguna'}; adicionales: {unexpected or 'ninguna'}."
        )
    if feature_data.empty:
        raise ValueError("La tabla de features no contiene registros para entrenar.")
    if feature_data[TARGET].isna().any():
        raise ValueError("La variable objetivo disease contiene valores faltantes.")

    target_values = set(pd.to_numeric(feature_data[TARGET], errors="coerce").unique())
    if target_values != {NEGATIVE_CLASS, POSITIVE_CLASS}:
        raise ValueError("La variable objetivo disease debe contener exactamente las clases 0 y 1.")

    return feature_data.loc[:, EXPECTED_COLUMNS]


def split_training_data(
    feature_data: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> DataSplit:
    """Create a reproducible stratified split without fitting transformations."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size debe estar entre 0 y 1.")

    X = feature_data.loc[:, MODEL_FEATURES]
    y = feature_data[TARGET].astype("int8")
    if int(y.value_counts().min()) < MIN_CLASS_ROWS_FOR_SPLIT:
        raise ValueError("Cada clase necesita al menos dos registros para separar train y test.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return DataSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def build_training_pipeline(
    *,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build the preprocessing and Random Forest pipeline selected in the POC."""
    numeric_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", RobustScaler()),
        ]
    )
    binary_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True))]
    )
    ordinal_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True))]
    )
    nominal_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="missing",
                    keep_empty_features=True,
                ),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("binary", binary_pipeline, list(BINARY_FEATURES)),
            ("ordinal", ordinal_pipeline, list(ORDINAL_FEATURES)),
            ("nominal", nominal_pipeline, list(NOMINAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    classifier = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        n_jobs=PARALLEL_JOBS,
        random_state=random_state,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", classifier)])


def evaluate_classifier(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, object]:
    """Calculate classification metrics with recall as the primary measure."""
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)
    positive_positions = np.flatnonzero(pipeline.classes_ == POSITIVE_CLASS)
    if len(positive_positions) != 1:
        raise ValueError("El modelo no contiene una única clase positiva identificada como 1.")
    positive_probability = probabilities[:, int(positive_positions[0])]
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[NEGATIVE_CLASS, POSITIVE_CLASS],
    ).ravel()

    return {
        "primary_metric": "recall",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "specificity": float(recall_score(y_test, predictions, pos_label=0, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, positive_probability)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def build_evaluation_report(
    split: DataSplit,
    metrics: dict[str, object],
    split_validation: TrainTestValidationReport,
    *,
    random_state: int,
    test_size: float,
) -> dict[str, object]:
    """Build a JSON-serializable record of the model, split and evaluation."""
    return {
        "model": {
            "algorithm": "RandomForestClassifier",
            "parameters": {
                "n_estimators": N_ESTIMATORS,
                "min_samples_leaf": MIN_SAMPLES_LEAF,
                "class_weight": "balanced",
                "random_state": random_state,
            },
        },
        "split": {
            "test_size": test_size,
            "random_state": random_state,
            "stratified": True,
            "train_rows": len(split.X_train),
            "test_rows": len(split.X_test),
            "train_positive_rate": float(split.y_train.mean()),
            "test_positive_rate": float(split.y_test.mean()),
        },
        "split_validation": split_validation.to_dict(),
        "metrics": metrics,
    }


def persist_training_artifacts(
    pipeline: Pipeline,
    report: dict[str, object],
    model_path: Path,
    metrics_path: Path,
) -> None:
    """Persist the fitted pipeline and its evaluation report."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_training_pipeline(
    input_path: Path = DEFAULT_INPUT_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> TrainingPipelineResult:
    """Execute feature loading, splitting, training, evaluation and persistence."""
    feature_data = load_feature_data(input_path)
    split = split_training_data(
        feature_data,
        test_size=test_size,
        random_state=random_state,
    )
    split_validation = validate_train_test_split(
        split,
        SplitValidationConfig(
            expected_features=MODEL_FEATURES,
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_SPLIT_FEATURES,
            expected_test_size=test_size,
        ),
    )
    pipeline = build_training_pipeline(random_state=random_state)
    pipeline.fit(split.X_train, split.y_train)
    metrics = evaluate_classifier(pipeline, split.X_test, split.y_test)
    report = build_evaluation_report(
        split,
        metrics,
        split_validation,
        random_state=random_state,
        test_size=test_size,
    )
    persist_training_artifacts(pipeline, report, model_path, metrics_path)

    return TrainingPipelineResult(
        input_rows=len(feature_data),
        train_rows=len(split.X_train),
        test_rows=len(split.X_test),
        metrics=metrics,
        split_validation=split_validation.to_dict(),
        model_path=model_path,
        metrics_path=metrics_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface parser."""
    parser = argparse.ArgumentParser(
        description="Entrena y evalúa el modelo de clasificación de enfermedad cardiaca."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the training pipeline from the command line."""
    arguments = _build_parser().parse_args(argv)
    result = run_training_pipeline(
        arguments.input,
        arguments.model_output,
        arguments.metrics_output,
        test_size=arguments.test_size,
        random_state=arguments.random_state,
    )
    print("Training Pipeline completado correctamente.")
    print(f"- Registros disponibles: {result.input_rows:,}")
    print(f"- Train: {result.train_rows:,}; test: {result.test_rows:,}")
    recall = cast(float, result.metrics["recall"])
    roc_auc = cast(float, result.metrics["roc_auc"])
    print(f"- Recall: {recall:.3f}")
    print(f"- ROC AUC: {roc_auc:.3f}")
    print(f"- Validación train/test: {result.split_validation['status']}")
    split_warnings = cast(list[str], result.split_validation["warnings"])
    if split_warnings:
        print(f"- Advertencias de distribución: {len(split_warnings)}")
    print(f"- Modelo generado: {result.model_path}")
    print(f"- Métricas generadas: {result.metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
