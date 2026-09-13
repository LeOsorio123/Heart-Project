"""Validate classifier performance and generalization without using the test set."""

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

DEFAULT_CV_FOLDS = 5
DEFAULT_MIN_ACCEPTABLE_RECALL = 0.75
DEFAULT_OVERFIT_GAP = 0.10
DEFAULT_TEST_GAP = 0.10
MIN_CV_FOLDS = 2

VALIDATION_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "recall",
    "precision",
    "f1",
    "roc_auc",
)


@dataclass(frozen=True)
class ModelValidationConfig:
    """Thresholds and reproducibility settings for model validation."""

    cv_folds: int = DEFAULT_CV_FOLDS
    random_state: int = 42
    min_acceptable_recall: float = DEFAULT_MIN_ACCEPTABLE_RECALL
    overfit_gap: float = DEFAULT_OVERFIT_GAP
    test_gap: float = DEFAULT_TEST_GAP


@dataclass(frozen=True)
class ModelValidationReport:
    """Serializable summary of cross-validation and generalization checks."""

    status: str
    cross_validation: dict[str, object]
    comparison: dict[str, dict[str, float]]
    diagnosis: str
    warnings: tuple[str, ...]
    recommendations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the report."""
        return {
            "status": self.status,
            "cross_validation": self.cross_validation,
            "comparison": self.comparison,
            "diagnosis": self.diagnosis,
            "warnings": list(self.warnings),
            "recommendations": list(self.recommendations),
        }


@dataclass(frozen=True)
class ModelMetricSets:
    """Train and test metrics used to compare model generalization."""

    train: dict[str, object]
    test: dict[str, object]


def _validate_config(config: ModelValidationConfig) -> None:
    """Reject invalid validation settings before starting expensive work."""
    if config.cv_folds < MIN_CV_FOLDS:
        raise ValueError("cv_folds debe ser al menos 2.")
    for field_name, threshold in (
        ("min_acceptable_recall", config.min_acceptable_recall),
        ("overfit_gap", config.overfit_gap),
        ("test_gap", config.test_gap),
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"{field_name} debe estar entre 0 y 1.")


def run_cross_validation(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelValidationConfig,
) -> dict[str, object]:
    """Evaluate the complete pipeline using only stratified training folds."""
    _validate_config(config)
    minimum_class_rows = int(y_train.value_counts().min())
    if minimum_class_rows < config.cv_folds:
        raise ValueError(
            "Cada clase de train necesita al menos tantos registros como folds de validación. "
            f"Mínimo observado: {minimum_class_rows}; folds solicitados: {config.cv_folds}."
        )

    splitter = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    raw_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=splitter,
        scoring=VALIDATION_METRICS,
        return_train_score=False,
        n_jobs=1,
    )

    metric_summaries: dict[str, object] = {}
    for metric_name in VALIDATION_METRICS:
        fold_scores = np.asarray(raw_results[f"test_{metric_name}"], dtype=float)
        metric_summaries[metric_name] = {
            "fold_scores": [float(score) for score in fold_scores],
            "mean": float(fold_scores.mean()),
            "std": float(fold_scores.std(ddof=0)),
        }

    return {
        "strategy": "StratifiedKFold",
        "folds": config.cv_folds,
        "shuffle": True,
        "random_state": config.random_state,
        "uses_training_data_only": True,
        "metrics": metric_summaries,
    }


def _comparison_table(
    train_metrics: dict[str, object],
    cross_validation: dict[str, object],
    test_metrics: dict[str, object],
) -> dict[str, dict[str, float]]:
    """Compare final-fit train, cross-validation and untouched test metrics."""
    cv_metrics = cross_validation["metrics"]
    if not isinstance(cv_metrics, dict):
        raise TypeError("Las métricas de validación cruzada tienen un formato inválido.")

    comparison: dict[str, dict[str, float]] = {}
    for metric_name in VALIDATION_METRICS:
        cv_summary = cv_metrics[metric_name]
        if not isinstance(cv_summary, dict):
            raise TypeError(f"El resumen de {metric_name} tiene un formato inválido.")
        train_value = float(cast(float, train_metrics[metric_name]))
        cv_value = float(cv_summary["mean"])
        test_value = float(cast(float, test_metrics[metric_name]))
        comparison[metric_name] = {
            "train": train_value,
            "cv_mean": cv_value,
            "cv_std": float(cv_summary["std"]),
            "test": test_value,
            "train_cv_gap": train_value - cv_value,
            "cv_test_gap": abs(cv_value - test_value),
        }
    return comparison


def analyze_generalization(
    train_metrics: dict[str, object],
    cross_validation: dict[str, object],
    test_metrics: dict[str, object],
    config: ModelValidationConfig,
) -> ModelValidationReport:
    """Classify the observed recall pattern and suggest follow-up actions."""
    _validate_config(config)
    comparison = _comparison_table(train_metrics, cross_validation, test_metrics)
    recall = comparison["recall"]
    warnings: list[str] = []
    recommendations: list[str] = []

    if (
        recall["cv_mean"] < config.min_acceptable_recall
        and recall["test"] < config.min_acceptable_recall
    ):
        diagnosis = "underfitting"
        warnings.append(
            "El recall de validación cruzada y test está por debajo del mínimo esperado."
        )
        recommendations.append(
            "Revisar la representación de variables, la complejidad del modelo y el umbral de "
            "clasificación en una tarea posterior de experimentación."
        )
    elif recall["train_cv_gap"] > config.overfit_gap:
        diagnosis = "overfitting"
        warnings.append(
            "La diferencia de recall entre train y validación cruzada sugiere sobreajuste."
        )
        recommendations.append(
            "Evaluar mayor regularización, menor complejidad o ajuste de hiperparámetros usando "
            "solo train y validación cruzada."
        )
    elif recall["cv_test_gap"] > config.test_gap:
        diagnosis = "unstable_generalization"
        warnings.append(
            "El recall de test difiere de la media de validación cruzada más de lo esperado."
        )
        recommendations.append(
            "Revisar la variabilidad entre folds y la representatividad del test sin usarlo para "
            "ajustar el modelo."
        )
    else:
        diagnosis = "stable_generalization"
        recommendations.append(
            "Conservar esta evaluación como línea base y monitorear las mismas métricas en datos "
            "futuros."
        )

    return ModelValidationReport(
        status="passed_with_warnings" if warnings else "passed",
        cross_validation=cross_validation,
        comparison=comparison,
        diagnosis=diagnosis,
        warnings=tuple(warnings),
        recommendations=tuple(recommendations),
    )


def validate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    metric_sets: ModelMetricSets,
    config: ModelValidationConfig,
) -> ModelValidationReport:
    """Run reproducible cross-validation and analyze model generalization."""
    cross_validation = run_cross_validation(pipeline, X_train, y_train, config)
    return analyze_generalization(
        metric_sets.train,
        cross_validation,
        metric_sets.test,
        config,
    )
