"""Validate train/test integrity and distribution representativeness."""

from dataclasses import dataclass
from typing import Protocol

import pandas as pd
from scipy.stats import ks_2samp

DEFAULT_SIZE_TOLERANCE = 0.02
DEFAULT_TARGET_RATE_TOLERANCE = 0.05
DEFAULT_NUMERIC_KS_THRESHOLD = 0.20
DEFAULT_CATEGORICAL_TVD_THRESHOLD = 0.20
MISSING_CATEGORY = "__MISSING__"
TARGET_HASH_COLUMN = "__target__"


class SplitData(Protocol):
    """Minimal interface required from a train/test partition."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class TrainTestValidationError(ValueError):
    """Raised when a train/test split violates a critical integrity rule."""


@dataclass(frozen=True)
class SplitValidationConfig:
    """Expected schema and configurable distribution thresholds."""

    expected_features: tuple[str, ...]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    expected_test_size: float
    size_tolerance: float = DEFAULT_SIZE_TOLERANCE
    target_rate_tolerance: float = DEFAULT_TARGET_RATE_TOLERANCE
    numeric_ks_threshold: float = DEFAULT_NUMERIC_KS_THRESHOLD
    categorical_tvd_threshold: float = DEFAULT_CATEGORICAL_TVD_THRESHOLD


@dataclass(frozen=True)
class TrainTestValidationReport:
    """JSON-serializable results from train/test validation."""

    status: str
    critical_checks: dict[str, object]
    distribution_checks: dict[str, object]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a representation suitable for the training metrics JSON."""
        return {
            "status": self.status,
            "critical_checks": self.critical_checks,
            "distribution_checks": self.distribution_checks,
            "warnings": list(self.warnings),
        }


def _validate_probability_threshold(name: str, value: float) -> None:
    """Reject thresholds outside the probability interval."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} debe estar entre 0 y 1.")


def _validate_config(config: SplitValidationConfig) -> None:
    """Validate every configurable probability or tolerance."""
    for name, value in (
        ("expected_test_size", config.expected_test_size),
        ("size_tolerance", config.size_tolerance),
        ("target_rate_tolerance", config.target_rate_tolerance),
        ("numeric_ks_threshold", config.numeric_ks_threshold),
        ("categorical_tvd_threshold", config.categorical_tvd_threshold),
    ):
        _validate_probability_threshold(name, value)


def _sample_hashes(features: pd.DataFrame, target: pd.Series) -> set[int]:
    """Hash complete labeled samples without including their DataFrame index."""
    labeled = features.copy()
    labeled[TARGET_HASH_COLUMN] = target.to_numpy()
    hashes = pd.util.hash_pandas_object(labeled, index=False)
    return {int(value) for value in hashes}


def _category_values(series: pd.Series) -> set[str]:
    """Return normalized non-null categorical values."""
    values = series.astype("string").dropna().unique().tolist()
    return {str(value) for value in values}


def _total_variation_distance(train: pd.Series, test: pd.Series) -> float:
    """Calculate categorical distribution distance on a zero-to-one scale."""
    train_values = train.astype("string").fillna(MISSING_CATEGORY)
    test_values = test.astype("string").fillna(MISSING_CATEGORY)
    train_rates = train_values.value_counts(normalize=True)
    test_rates = test_values.value_counts(normalize=True)
    categories = train_rates.index.union(test_rates.index)
    aligned_train = train_rates.reindex(categories, fill_value=0.0)
    aligned_test = test_rates.reindex(categories, fill_value=0.0)
    return float((aligned_train - aligned_test).abs().sum() / 2.0)


def _raise_critical_errors(errors: list[str]) -> None:
    """Raise one actionable exception containing every critical failure."""
    if errors:
        details = "\n- ".join(errors)
        raise TrainTestValidationError(
            f"La validación crítica del split train/test falló:\n- {details}"
        )


def _validate_partition_structure(
    split: SplitData,
    config: SplitValidationConfig,
) -> dict[str, object]:
    """Validate schema, emptiness and alignment between features and targets."""
    train_schema = tuple(str(column) for column in split.X_train.columns)
    test_schema = tuple(str(column) for column in split.X_test.columns)
    train_rows = len(split.X_train)
    test_rows = len(split.X_test)
    schema_matches = (
        train_schema == config.expected_features and test_schema == config.expected_features
    )
    non_empty = train_rows > 0 and test_rows > 0
    aligned_lengths = train_rows == len(split.y_train) and test_rows == len(split.y_test)

    errors: list[str] = []
    if not schema_matches:
        errors.append("train y test deben conservar exactamente el esquema esperado")
    if not non_empty:
        errors.append("train y test deben contener al menos un registro")
    if not aligned_lengths:
        errors.append("las longitudes de X e y no coinciden en train o test")
    _raise_critical_errors(errors)

    return {
        "schema": {
            "passed": schema_matches,
            "expected": list(config.expected_features),
            "train": list(train_schema),
            "test": list(test_schema),
        },
        "non_empty_partitions": {
            "passed": non_empty,
            "train_rows": train_rows,
            "test_rows": test_rows,
        },
        "aligned_feature_target_lengths": {"passed": aligned_lengths},
    }


def _find_new_categories(
    split: SplitData,
    categorical_features: tuple[str, ...],
) -> dict[str, list[str]]:
    """Return test categories that were unavailable during training."""
    new_categories: dict[str, list[str]] = {}
    for column in categorical_features:
        unseen = sorted(
            _category_values(split.X_test[column]).difference(
                _category_values(split.X_train[column])
            )
        )
        if unseen:
            new_categories[column] = unseen
    return new_categories


def _validate_leakage_and_domains(
    split: SplitData,
    config: SplitValidationConfig,
) -> dict[str, object]:
    """Detect index leakage, duplicated samples and unseen target domains."""
    overlap_count = len(split.X_train.index.intersection(split.X_test.index))
    duplicate_sample_count = len(
        _sample_hashes(split.X_train, split.y_train).intersection(
            _sample_hashes(split.X_test, split.y_test)
        )
    )
    train_labels = {int(value) for value in split.y_train.dropna().unique()}
    test_labels = {int(value) for value in split.y_test.dropna().unique()}
    new_labels = sorted(test_labels.difference(train_labels))
    new_categories = _find_new_categories(split, config.categorical_features)

    errors: list[str] = []
    if overlap_count:
        errors.append(f"se encontraron {overlap_count} índice(s) compartidos entre train y test")
    if duplicate_sample_count:
        errors.append(
            f"se encontraron {duplicate_sample_count} muestra(s) idénticas en train y test"
        )
    if new_labels:
        errors.append(f"test contiene etiquetas nuevas: {new_labels}")
    if new_categories:
        errors.append(f"test contiene categorías nuevas: {new_categories}")
    _raise_critical_errors(errors)

    return {
        "index_leakage": {
            "passed": overlap_count == 0,
            "overlap_count": overlap_count,
        },
        "duplicate_samples": {
            "passed": duplicate_sample_count == 0,
            "shared_hash_count": duplicate_sample_count,
        },
        "new_labels_in_test": {"passed": not new_labels, "values": new_labels},
        "new_categories_in_test": {
            "passed": not new_categories,
            "columns": new_categories,
        },
    }


def _validate_partition_and_target_rates(
    split: SplitData,
    config: SplitValidationConfig,
) -> tuple[dict[str, object], list[str]]:
    """Compare split size and target prevalence against configured tolerances."""
    train_rows = len(split.X_train)
    test_rows = len(split.X_test)
    observed_test_size = test_rows / (train_rows + test_rows)
    size_deviation = abs(observed_test_size - config.expected_test_size)
    size_passed = size_deviation <= config.size_tolerance

    train_positive_rate = float(split.y_train.mean())
    test_positive_rate = float(split.y_test.mean())
    target_rate_difference = abs(train_positive_rate - test_positive_rate)
    target_rate_passed = target_rate_difference <= config.target_rate_tolerance

    warnings: list[str] = []
    if not size_passed:
        warnings.append(
            "La proporción observada de test difiere del valor configurado "
            f"en {size_deviation:.3f}."
        )
    if not target_rate_passed:
        warnings.append(
            "La prevalencia de la clase positiva difiere entre train y test "
            f"en {target_rate_difference:.3f}."
        )

    checks: dict[str, object] = {
        "partition_size": {
            "passed": size_passed,
            "expected_test_size": config.expected_test_size,
            "observed_test_size": observed_test_size,
            "absolute_difference": size_deviation,
            "tolerance": config.size_tolerance,
        },
        "target_rate": {
            "passed": target_rate_passed,
            "train_positive_rate": train_positive_rate,
            "test_positive_rate": test_positive_rate,
            "absolute_difference": target_rate_difference,
            "tolerance": config.target_rate_tolerance,
        },
    }
    return checks, warnings


def _validate_numeric_drift(
    split: SplitData,
    config: SplitValidationConfig,
) -> tuple[dict[str, object], list[str]]:
    """Compare numeric distributions with the Kolmogorov-Smirnov statistic."""
    checks: dict[str, object] = {}
    warnings: list[str] = []
    for column in config.numeric_features:
        train_values = pd.to_numeric(split.X_train[column], errors="coerce").dropna()
        test_values = pd.to_numeric(split.X_test[column], errors="coerce").dropna()
        statistic = (
            None
            if train_values.empty or test_values.empty
            else float(ks_2samp(train_values, test_values).statistic)
        )
        passed = statistic is not None and statistic <= config.numeric_ks_threshold
        checks[column] = {
            "passed": passed,
            "ks_statistic": statistic,
            "threshold": config.numeric_ks_threshold,
            "train_non_null": len(train_values),
            "test_non_null": len(test_values),
        }
        if not passed:
            detail = "sin datos suficientes" if statistic is None else f"KS={statistic:.3f}"
            warnings.append(f"La variable numérica {column} presenta drift ({detail}).")
    return checks, warnings


def _validate_categorical_drift(
    split: SplitData,
    config: SplitValidationConfig,
) -> tuple[dict[str, object], list[str]]:
    """Compare categorical distributions with total variation distance."""
    checks: dict[str, object] = {}
    warnings: list[str] = []
    for column in config.categorical_features:
        distance = _total_variation_distance(split.X_train[column], split.X_test[column])
        passed = distance <= config.categorical_tvd_threshold
        checks[column] = {
            "passed": passed,
            "total_variation_distance": distance,
            "threshold": config.categorical_tvd_threshold,
        }
        if not passed:
            warnings.append(f"La variable categórica {column} presenta drift (TVD={distance:.3f}).")
    return checks, warnings


def validate_train_test_split(
    split: SplitData,
    config: SplitValidationConfig,
) -> TrainTestValidationReport:
    """Validate split integrity and compare train/test distributions.

    Critical integrity failures stop training. Distribution differences are
    recorded as warnings because they require review but are not proof of
    leakage by themselves.
    """
    _validate_config(config)
    critical_checks = _validate_partition_structure(split, config)
    critical_checks.update(_validate_leakage_and_domains(split, config))

    rate_checks, rate_warnings = _validate_partition_and_target_rates(split, config)
    numeric_checks, numeric_warnings = _validate_numeric_drift(split, config)
    categorical_checks, categorical_warnings = _validate_categorical_drift(split, config)
    warnings = (*rate_warnings, *numeric_warnings, *categorical_warnings)
    distribution_checks = {
        **rate_checks,
        "numeric_drift": numeric_checks,
        "categorical_drift": categorical_checks,
    }
    status = "passed_with_warnings" if warnings else "passed"
    return TrainTestValidationReport(
        status=status,
        critical_checks=critical_checks,
        distribution_checks=distribution_checks,
        warnings=warnings,
    )
