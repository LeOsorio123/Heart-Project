"""Custom, importable transformers used by the modeling pipelines."""

from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class HeartFeatureEngineer(TransformerMixin, BaseEstimator):
    """Normalize input types and derive the features defined in Task 4."""

    derived_features = (
        "age_squared",
        "old_peak_slope_interaction",
        "max_hr_old_peak_interaction",
        "chest_pain_exang",
    )
    numeric_input = (
        "age",
        "rest_bp",
        "chol",
        "max_hr",
        "old_peak",
        "slope",
        "ca",
    )
    binary_input = ("fbs", "exang")
    nominal_input = ("sex", "chest_pain", "rest_ecg", "thal")

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Record the input schema; no statistics are learned here."""
        del y
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HeartFeatureEngineer requires a pandas DataFrame.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a normalized copy with the four derived attributes."""
        check_is_fitted(self, "feature_names_in_")
        transformed = X.copy()

        for column in self.numeric_input:
            transformed[column] = pd.to_numeric(transformed[column], errors="coerce").astype(float)

        for column in self.binary_input:
            mapped = transformed[column].map({False: 0.0, True: 1.0})
            transformed[column] = mapped.to_numpy(dtype=float, na_value=np.nan)

        for column in self.nominal_input:
            values = transformed[column].astype("string")
            transformed[column] = values.astype(object).where(values.notna(), np.nan)

        transformed["age_squared"] = transformed["age"] ** 2
        transformed["old_peak_slope_interaction"] = transformed["old_peak"] * transformed["slope"]
        transformed["max_hr_old_peak_interaction"] = transformed["max_hr"] * transformed["old_peak"]

        exang_label = transformed["exang"].map({0.0: "no", 1.0: "yes"})
        valid_interaction = transformed["chest_pain"].notna() & exang_label.notna()
        chest_pain_exang = pd.Series(np.nan, index=transformed.index, dtype=object)
        chest_pain_exang.loc[valid_interaction] = (
            transformed.loc[valid_interaction, "chest_pain"].astype(str)
            + "__"
            + exang_label.loc[valid_interaction]
        )
        transformed["chest_pain_exang"] = chest_pain_exang
        return transformed

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Expose base and derived names for downstream transformers."""
        check_is_fitted(self, "feature_names_in_")
        base_features = (
            self.feature_names_in_
            if input_features is None
            else np.asarray(input_features, dtype=object)
        )
        return np.asarray([*base_features, *self.derived_features], dtype=object)
