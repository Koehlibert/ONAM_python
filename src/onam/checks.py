"""Input validation, mirroring ``input_checks.R``."""

from __future__ import annotations

import warnings
from typing import Mapping, Sequence

import numpy as np

from .terms import Term

__all__ = ["check_terms", "check_fit_args", "check_y_features"]


def check_terms(
    terms: Sequence[Term],
    deep_models: Mapping[str, object],
    feature_names: Sequence[str],
    categorical_features: Sequence[str],
) -> None:
    feature_names = list(feature_names)
    used_features: list[str] = []

    for term in terms:
        if term.model not in deep_models:
            raise ValueError(
                f"Term references model {term.model!r}, but it is not in "
                f"`deep_models`."
            )
        used_features.extend(term.features)

    missing = sorted({f for f in used_features if f not in feature_names})
    if missing:
        raise ValueError(
            f"Feature(s) {', '.join(missing)} in the terms, but not present in "
            f"data. Make sure feature names align with the data columns."
        )

    cat_missing = [f for f in categorical_features if f not in feature_names]
    if cat_missing:
        raise ValueError(
            f"{', '.join(cat_missing)} provided in categorical_features, but "
            f"not present in data."
        )

    cat_unused = [f for f in categorical_features if f not in used_features]
    if cat_unused:
        warnings.warn(
            f"Feature(s) {', '.join(cat_unused)} stated as categorical, but "
            f"not present in any term.",
            stacklevel=2,
        )

    highest = max(terms, key=lambda t: t.order)
    highest_features = set(highest.features)
    for term in terms:
        if term is highest:
            continue
        if any(f not in highest_features for f in term.features):
            warnings.warn(
                "Features in lower order effects do not appear in higher order "
                "effects. We recommend fitting a residual term that includes "
                "all lower order terms.",
                stacklevel=2,
            )
            break


def check_fit_args(target: str, n_ensemble: int, epochs: int) -> None:
    if target not in ("continuous", "binary"):
        raise ValueError("`target` must be either 'continuous' or 'binary'.")
    if int(n_ensemble) != n_ensemble or n_ensemble < 1:
        raise ValueError("`n_ensemble` must be a positive integer.")
    if int(epochs) != epochs or epochs < 1:
        raise ValueError("`epochs` must be a positive integer.")


def check_y_features(data, y: np.ndarray, model_info, target: str) -> None:
    """Warn if a data column all but equals the outcome (R ``check_y_features``)."""
    if not (model_info.all_feature_indic and target == "continuous"):
        return
    outcome = model_info.outcome
    for col in data.columns:
        if col == outcome:
            continue
        values = data[col].to_numpy()
        if not np.issubdtype(values.dtype, np.number):
            continue
        if np.std(values) == 0:
            continue
        corr = float(np.corrcoef(values, y)[0, 1])
        if corr > 0.99:
            warnings.warn(
                f"Terms include a residual `model(.)` term and data contains "
                f"column {col!r} with correlation {corr:.4f} with the outcome. "
                f"If supplying a `model` to generate the response, make sure "
                f"the original outcome is not contained in `data`.",
                stacklevel=2,
            )
            return
