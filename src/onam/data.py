"""Data preparation: outcome extraction, dummy encoding, per-term input matrices.

Python counterpart of the helpers in ``model_setup.R``
(``prepare_data``, ``encode_dummy``, ``get_category_counts``, ``get_output``).
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .terms import ModelInfo, Term

__all__ = [
    "as_frame",
    "category_counts",
    "fit_categories",
    "encode_dummy",
    "prepare_data",
    "resolve_outcome",
]


def as_frame(data, feature_names: Sequence[str] | None = None) -> pd.DataFrame:
    """Coerce ``data`` to a :class:`pandas.DataFrame`.

    Accepts a DataFrame (returned as-is, copy-free) or a 2-D array-like, in
    which case ``feature_names`` supplies the column names.
    """
    if isinstance(data, pd.DataFrame):
        return data
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("`data` must be 2-dimensional (n_samples, n_features).")
    if feature_names is None:
        feature_names = [f"x{i + 1}" for i in range(arr.shape[1])]
    if len(feature_names) != arr.shape[1]:
        raise ValueError(
            f"feature_names has length {len(feature_names)} but data has "
            f"{arr.shape[1]} columns."
        )
    return pd.DataFrame(arr, columns=list(feature_names))


def fit_categories(
    data: pd.DataFrame, categorical_features: Sequence[str]
) -> dict[str, list]:
    """Record the category levels of each categorical feature.

    The first level (in order of appearance) is the reference and is dropped
    during dummy encoding, matching R's ``encode_dummy``.
    """
    categories: dict[str, list] = {}
    for feat in categorical_features:
        seen: list = []
        for val in data[feat].tolist():
            if val not in seen:
                seen.append(val)
        categories[feat] = seen
    return categories


def category_counts(categories: Mapping[str, Sequence]) -> dict[str, int]:
    """Number of dummy columns per categorical feature (``n_levels - 1``)."""
    return {feat: max(len(levels) - 1, 0) for feat, levels in categories.items()}


def encode_dummy(values, levels: Sequence) -> np.ndarray:
    """One-hot encode ``values`` against ``levels[1:]`` (reference-dropped)."""
    values = np.asarray(values)
    cols = [(values == lvl).astype(np.float64) for lvl in levels[1:]]
    if not cols:
        return np.zeros((len(values), 0), dtype=np.float64)
    return np.column_stack(cols)


def _term_matrix(
    data: pd.DataFrame,
    term: Term,
    categories: Mapping[str, Sequence],
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for feat in term.features:
        if feat in categories:
            blocks.append(encode_dummy(data[feat].to_numpy(), categories[feat]))
        else:
            blocks.append(data[feat].to_numpy(dtype=np.float64).reshape(-1, 1))
    return np.column_stack(blocks).astype(np.float64)


def prepare_data(
    data: pd.DataFrame,
    model_info: ModelInfo,
    categories: Mapping[str, Sequence],
) -> list[np.ndarray]:
    """Build one input matrix per term, in ``model_info.flat_terms`` order."""
    return [_term_matrix(data, term, categories) for term in model_info.flat_terms]


def resolve_outcome(
    data: pd.DataFrame,
    outcome,
    *,
    model=None,
    prediction_function: Callable | None = None,
    model_data=None,
    target: str = "continuous",
) -> np.ndarray:
    """Return the training target as a 1-D float array.

    ``outcome`` may be a column name present in ``data`` or an array-like of
    length ``n_samples``.  If ``model`` is given, its predictions on
    ``model_data`` (or ``data``) are used instead and ``outcome`` is ignored.
    """
    if model is not None:
        if prediction_function is None:
            if hasattr(model, "predict"):
                prediction_function = lambda m, d: m.predict(d)  # noqa: E731
            else:
                raise ValueError(
                    "`model` has no `.predict`; supply `prediction_function`."
                )
        src = data if model_data is None else model_data
        y = np.asarray(prediction_function(model, src))
        y = np.squeeze(y)
        if y.ndim != 1:
            raise ValueError("prediction_function must return a 1-D vector.")
        return y.astype(np.float64)

    if isinstance(outcome, str):
        if outcome not in data.columns:
            raise ValueError(
                f"Outcome {outcome!r} specified, but not present in data."
            )
        return data[outcome].to_numpy(dtype=np.float64)

    if outcome is None:
        raise ValueError(
            "No target given. Pass `y` (an array or a column name), fit with a "
            "`formula=` whose outcome column is in the data, or pass `model=`."
        )

    y = np.squeeze(np.asarray(outcome))
    if y.ndim != 1 or len(y) != len(data):
        raise ValueError("`y` must be 1-D with length n_samples.")
    return y.astype(np.float64)
