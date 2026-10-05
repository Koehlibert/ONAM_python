"""Evaluate fitted ensemble members on data.

Port of ``evaluation.R`` (``evaluate_onam_single``, ``evaluate_onam_pre`` and
the aggregation inside ``predict.onam``).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .data import prepare_data
from .orthogonalization import EnsembleMember, _get_u
from .terms import ModelInfo

__all__ = ["member_effects", "ensemble_effects"]


def member_effects(
    member: EnsembleMember,
    model_info: ModelInfo,
    categories: Mapping[str, Sequence],
    data: pd.DataFrame,
    *,
    orthogonalized: bool = True,
) -> np.ndarray:
    """``(n_samples, n_terms)`` effect matrix for one ensemble member."""
    data_fit = prepare_data(data, model_info, categories)
    u, _, _ = _get_u(member.submodels, data_fit)
    w_list = member.w_list if orthogonalized else member.w_list_old
    return np.column_stack([u @ w for w in w_list])


def ensemble_effects(
    members: list[EnsembleMember],
    model_info: ModelInfo,
    categories: Mapping[str, Sequence],
    data: pd.DataFrame,
) -> np.ndarray:
    """Ensemble-averaged orthogonalized effects, ``(n_samples, n_terms)``.

    This is R's ``predictions_features_ensemble`` stacked into a matrix.
    """
    per_member = np.stack(
        [member_effects(m, model_info, categories, data) for m in members],
        axis=0,
    )
    return per_member.mean(axis=0)
