"""The :class:`ONAM` estimator and its prediction / summary result types."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import keras
import numpy as np
import pandas as pd

from .build import build_model
from .checks import check_fit_args, check_terms, check_y_features
from .data import (
    as_frame,
    category_counts,
    fit_categories,
    prepare_data,
    resolve_outcome,
)
from .decomposition import GenSobol, VarDecomp, decompose, gen_sobol
from .evaluate import ensemble_effects
from .orthogonalization import EnsembleMember, pho, pho_ensemble
from .terms import ModelInfo, Term, parse_formula

__all__ = ["ONAM", "ONAMPrediction", "Summary"]


# --------------------------------------------------------------------------- #
def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic."""
    y_true = np.asarray(y_true)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    s = scores[order]
    ranks_sorted = np.arange(1, len(scores) + 1, dtype=float)
    # average ranks for ties
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks_sorted[i : j + 1] = (i + j) / 2 + 1
        i = j + 1
    ranks[order] = ranks_sorted
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# --------------------------------------------------------------------------- #
@dataclass
class ONAMPrediction:
    """Result of :meth:`ONAM.predict` on (new) data."""

    data: pd.DataFrame
    predictions: np.ndarray
    feature_effects: pd.DataFrame
    model_info: ModelInfo

    def decompose(self, data=None) -> VarDecomp:
        if data is not None:
            raise ValueError(
                "Pass `data` to ONAM.decompose, not to a prediction's."
            )
        return decompose(self)

    def gen_sobol(self, data=None) -> GenSobol:
        if data is not None:
            raise ValueError(
                "Pass `data` to ONAM.gen_sobol, not to a prediction's."
            )
        return gen_sobol(self)


@dataclass
class Summary:
    """Human-readable summary of a fitted model (R ``summary.onam``)."""

    formula: str | None
    n_ensemble: int
    conv_metric: float
    conv_met_kind: str  # "cor" | "cor_p" | "auc"
    i_1: float
    i_2: float
    degree_expl: float
    target: str

    def __repr__(self) -> str:
        lines = []
        if self.formula:
            lines.append(f"Formula: {self.formula}")
        if self.conv_met_kind == "cor":
            lines.append(
                f"Correlation of model prediction with outcome variable: "
                f"{self.conv_metric:.4f}"
            )
        elif self.conv_met_kind == "cor_p":
            lines.append(
                f"Correlation of onam predictions with original model "
                f"predictions: {self.conv_metric:.4f}"
            )
        else:
            lines.append(f"Prediction AUC: {self.conv_metric:.4f}")
        lines.append(f"Number of ensemble members: {self.n_ensemble}")
        lines.append(f"I_1: {self.i_1:.4f}; I_2: {self.i_2:.4f}")
        lines.append(f"Degree of interpretability: {self.degree_expl:.4f}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
class ONAM:
    """Orthogonal Neural Additive Model.

    Parameters
    ----------
    terms:
        Effect specification -- a list of :class:`onam.Term` or
        ``(model_name, features)`` pairs.  ``features`` may be the single value
        ``"."`` for a residual term over all features.  Mutually exclusive with
        ``formula``.
    deep_models:
        Mapping ``name -> builder``, where ``builder(inputs) -> keras.Model``
        (e.g. an :class:`onam.DNN` instance).
    formula:
        Optional R-style formula string, e.g.
        ``"y ~ m1(x1) + m1(x2) + m1(x1, x2) + m2(.)"``.
    target:
        ``"continuous"`` (default) or ``"binary"``.
    categorical_features:
        Names of features to dummy-encode.
    """

    def __init__(
        self,
        terms: Sequence | None = None,
        deep_models: Mapping[str, Callable] | None = None,
        *,
        formula: str | None = None,
        target: str = "continuous",
        categorical_features: Sequence[str] | None = None,
    ) -> None:
        if (terms is None) == (formula is None):
            raise ValueError("Provide exactly one of `terms` or `formula`.")
        if deep_models is None:
            raise ValueError("`deep_models` is required.")

        self._formula = formula
        self._outcome_from_formula: str | None = None
        if formula is not None:
            self._outcome_from_formula, terms = parse_formula(formula)
        self._raw_terms = list(terms)
        self.deep_models = dict(deep_models)
        self.target = target
        self.categorical_features = list(categorical_features or [])

        # populated by fit()
        self.model_info: ModelInfo | None = None
        self.ensemble: list[EnsembleMember] = []
        self.categories_: dict[str, list] = {}
        self.feature_names_: list[str] = []
        self.data_: pd.DataFrame | None = None
        self.w_post_ensemble: np.ndarray | None = None
        self.feature_effects: pd.DataFrame | None = None
        self.predictions: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self._y_from_model = False

    # ------------------------------------------------------------------ #
    @classmethod
    def from_formula(
        cls,
        formula: str,
        deep_models: Mapping[str, Callable],
        *,
        target: str = "continuous",
        categorical_features: Sequence[str] | None = None,
    ) -> "ONAM":
        return cls(
            deep_models=deep_models,
            formula=formula,
            target=target,
            categorical_features=categorical_features,
        )

    # ------------------------------------------------------------------ #
    def fit(
        self,
        X,
        y=None,
        *,
        feature_names: Sequence[str] | None = None,
        model=None,
        prediction_function: Callable | None = None,
        model_data=None,
        n_ensemble: int = 10,
        epochs: int = 500,
        learning_rate: float = 1e-3,
        callbacks: list | None = None,
        seed: int | None = None,
        progress: bool = False,
        verbose: int = 0,
    ) -> "ONAM":
        """Fit the model.

        Parameters
        ----------
        X:
            Feature data -- a :class:`pandas.DataFrame` or a 2-D array-like
            (with ``feature_names``).  May also contain the outcome column,
            R-style.
        y:
            The training target, one of:

            * an array-like / Series of length ``n_samples`` (scikit-learn
              style: ``model.fit(X, y)``),
            * the name of a column in ``X`` holding the outcome,
            * ``None`` -- then the outcome column named in a ``formula`` must be
              present in ``X``, or ``model`` must be given.
        model, prediction_function, model_data:
            Explain an existing model: the target becomes
            ``prediction_function(model, model_data or X)`` (defaults to
            ``model.predict``).  ``y`` is ignored in this case.
        """
        frame = as_frame(X, feature_names)
        self.feature_names_ = list(frame.columns)

        y_name = y if isinstance(y, str) else None
        if y_name is None and self._outcome_from_formula is not None:
            y_name = self._outcome_from_formula

        model_info = ModelInfo.build(
            self._raw_terms,
            self.feature_names_,
            outcome=y_name,
            target=self.target,
            categorical_features=self.categorical_features,
            formula=self._formula,
        )
        check_fit_args(self.target, n_ensemble, epochs)
        check_terms(
            model_info.flat_terms,
            self.deep_models,
            self.feature_names_,
            self.categorical_features,
        )

        if seed is not None:
            keras.utils.set_random_seed(int(seed))

        self.categories_ = fit_categories(frame, self.categorical_features)
        cat_counts = category_counts(self.categories_)

        y_spec = y_name if isinstance(y, str) or y is None else y
        y = resolve_outcome(
            frame,
            y_spec,
            model=model,
            prediction_function=prediction_function,
            model_data=model_data,
            target=self.target,
        )
        self._y_from_model = model is not None
        check_y_features(frame, y, model_info, self.target)

        data_fit = prepare_data(frame, model_info, self.categories_)

        ensemble: list[EnsembleMember] = []
        for i in range(int(n_ensemble)):
            if progress:
                print(f"\rFitting model {i + 1} of {n_ensemble}",
                      end="", file=sys.stderr, flush=True)
            built = build_model(
                model_info, self.deep_models, cat_counts,
                learning_rate=learning_rate,
            )
            built.model.fit(
                data_fit, y, epochs=int(epochs),
                callbacks=callbacks, verbose=verbose,
            )
            ensemble.append(pho(built.submodels, model_info, data_fit))
        if progress:
            print("", file=sys.stderr)

        self.model_info = model_info
        self.ensemble = ensemble
        self.data_ = frame
        self.y = y

        u_ens = ensemble_effects(ensemble, model_info, self.categories_, frame)
        w_post, effects = pho_ensemble(u_ens, model_info)
        self.w_post_ensemble = w_post
        self.feature_effects = pd.DataFrame(
            effects, columns=model_info.effect_names, index=frame.index
        )
        self.predictions = effects.sum(axis=1)
        return self

    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if self.model_info is None:
            raise RuntimeError("Model is not fitted yet; call `.fit()` first.")

    def predict(self, newdata=None) -> ONAMPrediction:
        self._check_fitted()
        if newdata is None:
            frame = self.data_
        else:
            frame = as_frame(newdata, self.feature_names_)

        u_ens = ensemble_effects(
            self.ensemble, self.model_info, self.categories_, frame
        )
        effects = u_ens @ self.w_post_ensemble
        fe = pd.DataFrame(
            effects, columns=self.model_info.effect_names, index=frame.index
        )
        return ONAMPrediction(
            data=frame,
            predictions=effects.sum(axis=1),
            feature_effects=fe,
            model_info=self.model_info,
        )

    # ------------------------------------------------------------------ #
    def decompose(self, data=None) -> VarDecomp:
        self._check_fitted()
        return decompose(self, data)

    def gen_sobol(self, data=None) -> GenSobol:
        self._check_fitted()
        return gen_sobol(self, data)

    def summary(self) -> Summary:
        self._check_fitted()
        vd = decompose(self)
        preds = np.asarray(self.predictions, dtype=np.float64)
        y = np.asarray(self.y, dtype=np.float64)

        if self.target == "continuous":
            metric = float(np.corrcoef(preds, y)[0, 1])
            kind = "cor"
        else:
            is_labels = np.all(np.isin(np.unique(y), [0.0, 1.0]))
            if is_labels and not self._y_from_model:
                metric = _auc(y, preds)
                kind = "auc"
            else:
                metric = float(np.corrcoef(preds, y)[0, 1])
                kind = "cor_p"

        i_1 = float(vd.var_decomp.get(1, 0.0))
        i_2 = float(vd.var_decomp.get(2, 0.0))
        return Summary(
            formula=self._formula,
            n_ensemble=len(self.ensemble),
            conv_metric=metric,
            conv_met_kind=kind,
            i_1=i_1,
            i_2=i_2,
            degree_expl=i_1 + i_2,
            target=self.target,
        )

    # ------------------------------------------------------------------ #
    def save(self, path, overwrite: bool = False) -> None:
        from .persistence import save_onam

        save_onam(self, path, overwrite=overwrite)

    @classmethod
    def load(cls, path) -> "ONAM":
        from .persistence import load_onam

        return load_onam(path)

    def __repr__(self) -> str:
        if self.model_info is None:
            return f"ONAM(target={self.target!r}, <unfitted>)"
        return (
            f"ONAM(target={self.target!r}, n_ensemble={len(self.ensemble)}, "
            f"effects={self.model_info.effect_names})"
        )
