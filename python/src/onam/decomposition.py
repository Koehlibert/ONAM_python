"""Functional-decomposition diagnostics: variance decomposition & Sobol indices.

Port of ``var_decomposition.R`` (``decompose``, ``gen_sobol`` and their
``print`` / ``plot`` methods).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["VarDecomp", "GenSobol", "decompose", "gen_sobol"]


def _cov(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape[1] == 1:
        return np.array([[np.var(mat[:, 0], ddof=1)]])
    return np.cov(mat, rowvar=False)


# --------------------------------------------------------------------------- #
@dataclass
class VarDecomp:
    """Fraction of total effect variance carried by each interaction order."""

    var_decomp: "pd.Series"       # index = interaction order (int), values sum to 1
    sens_info: dict[int, int]     # number of effects per order
    target: str

    def __getitem__(self, order):
        return self.var_decomp.loc[order]

    def __repr__(self) -> str:
        scale = " (on logit level)" if self.target == "binary" else ""
        lines = [f"Fraction of total variance explained per order{scale}:"]
        for order, val in self.var_decomp.items():
            if order == 1:
                lines.append(f"  Main effects: {val:.4f}")
            else:
                lines.append(f"  Interactions of order {order}: {val:.4f}")
        return "\n".join(lines)

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        labels = [
            "Main effects" if o == 1 else f"Order {o} interactions"
            for o in self.var_decomp.index
        ]
        vals = self.var_decomp.to_numpy()
        ax.bar(labels, vals, color="#4c4c4c")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v * 100:.1f}%", ha="center", va="bottom")
        ylab = "Fraction of explained variance"
        if self.target == "binary":
            ylab += " (logit scale)"
        ax.set_ylabel(ylab)
        ax.set_ylim(0, max(vals) * 1.15 if len(vals) else 1)
        ax.margins(x=0.05)
        return ax


@dataclass
class GenSobol:
    """Generalized Sobol indices for (possibly dependent) effects."""

    table: "pd.DataFrame"  # index=effect name; cols: effect_order, gsi_1, gsi_2, total
    target: str

    @property
    def gen_sobol_index_1(self) -> "pd.Series":
        return self.table["gen_sobol_index_1"]

    @property
    def gen_sobol_index_2(self) -> "pd.Series":
        return self.table["gen_sobol_index_2"]

    @property
    def total_sobol_index(self) -> "pd.Series":
        return self.table["total_sobol_index"]

    def __repr__(self) -> str:
        scale = " (on logit level)" if self.target == "binary" else ""
        lines = [f"Generalized Sobol Indices{scale}:"]
        tbl = self.table.sort_values("effect_order")
        for order in sorted(tbl["effect_order"].unique()):
            if order == 1:
                lines.append("Main effects:")
            else:
                lines.append(f"Interactions of order {order}:")
            sub = tbl[tbl["effect_order"] == order]
            for name, row in sub.iterrows():
                lines.append(f"  {name}: {row['total_sobol_index']:.4f}")
        return "\n".join(lines)

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        tbl = self.table.sort_values(
            ["effect_order", "total_sobol_index"], ascending=[True, False]
        )
        names = list(tbl.index)
        ax.bar(names, tbl["gen_sobol_index_1"], label="Effect contribution",
               color="#3b6ea5")
        ax.bar(names, tbl["gen_sobol_index_2"],
               bottom=tbl["gen_sobol_index_1"],
               label="Effect induced by other terms of the same order",
               color="#a5c8e1")
        ylab = "Generalized Sobol Index"
        if self.target == "binary":
            ylab += " (logit scale)"
        ax.set_ylabel(ylab)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.legend(loc="upper center")
        return ax


# --------------------------------------------------------------------------- #
def decompose(obj, data=None) -> VarDecomp:
    """Variance decomposition of a fitted :class:`onam.ONAM` (or its prediction).

    ``obj`` must expose ``feature_effects`` (a DataFrame) and ``model_info``.
    A non-default ``data`` requires ``obj`` to be the fitted model.
    """
    model_info = obj.model_info
    if data is None:
        effects = obj.feature_effects
    else:
        effects = obj.predict(data).feature_effects

    max_order = model_info.max_order
    orders = sorted(set([1, 2, *model_info.orders]))

    cols = np.zeros((len(effects), len(orders)))
    sens_info: dict[int, int] = {}
    for j, order in enumerate(orders):
        names = model_info.effect_names_for_order(order)
        sens_info[order] = len(names)
        if names:
            cols[:, j] = effects.loc[:, names].to_numpy().sum(axis=1)

    tmp_var = _cov(cols)
    diag = np.diag(tmp_var)
    frac = diag / diag.sum()
    series = pd.Series(frac, index=orders, name="var_decomp")
    return VarDecomp(
        var_decomp=series,
        sens_info=dict(reversed(list(sens_info.items()))),
        target=model_info.target,
    )


def gen_sobol(obj, data=None) -> GenSobol:
    """Generalized Sobol indices (Chastaing et al. 2012)."""
    model_info = obj.model_info
    if data is None:
        effects = obj.feature_effects
    else:
        effects = obj.predict(data).feature_effects

    tmp_var = _cov(effects.to_numpy())
    total_var = tmp_var.sum()
    diag = np.diag(tmp_var)
    gsi_1 = diag / total_var
    gsi_2 = (tmp_var.sum(axis=1) - diag) / total_var
    total = gsi_1 + gsi_2

    effect_orders = [t.order for t in model_info.flat_terms]
    table = pd.DataFrame(
        {
            "effect_order": effect_orders,
            "gen_sobol_index_1": gsi_1,
            "gen_sobol_index_2": gsi_2,
            "total_sobol_index": total,
        },
        index=list(effects.columns),
    )
    return GenSobol(table=table, target=model_info.target)
