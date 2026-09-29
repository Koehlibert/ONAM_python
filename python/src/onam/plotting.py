"""Effect visualization with matplotlib (port of ``visualization.R``).

``plot_main_effect`` and ``plot_inter_effect`` accept either a fitted
:class:`onam.ONAM` or an :class:`onam.ONAMPrediction`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

__all__ = ["plot_main_effect", "plot_inter_effect"]

_SPECTRAL = [
    "#9e0142", "#d53e4f", "#f46d43", "#fdae61", "#fee08b", "#ffffbf",
    "#e6f598", "#abdda4", "#66c2a5", "#3288bd", "#5e4fa2",
]


def _spectral_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("spectral", _SPECTRAL[::-1], N=256)


def _unpack(obj):
    """Return ``(data_frame, feature_effects_df, model_info)``."""
    effects = obj.feature_effects
    model_info = obj.model_info
    data = getattr(obj, "data_", None)
    if data is None:
        data = getattr(obj, "data")
    return data, effects, model_info


def _eff_label(target: str) -> str:
    return "Effect" if target == "continuous" else "Effect on logit scale"


# --------------------------------------------------------------------------- #
def plot_main_effect(obj, feature, reference_level=None, labs=None, ax=None):
    import matplotlib.pyplot as plt

    data, effects, model_info = _unpack(obj)
    labs = labs or {}

    if feature not in effects.columns:
        raise ValueError(
            f"{feature} is not present in the fitted model effects."
        )
    is_cat = feature in model_info.categorical_features
    if reference_level is not None and not is_cat:
        warnings.warn(
            f"Reference level given, but {feature} is not modeled as "
            f"categorical.",
            stacklevel=2,
        )

    x = np.asarray(data[feature])
    y = np.asarray(effects[feature])

    if ax is None:
        _, ax = plt.subplots()

    if is_cat:
        levels, first_idx = np.unique(x, return_index=True)
        vals = y[first_idx]
        labels = [str(lvl) for lvl in levels]
        if reference_level is not None:
            if reference_level not in set(x.tolist()):
                raise ValueError(
                    f"{reference_level} is not a level of feature {feature}."
                )
            ref_pos = labels.index(str(reference_level))
            vals = vals - vals[ref_pos]
            labels[ref_pos] = f"{reference_level} (Reference)"
        ax.bar(labels, vals, color="#4c4c4c")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
    else:
        ax.scatter(x, y, s=12, color="#1f1f1f")

    ax.set_xlabel(labs.get("xlab", feature))
    ax.set_ylabel(
        labs.get("effect", labs.get("ylab", _eff_label(model_info.target)))
    )
    return ax


# --------------------------------------------------------------------------- #
def _interaction_column(effects, feature1, feature2):
    for a, b in ((feature1, feature2), (feature2, feature1)):
        name = f"{a}_{b}"
        if name in effects.columns:
            return name, a, b
    return None, feature1, feature2


def plot_inter_effect(
    obj,
    feature1,
    feature2,
    interpolate=False,
    labs=None,
    n_interpolate=200,
    include_main=False,
    cmap=None,
    ax=None,
):
    import matplotlib.pyplot as plt

    data, effects, model_info = _unpack(obj)
    labs = labs or {}
    cat = model_info.categorical_features
    n_cat = sum(f in cat for f in (feature1, feature2))
    if n_cat and interpolate:
        warnings.warn(
            "Interaction contains categorical feature. No interpolation will "
            "be performed.",
            stacklevel=2,
        )
        interpolate = False

    name, feature1, feature2 = _interaction_column(effects, feature1, feature2)
    if name is None:
        raise ValueError(
            f"No interaction effect fitted for {feature1} and {feature2}."
        )

    eff = np.asarray(effects[name], dtype=float)
    if include_main:
        for f in (feature1, feature2):
            if f in effects.columns:
                eff = eff + np.asarray(effects[f], dtype=float)
            else:
                warnings.warn(
                    "`include_main=True`, but not all features are fitted as "
                    "main effects.",
                    stacklevel=2,
                )

    x = np.asarray(data[feature1])
    y = np.asarray(data[feature2])
    cmap = cmap or _spectral_cmap()
    effect_label = labs.get("effect", _eff_label(model_info.target))

    if ax is None:
        _, ax = plt.subplots()

    if interpolate and n_cat == 0:
        from scipy.interpolate import griddata

        xf = x.astype(float)
        yf = y.astype(float)
        xi = np.linspace(xf.min(), xf.max(), n_interpolate)
        yi = np.linspace(yf.min(), yf.max(), n_interpolate)
        gx, gy = np.meshgrid(xi, yi)
        gz = griddata((xf, yf), eff, (gx, gy), method="linear")
        mesh = ax.pcolormesh(gx, gy, gz, cmap=cmap, shading="auto")
        plt.colorbar(mesh, ax=ax, label=effect_label)
    elif n_cat == 1:
        df = pd.DataFrame({"x": x, "y": y, "eff": eff})
        if feature1 in cat:
            grouped = df.groupby(["x", "y"], as_index=False)["eff"].mean()
            for lvl, sub in grouped.groupby("x"):
                ax.plot(sub["y"], sub["eff"], marker="o", label=str(lvl))
            ax.set_xlabel(labs.get("ylab", feature2))
            ax.legend(title=feature1)
        else:
            grouped = df.groupby(["x", "y"], as_index=False)["eff"].mean()
            for lvl, sub in grouped.groupby("y"):
                ax.plot(sub["x"], sub["eff"], marker="o", label=str(lvl))
            ax.set_xlabel(labs.get("xlab", feature1))
            ax.legend(title=feature2)
        ax.set_ylabel(effect_label)
        return ax
    else:
        sc = ax.scatter(x, y, c=eff, cmap=cmap, s=14)
        plt.colorbar(sc, ax=ax, label=effect_label)

    ax.set_xlabel(labs.get("xlab", feature1))
    ax.set_ylabel(labs.get("ylab", feature2))
    return ax
