"""onam -- Orthogonal Neural Additive Models.

Python port of the R package ``ONAM`` (Köhler et al. 2025,
doi:10.1038/s44387-025-00033-7).  Fit interpretable additive neural networks
with identifiable, visualizable feature effects via post-hoc orthogonalization,
including interaction effects of arbitrary order, and decompose the prediction
function into explainable predictor effects.

Quick start
-----------
>>> import numpy as np, pandas as pd
>>> from onam import ONAM, DNN
>>> rng = np.random.default_rng(0)
>>> n = 500
>>> x1, x2 = rng.uniform(-2, 2, n), rng.uniform(-2, 2, n)
>>> y = np.sin(x1) + 2 * x2 + x1 * x2 + rng.normal(0, 0.1, n)
>>> df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
>>> model = ONAM(
...     terms=[("m", ["x1"]), ("m", ["x2"]), ("m", ["x1", "x2"])],
...     deep_models={"m": DNN([16, 8, 1])},
... ).fit(df, "y", n_ensemble=2, epochs=50, seed=1)
>>> model.summary()                                  # doctest: +SKIP
>>> model.decompose()                                # doctest: +SKIP
"""

from __future__ import annotations

from .core import ONAM, ONAMPrediction, Summary
from .decomposition import GenSobol, VarDecomp, decompose, gen_sobol
from .dnn import DNN
from .plotting import plot_inter_effect, plot_main_effect
from .terms import RESIDUAL, ModelInfo, Term, parse_formula

__version__ = "1.1.0"

__all__ = [
    "ONAM",
    "ONAMPrediction",
    "Summary",
    "DNN",
    "Term",
    "ModelInfo",
    "RESIDUAL",
    "parse_formula",
    "decompose",
    "gen_sobol",
    "VarDecomp",
    "GenSobol",
    "plot_main_effect",
    "plot_inter_effect",
]
