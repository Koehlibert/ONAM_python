"""Explain a black-box model with ONAM (the walkthrough from the README).

    python examples/black_box.py

Fits a gradient boosting model to simulated data, then decomposes its
prediction function into main effects, two-way interactions and a residual.
Requires scikit-learn in addition to onam.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from onam import DNN, ONAM, plot_inter_effect, plot_main_effect

FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Data ------------------------------------------------------------------- #
rng = np.random.default_rng(1)
n = 3000
X = pd.DataFrame({f"x{i}": rng.uniform(-2, 2, n) for i in (1, 2, 3)})
y = (
    np.sin(2 * X["x1"])
    + 4 / (1 + X["x2"] ** 2)
    + X["x3"]
    + X["x1"] * X["x2"]
    + rng.normal(0, 0.5, n)
)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# 2. Black box -------------------------------------------------------------- #
black_box = HistGradientBoostingRegressor(random_state=1).fit(X_train, y_train)
print(f"Black box test R^2: {r2_score(y_test, black_box.predict(X_test)):.3f}")

# 3. ONAM ------------------------------------------------------------------- #
deep_models = {
    "main": DNN([64, 32, 16, 1]),
    "inter": DNN([128, 64, 32, 1]),
}

model = ONAM(
    terms=[
        ("main", ["x1"]),
        ("main", ["x2"]),
        ("main", ["x3"]),
        ("inter", ["x1", "x2"]),
        ("inter", ["x1", "x3"]),
        ("inter", ["x2", "x3"]),
        ("inter", "."),  # residual: whatever orders 1 and 2 cannot capture
    ],
    deep_models=deep_models,
).fit(X_train, model=black_box, n_ensemble=5, epochs=200, seed=1, progress=True)

# 4. How much of the black box is interpretable? ---------------------------- #
print(model.summary())
print()
print(model.decompose())
print()
print(model.gen_sobol())

# 5. Effects ---------------------------------------------------------------- #
truth = {
    "x1": lambda x: np.sin(2 * x),
    "x2": lambda x: 4 / (1 + x**2),
    "x3": lambda x: x,
}
grid = np.linspace(-2, 2, 200)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, (feature, f) in zip(axes, truth.items()):
    plot_main_effect(model, feature, ax=ax)
    # effects are mean-centred, so centre the data-generating function too
    ax.plot(grid, f(grid) - f(X_train[feature]).mean(), color="C3", lw=2,
            label="data-generating function")
axes[-1].legend(loc="upper left")
fig.tight_layout()
fig.savefig(FIG_DIR / "main_effects.png", dpi=120)

fig, ax = plt.subplots(figsize=(5.5, 4.2))
plot_inter_effect(model, "x1", "x2", interpolate=True, ax=ax)
fig.tight_layout()
fig.savefig(FIG_DIR / "interaction_x1_x2.png", dpi=120)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
model.decompose().plot(ax=axes[0])
model.gen_sobol().plot(ax=axes[1])
axes[1].set_ylim(top=axes[1].get_ylim()[1] * 1.3)  # headroom for the legend
fig.tight_layout()
fig.savefig(FIG_DIR / "decomposition.png", dpi=120)

# 6. New data --------------------------------------------------------------- #
pred = model.predict(X_test)
print()
print(pred.feature_effects.head().round(3))
print(
    "\nFidelity to the black box on held-out data (R^2): "
    f"{r2_score(black_box.predict(X_test), pred.predictions):.3f}"
)
print(f"\nwrote figures to {FIG_DIR}")
