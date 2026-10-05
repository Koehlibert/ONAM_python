"""Reproduces the README workflow from the R package on simulated data.

    python examples/simulation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from onam import DNN, ONAM, plot_inter_effect, plot_main_effect

rng = np.random.default_rng(1)
n = 1000
x1 = rng.uniform(-2, 2, n)
x2 = rng.uniform(-2, 2, n)
x3 = rng.uniform(-2, 2, n)
noise = rng.normal(0, 1, n)
y = np.sin(x1) + 4 / (1 + x2 ** 2) + 2 * x3 + x1 * x2 + noise
train = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})

deep_models = {
    "simple": DNN([128, 64, 32, 16, 8, 1]),
    "complex": DNN([256, 128, 64, 32, 16, 8, 1]),
}

model = ONAM(
    terms=[
        ("simple", ["x1"]),
        ("simple", ["x2"]),
        ("simple", ["x3"]),
        ("complex", ["x1", "x2"]),
        ("complex", ["x1", "x2", "x3"]),
    ],
    deep_models=deep_models,
).fit(train, "y", n_ensemble=2, epochs=100, seed=1, progress=True)

print(model.summary())
print()
print(model.decompose())
print()
print(model.gen_sobol())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
plot_main_effect(model, "x1", ax=axes[0])
plot_main_effect(model, "x2", ax=axes[1])
plot_inter_effect(model, "x1", "x2", interpolate=True, ax=axes[2])
fig.tight_layout()
fig.savefig("onam_effects.png", dpi=120)
print("\nwrote onam_effects.png")
