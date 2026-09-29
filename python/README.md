# onam — Orthogonal Neural Additive Models (Python)

A Python port of the R package [**ONAM**](https://github.com/Koehlibert/ONAM_R).
Fit interpretable additive neural networks with identifiable, visualizable
feature effects via **post-hoc orthogonalization**, including interaction effects
of arbitrary order, and decompose a prediction function into explainable
predictor effects.

> Köhler et al. (2025), *Orthogonal neural additive models*,
> <https://doi.org/10.1038/s44387-025-00033-7>

## Installation

```bash
pip install -e python/            # from this repo
# choose a Keras 3 backend (TensorFlow is the default/reference):
pip install "onam[tensorflow]"
```

Requires Python ≥ 3.9, `keras>=3`, plus one Keras backend
(`tensorflow`, `torch` or `jax`). The orthogonalization math runs in NumPy/SciPy
regardless of backend.

## Quick start

```python
import numpy as np, pandas as pd
from onam import ONAM, DNN

rng = np.random.default_rng(0)
n = 1000
x1, x2, x3 = (rng.uniform(-2, 2, n) for _ in range(3))
y = np.sin(x1) + 4 / (1 + x2**2) + 2 * x3 + x1 * x2 + rng.normal(0, 1, n)
train = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})

deep_models = {
    "simple":  DNN([128, 64, 32, 16, 8, 1]),
    "complex": DNN([512, 256, 128, 64, 32, 16, 8, 1]),
}

model = ONAM(
    terms=[
        ("simple", ["x1"]),
        ("simple", ["x2"]),
        ("simple", ["x3"]),
        ("complex", ["x1", "x2"]),
        ("complex", ["x1", "x2", "x3"]),   # highest-order / residual bucket
    ],
    deep_models=deep_models,
    target="continuous",                   # or "binary"
).fit(train, "y", n_ensemble=2, epochs=100, seed=1)   # "y" = outcome column

print(model.summary())
print(model.decompose())
print(model.gen_sobol())
```

### Passing the target

`fit` accepts the outcome three ways:

```python
model.fit(train, "y")                 # name of a column in the data (R-style)
model.fit(X, y)                       # scikit-learn style: features + target array/Series
model.fit(train)                      # y omitted -> use the formula's outcome column
```

`X` may be a DataFrame or a 2-D array (add `feature_names=[...]` for arrays).

An R-style string formula is also accepted:

```python
model = ONAM.from_formula(
    "y ~ simple(x1) + simple(x2) + simple(x3) + complex(x1, x2) + complex(.)",
    deep_models,
)
model.fit(train, n_ensemble=2, epochs=100)   # outcome "y" taken from the formula
```

`complex(.)` (or `("complex", ".")`) is the **residual term** over every feature
except the outcome.

## Explaining an existing (black-box) model

```python
model = ONAM(terms=..., deep_models=...).fit(
    X,                            # features only
    model=xgb_model,              # anything with .predict, or pass prediction_function
    prediction_function=lambda m, d: m.predict(d),
    model_data=X_for_model,       # optional
    n_ensemble=10, epochs=500,
)
```

## Visualization

```python
from onam import plot_main_effect, plot_inter_effect
import matplotlib.pyplot as plt

plot_main_effect(model, "x1")
plot_inter_effect(model, "x1", "x2", interpolate=True)
model.decompose().plot()
model.gen_sobol().plot()
plt.show()
```

Categorical features are dummy-encoded (first level = reference); pass their
names in `categorical_features=[...]` and use `reference_level=` in
`plot_main_effect`.

## Persistence

```python
model.save("my_model", overwrite=True)     # dir of .keras files + metadata
model = ONAM.load("my_model")
```

## Custom sub-networks

`deep_models` values are callables `inputs -> keras.Model`. Use `DNN` for dense
stacks, or write your own:

```python
import keras

def resnet_block(inputs):
    h = keras.layers.Dense(64, activation="relu")(inputs)
    h = keras.layers.Dense(64, activation="relu")(h)
    out = keras.layers.Dense(1, activation="linear")(h)
    return keras.Model(inputs, out)

deep_models = {"res": resnet_block, "lin": DNN.linear()}
```

The final layer must have a single linear output unit (`DNN` enforces this with
a warning, matching the R `build_dnn`).

## API map (R → Python)

| R (`ONAM`)                       | Python (`onam`)                                   |
|---------------------------------|---------------------------------------------------|
| `onam(formula, list_of_deep_models, data, ...)` | `ONAM(terms=, deep_models=, ...).fit(X, y, ...)` or `ONAM.from_formula(...)` |
| `build_dnn(units=, layer_df=)`   | `DNN(units=, layers=)`                             |
| `predict(mod, newdata)`          | `model.predict(newdata)` → `ONAMPrediction`        |
| `summary(mod)`                   | `model.summary()` → `Summary`                      |
| `decompose(mod, data)`           | `model.decompose(data)` → `VarDecomp` (`.plot()`)  |
| `gen_sobol(mod, data)`           | `model.gen_sobol(data)` → `GenSobol` (`.plot()`)   |
| `plot_main_effect(mod, "x1")`    | `plot_main_effect(model, "x1")`                    |
| `plot_inter_effect(mod, "x1", "x2")` | `plot_inter_effect(model, "x1", "x2")`         |
| `save_onam(mod, dir)` / `load_onam(dir)` | `model.save(dir)` / `ONAM.load(dir)`       |
| `install_conda_env()`            | not needed — install a Keras backend with pip     |

## How it works

1. **Fit** an additive model: one sub-network per (interaction) term, summed
   (then a sigmoid for `target="binary"`), trained by Adam on MSE / BCE.
2. **Orthogonalize each ensemble member** (`pho`): project every interaction
   order onto the span of all lower orders (penultimate-layer features + an
   intercept) via pivoted-QR least squares and move the shared component down,
   then mean-center every effect but the first.
3. **Ensemble-orthogonalize** (`pho_ensemble`): repeat the projection on the
   ensemble-averaged effects.
4. The resulting `feature_effects` are identifiable and sum to the prediction,
   enabling the variance decomposition and generalized Sobol indices.

## Tests

```bash
cd python && pip install -e ".[test,tensorflow]" && pytest
```
