# onam — Orthogonal Neural Additive Models (Python)

A Python port of the R package [**ONAM**](https://github.com/Koehlibert/ONAM_R).
Fit interpretable additive neural networks with identifiable, visualizable
feature effects via **post-hoc orthogonalization**, including interaction effects
of arbitrary order, and decompose a prediction function into explainable
predictor effects.

> Köhler et al. (2025), *Achieving interpretable machine learning by functional
> decomposition of black-box models into explainable predictor effects*,
> npj Artificial Intelligence, <https://doi.org/10.1038/s44387-025-00033-7>

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
model.fit(train)                      # y omitted -> use the formula's outcome column (Common R syntax)
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

`complex(.)` (or `("complex", ".")`) is a term including every feature
except the outcome.

## Explaining an existing (black-box) model

```python
model = ONAM(terms=..., deep_models=...).fit(
    X,                            # features only
    model=black_box_model,              # anything with .predict, or pass prediction_function
    prediction_function=lambda m, d: m.predict(d),
    model_data=X_for_model,       # optional, used by model.predict or prediction_function
    n_ensemble=4, epochs=500,
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
a warning).

## How it works

1. **Fit** an additive model: one sub-network per (interaction) term, summed
   (then a sigmoid for `target="binary"`), trained by Adam on MSE / BCE.
2. **Orthogonalize each ensemble member**: project every interaction
   order onto the span of all lower order terms via pivoted-QR least squares and 
   move the shared component down, then mean-center every effect but the first.
3. **Ensemble-orthogonalize** (`pho_ensemble`): repeat the projection on the
   ensemble-averaged effects.
4. The resulting `feature_effects` are identifiable and sum to the prediction,
   enabling the variance decomposition and generalized Sobol indices.

## Tests

```bash
cd python && pip install -e ".[test,tensorflow]" && pytest
```
