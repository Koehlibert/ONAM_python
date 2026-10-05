import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sim_data():
    rng = np.random.default_rng(42)
    n = 120
    x1 = rng.uniform(-2, 2, n)
    x2 = rng.uniform(-2, 2, n)
    x3 = rng.uniform(-2, 2, n)
    x4 = rng.uniform(-2, 2, n)
    y = (
        np.sin(x1)
        + np.where(x2 > 0, x2 ** 2, -x2)
        + 4 * x3 / (1 + x3 ** 2)
        + x1 * x2
        + rng.normal(0, 0.3, n)
    )
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y})


@pytest.fixture
def deep_models():
    from onam import DNN

    return {"mod1": DNN([16, 8, 1])}


@pytest.fixture
def terms():
    return [
        ("mod1", ["x1"]),
        ("mod1", ["x2"]),
        ("mod1", ["x3"]),
        ("mod1", ["x1", "x2"]),
        ("mod1", ["x1", "x2", "x3"]),
    ]
