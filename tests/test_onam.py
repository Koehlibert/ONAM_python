"""Port of tests/testthat/test-onam.R."""

import numpy as np
import pandas as pd
import pytest

from onam import DNN, ONAM


def test_onam_fits(sim_data, deep_models, terms):
    model = ONAM(terms=terms, deep_models=deep_models).fit(
        sim_data, "y", n_ensemble=1, epochs=10, seed=1
    )
    assert model.feature_effects.shape == (len(sim_data), 5)
    assert np.isfinite(model.predictions).all()


def test_onam_predict_differs_on_new_data(sim_data, deep_models, terms):
    model = ONAM(terms=terms, deep_models=deep_models).fit(
        sim_data, "y", n_ensemble=1, epochs=10, seed=1
    )
    rng = np.random.default_rng(7)
    newdata = pd.DataFrame(
        {c: rng.uniform(-2, 2, len(sim_data)) for c in ["x1", "x2", "x3", "x4"]}
    )
    new_pred = model.predict(newdata).predictions
    assert not np.allclose(new_pred, model.predictions)


def test_onam_predict_matches_training_data(sim_data, deep_models, terms):
    model = ONAM(terms=terms, deep_models=deep_models).fit(
        sim_data, "y", n_ensemble=2, epochs=8, seed=1
    )
    again = model.predict(sim_data.drop(columns="y"))
    np.testing.assert_allclose(again.predictions, model.predictions, atol=1e-6)
    np.testing.assert_allclose(
        again.feature_effects.to_numpy(),
        model.feature_effects.to_numpy(),
        atol=1e-6,
    )


def test_onam_explains_external_model(sim_data, deep_models):
    # A minimal "black box": OLS fit by hand, exposing only .predict.
    x = np.column_stack([np.ones(len(sim_data)), sim_data[["x1", "x2"]]])
    beta, *_ = np.linalg.lstsq(x, sim_data["y"].to_numpy(), rcond=None)

    class BlackBox:
        def predict(self, d):
            xd = np.column_stack([np.ones(len(d)), d[["x1", "x2"]]])
            return xd @ beta

    model = ONAM(
        terms=[("mod1", ["x1"]), ("mod1", ["x2"]), ("mod1", ["x1", "x2"])],
        deep_models=deep_models,
    ).fit(
        sim_data[["x1", "x2"]],
        model=BlackBox(),
        prediction_function=lambda m, d: m.predict(d),
        n_ensemble=1,
        epochs=10,
        seed=1,
    )
    assert model.feature_effects.shape[1] == 3
    assert model.summary().conv_met_kind == "cor"


def test_residual_term_warns_on_leaked_outcome(sim_data, deep_models):
    # y is left in the data and the residual term `.` sweeps it up as a
    # feature, while the "black box" target essentially reproduces it.
    df = sim_data.assign(y=2 * sim_data["x1"] - 2 * sim_data["x2"])

    class Echo:
        def predict(self, d):
            return d["y"].to_numpy()

    with pytest.warns(UserWarning, match="correlation"):
        ONAM(
            terms=[("mod1", ["x1"]), ("mod1", ".")],
            deep_models=deep_models,
        ).fit(
            df,
            y="response",  # not a column -> `.` keeps every column
            model=Echo(),
            prediction_function=lambda m, d: m.predict(d),
            n_ensemble=1,
            epochs=5,
            seed=1,
        )


def test_fit_sklearn_style_dataframe_and_series(sim_data, deep_models, terms):
    X = sim_data.drop(columns="y")
    y = sim_data["y"]
    model = ONAM(terms=terms, deep_models=deep_models).fit(
        X, y, n_ensemble=1, epochs=10, seed=1
    )
    assert model.feature_effects.shape == (len(sim_data), 5)


def test_fit_sklearn_style_numpy_arrays(sim_data, deep_models):
    X = sim_data[["x1", "x2"]].to_numpy()
    y = sim_data["y"].to_numpy()
    model = ONAM(
        terms=[("mod1", ["a"]), ("mod1", ["b"]), ("mod1", ["a", "b"])],
        deep_models=deep_models,
    ).fit(X, y, feature_names=["a", "b"], n_ensemble=1, epochs=10, seed=1)
    assert list(model.feature_effects.columns) == ["Interaction 2", "a", "b"]


def test_fit_column_name_and_array_agree(sim_data, deep_models, terms):
    by_name = ONAM(terms=terms, deep_models=deep_models).fit(
        sim_data, "y", n_ensemble=1, epochs=10, seed=1
    )
    by_array = ONAM(terms=terms, deep_models=deep_models).fit(
        sim_data.drop(columns="y"), sim_data["y"].to_numpy(),
        n_ensemble=1, epochs=10, seed=1,
    )
    np.testing.assert_allclose(
        by_name.feature_effects.to_numpy(),
        by_array.feature_effects.to_numpy(),
        atol=1e-6,
    )


def test_fit_without_target_raises(sim_data, deep_models, terms):
    with pytest.raises(ValueError, match="No target"):
        ONAM(terms=terms, deep_models=deep_models).fit(
            sim_data.drop(columns="y"), n_ensemble=1, epochs=5
        )


def test_binary_outcome(sim_data, deep_models):
    rng = np.random.default_rng(3)
    eta = sim_data["x1"] + sim_data["x2"]
    y = (rng.uniform(size=len(sim_data)) < 1 / (1 + np.exp(-eta))).astype(int)
    df = sim_data.assign(y=y)

    model = ONAM(
        terms=[("mod1", ["x1"]), ("mod1", ["x2"]), ("mod1", ["x1", "x2"])],
        deep_models=deep_models,
        target="binary",
    ).fit(df, "y", n_ensemble=1, epochs=10, seed=1)

    s = model.summary()
    assert s.conv_met_kind == "auc"
    assert 0.0 <= s.conv_metric <= 1.0
