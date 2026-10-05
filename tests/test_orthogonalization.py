"""Port of tests/testthat/test-utils.R (orthogonalization + seed)."""

import numpy as np

from onam import ONAM


def test_effects_are_mean_centered_and_orthogonal(sim_data, deep_models, terms):
    model = ONAM(terms=terms, deep_models=deep_models).fit(
        sim_data, "y", n_ensemble=1, epochs=12, seed=1
    )
    eff = model.feature_effects
    # every effect except the first (which absorbs the intercept) is centered
    means = eff.iloc[:, 1:].mean().to_numpy()
    np.testing.assert_allclose(means, 0.0, atol=1e-8)

    # main effects are orthogonal to the interaction bucket and to x1_x2
    cov = np.cov(eff.to_numpy(), rowvar=False)
    names = list(eff.columns)
    inter = names.index("Interaction 3")
    x1x2 = names.index("x1_x2")
    mains = [names.index(c) for c in ("x1", "x2", "x3")]
    for m in mains:
        assert abs(cov[m, inter]) < 1e-6
        assert abs(cov[m, x1x2]) < 1e-6
    assert abs(cov[inter, x1x2]) < 1e-6


def test_seed_is_reproducible(sim_data, deep_models):
    spec = dict(
        terms=[("mod1", ["x1"]), ("mod1", ["x2"]), ("mod1", ["x1", "x2"])],
        deep_models=deep_models,
    )
    a = ONAM(**spec).fit(sim_data, "y", n_ensemble=1, epochs=10, seed=1)
    b = ONAM(**spec).fit(sim_data, "y", n_ensemble=1, epochs=10, seed=1)
    np.testing.assert_allclose(
        a.feature_effects["x1"].to_numpy(),
        b.feature_effects["x1"].to_numpy(),
    )


def test_predictions_equal_row_sum_of_effects(sim_data, deep_models, terms):
    model = ONAM(terms=terms, deep_models=deep_models).fit(
        sim_data, "y", n_ensemble=2, epochs=8, seed=1
    )
    np.testing.assert_allclose(
        model.predictions,
        model.feature_effects.to_numpy().sum(axis=1),
        atol=1e-8,
    )
