"""Port of tests/testthat/test-inputs.R."""

import pytest

from onam import ONAM, plot_inter_effect, plot_main_effect


def _fit(sim_data, deep_models, terms, **kw):
    return ONAM(terms=terms, deep_models=deep_models).fit(
        sim_data, "y", n_ensemble=1, epochs=8, seed=1, **kw
    )


def test_missing_feature_raises(sim_data, deep_models):
    terms = [("mod1", ["x1"]), ("mod1", ["x5"])]
    with pytest.raises(ValueError, match="x5"):
        _fit(sim_data, deep_models, terms)


def test_unknown_model_raises(sim_data, deep_models):
    terms = [("mod1", ["x1"]), ("mod4", ["x2"])]
    with pytest.raises(ValueError, match="mod4"):
        _fit(sim_data, deep_models, terms)


def test_categorical_missing(sim_data, deep_models, terms):
    with pytest.raises(ValueError, match="x5"):
        ONAM(
            terms=terms, deep_models=deep_models, categorical_features=["x5"]
        ).fit(sim_data, "y", n_ensemble=1, epochs=5, seed=1)


def test_lower_order_not_in_higher_warns(sim_data, deep_models):
    terms = [("mod1", ["x1"]), ("mod1", ["x2"]), ("mod1", ["x3"]),
             ("mod1", ["x1", "x2"])]
    with pytest.warns(UserWarning, match="lower order effects"):
        _fit(sim_data, deep_models, terms)


def test_categorical_not_in_terms_warns(sim_data, deep_models, terms):
    with pytest.warns(UserWarning, match="categorical"):
        ONAM(
            terms=terms, deep_models=deep_models, categorical_features=["x4"]
        ).fit(sim_data, "y", n_ensemble=1, epochs=5, seed=1)


def test_plot_unknown_effect_raises(sim_data, deep_models, terms):
    model = _fit(sim_data, deep_models, terms)
    with pytest.raises(ValueError, match="x4"):
        plot_main_effect(model, "x4")
    with pytest.raises(ValueError, match="[Nn]o interaction"):
        plot_inter_effect(model, "x1", "x3")


def test_plot_reference_level_on_numeric_warns(sim_data, deep_models, terms):
    model = _fit(sim_data, deep_models, terms)
    with pytest.warns(UserWarning, match="not modeled as categorical"):
        plot_main_effect(model, "x1", reference_level="1")
