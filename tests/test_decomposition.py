"""Variance decomposition and generalized Sobol indices."""

import numpy as np

from onam import ONAM, decompose, gen_sobol


def test_var_decomp_sums_to_one(sim_data, deep_models, terms):
    model = ONAM(terms=terms, deep_models=deep_models).fit(
        sim_data, "y", n_ensemble=1, epochs=10, seed=1
    )
    vd = decompose(model)
    assert np.isclose(vd.var_decomp.sum(), 1.0)
    assert set(vd.var_decomp.index) == {1, 2, 3}
    assert 0 <= vd[1] <= 1


def test_gen_sobol_shape_and_orders(sim_data, deep_models, terms):
    model = ONAM(terms=terms, deep_models=deep_models).fit(
        sim_data, "y", n_ensemble=1, epochs=10, seed=1
    )
    gs = gen_sobol(model)
    assert list(gs.table.index) == model.model_info.effect_names
    assert gs.table.loc["x1", "effect_order"] == 1
    assert gs.table.loc["Interaction 3", "effect_order"] == 3
    np.testing.assert_allclose(
        gs.table["total_sobol_index"],
        gs.table["gen_sobol_index_1"] + gs.table["gen_sobol_index_2"],
    )


def test_decompose_on_new_data(sim_data, deep_models, terms):
    model = ONAM(terms=terms, deep_models=deep_models).fit(
        sim_data, "y", n_ensemble=1, epochs=8, seed=1
    )
    vd = model.decompose(data=sim_data.drop(columns="y").iloc[:50])
    assert np.isclose(vd.var_decomp.sum(), 1.0)
