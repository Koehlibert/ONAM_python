"""Port of tests/testthat/test-persistence.R."""

import numpy as np
import pytest

from onam import DNN, ONAM


def test_save_load_roundtrip(tmp_path, sim_data, terms):
    model = ONAM(
        terms=terms, deep_models={"mod1": DNN([16, 8, 1])}
    ).fit(sim_data, "y", n_ensemble=1, epochs=10, seed=1)

    out = tmp_path / "mod"
    model.save(out)
    reloaded = ONAM.load(out)

    a = model.predict(sim_data.drop(columns="y"))
    b = reloaded.predict(sim_data.drop(columns="y"))
    np.testing.assert_allclose(a.predictions, b.predictions, atol=1e-6)
    np.testing.assert_allclose(
        a.feature_effects.to_numpy(), b.feature_effects.to_numpy(), atol=1e-6
    )


def test_save_refuses_existing_dir(tmp_path, sim_data, terms):
    model = ONAM(
        terms=terms, deep_models={"mod1": DNN([16, 8, 1])}
    ).fit(sim_data, "y", n_ensemble=1, epochs=5, seed=1)
    out = tmp_path / "mod"
    model.save(out)
    with pytest.raises(FileExistsError):
        model.save(out)
    model.save(out, overwrite=True)  # ok
