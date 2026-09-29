"""Port of the build_dnn checks in tests/testthat/test-utils.R."""

import keras
import pytest

from onam import DNN


def _dense_specs(model):
    return [
        {
            "units": l.units,
            "activation": keras.activations.serialize(l.activation),
            "use_bias": l.use_bias,
        }
        for l in model.layers
        if isinstance(l, keras.layers.Dense)
    ]


def _reference(inputs, units, use_bias, activations):
    x = inputs
    for u, b, a in zip(units, use_bias, activations):
        x = keras.layers.Dense(units=u, activation=a, use_bias=b)(x)
    return keras.Model(inputs, x)


def test_units_vector_matches_manual_stack():
    inp = keras.Input(shape=(10,))
    ref = _reference(inp, [16, 8, 1], [True, True, True],
                     ["relu", "relu", "linear"])
    built = DNN([16, 8, 1])(inp)
    assert _dense_specs(built) == _dense_specs(ref)
    assert built.count_params() == ref.count_params()
    assert built.output_shape == ref.output_shape


def test_layer_dicts_match_manual_stack():
    inp = keras.Input(shape=(10,))
    ref = _reference(
        inp, [32, 16, 8, 1], [False, False, False, True],
        ["relu", "relu", "relu", "linear"],
    )
    built = DNN(layers=[
        {"units": 32, "activation": "relu", "use_bias": False},
        {"units": 16, "activation": "relu", "use_bias": False},
        {"units": 8, "activation": "relu", "use_bias": False},
        {"units": 1, "activation": "linear", "use_bias": True},
    ])(inp)
    assert _dense_specs(built) == _dense_specs(ref)
    assert built.count_params() == ref.count_params()


def test_non_unit_last_layer_is_repaired_with_warning():
    with pytest.warns(UserWarning, match="1 output unit"):
        dnn = DNN([16, 8])
    assert dnn.specs[-1]["units"] == 1


def test_non_linear_last_activation_is_repaired_with_warning():
    with pytest.warns(UserWarning, match="linear activation"):
        dnn = DNN(layers=[{"units": 8}, {"units": 1, "activation": "relu"}])
    assert dnn.specs[-1]["activation"] == "linear"


def test_requires_one_of_units_or_layers():
    with pytest.raises(ValueError):
        DNN()
    with pytest.raises(ValueError):
        DNN(units=[8, 1], layers=[{"units": 1}])
