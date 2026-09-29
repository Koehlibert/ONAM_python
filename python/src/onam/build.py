"""Assemble the trainable additive Keras model from the effect structure.

Python counterpart of ``create_model`` / ``create_inputs`` / ``create_models``
/ ``compile_model`` in ``model_setup.R``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import keras

from .terms import ModelInfo

__all__ = ["OnamKerasModel", "build_model"]


@dataclass
class OnamKerasModel:
    model: keras.Model
    submodels: list[keras.Model]  # one per flat term, same order


def _target_activation(target: str) -> str:
    return "linear" if target == "continuous" else "sigmoid"


def build_model(
    model_info: ModelInfo,
    deep_models: Mapping[str, Callable],
    cat_counts: Mapping[str, int],
    *,
    learning_rate: float = 1e-3,
) -> OnamKerasModel:
    """Create and compile the additive ensemble member."""
    inputs: list[keras.KerasTensor] = []
    submodels: list[keras.Model] = []

    for term in model_info.flat_terms:
        n_in = model_info.n_inputs(term, dict(cat_counts))
        inp = keras.Input(shape=(n_in,))
        builder = deep_models[term.model]
        submodels.append(builder(inp))
        inputs.append(inp)

    outputs = [m.output for m in submodels]
    summed = keras.layers.Add()(outputs) if len(outputs) > 1 else outputs[0]

    if model_info.target == "continuous":
        final = summed
        loss = keras.losses.MeanSquaredError()
    elif model_info.target == "binary":
        final = keras.layers.Activation("sigmoid")(summed)
        loss = keras.losses.BinaryCrossentropy()
    else:  # pragma: no cover - guarded upstream
        raise ValueError(f"Unknown target: {model_info.target!r}")

    whole = keras.Model(inputs, final)
    whole.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    )
    _ = _target_activation  # kept for parity/reference
    return OnamKerasModel(model=whole, submodels=submodels)
