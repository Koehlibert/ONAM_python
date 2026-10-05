"""``DNN`` -- a reusable stacked-dense sub-network constructor.

Python counterpart of ``build_dnn()`` in ``model_architecture.R``.  A ``DNN``
instance is *callable*: given a Keras input tensor it builds a feed-forward
stack of ``Dense`` layers and returns the resulting ``keras.Model``.  This is
exactly the shape of object expected in the ``deep_models`` mapping passed to
:class:`onam.ONAM`.
"""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import keras

__all__ = ["DNN"]

_SCALAR_DEFAULTS = {"activation": "relu", "use_bias": True}
_OBJECT_KEYS = (
    "kernel_initializer",
    "bias_initializer",
    "kernel_regularizer",
    "bias_regularizer",
    "activity_regularizer",
    "name",
)


class DNN:
    """Build a stack of ``Dense`` layers on top of an input tensor.

    Parameters
    ----------
    units:
        Integer sequence of layer sizes.  Every layer gets
        ``activation=default_activation`` and ``use_bias=True``, except the
        final layer which is forced to ``activation="linear"``.  Mutually
        exclusive with ``layers``.
    layers:
        Sequence of per-layer dicts giving full control.  Each dict must
        contain ``"units"`` and may contain ``activation``, ``use_bias``,
        ``kernel_initializer``, ``bias_initializer``, ``kernel_regularizer``,
        ``bias_regularizer``, ``activity_regularizer`` and ``name``.
    default_activation:
        Activation used where none is specified (default ``"relu"``).

    Notes
    -----
    The last layer must have exactly one output unit and linear activation;
    as in the R package, a violation is repaired with a warning rather than an
    error.
    """

    def __init__(
        self,
        units: Sequence[int] | None = None,
        layers: Sequence[dict] | None = None,
        default_activation: str = "relu",
    ) -> None:
        if units is None and layers is None:
            raise ValueError("Either `units` or `layers` must be supplied.")
        if units is not None and layers is not None:
            raise ValueError("Supply only one of `units` or `layers`.")

        self.default_activation = default_activation or "relu"

        if units is not None:
            specs = [
                {"units": int(u), "activation": self.default_activation,
                 "use_bias": True}
                for u in units
            ]
            from_units = True
        else:
            specs = [dict(layer) for layer in layers]  # shallow copies
            if any("units" not in s for s in specs):
                raise ValueError("Every layer spec must contain `units`.")
            for s in specs:
                s["units"] = int(s["units"])
            from_units = False

        if not specs:
            raise ValueError("At least one layer is required.")

        # Last layer must produce a single output unit.
        if specs[-1]["units"] != 1:
            warnings.warn(
                "Last layer must have exactly 1 output unit. An additional "
                "layer with 1 output unit was added.",
                stacklevel=2,
            )
            specs = specs + [dict(s) for s in specs[1:]]
            specs[-1]["units"] = 1

        if from_units:
            specs[-1]["activation"] = "linear"

        for s in specs:
            for key, default in _SCALAR_DEFAULTS.items():
                s.setdefault(key, default)

        if specs[-1].get("activation") != "linear":
            warnings.warn(
                "Last layer must have linear activation. Activation in last "
                "layer was set to linear.",
                stacklevel=2,
            )
            specs[-1]["activation"] = "linear"

        self.specs: list[dict[str, Any]] = specs

    # ------------------------------------------------------------------ #
    def __call__(self, inputs) -> keras.Model:
        x = inputs
        for spec in self.specs:
            kwargs = {"units": spec["units"], "activation": spec["activation"],
                      "use_bias": spec["use_bias"]}
            for key in _OBJECT_KEYS:
                if spec.get(key) is not None:
                    kwargs[key] = spec[key]
            x = keras.layers.Dense(**kwargs)(x)
        return keras.Model(inputs, x)

    def __repr__(self) -> str:
        shape = " -> ".join(str(s["units"]) for s in self.specs)
        return f"DNN({shape})"

    # ------------------------------------------------------------------ #
    @classmethod
    def linear(cls) -> "DNN":
        """A single bias-free linear unit (R ``get_linear_submodel``)."""
        return cls(layers=[{"units": 1, "activation": "linear",
                            "use_bias": False}])
