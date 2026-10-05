"""Save / load fitted models (port of ``persistence.R``).

Keras sub-networks cannot round-trip through :mod:`pickle`, so each is written
to its own ``.keras`` file under ``<dir>/submodels`` while the orthogonalization
metadata goes into ``<dir>/onam_meta.pkl``.
"""

from __future__ import annotations

import os
import pickle
import shutil

import keras
import numpy as np

from .orthogonalization import EnsembleMember

__all__ = ["save_onam", "load_onam"]

_META = "onam_meta.pkl"
_SUB = "submodels"


def save_onam(model, path, overwrite: bool = False) -> None:
    path = os.fspath(path)
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(
                f"{path!r} already exists. Pass overwrite=True to replace it."
            )
        shutil.rmtree(path)
    sub_dir = os.path.join(path, _SUB)
    os.makedirs(sub_dir, exist_ok=True)

    n_ensemble = len(model.ensemble)
    for i, member in enumerate(model.ensemble):
        for j, submodel in enumerate(member.submodels):
            submodel.save(os.path.join(sub_dir, f"ensemble{i}_term{j}.keras"))

    meta = {
        "version": 1,
        "raw_terms": model._raw_terms,
        "formula": model._formula,
        "outcome_from_formula": model._outcome_from_formula,
        "target": model.target,
        "categorical_features": model.categorical_features,
        "model_info": model.model_info,
        "categories_": model.categories_,
        "feature_names_": model.feature_names_,
        "data_": model.data_,
        "w_post_ensemble": model.w_post_ensemble,
        "feature_effects": model.feature_effects,
        "predictions": np.asarray(model.predictions),
        "y": np.asarray(model.y),
        "y_from_model": model._y_from_model,
        "n_ensemble": n_ensemble,
        "members": [
            {
                "w_list": [np.asarray(w) for w in m.w_list],
                "w_list_old": [np.asarray(w) for w in m.w_list_old],
                "u_dims": list(m.u_dims),
                "n_submodels": len(m.submodels),
            }
            for m in model.ensemble
        ],
    }
    with open(os.path.join(path, _META), "wb") as fh:
        pickle.dump(meta, fh)


def load_onam(path):
    from .core import ONAM

    path = os.fspath(path)
    meta_path = os.path.join(path, _META)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No onam model found in {path!r}.")
    with open(meta_path, "rb") as fh:
        meta = pickle.load(fh)

    sub_dir = os.path.join(path, _SUB)
    model = ONAM(
        terms=meta["raw_terms"],
        deep_models={},  # not needed once fitted
        target=meta["target"],
        categorical_features=meta["categorical_features"],
    )
    model._formula = meta["formula"]
    model._outcome_from_formula = meta["outcome_from_formula"]
    model.model_info = meta["model_info"]
    model.categories_ = meta["categories_"]
    model.feature_names_ = meta["feature_names_"]
    model.data_ = meta["data_"]
    model.w_post_ensemble = meta["w_post_ensemble"]
    model.feature_effects = meta["feature_effects"]
    model.predictions = meta["predictions"]
    model.y = meta["y"]
    model._y_from_model = meta["y_from_model"]

    ensemble = []
    for i, m in enumerate(meta["members"]):
        submodels = [
            keras.models.load_model(
                os.path.join(sub_dir, f"ensemble{i}_term{j}.keras")
            )
            for j in range(m["n_submodels"])
        ]
        ensemble.append(
            EnsembleMember(
                submodels=submodels,
                w_list=m["w_list"],
                w_list_old=m["w_list_old"],
                u_dims=m["u_dims"],
            )
        )
    model.ensemble = ensemble
    return model
