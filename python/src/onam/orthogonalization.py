"""Post-hoc orthogonalization (PHO) of the fitted sub-networks.

Direct port of ``orthogonalization.R``.  Two stages:

* :func:`pho` -- orthogonalize the effects of a single ensemble member by
  projecting each interaction order onto the span of all lower orders and
  moving the shared component down.
* :func:`pho_ensemble` -- a second, effect-level orthogonalization applied to
  the ensemble-averaged effects.

All linear algebra is done in float64 on NumPy arrays; only the penultimate
activations come from Keras (float32), matching the R/reticulate behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

import keras
import numpy as np
import scipy.linalg

from .terms import ModelInfo

__all__ = ["EnsembleMember", "pho", "pho_ensemble", "penultimate_output"]


# --------------------------------------------------------------------------- #
@dataclass
class EnsembleMember:
    """One fitted + orthogonalized additive model."""

    submodels: list[keras.Model]
    w_list: list[np.ndarray]       # block-sparse, orthogonalized
    w_list_old: list[np.ndarray]   # block-sparse, pre-orthogonalization
    u_dims: list[int]


# --------------------------------------------------------------------------- #
def penultimate_output(submodel: keras.Model, x: np.ndarray) -> np.ndarray:
    """Activations feeding the final ``Dense(1)`` layer (R ``get_intermediate_model``)."""
    last = submodel.layers[-1]
    try:
        intermediate = keras.Model(submodel.inputs, last.input)
    except Exception:  # pragma: no cover - fallback for exotic graphs
        intermediate = keras.Model(submodel.inputs, submodel.layers[-2].output)
    return np.asarray(intermediate.predict(x, verbose=0), dtype=np.float64)


def _get_u(
    submodels: list[keras.Model], data_fit: list[np.ndarray]
) -> tuple[np.ndarray, list[int], list[np.ndarray]]:
    add_bias_col = bool(submodels[0].layers[-1].use_bias)
    blocks: list[np.ndarray] = []
    for submodel, x in zip(submodels, data_fit):
        u_i = penultimate_output(submodel, x)
        if add_bias_col:
            u_i = np.column_stack([u_i, np.ones(len(u_i))])
        blocks.append(u_i)
    u_dims = [b.shape[1] for b in blocks]
    u = np.column_stack(blocks + [np.ones(len(blocks[0]))])
    idx_list: list[np.ndarray] = []
    start = 0
    for d in u_dims:
        idx_list.append(np.arange(start, start + d))
        start += d
    return u, u_dims, idx_list


def _get_w_list(
    submodels: list[keras.Model], idx_list: list[np.ndarray], n_cols: int
) -> list[np.ndarray]:
    sep = []
    for submodel in submodels:
        weights = submodel.layers[-1].get_weights()
        sep.append(np.concatenate([np.ravel(w) for w in weights]))
    full = np.concatenate([np.concatenate(sep), [0.0]])
    if len(full) != n_cols:  # pragma: no cover - non-uniform bias config
        raise ValueError(
            "Sub-networks disagree on final-layer bias; all must match."
        )
    out = []
    for idx in idx_list:
        w = np.zeros(n_cols)
        w[idx] = full[idx]
        out.append(w)
    return out


def _pivot_order(mat: np.ndarray) -> np.ndarray:
    """Columns of ``mat`` ordered most- to least-independent, rank-truncated.

    Emulates ``qr(crossprod(mat))$pivot[seq_len(rank)]`` from the R code.
    """
    h = mat.T @ mat
    _, r, p = scipy.linalg.qr(h, pivoting=True, mode="economic")
    diag = np.abs(np.diag(r))
    if diag.size == 0 or diag[0] == 0:
        return np.array([], dtype=int)
    rank = int(np.sum(diag > 1e-7 * diag[0]))
    return p[:rank]


def _solve_singular(mat: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    """Drop trailing columns until ``mat[:, pivot]`` has full column rank."""
    pivot = np.asarray(pivot, dtype=int)
    while pivot.size > 0:
        sub = mat[:, pivot]
        gram = sub.T @ sub
        try:
            inv = np.linalg.inv(gram)
            if np.all(np.isfinite(inv)):
                return pivot
        except np.linalg.LinAlgError:
            pass
        pivot = pivot[:-1]
    return pivot


def _project(tmp_u: np.ndarray, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(pivot, inverse)`` for least squares onto ``tmp_u``'s columns."""
    pivot = _solve_singular(tmp_u, _pivot_order(tmp_u))
    reduced = tmp_u[:, pivot]
    inverse = np.linalg.inv(reduced.T @ reduced)
    return pivot, inverse


# --------------------------------------------------------------------------- #
def pho(
    submodels: list[keras.Model],
    model_info: ModelInfo,
    data_fit: list[np.ndarray],
) -> EnsembleMember:
    """Orthogonalize one ensemble member in place (returns an EnsembleMember)."""
    model_order = np.asarray(model_info.group_index_of_term)  # 1-based group pos
    n_terms = len(submodels)
    n_groups = len(model_info.order_groups)

    u, u_dims, idx_list = _get_u(submodels, data_fit)
    n_cols = u.shape[1]
    w_list = _get_w_list(submodels, idx_list, n_cols)
    w_list_old = [w.copy() for w in w_list]

    for idx_ortho in range(2, n_groups + 1):
        lower = np.where(model_order >= idx_ortho)[0]
        higher = np.where(model_order == idx_ortho - 1)[0]
        rel_cols = np.concatenate([idx_list[i] for i in lower])

        tmp_u = u.copy()
        mask = np.ones(n_cols, dtype=bool)
        mask[rel_cols] = False
        tmp_u[:, mask] = 0.0
        tmp_u[:, -1] = 1.0

        pivot, inverse = _project(tmp_u, n_cols)
        reduced = tmp_u[:, pivot]

        z_list = []
        for h_idx in higher:
            outputs = u @ w_list[h_idx]
            tmp_z = inverse @ reduced.T @ outputs
            z = np.zeros(n_cols)
            z[pivot] = tmp_z
            z_list.append(z)

        for k, h_idx in enumerate(higher):
            w_list[h_idx] = w_list[h_idx] - z_list[k]

        for l_idx in lower:
            block = idx_list[l_idx]
            acc = np.zeros(len(block))
            for z in z_list:
                acc += z[block]
            upd = np.zeros(n_cols)
            upd[block] = acc
            w_list[l_idx] = w_list[l_idx] + upd

    means = [float((u @ w).mean()) for w in w_list]
    for i, w in enumerate(w_list):
        w = w.copy()
        w[-1] = sum(means[1:]) if i == 0 else -means[i]
        w_list[i] = w

    return EnsembleMember(
        submodels=submodels,
        w_list=w_list,
        w_list_old=w_list_old,
        u_dims=u_dims,
    )


# --------------------------------------------------------------------------- #
def pho_ensemble(
    effects_ensemble: np.ndarray, model_info: ModelInfo
) -> tuple[np.ndarray, np.ndarray]:
    """Effect-level orthogonalization of ensemble-averaged effects.

    Parameters
    ----------
    effects_ensemble:
        ``(n_samples, n_terms)`` matrix; column ``j`` is the ensemble mean of
        term ``j``'s orthogonalized effect.

    Returns
    -------
    (w_post, effects) where ``effects = effects_ensemble @ w_post``.
    """
    u = np.asarray(effects_ensemble, dtype=np.float64)
    n_terms = u.shape[1]
    model_order = np.asarray(model_info.group_index_of_term)
    n_groups = len(model_info.order_groups)
    w = np.eye(n_terms)

    for idx_ortho in range(2, n_groups + 1):
        lower = np.where(model_order >= idx_ortho)[0]
        higher = np.where(model_order == idx_ortho - 1)[0]

        tmp_u = u.copy()
        mask = np.ones(n_terms, dtype=bool)
        mask[lower] = False
        tmp_u[:, mask] = 0.0

        pivot = _solve_singular(tmp_u, _pivot_order(tmp_u))
        reduced = tmp_u[:, pivot]
        inverse = np.linalg.inv(reduced.T @ reduced)

        z_list = []
        for h_idx in higher:
            outputs = u @ w[:, h_idx]
            tmp_z = inverse @ reduced.T @ outputs
            z = np.zeros(n_terms)
            z[pivot] = tmp_z
            z_list.append(z)

        for k, h_idx in enumerate(higher):
            w[:, h_idx] = w[:, h_idx] - z_list[k]

        for l_idx in lower:
            w[l_idx, l_idx] += sum(z[l_idx] for z in z_list)

    return w, u @ w
