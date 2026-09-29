"""Effect-term specification and the derived model structure (``ModelInfo``).

This is the Python counterpart of ``get_theta()`` in the R package.  Instead of
parsing an R ``formula`` object, effects are given explicitly as a list of
``(model_name, features)`` pairs (see :class:`Term`).  A string formula in the
R style (``"y ~ m1(x1) + m1(x1, x2) + m1(.)"``) is still accepted through
:func:`parse_formula` for convenience when migrating existing code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

__all__ = ["Term", "ModelInfo", "parse_formula", "RESIDUAL"]

# Sentinel used in a term's feature list to mean "every feature except the
# outcome" -- the R ``deep_model(.)`` residual term.
RESIDUAL = "."


@dataclass(frozen=True)
class Term:
    """A single (interaction) effect fitted by its own sub-network.

    Parameters
    ----------
    model:
        Key into the ``deep_models`` mapping given to :class:`onam.ONAM`.
    features:
        Ordered tuple of feature names.  A one-element tuple is a main effect,
        longer tuples are interaction effects.  The single value ``"."``
        (``onam.terms.RESIDUAL``) expands, at fit time, to every column of the
        data except the outcome.
    """

    model: str
    features: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.features, str):
            object.__setattr__(self, "features", (self.features,))
        else:
            object.__setattr__(self, "features", tuple(self.features))
        if not self.features:
            raise ValueError("A term must reference at least one feature.")

    @property
    def is_residual(self) -> bool:
        return self.features == (RESIDUAL,)

    @property
    def order(self) -> int:
        return len(self.features)


def _coerce_term(spec: "Term | tuple | list") -> Term:
    if isinstance(spec, Term):
        return spec
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        model, features = spec
        if isinstance(features, str):
            features = (features,)
        return Term(model=model, features=tuple(features))
    raise TypeError(
        "Each term must be a Term or a (model_name, features) pair, got: "
        f"{spec!r}"
    )


# --------------------------------------------------------------------------- #
# String-formula parsing (optional convenience)
# --------------------------------------------------------------------------- #
_FORMULA_CALL = re.compile(r"([A-Za-z_.][\w.]*)\s*\(([^()]*)\)")


def parse_formula(formula: str) -> tuple[str | None, list[Term]]:
    """Parse an R-style formula string into ``(outcome, terms)``.

    ``"y ~ m1(x1) + m1(x2) + m2(x1, x2) + m2(.)"`` ->
    ``("y", [Term("m1", ("x1",)), Term("m1", ("x2",)),
             Term("m2", ("x1", "x2")), Term("m2", (".",))])``
    """
    if "~" in formula:
        lhs, rhs = formula.split("~", 1)
        outcome = lhs.strip() or None
    else:
        outcome, rhs = None, formula
    terms: list[Term] = []
    for match in _FORMULA_CALL.finditer(rhs):
        model = match.group(1)
        raw = match.group(2).strip()
        if raw == RESIDUAL or raw == "":
            features: tuple[str, ...] = (RESIDUAL,)
        else:
            features = tuple(f.strip() for f in raw.split(",") if f.strip())
        terms.append(Term(model=model, features=features))
    if not terms:
        raise ValueError(f"No effect terms found in formula: {formula!r}")
    return outcome, terms


# --------------------------------------------------------------------------- #
# ModelInfo -- the resolved structure used throughout fitting/evaluation
# --------------------------------------------------------------------------- #
@dataclass
class ModelInfo:
    """Resolved effect structure, grouped by descending interaction order.

    Attributes
    ----------
    order_groups:
        ``order_groups[0]`` holds every term of the highest interaction order,
        ``order_groups[-1]`` the main effects.  Mirrors ``model_info$theta`` in
        the R package (which is likewise ordered high -> low).
    orders:
        The actual interaction order of each group, e.g. ``[3, 2, 1]``.
    outcome, target, categorical_features, all_feature_indic:
        As passed to / derived by :class:`onam.ONAM`.
    """

    order_groups: list[list[Term]]
    orders: list[int]
    outcome: str | None
    target: str
    categorical_features: list[str] = field(default_factory=list)
    all_feature_indic: bool = False
    formula: str | None = None

    # -- construction ------------------------------------------------------- #
    @classmethod
    def build(
        cls,
        terms: Iterable["Term | tuple | list"],
        feature_names: Sequence[str],
        *,
        outcome: str | None,
        target: str,
        categorical_features: Sequence[str] | None = None,
        formula: str | None = None,
    ) -> "ModelInfo":
        categorical_features = list(categorical_features or [])
        resolved: list[Term] = []
        all_feature_indic = False
        for spec in terms:
            term = _coerce_term(spec)
            if term.is_residual:
                all_feature_indic = True
                if outcome is not None and outcome in feature_names:
                    feats = tuple(f for f in feature_names if f != outcome)
                else:
                    feats = tuple(feature_names)
                term = Term(model=term.model, features=feats)
            resolved.append(term)

        # group by interaction order, descending
        unique_orders = sorted({t.order for t in resolved}, reverse=True)
        order_groups = [
            [t for t in resolved if t.order == order] for order in unique_orders
        ]
        return cls(
            order_groups=order_groups,
            orders=unique_orders,
            outcome=outcome,
            target=target,
            categorical_features=categorical_features,
            all_feature_indic=all_feature_indic,
            formula=formula,
        )

    # -- convenience views ----------------------------------------------------#
    @property
    def flat_terms(self) -> list[Term]:
        """All terms, groups high -> low order, terms in within-group order."""
        return [t for group in self.order_groups for t in group]

    @property
    def n_terms(self) -> int:
        return len(self.flat_terms)

    @property
    def max_order(self) -> int:
        return max(self.orders)

    @property
    def group_index_of_term(self) -> list[int]:
        """1-based group position for each flat term (R's ``model_order``)."""
        out: list[int] = []
        for gi, group in enumerate(self.order_groups, start=1):
            out.extend([gi] * len(group))
        return out

    def effect_name(self, term: Term) -> str:
        """Display / column name for one effect (R ``get_effect_name``)."""
        if term.order == self.max_order:
            return f"Interaction {term.order}"
        return "_".join(term.features)

    @property
    def effect_names(self) -> list[str]:
        return [self.effect_name(t) for t in self.flat_terms]

    def effect_names_for_order(self, order: int) -> list[str]:
        for grp_order, group in zip(self.orders, self.order_groups):
            if grp_order == order:
                return [self.effect_name(t) for t in group]
        return []

    def n_inputs(self, term: Term, cat_counts: dict[str, int]) -> int:
        n = term.order
        for feat in term.features:
            if feat in cat_counts:
                n += cat_counts[feat] - 1
        return n
