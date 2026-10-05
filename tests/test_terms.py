"""Formula parsing and ModelInfo structure (port of get_theta logic)."""

from onam import ModelInfo, Term, parse_formula
from onam.terms import RESIDUAL


def test_parse_formula():
    outcome, terms = parse_formula(
        "y ~ m1(x1) + m1(x2) + m2(x1, x2) + m2(.)"
    )
    assert outcome == "y"
    assert terms == [
        Term("m1", ("x1",)),
        Term("m1", ("x2",)),
        Term("m2", ("x1", "x2")),
        Term("m2", (RESIDUAL,)),
    ]


def test_modelinfo_groups_by_descending_order():
    info = ModelInfo.build(
        [("m", ["x1"]), ("m", ["x2"]), ("m", ["x1", "x2"]),
         ("m", ["x1", "x2", "x3"])],
        feature_names=["x1", "x2", "x3", "y"],
        outcome="y",
        target="continuous",
    )
    assert info.orders == [3, 2, 1]
    assert info.group_index_of_term == [1, 2, 3, 3]
    assert info.effect_names == ["Interaction 3", "x1_x2", "x1", "x2"]


def test_residual_expands_to_all_non_outcome_columns():
    info = ModelInfo.build(
        [("m", ["x1"]), ("m", ".")],
        feature_names=["x1", "x2", "x3", "y"],
        outcome="y",
        target="continuous",
    )
    residual = info.order_groups[0][0]
    assert residual.features == ("x1", "x2", "x3")
    assert info.all_feature_indic is True


def test_n_inputs_accounts_for_dummies():
    info = ModelInfo.build(
        [("m", ["x1", "c"])],
        feature_names=["x1", "c", "y"],
        outcome="y",
        target="continuous",
        categorical_features=["c"],
    )
    term = info.flat_terms[0]
    assert info.n_inputs(term, {"c": 3}) == 1 + 3
