"""Tests for readable rule / branch rendering."""
import numpy as np

import setup_tests  # noqa: F401
from chaid_segmenter import ManualBinner, PassthroughNominal
from chaid_segmenter.rules import branch_label, describe_rules, render_nominal


def _binner():
    return ManualBinner(edges=[25, 40, 60], name="age").fit(
        np.array([10, 30, 50, 70], dtype=float)
    )


def test_describe_rules_combines_ordinal_and_nominal():
    binners = {"age": _binner(), "region": PassthroughNominal(name="region")}
    rules = [
        {"variable": "age", "data": [0.0, 1.0]},   # < 40
        {"variable": "region", "data": ["PP"]},
    ]
    description, structured = describe_rules(rules, binners)
    assert description == "age < 40 AND region = PP"
    assert structured[0]["label"] == "age < 40"


def test_nominal_multi_category_and_missing():
    assert render_nominal("region", ["PP", "SR"]) == "region in {PP, SR}"
    assert render_nominal("region", ["PP", float("nan")]) == "region in {PP, missing}"


def test_nominal_truncates_long_lists():
    cats = [str(i) for i in range(10)]
    out = render_nominal("region", cats, max_cats=3)
    assert "+7" in out


def test_branch_label_uses_range_for_ordinal():
    assert branch_label("age", [2.0, 3.0], _binner()) == "age >= 40"
    assert branch_label("region", ["PP"], PassthroughNominal()) == "region = PP"
