"""Tests for automatic predictor inference (dict / list / full-auto)."""
import os

import pandas as pd
import pytest

import setup_tests  # noqa: F401
from chaid_segmenter import ChaidSegmenter

DATA = os.path.join(os.path.dirname(__file__), "data", "titanic.csv")


def _method(spec):
    return spec if isinstance(spec, str) else spec["method"]


def _fit(predictors, **kw):
    df = pd.read_csv(DATA)
    seg = ChaidSegmenter(
        target="survived", positive_class=1, predictors=predictors,
        default_numeric_method="equal_frequency",  # avoid optbinning dependency
        max_depth=2, min_child_node_size=0.05, **kw,
    )
    return seg.fit(df), df


def test_full_auto_selects_and_drops_high_cardinality():
    seg, df = _fit(None)
    resolved = seg.resolved_predictors
    # high-cardinality text columns are dropped
    for junk in ("name", "ticket", "cabin", "home.dest", "boat"):
        assert junk not in resolved
    # target is never a predictor
    assert "survived" not in resolved
    # dtype-based methods
    assert _method(resolved["sex"]) == "nominal"
    assert _method(resolved["age"]) == "equal_frequency"


def test_list_form_infers_methods_from_dtype():
    seg, _ = _fit(["age", "fare", "sex", "embarked"])
    resolved = seg.resolved_predictors
    assert set(resolved) == {"age", "fare", "sex", "embarked"}
    assert _method(resolved["age"]) == "equal_frequency"
    assert _method(resolved["sex"]) == "nominal"


def test_dict_auto_and_explicit_override():
    seg, _ = _fit({"age": "auto", "fare": {"method": "equal_width", "bins": 3}, "sex": "auto"})
    resolved = seg.resolved_predictors
    assert _method(resolved["age"]) == "equal_frequency"   # inferred
    assert _method(resolved["fare"]) == "equal_width"       # honoured override
    assert _method(resolved["sex"]) == "nominal"


def test_default_numeric_method_target_when_requested():
    pytest.importorskip("optbinning")
    df = pd.read_csv(DATA)
    seg = ChaidSegmenter(
        target="survived", positive_class=1, predictors=["age"],
        default_numeric_method="target", default_bins=4,
        max_depth=2, min_child_node_size=0.05,
    ).fit(df)
    assert seg.resolved_predictors["age"]["method"] == "target"
    assert seg.resolved_predictors["age"]["max_bins"] == 4


def test_target_and_weight_are_never_predictors():
    with pytest.warns(UserWarning):
        seg, _ = _fit(["age", "survived"])
    assert "survived" not in seg.resolved_predictors


def test_bad_predictors_type_raises():
    df = pd.read_csv(DATA)
    seg = ChaidSegmenter(target="survived", positive_class=1, predictors=42)
    with pytest.raises(TypeError):
        seg.fit(df)


def test_no_usable_predictors_raises():
    df = pd.read_csv(DATA)
    seg = ChaidSegmenter(target="survived", positive_class=1, predictors=[])
    with pytest.raises(ValueError):
        seg.fit(df)
