"""Tests for high-cardinality categorical (ID) predictors via supervised grouping."""
import numpy as np
import pandas as pd
import pytest

import setup_tests  # noqa: F401
from chaid_segmenter import ChaidSegmenter
from chaid_segmenter.binning import CategoricalTargetBinner, TargetBinner


def _make_df(n=6000, n_members=80, seed=3):
    rng = np.random.default_rng(seed)
    member_risk = np.clip(rng.normal(0.15, 0.10, n_members + 1), 0.02, 0.7)
    member_num = rng.integers(1, n_members + 1, n)
    member_id = np.array([f"{m:03d}" for m in member_num])
    target = (rng.uniform(0, 1, n) < member_risk[member_num]).astype(int)
    return pd.DataFrame({"member": member_id, "dpd": target})


def test_categorical_target_binner_groups_by_rate():
    pytest.importorskip("optbinning")
    df = _make_df()
    b = CategoricalTargetBinner(max_bins=5, mode="binary", positive_class=1)
    b.fit(df["member"].values, df["dpd"].values)
    assert 1 < b.n_bins <= 5
    # every category maps to a code; codes are contiguous 0..k-1
    assert set(b.cat_to_code.values()) == set(range(b.n_bins))
    # mean target rate is monotonic in the (rate-ordered) group code
    codes = b.transform(df["member"].values)
    rates = [df["dpd"].values[codes == c].mean() for c in range(b.n_bins)]
    assert rates == sorted(rates)


def test_range_label_lists_members_capped():
    pytest.importorskip("optbinning")
    df = _make_df()
    b = CategoricalTargetBinner(max_bins=5).fit(df["member"].values, df["dpd"].values)
    # the largest group should render as a capped "in {...}" list
    biggest = max(b.code_to_cats, key=lambda c: len(b.code_to_cats[c]))
    label = b.range_label([biggest])
    assert label.startswith("in {")
    if len(b.code_to_cats[biggest]) > 6:
        assert "+" in label


def test_segmenter_uses_listed_high_cardinality_id():
    pytest.importorskip("optbinning")
    df = _make_df()
    seg = ChaidSegmenter(target="dpd", positive_class=1, predictors=["member"],
                         max_depth=2, min_child_node_size=0.03).fit(df)
    # listed high-cardinality categorical -> inferred to supervised grouping
    assert seg.resolved_predictors["member"]["method"] == "target"
    assert isinstance(seg.binners["member"], CategoricalTargetBinner)
    assert seg.var_types["member"] == "ordinal"
    descriptions = " ".join(s.description for s in seg.segments())
    assert "member in {" in descriptions
    # predict assigns rows by the same grouping
    assert seg.predict(df).notna().all()


def test_full_auto_still_drops_high_cardinality_id():
    df = _make_df()
    rng = np.random.default_rng(0)
    df["region"] = rng.choice(["PP", "SR", "BB"], len(df))  # usable low-card column
    with pytest.warns(UserWarning, match="high-cardinality"):
        seg = ChaidSegmenter(target="dpd", positive_class=1,
                             default_numeric_method="equal_frequency",
                             max_depth=2, min_child_node_size=0.05).fit(df)
    assert "member" not in seg.resolved_predictors  # dropped as high-cardinality
    assert "region" in seg.resolved_predictors


def test_numeric_id_defaults_to_numeric_but_can_be_forced_categorical():
    pytest.importorskip("optbinning")
    df = _make_df()
    df["member"] = df["member"].astype(int)  # e.g. an id parsed from CSV as int64

    # default: a numeric column is treated as numeric (range bins) -- the gotcha
    num = ChaidSegmenter(target="dpd", positive_class=1, predictors=["member"],
                         max_depth=2, min_child_node_size=0.03).fit(df)
    assert isinstance(num.binners["member"], TargetBinner)

    # forcing categorical groups it like an id, regardless of dtype
    cat = ChaidSegmenter(
        target="dpd", positive_class=1,
        predictors={"member": {"method": "target", "categorical": True}},
        max_depth=2, min_child_node_size=0.03).fit(df)
    assert isinstance(cat.binners["member"], CategoricalTargetBinner)
    assert "member in {" in " ".join(s.description for s in cat.segments())


def test_nominal_high_cardinality_warns():
    pytest.importorskip("optbinning")
    df = _make_df(n=2000, n_members=60)
    seg = ChaidSegmenter(target="dpd", positive_class=1,
                         predictors={"member": "nominal"},
                         max_depth=1, min_child_node_size=0.05)
    with pytest.warns(UserWarning, match="nominal with"):
        seg.fit(df)
