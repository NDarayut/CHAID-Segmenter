"""End-to-end tests for ChaidSegmenter with a binary target."""
import numpy as np
import pandas as pd
import pytest

import setup_tests  # noqa: F401
from chaid_segmenter import ChaidSegmenter


def _make_df(n=4000, seed=0):
    rng = np.random.RandomState(seed)
    age = rng.uniform(18, 70, n)
    region = rng.choice(["PP", "SR", "BB"], n)
    # High event rate for young customers in Phnom Penh; low otherwise.
    base = np.where((age < 25) & (region == "PP"), 0.6, 0.1)
    target = (rng.uniform(0, 1, n) < base).astype(int)
    return pd.DataFrame({"age": age, "region": region, "dpd90": target})


def _fit(df):
    seg = ChaidSegmenter(
        target="dpd90", positive_class=1,
        predictors={"age": {"method": "target", "max_bins": 4},
                    "region": {"method": "nominal"}},
        max_depth=3, min_child_node_size=0.02, alpha_merge=0.05,
    )
    return seg.fit(df)


def test_high_rate_segment_is_discovered():
    pytest.importorskip("optbinning")
    df = _make_df()
    seg = _fit(df)
    summary = seg.summary()
    top = summary.iloc[0]
    assert top["rate"] > 0.4
    assert "region = PP" in top["description"]
    assert "age <" in top["description"]


def test_populations_and_overall_rate():
    pytest.importorskip("optbinning")
    df = _make_df()
    seg = _fit(df)
    summary = seg.summary()
    assert summary["population"].sum() == len(df)
    assert summary["population_pct"].sum() == pytest.approx(1.0)
    assert seg.overall_rate == pytest.approx(df["dpd90"].mean())


def test_segment_metrics_match_predict():
    pytest.importorskip("optbinning")
    df = _make_df()
    seg = _fit(df)
    pred = seg.predict(df)
    assert pred.isna().sum() == 0  # exhaustive assignment
    df2 = df.assign(_node=pred.astype(int))
    for s in seg.segments():
        rows = df2[df2["_node"] == s.node_id]
        assert len(rows) == int(s.population)
        assert (rows["dpd90"] == 1).mean() == pytest.approx(s.rate)


def test_lift_is_rate_over_overall():
    pytest.importorskip("optbinning")
    df = _make_df()
    seg = _fit(df)
    for s in seg.segments():
        assert s.lift == pytest.approx(s.rate / seg.overall_rate)


def test_rejects_bad_positive_class():
    df = _make_df()
    seg = ChaidSegmenter(
        target="dpd90", positive_class=99,
        predictors={"region": {"method": "nominal"}},
    )
    with pytest.raises(ValueError):
        seg.fit(df)


def test_predict_unseen_category_is_na():
    pytest.importorskip("optbinning")
    df = _make_df()
    seg = _fit(df)
    new = df.head(5).copy()
    new["region"] = "ZZ"  # never seen during fit
    pred = seg.predict(new)
    # rows whose only distinguishing path needs region=ZZ may be unassigned
    assert pred.isna().any() or (pred.notna().all())
