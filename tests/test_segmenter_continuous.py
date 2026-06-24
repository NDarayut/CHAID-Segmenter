"""End-to-end tests for ChaidSegmenter with a continuous KPI."""
import numpy as np
import pandas as pd
import pytest

import setup_tests  # noqa: F401
from chaid_segmenter import ChaidSegmenter


def _make_df(n=3000, seed=1):
    rng = np.random.RandomState(seed)
    tenure = rng.uniform(0, 60, n)
    region = rng.choice(["PP", "SR"], n)
    balance = np.where(tenure > 30, 1000, 300) + rng.normal(0, 50, n)
    return pd.DataFrame({"tenure": tenure, "region": region, "balance": balance})


def test_continuous_rate_is_node_mean():
    df = _make_df()
    seg = ChaidSegmenter(
        target="balance",  # no positive_class -> continuous mode
        predictors={"tenure": {"method": "equal_frequency", "bins": 4},
                    "region": {"method": "nominal"}},
        max_depth=3, min_child_node_size=0.05,
    )
    seg.fit(df)
    assert seg.mode == "continuous"
    assert seg.overall_rate == pytest.approx(df["balance"].mean())

    pred = seg.predict(df)
    df2 = df.assign(_node=pred.astype(int))
    for s in seg.segments():
        rows = df2[df2["_node"] == s.node_id]
        assert len(rows) == int(s.population)
        assert rows["balance"].mean() == pytest.approx(s.rate, rel=1e-6)
        assert s.lift == pytest.approx(s.rate / seg.overall_rate)
