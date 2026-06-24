"""Smoke tests for the static matplotlib/seaborn renderer."""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import setup_tests  # noqa: F401,E402
from chaid_segmenter import ChaidSegmenter  # noqa: E402


def _fit():
    rng = np.random.RandomState(2)
    n = 1500
    age = rng.uniform(18, 70, n)
    region = rng.choice(["PP", "SR"], n)
    target = (rng.uniform(0, 1, n) < np.where(age < 30, 0.5, 0.1)).astype(int)
    df = pd.DataFrame({"age": age, "region": region, "dpd90": target})
    seg = ChaidSegmenter(
        target="dpd90", positive_class=1,
        predictors={"age": {"method": "equal_width", "bins": 4},
                    "region": {"method": "nominal"}},
        max_depth=2, min_child_node_size=0.05,
    )
    return seg.fit(df)


def test_plot_writes_png(tmp_path):
    seg = _fit()
    out = tmp_path / "tree.png"
    seg.plot(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_writes_svg(tmp_path):
    seg = _fit()
    out = tmp_path / "tree.svg"
    seg.plot(str(out))
    assert out.exists() and out.stat().st_size > 0
