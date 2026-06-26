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


def _fit_wide():
    """A deeper tree with many leaves and long nominal category merges."""
    rng = np.random.RandomState(7)
    n = 6000
    age = rng.uniform(18, 70, n)
    region = rng.choice([f"R{i:02d}" for i in range(12)], n)   # high-ish cardinality
    product = rng.choice(["A", "B", "C", "D"], n)
    target = (rng.uniform(0, 1, n) < np.where(age < 30, 0.5, 0.15)).astype(int)
    df = pd.DataFrame({"age": age, "region": region, "product": product, "dpd": target})
    seg = ChaidSegmenter(
        target="dpd", positive_class=1,
        predictors={"age": {"method": "equal_frequency", "bins": 4},
                    "region": "nominal", "product": "nominal"},
        max_depth=3, min_child_node_size=0.02,
    )
    return seg.fit(df)


def test_wide_tree_auto_sizes_and_writes(tmp_path):
    seg = _fit_wide()
    # auto figsize / auto fonts path; must not raise and must produce a real file
    fig = seg.plot(str(tmp_path / "wide.png"))
    w, h = fig.get_size_inches()
    assert w <= 55.0 and w >= 8.0 and h >= 5.0
    assert (tmp_path / "wide.png").stat().st_size > 0
