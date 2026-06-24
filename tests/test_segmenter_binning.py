"""Tests for the chaid_segmenter binning strategies."""
import numpy as np
import pytest

import setup_tests  # noqa: F401  (bootstraps sys.path to repo root)
from chaid_segmenter.binning import (
    EqualFrequencyBinner,
    EqualWidthBinner,
    ManualBinner,
    PassthroughNominal,
    TargetBinner,
    make_binner,
)


def test_equal_width_edges_and_labels():
    b = EqualWidthBinner(bins=4).fit(np.arange(0, 101))
    np.testing.assert_allclose(b.edges, [25, 50, 75])
    assert b.n_bins == 4
    assert b.code_to_label[0] == "< 25"
    assert b.code_to_label[1] == "in [25, 50)"
    assert b.code_to_label[3] == ">= 75"


def test_equal_width_transform_codes():
    b = EqualWidthBinner(bins=4).fit(np.arange(0, 101))
    codes = b.transform([0, 24, 25, 50, 99])
    np.testing.assert_array_equal(codes, [0, 0, 1, 2, 3])


def test_equal_frequency_quantile_edges():
    b = EqualFrequencyBinner(bins=4).fit(np.arange(100))
    np.testing.assert_allclose(b.edges, np.nanquantile(np.arange(100), [.25, .5, .75]))
    assert b.n_bins == 4


def test_equal_frequency_tie_collapse_warns():
    # A heavily tied column: most quantiles land on the same value.
    data = np.array([1.0] * 20 + [5.0, 5.0])
    with pytest.warns(UserWarning):
        b = EqualFrequencyBinner(bins=4).fit(data)
    assert b.n_bins < 4  # tied edges collapsed


def test_manual_validation_rejects_unsorted():
    with pytest.raises(ValueError):
        ManualBinner(edges=[40, 25])


def test_manual_codes_and_labels():
    b = ManualBinner(edges=[25, 40]).fit(np.array([10, 30, 50], dtype=float))
    np.testing.assert_array_equal(b.transform([10, 30, 50]), [0, 1, 2])
    assert b.code_to_label[0] == "< 25"
    assert b.code_to_label[2] == ">= 40"


def test_nan_is_preserved_as_missing():
    b = ManualBinner(edges=[25]).fit(np.array([1, 50], dtype=float))
    codes = b.transform([1, np.nan, 50])
    assert codes[0] == 0 and codes[2] == 1
    assert np.isnan(codes[1])


def test_range_label_merged_contiguous_groups():
    b = ManualBinner(edges=[25, 40, 60]).fit(np.array([10, 30, 50, 70], dtype=float))
    assert b.range_label([0, 1]) == "< 40"
    assert b.range_label([1, 2]) == "in [25, 60)"
    assert b.range_label([2, 3]) == ">= 40"
    assert b.range_label([1, "<missing>"]) == "in [25, 40) or missing"
    assert b.range_label(["<missing>"]) == "is missing"


def test_transform_reuse_does_not_refit():
    b = EqualWidthBinner(bins=4).fit(np.arange(0, 101))
    before = b.edges.copy()
    out = b.transform([-10, 200])  # out-of-range values clamp to end bins
    np.testing.assert_array_equal(out, [0, 3])
    np.testing.assert_array_equal(b.edges, before)


def test_single_bin_warns_for_constant_column():
    with pytest.warns(UserWarning):
        b = EqualWidthBinner(bins=4).fit(np.array([5.0, 5.0, 5.0]))
    assert b.n_bins == 1


def test_make_binner_factory_dispatch():
    assert isinstance(make_binner({"method": "equal_width", "bins": 3}), EqualWidthBinner)
    assert isinstance(make_binner({"method": "equal_frequency"}), EqualFrequencyBinner)
    assert isinstance(make_binner({"method": "manual", "edges": [1, 2]}), ManualBinner)
    assert isinstance(make_binner({"method": "nominal"}), PassthroughNominal)
    assert isinstance(make_binner("nominal"), PassthroughNominal)
    assert isinstance(make_binner({"method": "target"}), TargetBinner)
    with pytest.raises(ValueError):
        make_binner({"method": "bogus"})


def test_target_binner_monotonic_and_capped():
    optbinning = pytest.importorskip("optbinning")  # noqa: F841
    rng = np.random.RandomState(0)
    x = np.concatenate([rng.normal(0, 1, 600), rng.normal(4, 1, 600)])
    y = np.concatenate([np.zeros(600), np.ones(600)]).astype(int)
    b = TargetBinner(max_bins=3, mode="binary", positive_class=1).fit(x, y)
    assert b.n_bins <= 3
    assert len(b.edges) <= 2
    # event rate per resulting bin should be monotonic
    codes = b.transform(x)
    rates = [y[codes == c].mean() for c in range(b.n_bins) if (codes == c).any()]
    assert rates == sorted(rates) or rates == sorted(rates, reverse=True)
