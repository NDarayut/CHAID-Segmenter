"""Tests for the DataFrame / file loader entry points."""
import os

import pandas as pd
import pytest

import setup_tests  # noqa: F401
from chaid_segmenter import ChaidSegmenter

DATA = os.path.join(os.path.dirname(__file__), "data", "titanic.csv")
PREDICTORS = {
    "fare": {"method": "equal_frequency", "bins": 4},
    "sex": {"method": "nominal"},
}
KW = dict(positive_class=1, max_depth=2, min_child_node_size=0.05)


def test_from_csv_matches_manual_fit():
    via_csv = ChaidSegmenter.from_csv(DATA, "survived", PREDICTORS, **KW)
    df = pd.read_csv(DATA)
    manual = ChaidSegmenter("survived", PREDICTORS, **KW).fit(df)
    pd.testing.assert_frame_equal(via_csv.summary(), manual.summary())


def test_from_parquet_matches_manual_fit(tmp_path):
    pytest.importorskip("pyarrow")
    df = pd.read_csv(DATA)
    path = tmp_path / "titanic.parquet"
    df.to_parquet(path)
    via_parquet = ChaidSegmenter.from_parquet(str(path), "survived", PREDICTORS, **KW)
    manual = ChaidSegmenter("survived", PREDICTORS, **KW).fit(df)
    pd.testing.assert_frame_equal(via_parquet.summary(), manual.summary())
