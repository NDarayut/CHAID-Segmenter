"""CHAID-based segmentation with automatic predictor binning.

A thin wrapper over the in-repo ``CHAID`` package that:

* auto-bins continuous predictors (target-based, equal-width, equal-frequency,
  manual cut points),
* builds a CHAID tree on the binned predictors,
* exposes terminal nodes as interpretable segments with rate / population /
  population share / lift,
* renders a static matplotlib + seaborn tree (node = population, branch = choice).
"""
from .binning import (
    Binner,
    CategoricalTargetBinner,
    EqualFrequencyBinner,
    EqualWidthBinner,
    ManualBinner,
    PassthroughNominal,
    TargetBinner,
    make_binner,
)
from .segmenter import ChaidSegmenter, Segment

__all__ = [
    "ChaidSegmenter",
    "Segment",
    "Binner",
    "EqualWidthBinner",
    "EqualFrequencyBinner",
    "ManualBinner",
    "TargetBinner",
    "CategoricalTargetBinner",
    "PassthroughNominal",
    "make_binner",
]

__version__ = "0.1.2"
