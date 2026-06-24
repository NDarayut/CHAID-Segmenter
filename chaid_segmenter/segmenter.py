"""High-level CHAID segmentation built on top of the in-repo ``CHAID`` package.

``ChaidSegmenter`` auto-bins continuous predictors, builds a CHAID tree, and
exposes the terminal nodes as interpretable *segments* with an event rate /
mean, population, population share and lift -- e.g.

    AGE < 25 AND region = Phnom Penh AND bank = ABA -> rate 60%, pop 10%
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from CHAID import Tree

from . import metrics, rules
from .binning import PassthroughNominal, make_binner


@dataclass
class Segment:
    """A terminal segment of the tree."""

    node_id: int
    description: str
    population: float
    population_pct: float
    rate: float
    lift: float
    rules: List[dict] = field(default_factory=list)

    def __repr__(self):
        return "Segment(node={}, rate={:.4g}, pop_pct={:.2%}, {!r})".format(
            self.node_id, self.rate, self.population_pct, self.description
        )


class ChaidSegmenter:
    """Fit a CHAID segmentation tree with automatic predictor binning.

    Parameters
    ----------
    target : str
        Name of the KPI/target column.
    predictors : dict
        ``{column: spec}`` where ``spec`` is a binning spec dict (or method
        string). Methods: ``target``, ``equal_width``, ``equal_frequency``,
        ``manual``, ``nominal``. See :func:`chaid_segmenter.binning.make_binner`.
    positive_class : optional
        The event value of a binary target (e.g. ``1``). When given, the tree is
        categorical and node ``rate`` is ``P(target == positive_class)``. When
        ``None`` the target is treated as continuous and ``rate`` is the mean.
    max_depth, min_child_node_size, min_parent_node_size, alpha_merge,
    split_threshold, max_splits, weight :
        Passed through to ``CHAID.Tree``. Node-size args accept fractions in
        ``(0, 1)``. ``min_parent_node_size`` defaults to ``min_child_node_size``.
    """

    def __init__(self, target, predictors, positive_class=None, max_depth=3,
                 min_child_node_size=30, min_parent_node_size=None,
                 alpha_merge=0.05, split_threshold=0, max_splits=None, weight=None):
        if not predictors:
            raise ValueError("At least one predictor must be supplied.")
        self.target = target
        self.predictors = predictors
        self.positive_class = positive_class
        self.mode = "binary" if positive_class is not None else "continuous"
        self.max_depth = max_depth
        self.min_child_node_size = min_child_node_size
        self.min_parent_node_size = min_parent_node_size
        self.alpha_merge = alpha_merge
        self.split_threshold = split_threshold
        self.max_splits = max_splits
        self.weight = weight

        self.binners: Dict[str, object] = {}
        self.var_types: Dict[str, str] = {}
        self.tree: Optional[Tree] = None
        self.root_total = None
        self.overall_rate = None
        self._terminals: List[tuple] = []

    # -- construction helpers --------------------------------------------
    @property
    def _weighted(self):
        return self.weight is not None

    def _effective_min_parent(self):
        if self.min_parent_node_size is None:
            return self.min_child_node_size
        return self.min_parent_node_size

    # -- fitting ----------------------------------------------------------
    def fit(self, df):
        """Fit binners and build the CHAID tree from a pandas DataFrame."""
        if self.target not in df.columns:
            raise KeyError("target {!r} not found in DataFrame.".format(self.target))
        if self.mode == "binary":
            uniques = pd.unique(df[self.target].dropna())
            if self.positive_class not in set(uniques):
                raise ValueError(
                    "positive_class {!r} is not a value of target {!r}; "
                    "observed values: {}".format(
                        self.positive_class, self.target, list(uniques)
                    )
                )

        work = {}
        i_variables = {}
        self.binners, self.var_types = {}, {}
        target_vals = df[self.target].values
        for col, spec in self.predictors.items():
            if col not in df.columns:
                raise KeyError("predictor {!r} not found in DataFrame.".format(col))
            binner = make_binner(
                spec, name=col, mode=self.mode, positive_class=self.positive_class
            )
            if isinstance(binner, PassthroughNominal):
                work[col] = df[col].values
                i_variables[col] = "nominal"
                self.var_types[col] = "nominal"
            else:
                binner.fit(df[col].values, target_vals)
                work[col] = binner.transform(df[col].values)
                i_variables[col] = "ordinal"
                self.var_types[col] = "ordinal"
            self.binners[col] = binner

        work[self.target] = target_vals
        if self.weight is not None:
            work[self.weight] = df[self.weight].values
        work_df = pd.DataFrame(work, index=df.index)

        dep_type = "categorical" if self.mode == "binary" else "continuous"
        self.tree = Tree.from_pandas_df(
            work_df, i_variables, self.target,
            alpha_merge=self.alpha_merge, max_depth=self.max_depth,
            min_parent_node_size=self._effective_min_parent(),
            min_child_node_size=self.min_child_node_size,
            split_threshold=self.split_threshold, weight=self.weight,
            dep_variable_type=dep_type, max_splits=self.max_splits,
        )
        _ = self.tree.tree_store  # force build

        root = self.tree.tree_store[0]
        self.root_total = metrics.node_population(root, weighted=self._weighted)
        self.overall_rate = metrics.node_rate(root, self.mode, self.positive_class)
        self._terminals = [
            (node.node_id, self.tree.classification_rules(node)[0]["rules"])
            for node in self.tree if node.is_terminal
        ]
        return self

    # -- outputs ----------------------------------------------------------
    def _require_fit(self):
        if self.tree is None:
            raise RuntimeError("Call fit() before requesting segments.")

    def segments(self, sort_by_rate=True):
        """Return one :class:`Segment` per terminal node."""
        self._require_fit()
        out = []
        for node in self.tree:
            if not node.is_terminal:
                continue
            r = self.tree.classification_rules(node)[0]["rules"]
            description, structured = rules.describe_rules(r, self.binners)
            stats = metrics.node_stats(
                node, self.root_total, self.overall_rate, self.mode,
                self.positive_class, weighted=self._weighted,
            )
            out.append(Segment(
                node_id=node.node_id, description=description, rules=structured,
                **stats,
            ))
        if sort_by_rate:
            out.sort(key=lambda s: (np.isnan(s.rate), -s.rate))
        return out

    def summary(self):
        """Return a tidy ``pandas.DataFrame`` of the segments, best rate first."""
        segs = self.segments(sort_by_rate=True)
        return pd.DataFrame([
            {
                "node_id": s.node_id,
                "description": s.description,
                "population": s.population,
                "population_pct": s.population_pct,
                "rate": s.rate,
                "lift": s.lift,
            }
            for s in segs
        ])

    @property
    def segment_rates(self):
        """Map terminal ``node_id`` -> rate."""
        return {s.node_id: s.rate for s in self.segments(sort_by_rate=False)}

    # -- prediction -------------------------------------------------------
    def _transform_frame(self, df):
        out = {}
        for col, binner in self.binners.items():
            if isinstance(binner, PassthroughNominal):
                out[col] = df[col].values
            else:
                out[col] = binner.transform(df[col].values)
        return pd.DataFrame(out, index=df.index)

    def _rule_mask(self, work_df, rule_list):
        mask = pd.Series(True, index=work_df.index)
        for rule in rule_list:
            var, data = rule["variable"], rule["data"]
            series = work_df[var]
            has_missing = any(rules.is_missing(c) for c in data)
            if self.var_types.get(var) == "ordinal":
                values = [float(c) for c in data if not rules.is_missing(c)]
            else:
                values = [c for c in data if not rules.is_missing(c)]
            m = series.isin(values)
            if has_missing:
                m = m | series.isna()
            mask &= m
        return mask

    def predict(self, df, with_rate=False):
        """Assign each row of ``df`` to a terminal segment.

        Returns a ``Series`` of terminal ``node_id`` (``<NA>`` if a row matches
        no segment, e.g. an unseen category). With ``with_rate=True`` returns a
        DataFrame adding the segment ``rate``.
        """
        self._require_fit()
        work_df = self._transform_frame(df)
        node_ids = pd.Series(pd.NA, index=df.index, dtype="object")
        for node_id, rule_list in self._terminals:
            mask = self._rule_mask(work_df, rule_list)
            node_ids[mask & node_ids.isna()] = node_id
        node_ids = node_ids.astype("Int64")
        if not with_rate:
            return node_ids
        rate_map = self.segment_rates
        out = pd.DataFrame({"node_id": node_ids}, index=df.index)
        out["rate"] = node_ids.map(rate_map).astype(float)
        return out

    # -- visualisation ----------------------------------------------------
    def plot(self, path=None, **kwargs):
        """Render the tree to a static matplotlib figure (see TreePlotter)."""
        self._require_fit()
        from .plotting import TreePlotter

        return TreePlotter(self).render(path=path, **kwargs)

    # -- loaders ----------------------------------------------------------
    @classmethod
    def from_csv(cls, path, target, predictors, *, read_csv_kwargs=None,
                 **segmenter_kwargs):
        """Read a CSV into a DataFrame, then construct and ``fit``."""
        df = pd.read_csv(path, **(read_csv_kwargs or {}))
        seg = cls(target=target, predictors=predictors, **segmenter_kwargs)
        return seg.fit(df)

    @classmethod
    def from_parquet(cls, path, target, predictors, *, read_parquet_kwargs=None,
                     **segmenter_kwargs):
        """Read a Parquet file into a DataFrame, then construct and ``fit``."""
        try:
            df = pd.read_parquet(path, **(read_parquet_kwargs or {}))
        except ImportError as exc:  # pragma: no cover - exercised via message
            raise ImportError(
                "Reading Parquet needs a parquet engine. Install it with "
                "`pip install 'CHAID[parquet]'` or `pip install pyarrow`."
            ) from exc
        seg = cls(target=target, predictors=predictors, **segmenter_kwargs)
        return seg.fit(df)
