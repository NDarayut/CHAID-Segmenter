"""Binning strategies for continuous predictors.

Each binner converts a continuous variable into small integer *bin codes*
(``0, 1, 2, ...``) that are fed into CHAID as an ``ordinal`` column. The binner
keeps the numeric ``edges`` and a ``code_to_label`` map so the tree's branch
choices (which come back as bin codes) can be rendered as human readable ranges
such as ``"< 25"`` / ``"in [25, 40)"`` / ``">= 40"``.

Continuous predictors must enter CHAID as integer codes because
``CHAID.column.OrdinalColumn`` casts its data to ``int`` and only ever merges
*contiguous* codes -- which is exactly what keeps merged bins readable as a
single range.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np

#: Token used for the missing/NaN branch (matches CHAID's default missing id).
MISSING_LABEL = "<missing>"


def _fmt_num(value):
    """Format a numeric edge compactly (drop trailing ``.0``)."""
    value = float(value)
    if value == int(value):
        return str(int(value))
    rounded = round(value, 4)
    text = ("%g" % rounded)
    return text


class Binner(ABC):
    """Base class for all continuous binners.

    Attributes
    ----------
    edges : numpy.ndarray
        Sorted interior cut points. ``n`` edges produce ``n + 1`` bins.
    code_to_label : dict[int, str]
        Maps each bin code to its standalone label (e.g. ``"[25, 40)"``).
    name : str or None
        The column name, used for nicer warnings.
    """

    def __init__(self, name=None):
        self.edges = np.array([], dtype=float)
        self.code_to_label = {}
        self.name = name

    # -- abstract ---------------------------------------------------------
    @abstractmethod
    def _compute_edges(self, x, y=None):
        """Return the interior cut points for ``x`` (and optionally ``y``)."""

    # -- public API -------------------------------------------------------
    @property
    def n_bins(self):
        return len(self.edges) + 1

    def fit(self, x, y=None):
        x = np.asarray(x, dtype=float)
        edges = np.asarray(self._compute_edges(x, y), dtype=float)
        edges = np.unique(edges[~np.isnan(edges)])
        self.edges = edges
        if len(edges) == 0:
            warnings.warn(
                "Binner for column {!r} produced a single bin (no usable cut "
                "points); this predictor cannot be split.".format(self.name),
                stacklevel=2,
            )
        self._build_labels()
        return self

    def transform(self, x):
        """Map values to float bin codes, preserving ``NaN`` for missing."""
        x = np.asarray(x, dtype=float)
        codes = np.digitize(x, self.edges).astype(float)
        codes[np.isnan(x)] = np.nan
        return codes

    def fit_transform(self, x, y=None):
        return self.fit(x, y).transform(x)

    def range_label(self, codes):
        """Render a (possibly merged, contiguous) group of codes as one range.

        ``codes`` may contain the missing token; it is described separately and
        appended with "or missing".
        """
        numeric, has_missing = self._split_missing(codes)
        if not numeric:
            return "is missing"

        lo_code, hi_code = min(numeric), max(numeric)
        last = len(self.edges)  # index of the final (open-topped) bin
        low = None if lo_code == 0 else self.edges[lo_code - 1]
        high = None if hi_code >= last else self.edges[hi_code]

        if low is None and high is None:
            fragment = "is any value"
        elif low is None:
            fragment = "< {}".format(_fmt_num(high))
        elif high is None:
            fragment = ">= {}".format(_fmt_num(low))
        else:
            fragment = "in [{}, {})".format(_fmt_num(low), _fmt_num(high))

        if has_missing:
            fragment += " or missing"
        return fragment

    # -- helpers ----------------------------------------------------------
    def _build_labels(self):
        labels = {}
        last = len(self.edges)
        for code in range(self.n_bins):
            labels[code] = self.range_label([code])
        self.code_to_label = labels

    @staticmethod
    def _split_missing(codes):
        """Split a list of choices into ``(numeric_codes, has_missing)``."""
        numeric, has_missing = [], False
        for c in codes:
            if isinstance(c, str):
                has_missing = True
                continue
            if isinstance(c, float) and np.isnan(c):
                has_missing = True
                continue
            numeric.append(int(round(float(c))))
        return numeric, has_missing


class EqualWidthBinner(Binner):
    """Fixed-interval bins of equal width across the value range."""

    def __init__(self, bins=5, name=None):
        super().__init__(name=name)
        if bins < 1:
            raise ValueError("bins must be >= 1, got {}".format(bins))
        self.bins = bins

    def _compute_edges(self, x, y=None):
        finite = x[~np.isnan(x)]
        if finite.size == 0:
            return []
        lo, hi = float(np.min(finite)), float(np.max(finite))
        if lo == hi:
            return []
        return np.linspace(lo, hi, self.bins + 1)[1:-1]


class EqualFrequencyBinner(Binner):
    """Quantile (equal-population) bins."""

    def __init__(self, bins=5, name=None):
        super().__init__(name=name)
        if bins < 1:
            raise ValueError("bins must be >= 1, got {}".format(bins))
        self.bins = bins

    def _compute_edges(self, x, y=None):
        finite = x[~np.isnan(x)]
        if finite.size == 0:
            return []
        quantiles = np.linspace(0, 1, self.bins + 1)[1:-1]
        edges = np.nanquantile(finite, quantiles)
        deduped = np.unique(edges)
        if len(deduped) < len(edges):
            warnings.warn(
                "Tied values in column {!r} collapsed equal-frequency edges "
                "from {} to {} bins.".format(self.name, self.bins, len(deduped) + 1),
                stacklevel=2,
            )
        return deduped


class ManualBinner(Binner):
    """Bins from user supplied interior cut points."""

    def __init__(self, edges, name=None):
        super().__init__(name=name)
        edges = np.asarray(edges, dtype=float)
        if edges.ndim != 1 or edges.size == 0:
            raise ValueError("Manual edges must be a non-empty 1-D sequence.")
        if np.any(np.diff(edges) <= 0):
            raise ValueError("Manual edges must be strictly increasing and unique.")
        self._manual_edges = edges

    def _compute_edges(self, x, y=None):
        return self._manual_edges


class TargetBinner(Binner):
    """Supervised optimal binning via the :mod:`optbinning` library.

    Uses :class:`optbinning.OptimalBinning` for binary targets and
    :class:`optbinning.ContinuousOptimalBinning` for continuous targets. Only
    the *cut points* (``binner.splits``) are taken; transformation reuses the
    shared :class:`Binner` machinery so range rendering is identical across all
    strategies.
    """

    def __init__(self, max_bins=5, mode="binary", positive_class=1,
                 monotonic_trend="auto", name=None):
        super().__init__(name=name)
        if max_bins < 1:
            raise ValueError("max_bins must be >= 1, got {}".format(max_bins))
        self.max_bins = max_bins
        self.mode = mode
        self.positive_class = positive_class
        self.monotonic_trend = monotonic_trend

    def _compute_edges(self, x, y=None):
        if y is None:
            raise ValueError("TargetBinner requires the target 'y' to fit.")
        try:
            from optbinning import ContinuousOptimalBinning, OptimalBinning
        except ImportError as exc:  # pragma: no cover - exercised via message
            raise ImportError(
                "Target-based binning needs optbinning. Install it with "
                "`pip install 'CHAID[segmenter-target]'` or `pip install optbinning`."
            ) from exc

        y = np.asarray(y)
        mask = ~np.isnan(x)
        x_fit = x[mask]
        y_fit = y[mask]

        if self.mode == "continuous":
            y_fit = np.asarray(y_fit, dtype=float)
            ob = ContinuousOptimalBinning(
                name=str(self.name or "x"), dtype="numerical",
                max_n_bins=self.max_bins, monotonic_trend=self.monotonic_trend,
            )
        else:
            y_fit = (y_fit == self.positive_class).astype(int)
            ob = OptimalBinning(
                name=str(self.name or "x"), dtype="numerical",
                max_n_bins=self.max_bins, monotonic_trend=self.monotonic_trend,
            )
        ob.fit(x_fit, y_fit)
        splits = np.asarray(ob.splits, dtype=float)
        return splits[~np.isnan(splits)]


class CategoricalTargetBinner(Binner):
    """Supervised grouping of a high-cardinality categorical (e.g. a member/ID).

    Uses :mod:`optbinning`'s categorical mode to group the categories into a few
    *rate-ordered* bins, so a column with hundreds of distinct values (which plain
    nominal CHAID can't handle well) becomes a handful of risk tiers. The bins are
    monotonic in the target, so they are fed to CHAID as an ordinal column and only
    contiguous (adjacent-risk) tiers merge.
    """

    def __init__(self, max_bins=8, mode="binary", positive_class=1, name=None):
        super().__init__(name=name)
        if max_bins < 1:
            raise ValueError("max_bins must be >= 1, got {}".format(max_bins))
        self.max_bins = max_bins
        self.mode = mode
        self.positive_class = positive_class
        self.cat_to_code = {}
        self.code_to_cats = {}
        self._n_groups = 1

    def _compute_edges(self, x, y=None):  # not used; categorical has no edges
        return []

    @property
    def n_bins(self):
        return max(self._n_groups, 1)

    @staticmethod
    def _is_na(v):
        return v is None or (isinstance(v, float) and isnan(v))

    def fit(self, x, y=None):
        if y is None:
            raise ValueError("CategoricalTargetBinner requires the target 'y' to fit.")
        try:
            from optbinning import ContinuousOptimalBinning, OptimalBinning
        except ImportError as exc:  # pragma: no cover - exercised via message
            raise ImportError(
                "Target-based binning needs optbinning. Install it with "
                "`pip install 'CHAID[segmenter-target]'` or `pip install optbinning`."
            ) from exc

        x_arr = np.asarray(x, dtype=object)
        y_arr = np.asarray(y)
        keep = np.array([not self._is_na(v) for v in x_arr])
        x_fit = x_arr[keep].astype(str)
        y_fit = y_arr[keep]
        if x_fit.size and len(np.unique(x_fit)) > 1000:
            warnings.warn(
                "Column {!r} has very high cardinality ({} categories); supervised "
                "grouping may be slow or overfit — is this a row identifier?".format(
                    self.name, len(np.unique(x_fit))),
                stacklevel=2,
            )

        if self.mode == "continuous":
            ob = ContinuousOptimalBinning(name=str(self.name or "x"),
                                          dtype="categorical", max_n_bins=self.max_bins)
            ob.fit(x_fit, y_fit.astype(float))
        else:
            ob = OptimalBinning(name=str(self.name or "x"),
                                dtype="categorical", max_n_bins=self.max_bins)
            ob.fit(x_fit, (y_fit == self.positive_class).astype(int))

        uniq = np.unique(x_fit)
        raw = [int(c) for c in ob.transform(uniq, metric="indices")]
        remap = {c: i for i, c in enumerate(sorted(set(raw)))}  # contiguous, rate-ordered
        self.cat_to_code = {str(cat): remap[code] for cat, code in zip(uniq, raw)}
        self.code_to_cats = {}
        for cat, code in self.cat_to_code.items():
            self.code_to_cats.setdefault(code, []).append(cat)
        for code in self.code_to_cats:
            self.code_to_cats[code] = sorted(self.code_to_cats[code])
        self._n_groups = len(remap)
        self.code_to_label = {c: self.range_label([c]) for c in self.code_to_cats}
        if self._n_groups <= 1:
            warnings.warn(
                "Supervised grouping of {!r} produced a single group; it cannot "
                "be split.".format(self.name), stacklevel=2)
        return self

    def transform(self, x):
        x_arr = np.asarray(x, dtype=object)
        out = np.full(len(x_arr), np.nan, dtype=float)
        for i, v in enumerate(x_arr):
            if self._is_na(v):
                continue
            code = self.cat_to_code.get(str(v))
            if code is not None:
                out[i] = float(code)
        return out

    def range_label(self, codes):
        numeric, has_missing = self._split_missing(codes)
        cats = []
        for c in sorted(set(numeric)):
            cats.extend(self.code_to_cats.get(c, []))
        cats = sorted(set(cats))
        if not cats:
            return "is missing"
        cap = 6
        shown = ", ".join(cats[:cap])
        if len(cats) > cap:
            shown += ", +{}".format(len(cats) - cap)
        fragment = "in {{{}}}".format(shown)
        if has_missing:
            fragment += " or missing"
        return fragment


class PassthroughNominal:
    """Marker for categorical predictors that bypass binning."""

    is_nominal = True

    def __init__(self, name=None):
        self.name = name


def make_binner(spec, name=None, mode="binary", positive_class=1, is_numeric=True):
    """Build a binner from a per-predictor spec dict.

    ``spec`` is ``{"method": "...", ...}`` where method is one of
    ``equal_width`` (``bins``), ``equal_frequency`` (``bins``), ``manual``
    (``edges``), ``target`` (``max_bins``, optional ``monotonic_trend``) or
    ``nominal`` (passthrough). ``is_numeric`` selects, for ``target``, between
    numeric optimal binning and categorical (high-cardinality) grouping; set
    ``spec['categorical'] = True`` to force categorical grouping even for a numeric
    column (e.g. an integer member/institution id that must not be binned as a
    number).
    """
    if isinstance(spec, str):
        spec = {"method": spec}
    method = spec.get("method")
    if method == "nominal":
        return PassthroughNominal(name=name)
    if method == "equal_width":
        return EqualWidthBinner(bins=spec.get("bins", 5), name=name)
    if method == "equal_frequency":
        return EqualFrequencyBinner(bins=spec.get("bins", 5), name=name)
    if method == "manual":
        if "edges" not in spec:
            raise ValueError("Manual binning for {!r} requires 'edges'.".format(name))
        return ManualBinner(edges=spec["edges"], name=name)
    if method == "target":
        if is_numeric and not spec.get("categorical", False):
            return TargetBinner(
                max_bins=spec.get("max_bins", 5), mode=mode,
                positive_class=positive_class,
                monotonic_trend=spec.get("monotonic_trend", "auto"), name=name,
            )
        return CategoricalTargetBinner(
            max_bins=spec.get("max_bins", 8), mode=mode,
            positive_class=positive_class, name=name,
        )
    raise ValueError("Unknown binning method {!r} for column {!r}.".format(method, name))
