"""Per-node segmentation metrics derived from CHAID node objects.

A node's ``members`` is ``{target_value: count}`` for categorical (binary)
targets -- counts are weighted floats when a weight column is used -- or
``{'mean': ..., 's.t.d': ...}`` for continuous targets. ``node.indices`` holds
the row indices that fell into the node.
"""
from __future__ import annotations

import math


def _is_continuous_members(members):
    return "mean" in members and "s.t.d" in members


def node_population(node, weighted=False):
    """Population of a node: row count, or weighted total when requested.

    Weighted totals are only available for categorical targets (from the
    weighted member sums); continuous targets fall back to the row count.
    """
    members = node.members
    if weighted and not _is_continuous_members(members):
        return float(sum(members.values()))
    return float(len(node.indices))


def node_rate(node, mode, positive_class=None):
    """Event rate for binary targets, or the mean for continuous targets."""
    members = node.members
    if mode == "continuous":
        return float(members["mean"])
    total = sum(members.values())
    if total == 0:
        return float("nan")
    return float(members.get(positive_class, 0)) / float(total)


def node_stats(node, root_total, overall_rate, mode, positive_class=None,
               weighted=False):
    """Return ``population``, ``population_pct``, ``rate`` and ``lift``."""
    population = node_population(node, weighted=weighted)
    rate = node_rate(node, mode, positive_class)
    population_pct = population / root_total if root_total else float("nan")
    if overall_rate in (0, None) or (isinstance(overall_rate, float) and math.isnan(overall_rate)):
        lift = float("nan")
    else:
        lift = rate / overall_rate
    return {
        "population": population,
        "population_pct": population_pct,
        "rate": rate,
        "lift": lift,
    }
