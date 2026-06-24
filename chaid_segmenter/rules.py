"""Translate CHAID split choices into human readable predicates.

A node's incoming branch is described by its parent's split variable plus the
node's ``choices`` (the grouped values that lead into it). For binned (ordinal)
predictors those choices are bin codes, rendered as ranges via the variable's
:class:`~chaid_segmenter.binning.Binner`; for nominal predictors they are the
original categories.
"""
from __future__ import annotations

import numpy as np

from .binning import MISSING_LABEL, Binner


def is_missing(value):
    if isinstance(value, str):
        return value == MISSING_LABEL
    return isinstance(value, float) and np.isnan(value)


def render_nominal(variable, data, max_cats=None):
    """e.g. ``region = Phnom Penh`` or ``region in {Phnom Penh, Siem Reap}``."""
    cats = [c for c in data if not is_missing(c)]
    has_missing = any(is_missing(c) for c in data)
    parts = [str(c) for c in cats]
    if has_missing:
        parts.append("missing")
    if max_cats is not None and len(parts) > max_cats:
        shown = parts[:max_cats]
        parts = shown + ["+{} more".format(len(parts) - max_cats)]
    if len(parts) == 1:
        return "{} = {}".format(variable, parts[0])
    return "{} in {{{}}}".format(variable, ", ".join(parts))


def render_predicate(variable, data, binner, max_cats=None):
    """Render a single ``{variable, data}`` condition as readable text."""
    if isinstance(binner, Binner):
        return "{} {}".format(variable, binner.range_label(data))
    return render_nominal(variable, data, max_cats=max_cats)


def describe_rules(rules, binners):
    """Build a ``(description, structured)`` pair for a node's rule list.

    ``rules`` is the ``rules`` list from ``Tree.classification_rules`` for one
    terminal node. ``structured`` is a list of ``{variable, label, data}``.
    """
    structured = []
    for rule in rules:
        variable, data = rule["variable"], rule["data"]
        label = render_predicate(variable, data, binners.get(variable))
        structured.append({"variable": variable, "label": label, "data": data})
    description = " AND ".join(s["label"] for s in structured) if structured else "(all)"
    return description, structured


def branch_label(variable, choices, binner, max_cats=4):
    """Compact label for the branch leading into a child node."""
    return render_predicate(variable, choices, binner, max_cats=max_cats)
