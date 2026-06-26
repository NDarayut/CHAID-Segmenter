"""Static tree visualisation with matplotlib + seaborn.

Each NODE is drawn as a box showing its population (count + % of total) and its
target rate; each BRANCH is labelled with the choice (bin range or category)
that leads into the child node. Node fill colour encodes the rate.

The layout is top-down. To stay readable as the number of leaves grows, the
figure width is sized from the actual node-box width so adjacent leaves never
overlap, fonts shrink with leaf count, and branch labels are wrapped/truncated.
"""
from __future__ import annotations

import textwrap

import numpy as np

from . import metrics, rules
from .layout import compute_layout


def _text_color(rgba):
    """Pick black/white text for contrast against a fill colour."""
    r, g, b = rgba[0], rgba[1], rgba[2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 0.55 else "white"


def _fmt_rate(rate, mode):
    if rate is None or (isinstance(rate, float) and np.isnan(rate)):
        return "n/a"
    return "{:.1%}".format(rate) if mode == "binary" else "{:.3g}".format(rate)


def _fmt_pop(pop):
    return "{:,}".format(int(round(pop)))


def _auto_fontsize(n_leaves):
    """Shrink the node font as the tree gets wider."""
    if n_leaves <= 8:
        return 9
    if n_leaves <= 14:
        return 8
    if n_leaves <= 22:
        return 7
    return 6


def _wrap(text, width):
    """Wrap onto as many lines as needed (full text, never truncated)."""
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)


class TreePlotter:
    """Render a fitted :class:`~chaid_segmenter.segmenter.ChaidSegmenter`."""

    def __init__(self, segmenter):
        self.seg = segmenter
        self.tree = segmenter.tree

    def render(self, path=None, figsize=None, cmap="flare", dpi=150,
               node_fontsize=None, branch_fontsize=None,
               max_branch_categories=None, show=False):
        import matplotlib

        if path is not None and not show:
            matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        layout = compute_layout(self.tree)
        pos = layout["pos"]
        n_leaves = max(layout["n_leaves"], 1)
        max_depth = layout["max_depth"]
        nodes = {n.node_id: n for n in self.tree}

        if node_fontsize is None:
            node_fontsize = _auto_fontsize(n_leaves)
        if branch_fontsize is None:
            branch_fontsize = max(5, node_fontsize - 1)

        # node rate / population
        rate = {nid: metrics.node_rate(n, self.seg.mode, self.seg.positive_class)
                for nid, n in nodes.items()}
        pop = {nid: metrics.node_population(n, weighted=self.seg._weighted)
               for nid, n in nodes.items()}
        finite_rates = [v for v in rate.values() if not np.isnan(v)]
        rmin = min(finite_rates) if finite_rates else 0.0
        rmax = max(finite_rates) if finite_rates else 1.0
        if rmin == rmax:
            rmax = rmin + 1e-9
        norm = Normalize(vmin=rmin, vmax=rmax)
        colormap = sns.color_palette(cmap, as_cmap=True)

        # node label text (target name lives in the title, not in every box)
        total = self.seg.root_total
        metric_word = "rate" if self.seg.mode == "binary" else "mean"
        node_text, max_line_chars = {}, 1
        for nid in nodes:
            pct = (pop[nid] / total * 100) if total else float("nan")
            text = "#{}\nn={} · {:.1f}%\n{} {}".format(
                nid, _fmt_pop(pop[nid]), pct, metric_word,
                _fmt_rate(rate[nid], self.seg.mode))
            node_text[nid] = text
            max_line_chars = max(max_line_chars, max(len(s) for s in text.split("\n")))

        # branch (choice) labels -- full text, wrapped onto as many lines as needed
        wrap_chars = 22
        branch_text, max_branch_lines = {}, 1
        for nid, node in nodes.items():
            if node.parent is None:
                continue
            parent = nodes[node.parent]
            binner = self.seg.binners.get(parent.split_variable)
            label = _wrap(
                rules.branch_label(parent.split_variable, node.choices, binner,
                                   max_cats=max_branch_categories),
                wrap_chars)
            branch_text[nid] = label
            max_branch_lines = max(max_branch_lines, label.count("\n") + 1)

        # size the canvas so one data unit (min gap between adjacent leaves) is at
        # least one node box wide -> leaf boxes don't overlap; grow the row height
        # to fit the (possibly multi-line) branch labels between rows.
        if figsize is None:
            gap_in, max_width = 0.5, 60.0
            for _ in range(2):  # one font step-down if it would get too wide
                node_w_in = max_line_chars * node_fontsize * 0.62 / 72.0 + 0.30
                label_w_in = wrap_chars * branch_fontsize * 0.62 / 72.0 + 0.20
                per_leaf = max(node_w_in, label_w_in) + gap_in
                width = per_leaf * (n_leaves + 1) / 0.85 + 0.6
                if width <= max_width:
                    break
                node_fontsize = max(5, node_fontsize - 1)
                branch_fontsize = max(4, node_fontsize - 1)
            width = min(width, max_width)

            node_h = 3 * node_fontsize * 1.4 / 72.0 + 0.30
            label_h = max_branch_lines * branch_fontsize * 1.4 / 72.0 + 0.18
            row_h = node_h + label_h + 0.5
            height = max(5.0, row_h * (max_depth + 1) + node_h)
            figsize = (width, height)

        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")

        # edges + branch labels (centred in the gap so tall labels clear both rows)
        for nid, node in nodes.items():
            if node.parent is None:
                continue
            px, py = pos[node.parent]
            cx, cy = pos[nid]
            ax.plot([px, cx], [py, cy], color="#9aa0a6", lw=1.0, zorder=1)
            # place the label above its own child so siblings separate by the full
            # child spacing (no horizontal collision); centred vertically in the gap.
            mx, my = cx, (py + cy) / 2.0
            ax.text(mx, my, branch_text[nid], ha="center", va="center",
                    fontsize=branch_fontsize, zorder=2,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="#c7c7c7", lw=0.5))

        # nodes
        for nid in nodes:
            x, y = pos[nid]
            color = colormap(norm(rate[nid])) if not np.isnan(rate[nid]) else (0.85, 0.85, 0.85, 1)
            ax.text(x, y, node_text[nid], ha="center", va="center", zorder=3,
                    fontsize=node_fontsize, color=_text_color(color),
                    bbox=dict(boxstyle="round,pad=0.35", fc=color, ec="#5f6368", lw=1.0))

        ax.set_xlim(-1.0, n_leaves)
        ax.set_ylim(-max_depth - 0.7, 0.85)

        overall = _fmt_rate(self.seg.overall_rate, self.seg.mode)
        ax.set_title(
            "CHAID segmentation — {} (overall {})".format(self.seg.target, overall),
            fontsize=node_fontsize + 3,
        )

        sm = ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label(metric_word)

        if path is not None:
            fig.savefig(path, bbox_inches="tight", dpi=dpi)
        return fig
