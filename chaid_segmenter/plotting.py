"""Static tree visualisation with matplotlib + seaborn.

Each NODE is drawn as a box showing its population (count + % of total) and its
target rate; each BRANCH is labelled with the choice (bin range or category)
that leads into the child node. Node fill colour encodes the rate.
"""
from __future__ import annotations

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


class TreePlotter:
    """Render a fitted :class:`~chaid_segmenter.segmenter.ChaidSegmenter`."""

    def __init__(self, segmenter):
        self.seg = segmenter
        self.tree = segmenter.tree

    def render(self, path=None, figsize=None, cmap="flare", dpi=150,
               node_fontsize=8, branch_fontsize=7, show=False):
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

        # node rates / populations
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

        if figsize is None:
            figsize = (max(8.0, 1.7 * n_leaves), max(5.0, 2.3 * (max_depth + 1)))
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")

        # edges first, with branch (choice) labels
        for nid, node in nodes.items():
            if node.parent is None:
                continue
            px, py = pos[node.parent]
            cx, cy = pos[nid]
            ax.plot([px, cx], [py, cy], color="#9aa0a6", lw=1.0, zorder=1)
            parent = nodes[node.parent]
            binner = self.seg.binners.get(parent.split_variable)
            label = rules.branch_label(parent.split_variable, node.choices, binner)
            # Place the label toward the child so sibling labels don't overlap.
            frac = 0.62
            mx, my = px + frac * (cx - px), py + frac * (cy - py)
            ax.text(mx, my, label, ha="center", va="center",
                    fontsize=branch_fontsize, zorder=2,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="#c7c7c7", lw=0.6))

        # nodes
        total = self.seg.root_total
        for nid, node in nodes.items():
            x, y = pos[nid]
            color = colormap(norm(rate[nid])) if not np.isnan(rate[nid]) else (0.85, 0.85, 0.85, 1)
            pct = pop[nid] / total if total else float("nan")
            text = "#{}\nn={} ({:.1%})\n{}: {}".format(
                nid, _fmt_pop(pop[nid]), pct, self.seg.target,
                _fmt_rate(rate[nid], self.seg.mode),
            )
            ax.text(x, y, text, ha="center", va="center", zorder=3,
                    fontsize=node_fontsize, color=_text_color(color),
                    bbox=dict(boxstyle="round,pad=0.4", fc=color, ec="#5f6368", lw=1.0))

        ax.set_xlim(-1.0, n_leaves)
        ax.set_ylim(-max_depth - 0.7, 0.8)

        overall = _fmt_rate(self.seg.overall_rate, self.seg.mode)
        ax.set_title(
            "CHAID segmentation — {} (overall {})".format(self.seg.target, overall),
            fontsize=node_fontsize + 3,
        )

        sm = ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("rate" if self.seg.mode == "binary" else "mean")

        if path is not None:
            fig.savefig(path, bbox_inches="tight", dpi=dpi)
        return fig
