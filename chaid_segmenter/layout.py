"""Pure geometry for laying out a CHAID tree on a 2-D canvas.

Leaves are spread left-to-right at increasing x; each internal node sits at the
mean x of its children; y is ``-depth`` so the root is at the top.
"""
from __future__ import annotations

from collections import defaultdict


def compute_layout(tree):
    """Return a layout dict for ``tree`` (any iterable of CHAID nodes).

    Keys: ``pos`` ({node_id: (x, y)}), ``children`` ({node_id: [child_id]}),
    ``depth`` ({node_id: int}), ``n_leaves``, ``max_depth``, ``root_id``.
    """
    nodes = {n.node_id: n for n in tree}
    children = defaultdict(list)
    root_id = None
    for n in tree:
        if n.parent is None:
            root_id = n.node_id
        else:
            children[n.parent].append(n.node_id)
    for kids in children.values():
        kids.sort()

    depth = {}

    def _depth(nid):
        node = nodes[nid]
        if node.parent is None:
            depth[nid] = 0
        elif nid not in depth:
            depth[nid] = _depth(node.parent) + 1
        return depth[nid]

    for nid in nodes:
        _depth(nid)

    pos = {}
    counter = [0]

    def _assign(nid):
        kids = children.get(nid, [])
        if not kids:
            x = float(counter[0])
            counter[0] += 1
        else:
            for c in kids:
                _assign(c)
            x = sum(pos[c][0] for c in kids) / len(kids)
        pos[nid] = (x, float(-depth[nid]))

    _assign(root_id)

    return {
        "pos": pos,
        "children": dict(children),
        "depth": depth,
        "n_leaves": counter[0],
        "max_depth": max(depth.values()) if depth else 0,
        "root_id": root_id,
    }
