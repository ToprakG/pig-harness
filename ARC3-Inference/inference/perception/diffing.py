"""Frame-to-frame difference analysis."""

from inference.perception.grid import _DIAG, _cell_char, _dims
from inference.perception.objects import _object_index, _pair_by_proximity


def _diff_objects(before_segmentation, after_segmentation, max_objects, background_pixels):
    before_groups, before_background = _object_index(before_segmentation, background_pixels)
    after_groups, after_background = _object_index(after_segmentation, background_pixels)

    moved = []
    appeared = []
    disappeared = []
    stable = 0

    for hash_key in set(before_groups) | set(after_groups):
        pairs, unmatched_before, unmatched_after = _pair_by_proximity(
            before_groups.get(hash_key, []), after_groups.get(hash_key, [])
        )
        for before, after in pairs:
            delta_row = after["anchor"][0] - before["anchor"][0]
            delta_col = after["anchor"][1] - before["anchor"][1]
            if delta_row == 0 and delta_col == 0:
                stable += 1
                continue
            moved.append(
                {
                    "hash": before["hash"],
                    "color": before["color"],
                    "pixels": before["pixels"],
                    "from": [before["anchor"][0], before["anchor"][1]],
                    "to": [after["anchor"][0], after["anchor"][1]],
                    "delta": [delta_row, delta_col],
                }
            )
        for node in unmatched_before:
            disappeared.append(
                {
                    "hash": node["hash"],
                    "color": node["color"],
                    "pixels": node["pixels"],
                    "at": [node["anchor"][0], node["anchor"][1]],
                }
            )
        for node in unmatched_after:
            appeared.append(
                {
                    "hash": node["hash"],
                    "color": node["color"],
                    "pixels": node["pixels"],
                    "at": [node["anchor"][0], node["anchor"][1]],
                }
            )

    # Largest objects first: a moved 40-cell shape matters more than a moved single cell.
    moved.sort(key=lambda item: (-item["pixels"], item["from"], item["to"]))
    appeared.sort(key=lambda item: (-item["pixels"], item["at"]))
    disappeared.sort(key=lambda item: (-item["pixels"], item["at"]))

    before_background_hashes = sorted(node["hash"] for node in before_background)
    after_background_hashes = sorted(node["hash"] for node in after_background)

    return {
        "moved": moved[:max_objects],
        "moved_total": len(moved),
        "appeared": appeared[:max_objects],
        "appeared_total": len(appeared),
        "disappeared": disappeared[:max_objects],
        "disappeared_total": len(disappeared),
        "stable": stable,
        "background_reshaped": before_background_hashes != after_background_hashes,
        "background_colors": sorted({node["color"] for node in after_background}),
    }


def _changed_cells(before, after, color_chars):
    before_rows, before_cols = _dims(before)
    after_rows, after_cols = _dims(after)
    rows = max(before_rows, after_rows)
    cols = max(before_cols, after_cols)

    changes = []
    for row in range(rows):
        for col in range(cols):
            before_char = _cell_char(before, row, col, color_chars)
            after_char = _cell_char(after, row, col, color_chars)
            if before_char == after_char:
                continue
            changes.append((row, col, before_char, after_char))
    return changes


def _cluster_regions(changes, max_regions):
    """Group changed cells into 8-connected regions, largest first."""
    remaining = {(row, col) for row, col, _before, _after in changes}
    regions = []
    while remaining:
        start = next(iter(remaining))
        remaining.discard(start)
        cells = [start]
        stack = [start]
        while stack:
            row, col = stack.pop()
            for delta_row, delta_col in _DIAG:
                neighbour = (row + delta_row, col + delta_col)
                if neighbour in remaining:
                    remaining.discard(neighbour)
                    cells.append(neighbour)
                    stack.append(neighbour)
        rows = [row for row, _col in cells]
        cols = [col for _row, col in cells]
        regions.append(
            {
                "bbox": [min(rows), min(cols), max(rows), max(cols)],
                "cells": len(cells),
            }
        )
    regions.sort(key=lambda item: (-item["cells"], item["bbox"]))
    return regions[:max_regions], len(regions)


def _group_transitions(changes, max_transitions, max_cells_per_transition):
    grouped = {}
    for row, col, before_char, after_char in changes:
        grouped.setdefault((before_char, after_char), []).append([row, col])

    transitions = []
    for (before_char, after_char), cells in grouped.items():
        transitions.append(
            {
                "from": before_char,
                "to": after_char,
                "count": len(cells),
                "cells": cells[:max_cells_per_transition],
                "cells_truncated": len(cells) > max_cells_per_transition,
            }
        )
    transitions.sort(key=lambda item: (-item["count"], item["from"], item["to"]))
    return transitions[:max_transitions], len(transitions)


def diff_grids(
    before,
    after,
    color_chars,
    before_segmentation=None,
    after_segmentation=None,
    max_transitions=8,
    max_cells_per_transition=6,
    max_regions=6,
    max_objects=6,
    background_fraction=0.25,
):
    """Summarize what changed between two frames, compactly enough to read in a prompt.

    ``before`` and ``after`` are grids of integer color values; ``color_chars`` is the
    ARC color-symbol mapping (integer color value -> single-char label). Cells outside a
    grid's bounds compare as ``"-"``, so mismatched shapes are reported rather than raised.

    Pass ``before_segmentation`` and ``after_segmentation`` (the output of
    ``segment_layer`` for each frame) to additionally get an object-level diff. Objects
    are matched across frames by shape hash, so an object that only moved is reported as
    ``moved`` with its translation, while an object that changed shape or color shows up
    as a ``disappeared``/``appeared`` pair.

    Components covering at least ``background_fraction`` of the grid are treated as
    background and reported only via ``background_reshaped``, since their shape changes
    whenever anything on top of them moves.

    Every list is capped and paired with a ``*_total`` count, so a truncated summary can
    never be mistaken for a complete one. The default caps keep a full dump of even a
    chaotic frame pair within a tool response's size budget; raise them when a specific
    question needs more detail.

    Returns a dict with:
      - ``changed``: whether anything differs at all.
      - ``changed_cells``: number of differing cells.
      - ``bbox``: ``[min_row, min_col, max_row, max_col]`` covering every change, or
        ``None`` when the frames are identical.
      - ``shape``: ``{"before": [rows, cols], "after": [rows, cols]}``.
      - ``transitions``: per color-pair change counts, each with sample ``cells``.
      - ``regions``: 8-connected clusters of changed cells as ``bbox`` + ``cells`` count.
      - ``objects``: present only when both segmentations are supplied; holds ``moved``,
        ``appeared``, ``disappeared`` lists, a ``stable`` count of unmoved objects, and
        ``background_reshaped``.
    """
    changes = _changed_cells(before, after, color_chars)
    transitions, transitions_total = _group_transitions(
        changes, max_transitions, max_cells_per_transition
    )
    regions, regions_total = _cluster_regions(changes, max_regions)

    if changes:
        rows = [row for row, _col, _before, _after in changes]
        cols = [col for _row, col, _before, _after in changes]
        bbox = [min(rows), min(cols), max(rows), max(cols)]
    else:
        bbox = None

    before_rows, before_cols = _dims(before)
    after_rows, after_cols = _dims(after)

    summary = {
        "changed": bool(changes),
        "changed_cells": len(changes),
        "bbox": bbox,
        "shape": {"before": [before_rows, before_cols], "after": [after_rows, after_cols]},
        "transitions": transitions,
        "transitions_total": transitions_total,
        "regions": regions,
        "regions_total": regions_total,
    }

    if before_segmentation is not None and after_segmentation is not None:
        area = max(before_rows * before_cols, after_rows * after_cols)
        summary["objects"] = _diff_objects(
            before_segmentation,
            after_segmentation,
            max_objects,
            int(area * background_fraction),
        )

    return summary
