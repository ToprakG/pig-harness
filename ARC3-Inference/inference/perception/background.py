"""Identifying which color is the background, and how confident that call is."""

from inference.perception.grid import _cell_char, _color_counts, _dims
from inference.perception.objects import _node_bbox


def identify_background(
    grid,
    color_chars,
    segmentation=None,
    other_grids=(),
    max_colors=4,
):
    """Identify the most likely background color, with the evidence behind the call.

    Area alone can mislead, so this also reports how the color is laid out and -- when
    ``other_grids`` are supplied (earlier or later frames) -- how static it is. A color
    that covers most of the board in one huge component and never changes between frames
    is background; a color that covers a lot of cells but keeps changing is gameplay.

    Pass ``segmentation`` (the output of ``segment_layer``) to get component structure.

    Returns a dict with:
      - ``color``: the dominant color symbol.
      - ``fraction``: its share of all cells.
      - ``color_counts``: the top colors as ``color``/``cells``/``fraction``, largest first.
      - ``components``: how many separate components the dominant color forms, or ``None``
        without ``segmentation``.
      - ``largest_component``: ``bbox``/``pixels``/``fraction`` of its biggest component,
        or ``None`` without ``segmentation``.
      - ``stable_fraction``: share of the dominant color's cells holding that same color in
        every frame in ``other_grids``; ``None`` when none are supplied.
    """
    rows, cols = _dims(grid)
    area = rows * cols
    if area == 0:
        return {
            "color": None,
            "fraction": 0.0,
            "color_counts": [],
            "components": None,
            "largest_component": None,
            "stable_fraction": None,
        }

    counts = _color_counts(grid, color_chars)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    dominant, dominant_cells = ranked[0]

    components = None
    largest_component = None
    if segmentation is not None:
        matching = [
            node for node in segmentation.get("nodes", []) if node.get("color") == dominant
        ]
        components = len(matching)
        if matching:
            biggest = max(matching, key=lambda node: int(node.get("pixels", 0)))
            bbox = _node_bbox(biggest)
            largest_component = {
                "bbox": list(bbox) if bbox else None,
                "pixels": int(biggest.get("pixels", 0)),
                "fraction": round(int(biggest.get("pixels", 0)) / area, 3),
            }

    stable_fraction = None
    if other_grids:
        same = 0
        for row_index in range(rows):
            for col_index in range(cols):
                if _cell_char(grid, row_index, col_index, color_chars) != dominant:
                    continue
                if all(
                    _cell_char(other, row_index, col_index, color_chars) == dominant
                    for other in other_grids
                ):
                    same += 1
        stable_fraction = round(same / dominant_cells, 3) if dominant_cells else None

    return {
        "color": dominant,
        "fraction": round(dominant_cells / area, 3),
        "color_counts": [
            {"color": color, "cells": cells, "fraction": round(cells / area, 3)}
            for color, cells in ranked[:max_colors]
        ],
        "components": components,
        "largest_component": largest_component,
        "stable_fraction": stable_fraction,
    }
