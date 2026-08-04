"""Detecting HUD elements -- timer bars and segmented strips -- along board edges."""

from inference.perception.grid import _color_counts, _dims
from inference.perception.objects import _node_bbox


_EDGES = ("top", "bottom", "left", "right")


def _in_edge_band(bbox, edge, rows, cols, thickness):
    """Whether a node's bbox sits wholly within ``thickness`` cells of one edge."""
    min_row, min_col, max_row, max_col = bbox
    if edge == "top":
        return max_row <= thickness - 1
    if edge == "bottom":
        return min_row >= rows - thickness
    if edge == "left":
        return max_col <= thickness - 1
    return min_col >= cols - thickness


def _along_axis(bbox, edge):
    """Extent along the edge, and thickness across it."""
    min_row, min_col, max_row, max_col = bbox
    if edge in ("top", "bottom"):
        return (min_col, max_col), max_row - min_row + 1
    return (min_row, max_row), max_col - min_col + 1


def detect_hud(
    grid,
    color_chars,
    segmentation,
    background_color=None,
    max_thickness=3,
    min_span_fraction=0.3,
    min_segments=3,
    max_candidates=4,
):
    """Find likely HUD elements: timer bars and segmented strips along the board edges.

    Two shapes are recognised, both confined to within ``max_thickness`` cells of an edge
    and spanning at least ``min_span_fraction`` of that edge:

    - ``bar``: one long thin component, the usual timer or progress bar.
    - ``segmented_bar``: at least ``min_segments`` identically shaped components in a row,
      the strip of small blocks that is easy to mistake for clickable puzzle pieces.

    These are candidates, not certainties. Confirm by checking that actions change only
    the candidate and leave the interior untouched. ``interior_bbox`` gives the board area
    with the flagged bands trimmed off, which is the region gameplay actually happens in.

    ``background_color`` defaults to the most common color, which is excluded so the
    unfilled part of a bar is not reported as its own element.

    Returns ``{"candidates": [...], "interior_bbox": [...] | None}`` where each candidate
    has ``kind``, ``edge``, ``bbox``, ``cells``, ``colors``, ``segments``, and
    ``span_fraction``.
    """
    rows, cols = _dims(grid)
    if rows == 0 or cols == 0:
        return {"candidates": [], "interior_bbox": None}

    if background_color is None:
        counts = _color_counts(grid, color_chars)
        background_color = max(sorted(counts.items()), key=lambda item: item[1])[0] if counts else None

    nodes = []
    for node in (segmentation or {}).get("nodes", []):
        if node.get("color") == background_color:
            continue
        bbox = _node_bbox(node)
        if bbox is None:
            continue
        nodes.append((node, bbox))

    candidates = []
    for edge in _EDGES:
        edge_length = cols if edge in ("top", "bottom") else rows
        minimum_span = min_span_fraction * edge_length
        in_band = [(node, bbox) for node, bbox in nodes if _in_edge_band(bbox, edge, rows, cols, max_thickness)]

        for node, bbox in in_band:
            (start, end), thickness = _along_axis(bbox, edge)
            span = end - start + 1
            if thickness <= max_thickness and span >= minimum_span:
                candidates.append(
                    {
                        "kind": "bar",
                        "edge": edge,
                        "bbox": list(bbox),
                        "cells": int(node.get("pixels", 0)),
                        "colors": [node.get("color")],
                        "segments": 1,
                        "span_fraction": round(span / edge_length, 3),
                    }
                )

        grouped = {}
        for node, bbox in in_band:
            grouped.setdefault(node.get("hash"), []).append((node, bbox))
        for members in grouped.values():
            if len(members) < min_segments:
                continue
            starts = []
            ends = []
            box = [rows, cols, -1, -1]
            for node, bbox in members:
                (start, end), _thickness = _along_axis(bbox, edge)
                starts.append(start)
                ends.append(end)
                box = [
                    min(box[0], bbox[0]),
                    min(box[1], bbox[1]),
                    max(box[2], bbox[2]),
                    max(box[3], bbox[3]),
                ]
            span = max(ends) - min(starts) + 1
            if span < minimum_span:
                continue
            candidates.append(
                {
                    "kind": "segmented_bar",
                    "edge": edge,
                    "bbox": box,
                    "cells": sum(int(node.get("pixels", 0)) for node, _bbox in members),
                    "colors": sorted({node.get("color") for node, _bbox in members}),
                    "segments": len(members),
                    "span_fraction": round(span / edge_length, 3),
                }
            )

    candidates.sort(key=lambda item: (-item["span_fraction"], -item["cells"], item["edge"]))
    candidates = candidates[:max_candidates]

    interior = [0, 0, rows - 1, cols - 1]
    for candidate in candidates:
        min_row, min_col, max_row, max_col = candidate["bbox"]
        if candidate["edge"] == "top":
            interior[0] = max(interior[0], max_row + 1)
        elif candidate["edge"] == "bottom":
            interior[2] = min(interior[2], min_row - 1)
        elif candidate["edge"] == "left":
            interior[1] = max(interior[1], max_col + 1)
        else:
            interior[3] = min(interior[3], min_col - 1)

    interior_bbox = interior if interior[0] <= interior[2] and interior[1] <= interior[3] else None
    return {"candidates": candidates, "interior_bbox": interior_bbox}
