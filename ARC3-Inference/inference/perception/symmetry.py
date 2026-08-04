"""Mirror, rotational, and diagonal symmetry tests over a frame or region."""

from inference.perception.grid import _clamp_bbox, _dims, _region_chars


# Each mapping sends (row, col) to the cell it must equal for that symmetry to hold,
# given region height and width. Square-only mappings are marked.
_SYMMETRY_AXES = (
    ("left_right", False, lambda row, col, height, width: (row, width - 1 - col)),
    ("top_bottom", False, lambda row, col, height, width: (height - 1 - row, col)),
    ("rotation_180", False, lambda row, col, height, width: (height - 1 - row, width - 1 - col)),
    ("rotation_90", True, lambda row, col, height, width: (col, height - 1 - row)),
    ("transpose", True, lambda row, col, height, width: (col, row)),
    ("anti_transpose", True, lambda row, col, height, width: (width - 1 - col, height - 1 - row)),
)


def detect_symmetry(grid, color_chars, bbox=None, ignore_colors=()):
    """Test a frame or region for mirror, rotational, and diagonal symmetry.

    Without ``bbox`` the whole grid is tested. Pass a node's ``bbox`` to ask whether that
    one object is symmetric, which is usually the more useful question.

    ``ignore_colors`` skips cells of those colors on either side of a comparison, so a
    pattern can be checked while ignoring the background or a piece sitting on top of it.

    ``mismatches`` counts cells that disagree with their image, so one odd cell in a
    mirrored pair counts as two. ``score`` is the share of compared cells that agree, which
    separates "nearly symmetric, one cell out of place" from "not symmetric at all" -- a
    near miss is often the puzzle's goal state.

    Returns ``bbox``, ``shape``, ``cells``, ``axes`` (per axis ``exact``, ``mismatches``,
    ``compared``, ``score``, ``applicable``), and ``symmetric`` listing the axes that hold
    exactly. ``rotation_90``, ``transpose``, and ``anti_transpose`` only apply to square
    regions and are reported as ``applicable: False`` otherwise.
    """
    rows, cols = _dims(grid)
    if rows == 0 or cols == 0:
        return {"bbox": None, "shape": [0, 0], "cells": 0, "axes": {}, "symmetric": []}

    clamped = _clamp_bbox(grid, bbox if bbox is not None else [0, 0, rows - 1, cols - 1])
    if clamped is None:
        return {"bbox": None, "shape": [0, 0], "cells": 0, "axes": {}, "symmetric": []}

    region = _region_chars(grid, color_chars, clamped)
    height = len(region)
    width = len(region[0])
    ignored = set(ignore_colors)
    is_square = height == width

    axes = {}
    for name, square_only, mapping in _SYMMETRY_AXES:
        if square_only and not is_square:
            axes[name] = {
                "exact": False,
                "mismatches": None,
                "compared": 0,
                "score": None,
                "applicable": False,
            }
            continue
        compared = 0
        mismatches = 0
        for row in range(height):
            for col in range(width):
                other_row, other_col = mapping(row, col, height, width)
                if region[row][col] in ignored or region[other_row][other_col] in ignored:
                    continue
                compared += 1
                if region[row][col] != region[other_row][other_col]:
                    mismatches += 1
        axes[name] = {
            "exact": compared > 0 and mismatches == 0,
            "mismatches": mismatches,
            "compared": compared,
            "score": round((compared - mismatches) / compared, 3) if compared else None,
            "applicable": True,
        }

    return {
        "bbox": list(clamped),
        "shape": [height, width],
        "cells": height * width,
        "axes": axes,
        "symmetric": [name for name, detail in axes.items() if detail["exact"]],
    }
