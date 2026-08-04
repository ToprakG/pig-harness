"""Shared grid primitives used across the perception helpers.

Reading cells, measuring dimensions, and clipping regions -- the operations every
other module in this package builds on.
"""


# 8-connected offsets: changed cells that touch diagonally are treated as one region,
# so a shape that shifted by one cell reads as a single region instead of two.
_DIAG = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

_MISSING_CHAR = "-"


def _dims(grid):
    rows = len(grid)
    cols = max((len(row) for row in grid), default=0)
    return rows, cols


def _cell_char(grid, row, col, color_chars):
    """Color symbol at ``(row, col)``, or ``_MISSING_CHAR`` when outside the grid."""
    if row < 0 or row >= len(grid):
        return _MISSING_CHAR
    line = grid[row]
    if col < 0 or col >= len(line):
        return _MISSING_CHAR
    return color_chars[max(0, min(15, int(line[col])))]


def _clamp_bbox(grid, bbox):
    """Intersect ``bbox`` with the grid, or ``None`` when they do not overlap."""
    rows, cols = _dims(grid)
    if rows == 0 or cols == 0:
        return None
    min_row = max(0, int(bbox[0]))
    min_col = max(0, int(bbox[1]))
    max_row = min(rows - 1, int(bbox[2]))
    max_col = min(cols - 1, int(bbox[3]))
    if min_row > max_row or min_col > max_col:
        return None
    return (min_row, min_col, max_row, max_col)


def _color_counts(grid, color_chars):
    counts = {}
    for row in grid:
        for value in row:
            char = color_chars[max(0, min(15, int(value)))]
            counts[char] = counts.get(char, 0) + 1
    return counts


def _region_chars(grid, color_chars, bbox):
    return [
        [_cell_char(grid, row, col, color_chars) for col in range(bbox[1], bbox[3] + 1)]
        for row in range(bbox[0], bbox[2] + 1)
    ]
