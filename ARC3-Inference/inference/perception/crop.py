"""Rendering a region of a frame as ASCII, with readable coordinates."""

from inference.perception.grid import _cell_char, _clamp_bbox


def crop_ascii(grid, color_chars, bbox, labels=True):
    """Render one rectangular region of a frame as ASCII, optionally with coordinates.

    ``bbox`` is ``[min_row, min_col, max_row, max_col]`` inclusive and is clipped to the
    grid, so asking for a window that runs off an edge is safe. Returns ``""`` when the
    region lies entirely outside the grid.

    With ``labels`` the row number precedes each line and the column numbers are stacked
    above as zero-padded digits, one line per digit, so a cell's coordinates can be read
    straight off the output instead of counted.
    """
    clamped = _clamp_bbox(grid, bbox)
    if clamped is None:
        return ""
    min_row, min_col, max_row, max_col = clamped

    body = [
        [_cell_char(grid, row, col, color_chars) for col in range(min_col, max_col + 1)]
        for row in range(min_row, max_row + 1)
    ]
    if not labels:
        return "\n".join("".join(line) for line in body)

    row_width = len(str(max_row))
    col_width = len(str(max_col))
    gutter = " " * (row_width + 1)
    lines = []
    for place in range(col_width):
        digits = [str(col).rjust(col_width, "0")[place] for col in range(min_col, max_col + 1)]
        lines.append(gutter + "".join(digits))
    for offset, line in enumerate(body):
        lines.append(str(min_row + offset).rjust(row_width) + " " + "".join(line))
    return "\n".join(lines)


def window_ascii(grid, color_chars, row, col, radius=3, labels=True):
    """Render the square region within ``radius`` cells of ``(row, col)``.

    A cheap way to look at what surrounds a specific coordinate without scanning the
    whole board.
    """
    return crop_ascii(
        grid,
        color_chars,
        [int(row) - int(radius), int(col) - int(radius), int(row) + int(radius), int(col) + int(radius)],
        labels=labels,
    )
