from inference.utils.grid_utils import ARC_COLOR_CHARS
from inference.perception import crop_ascii, window_ascii

WHITE = ARC_COLOR_CHARS[0]
BLUE = ARC_COLOR_CHARS[9]


def _blank(size=12):
    return [[0] * size for _ in range(size)]


def _grid_with_block():
    grid = _blank()
    for row in range(2, 4):
        for col in range(3, 6):
            grid[row][col] = 9
    return grid


def test_crop_without_labels_is_just_the_region():
    grid = _grid_with_block()

    assert crop_ascii(grid, ARC_COLOR_CHARS, [2, 3, 3, 5], labels=False) == "bbb\nbbb"


def test_crop_with_labels_shows_row_and_column_numbers():
    grid = _grid_with_block()

    assert crop_ascii(grid, ARC_COLOR_CHARS, [1, 2, 3, 4]) == "\n".join(
        [
            "  234",
            "1 WWW",
            "2 Wbb",
            "3 Wbb",
        ]
    )


def test_column_labels_stack_one_line_per_digit():
    grid = _blank()
    grid[9][10] = 9
    grid[10][11] = 9

    assert crop_ascii(grid, ARC_COLOR_CHARS, [9, 9, 10, 11]) == "\n".join(
        [
            "   011",
            "   901",
            " 9 WbW",
            "10 WWb",
        ]
    )


def test_crop_is_clipped_to_the_grid():
    grid = _blank(4)
    grid[0][0] = 9

    # Asking for a window that runs off the top-left edge is safe.
    assert crop_ascii(grid, ARC_COLOR_CHARS, [-3, -3, 1, 1], labels=False) == "bW\nWW"


def test_crop_entirely_outside_the_grid_is_empty():
    grid = _blank(4)

    assert crop_ascii(grid, ARC_COLOR_CHARS, [10, 10, 12, 12]) == ""
    assert crop_ascii(grid, ARC_COLOR_CHARS, [-5, -5, -2, -2]) == ""


def test_empty_grid_is_empty():
    assert crop_ascii([], ARC_COLOR_CHARS, [0, 0, 1, 1]) == ""


def test_single_cell_crop():
    grid = _grid_with_block()

    assert crop_ascii(grid, ARC_COLOR_CHARS, [2, 3, 2, 3], labels=False) == BLUE
    assert crop_ascii(grid, ARC_COLOR_CHARS, [0, 0, 0, 0], labels=False) == WHITE


def test_window_is_centred_on_the_coordinate():
    grid = _grid_with_block()

    assert window_ascii(grid, ARC_COLOR_CHARS, 3, 4, radius=1, labels=False) == "bbb\nbbb\nWWW"


def test_window_near_an_edge_is_truncated_not_shifted():
    grid = _blank(6)
    grid[0][0] = 9

    # Only the in-bounds part is returned; the centre stays at (0, 0).
    assert window_ascii(grid, ARC_COLOR_CHARS, 0, 0, radius=2, labels=False) == "bWW\nWWW\nWWW"


def test_window_labels_expose_absolute_coordinates():
    grid = _grid_with_block()

    labelled = window_ascii(grid, ARC_COLOR_CHARS, 2, 4, radius=1)

    assert labelled.splitlines()[0] == "  345"
    assert labelled.splitlines()[2].startswith("2 ")
