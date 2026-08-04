from inference.utils.grid_utils import ARC_COLOR_CHARS
from inference.perception import identify_background
from inference.utils.segmentation import segment_layer

WHITE = ARC_COLOR_CHARS[0]
BLUE = ARC_COLOR_CHARS[9]
RED = ARC_COLOR_CHARS[8]


def _blank(size=10, value=0):
    return [[value] * size for _ in range(size)]


def _with_block(size, top, left, height, width, value, base=0):
    grid = _blank(size, base)
    for row in range(top, top + height):
        for col in range(left, left + width):
            grid[row][col] = value
    return grid


def _identify(grid, **kwargs):
    return identify_background(
        grid,
        ARC_COLOR_CHARS,
        segmentation=segment_layer(grid, ARC_COLOR_CHARS),
        **kwargs,
    )


def test_dominant_color_is_reported_with_its_share():
    grid = _with_block(10, 2, 2, 2, 2, 9)

    result = _identify(grid)

    assert result["color"] == WHITE
    assert result["fraction"] == round(96 / 100, 3)
    assert result["components"] == 1
    assert result["largest_component"]["pixels"] == 96
    assert result["largest_component"]["bbox"] == [0, 0, 9, 9]


def test_color_counts_are_ranked_largest_first():
    grid = _blank(10)
    for col in range(10):
        grid[0][col] = 9
    for col in range(4):
        grid[1][col] = 8

    result = _identify(grid)
    counts = result["color_counts"]

    assert [entry["color"] for entry in counts[:3]] == [WHITE, BLUE, RED]
    assert [entry["cells"] for entry in counts[:3]] == [86, 10, 4]
    assert sum(entry["fraction"] for entry in counts) == 1.0


def test_non_white_background_is_detected():
    grid = _with_block(10, 4, 4, 2, 2, 0, base=9)

    result = _identify(grid)

    assert result["color"] == BLUE
    assert result["fraction"] == round(96 / 100, 3)


def test_fragmented_dominant_color_reports_many_components():
    grid = _blank(9, 9)
    # A white grid line splits the board into nine separate blue cells.
    for index in range(9):
        grid[index][3] = 0
        grid[index][6] = 0
        grid[3][index] = 0
        grid[6][index] = 0

    result = _identify(grid)

    assert result["color"] == BLUE
    assert result["components"] == 9
    # No single component dominates, which is the signal that area alone is misleading.
    assert result["largest_component"]["fraction"] < 0.2


def test_stability_is_reported_across_frames():
    first = _with_block(10, 2, 2, 2, 2, 9)
    second = _with_block(10, 2, 5, 2, 2, 9)

    result = _identify(first, other_grids=[second])

    # The four cells the block moves into stop being background in the later frame.
    assert result["stable_fraction"] == round(92 / 96, 3)


def test_unstable_dominant_color_scores_low():
    first = _blank(10)
    second = _blank(10)
    for row in range(10):
        for col in range(5):
            second[row][col] = 9

    result = _identify(first, other_grids=[second])

    assert result["stable_fraction"] == 0.5


def test_stability_is_none_without_other_frames():
    result = _identify(_blank(6))

    assert result["stable_fraction"] is None


def test_component_details_need_segmentation():
    grid = _with_block(10, 2, 2, 2, 2, 9)

    result = identify_background(grid, ARC_COLOR_CHARS)

    assert result["color"] == WHITE
    assert result["components"] is None
    assert result["largest_component"] is None


def test_empty_grid_is_handled():
    result = identify_background([], ARC_COLOR_CHARS)

    assert result["color"] is None
    assert result["fraction"] == 0.0
    assert result["color_counts"] == []


def test_ties_resolve_deterministically():
    grid = _blank(2)
    grid[0][0] = 9
    grid[0][1] = 9

    first = _identify(grid)
    second = _identify([row[:] for row in grid])

    assert first["color"] == second["color"]
    assert first["fraction"] == 0.5
