from inference.utils.grid_utils import ARC_COLOR_CHARS
from inference.utils.segmentation import segment_layer


def _blank(size=8):
    return [[0] * size for _ in range(size)]


def _nodes(grid):
    return segment_layer(grid, ARC_COLOR_CHARS)["nodes"]


def _node_of_color(grid, value):
    wanted = ARC_COLOR_CHARS[value]
    matches = [node for node in _nodes(grid) if node["color"] == wanted]
    assert len(matches) == 1, f"expected exactly one {wanted} component, got {len(matches)}"
    return matches[0]


def test_rectangle_bbox_and_centroid():
    grid = _blank()
    for row in range(2, 5):
        for col in range(3, 7):
            grid[row][col] = 9

    node = _node_of_color(grid, 9)

    assert node["bbox"] == [2, 3, 4, 6]
    assert node["centroid"] == [3.0, 4.5]
    assert node["pixels"] == 12


def test_single_cell_bbox_is_the_cell_itself():
    grid = _blank()
    grid[5][6] = 7

    node = _node_of_color(grid, 7)

    assert node["bbox"] == [5, 6, 5, 6]
    assert node["centroid"] == [5.0, 6.0]


def test_bbox_covers_full_extent_of_an_l_shape():
    grid = _blank()
    for row in range(1, 5):
        grid[row][1] = 9
    for col in range(1, 6):
        grid[4][col] = 9

    node = _node_of_color(grid, 9)

    assert node["bbox"] == [1, 1, 4, 5]


def test_ring_centroid_can_fall_on_a_non_member_cell():
    grid = _blank()
    for col in range(2, 5):
        grid[2][col] = 9
        grid[4][col] = 9
    grid[3][2] = 9
    grid[3][4] = 9

    node = _node_of_color(grid, 9)

    # The hole's centre is the centroid, which is why it is documented as a comparison
    # position rather than a guaranteed in-object click target.
    assert node["bbox"] == [2, 2, 4, 4]
    assert node["centroid"] == [3.0, 3.0]
    assert grid[3][3] == 0


def test_bbox_matches_boundary_extent_for_every_component():
    grid = _blank(10)
    for row in range(1, 4):
        grid[row][1] = 9
    grid[6][6] = 4
    for col in range(2, 9):
        grid[8][col] = 7

    for node in _nodes(grid):
        rows = [point[0] for point in node["boundary"]]
        cols = [point[1] for point in node["boundary"]]
        assert node["bbox"] == [min(rows), min(cols), max(rows), max(cols)]


def test_same_shape_in_different_colors_shares_a_shape_hash():
    first = _blank()
    _fill = [(2, 2), (2, 3), (3, 2)]
    for row, col in _fill:
        first[row][col] = 9
    second = _blank()
    for row, col in _fill:
        second[row][col] = 8

    blue = _node_of_color(first, 9)
    red = _node_of_color(second, 8)

    assert blue["shape_hash"] == red["shape_hash"]
    assert blue["hash"] != red["hash"]


def test_shape_hash_ignores_position_but_not_shape():
    grid = _blank(12)
    for row, col in [(1, 1), (1, 2), (2, 1)]:
        grid[row][col] = 9
    for row, col in [(7, 7), (7, 8), (8, 7)]:
        grid[row][col] = 8
    for row, col in [(4, 9), (5, 9), (5, 10)]:
        grid[row][col] = 3

    corner, moved, rotated = (
        _node_of_color(grid, 9),
        _node_of_color(grid, 8),
        _node_of_color(grid, 3),
    )

    assert corner["shape_hash"] == moved["shape_hash"]
    assert corner["shape_hash"] != rotated["shape_hash"]


def test_centroid_lies_within_bbox_for_every_component():
    grid = _blank(10)
    for row in range(2, 6):
        for col in range(2, 8):
            grid[row][col] = 3
    grid[4][4] = 9

    for node in _nodes(grid):
        min_row, min_col, max_row, max_col = node["bbox"]
        assert min_row <= node["centroid"][0] <= max_row
        assert min_col <= node["centroid"][1] <= max_col
