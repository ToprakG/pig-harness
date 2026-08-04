from inference.utils.grid_utils import ARC_COLOR_CHARS
from inference.perception import detect_symmetry

WHITE = ARC_COLOR_CHARS[0]
BLUE = ARC_COLOR_CHARS[9]


def _grid(rows):
    """Build a grid from strings where '.' is white and '#' is blue."""
    return [[9 if char == "#" else 0 for char in row] for row in rows]


def _symmetry(rows, **kwargs):
    return detect_symmetry(_grid(rows), ARC_COLOR_CHARS, **kwargs)


def test_plus_shape_is_symmetric_on_every_axis():
    result = _symmetry([".#.", "###", ".#."])

    assert sorted(result["symmetric"]) == sorted(
        ["left_right", "top_bottom", "rotation_180", "rotation_90", "transpose", "anti_transpose"]
    )
    assert result["shape"] == [3, 3]


def test_left_right_mirror_only():
    result = _symmetry(["#.#", "#.#", "###"])

    assert result["axes"]["left_right"]["exact"] is True
    assert result["axes"]["top_bottom"]["exact"] is False
    assert result["symmetric"] == ["left_right"]


def test_top_bottom_mirror_only():
    result = _symmetry(["###", "#..", "###"])

    assert result["axes"]["top_bottom"]["exact"] is True
    assert result["axes"]["left_right"]["exact"] is False


def test_s_shape_is_only_symmetric_under_180_rotation():
    result = _symmetry(["##.", ".#.", ".##"])

    assert result["symmetric"] == ["rotation_180"]


def test_transpose_symmetry_without_mirrors():
    result = _symmetry(["##.", "#..", "..."])

    assert result["axes"]["transpose"]["exact"] is True
    assert result["axes"]["left_right"]["exact"] is False
    assert result["axes"]["top_bottom"]["exact"] is False


def test_square_only_axes_are_marked_inapplicable_on_a_rectangle():
    result = _symmetry(["#..#", "#..#"])

    assert result["axes"]["left_right"]["exact"] is True
    for name in ("rotation_90", "transpose", "anti_transpose"):
        assert result["axes"][name]["applicable"] is False
        assert result["axes"][name]["exact"] is False
        assert result["axes"][name]["score"] is None


def test_near_miss_scores_high_but_is_not_exact():
    result = _symmetry(["###", "#.#", "##."])

    left_right = result["axes"]["left_right"]

    # One corner is out of place, so its mirrored partner disagrees too: two cells.
    assert left_right["exact"] is False
    assert left_right["mismatches"] == 2
    assert left_right["compared"] == 9
    assert left_right["score"] == round(7 / 9, 3)


def test_a_completely_asymmetric_pattern_scores_low():
    result = _symmetry(["##.", "##.", "##."])

    assert result["axes"]["left_right"]["score"] < 0.5
    assert result["symmetric"] == ["top_bottom"]


def test_bbox_restricts_the_test_to_one_object():
    grid = _grid(
        [
            "......",
            ".#.#..",
            "..#...",
            ".#.#..",
            "....##",
            "......",
        ]
    )

    whole_board = detect_symmetry(grid, ARC_COLOR_CHARS)
    just_the_x = detect_symmetry(grid, ARC_COLOR_CHARS, bbox=[1, 1, 3, 3])

    assert whole_board["symmetric"] == []
    assert just_the_x["bbox"] == [1, 1, 3, 3]
    assert "left_right" in just_the_x["symmetric"]
    assert "top_bottom" in just_the_x["symmetric"]


def test_bbox_is_clipped_to_the_grid():
    result = detect_symmetry(_grid(["#.#", "###", "#.#"]), ARC_COLOR_CHARS, bbox=[-2, -2, 9, 9])

    assert result["bbox"] == [0, 0, 2, 2]


def test_ignoring_a_color_lets_symmetry_show_through():
    rows = ["###", "#.#", "#Y#"]
    grid = [[9 if char == "#" else (4 if char == "Y" else 0) for char in row] for row in rows]
    yellow = ARC_COLOR_CHARS[4]

    strict = detect_symmetry(grid, ARC_COLOR_CHARS)
    lenient = detect_symmetry(grid, ARC_COLOR_CHARS, ignore_colors=[yellow])

    assert strict["axes"]["top_bottom"]["exact"] is False
    assert lenient["axes"]["top_bottom"]["exact"] is True
    # The yellow cell and the cell it is compared against both drop out.
    assert lenient["axes"]["top_bottom"]["compared"] == 7


def test_ignoring_every_color_compares_nothing():
    result = _symmetry(["#.", ".#"], ignore_colors=[WHITE, BLUE])

    assert result["cells"] == 4
    assert result["axes"]["left_right"]["compared"] == 0
    assert result["axes"]["left_right"]["exact"] is False
    assert result["axes"]["left_right"]["score"] is None


def test_single_cell_is_trivially_symmetric():
    result = _symmetry(["#"])

    assert len(result["symmetric"]) == 6


def test_uniform_region_is_symmetric_on_every_axis():
    result = _symmetry(["####", "####", "####", "####"])

    assert len(result["symmetric"]) == 6


def test_empty_grid_is_handled():
    result = detect_symmetry([], ARC_COLOR_CHARS)

    assert result["axes"] == {}
    assert result["symmetric"] == []


def test_bbox_outside_the_grid_is_handled():
    result = detect_symmetry(_grid(["##", "##"]), ARC_COLOR_CHARS, bbox=[5, 5, 7, 7])

    assert result["bbox"] is None
    assert result["cells"] == 0
