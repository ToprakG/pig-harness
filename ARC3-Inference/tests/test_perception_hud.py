from inference.utils.grid_utils import ARC_COLOR_CHARS
from inference.perception import detect_hud
from inference.utils.segmentation import segment_layer

BLUE = ARC_COLOR_CHARS[9]
RED = ARC_COLOR_CHARS[8]

SIZE = 32


def _blank(size=SIZE):
    return [[0] * size for _ in range(size)]


def _detect(grid, **kwargs):
    return detect_hud(grid, ARC_COLOR_CHARS, segment_layer(grid, ARC_COLOR_CHARS), **kwargs)


def _fill(grid, top, left, height, width, value):
    for row in range(top, top + height):
        for col in range(left, left + width):
            grid[row][col] = value


def test_partially_filled_bottom_bar_is_detected():
    grid = _blank()
    _fill(grid, SIZE - 1, 0, 1, 20, 9)

    result = _detect(grid)
    candidates = result["candidates"]

    assert len(candidates) == 1
    assert candidates[0]["kind"] == "bar"
    assert candidates[0]["edge"] == "bottom"
    assert candidates[0]["span_fraction"] == round(20 / SIZE, 3)
    assert candidates[0]["colors"] == [BLUE]


def test_segmented_strip_along_the_top_is_detected():
    grid = _blank()
    for index in range(10):
        _fill(grid, 0, index * 3, 2, 2, 9)

    result = _detect(grid)
    segmented = [item for item in result["candidates"] if item["kind"] == "segmented_bar"]

    assert len(segmented) == 1
    assert segmented[0]["edge"] == "top"
    assert segmented[0]["segments"] == 10
    assert segmented[0]["cells"] == 40


def test_gameplay_object_in_the_middle_is_not_flagged():
    grid = _blank()
    _fill(grid, 14, 14, 3, 3, 9)

    assert _detect(grid)["candidates"] == []


def test_long_line_away_from_any_edge_is_not_flagged():
    grid = _blank()
    _fill(grid, SIZE // 2, 0, 1, SIZE, 9)

    # Spans the full width, but it is not near an edge, so it is board content.
    assert _detect(grid)["candidates"] == []


def test_short_edge_object_is_not_flagged():
    grid = _blank()
    _fill(grid, 0, 0, 1, 4, 9)

    # Touches the top edge but spans only 12% of it.
    assert _detect(grid)["candidates"] == []


def test_few_segments_do_not_make_a_segmented_bar():
    grid = _blank()
    for index in range(2):
        _fill(grid, 0, index * 20, 2, 2, 9)

    assert _detect(grid)["candidates"] == []


def test_left_edge_vertical_bar_is_detected():
    grid = _blank()
    _fill(grid, 0, 0, 24, 1, 9)

    result = _detect(grid)

    assert result["candidates"][0]["edge"] == "left"
    assert result["candidates"][0]["kind"] == "bar"


def test_interior_bbox_trims_flagged_bands():
    grid = _blank()
    _fill(grid, SIZE - 1, 0, 1, 24, 9)
    _fill(grid, 0, 0, 2, 24, 8)

    result = _detect(grid)
    edges = {item["edge"] for item in result["candidates"]}

    assert edges == {"top", "bottom"}
    assert result["interior_bbox"] == [2, 0, SIZE - 2, SIZE - 1]


def test_interior_bbox_is_whole_board_without_candidates():
    grid = _blank()
    _fill(grid, 10, 10, 2, 2, 9)

    assert _detect(grid)["interior_bbox"] == [0, 0, SIZE - 1, SIZE - 1]


def test_background_color_is_excluded_from_candidates():
    grid = [[9] * SIZE for _ in range(SIZE)]
    _fill(grid, SIZE - 1, 0, 1, 20, 0)

    result = _detect(grid)

    # Blue is now the background, so the white bar is the element reported.
    assert [item["colors"] for item in result["candidates"]] == [[ARC_COLOR_CHARS[0]]]


def test_background_color_can_be_overridden():
    grid = _blank()
    _fill(grid, SIZE - 1, 0, 1, 20, 9)

    result = _detect(grid, background_color=BLUE)

    # Declaring blue as background hides the bar made of it.
    assert result["candidates"] == []


def test_thresholds_are_tunable():
    grid = _blank()
    _fill(grid, 0, 0, 1, 4, 9)

    relaxed = _detect(grid, min_span_fraction=0.1)

    assert relaxed["candidates"][0]["kind"] == "bar"
    assert relaxed["candidates"][0]["span_fraction"] == round(4 / SIZE, 3)


def test_thick_edge_block_is_not_treated_as_a_bar():
    grid = _blank()
    _fill(grid, 0, 0, 8, 20, 9)

    # 8 cells thick is a region, not a bar.
    assert _detect(grid)["candidates"] == []


def test_two_color_progress_bar_reports_both_segment_groups():
    grid = _blank()
    for index in range(6):
        _fill(grid, SIZE - 2, index * 3, 2, 2, 9)
    for index in range(6, 10):
        _fill(grid, SIZE - 2, index * 3, 2, 2, 8)

    result = _detect(grid)
    segmented = sorted(
        (item for item in result["candidates"] if item["kind"] == "segmented_bar"),
        key=lambda item: item["segments"],
    )

    assert [item["colors"] for item in segmented] == [[RED], [BLUE]]
    assert [item["segments"] for item in segmented] == [4, 6]


def test_empty_grid_is_handled():
    result = detect_hud([], ARC_COLOR_CHARS, {"nodes": []})

    assert result == {"candidates": [], "interior_bbox": None}
