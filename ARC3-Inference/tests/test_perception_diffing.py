from inference.utils.grid_utils import ARC_COLOR_CHARS
from inference.perception import diff_grids
from inference.utils.segmentation import segment_layer


def _blank(size=8):
    return [[0] * size for _ in range(size)]


def _with_block(size, top, left, height, width, value):
    grid = _blank(size)
    for row in range(top, top + height):
        for col in range(left, left + width):
            grid[row][col] = value
    return grid


def _diff(before, after, objects=True):
    kwargs = {}
    if objects:
        kwargs = {
            "before_segmentation": segment_layer(before, ARC_COLOR_CHARS),
            "after_segmentation": segment_layer(after, ARC_COLOR_CHARS),
        }
    return diff_grids(before, after, ARC_COLOR_CHARS, **kwargs)


def test_identical_frames_report_no_change():
    grid = _with_block(8, 2, 2, 2, 2, 9)
    result = _diff(grid, grid)

    assert result["changed"] is False
    assert result["changed_cells"] == 0
    assert result["bbox"] is None
    assert result["transitions"] == []
    assert result["regions"] == []
    assert result["objects"]["moved"] == []
    assert result["objects"]["stable"] > 0


def test_single_cell_recolor_is_localized():
    before = _blank(8)
    after = _blank(8)
    after[3][4] = 9

    result = _diff(before, after, objects=False)

    assert result["changed_cells"] == 1
    assert result["bbox"] == [3, 4, 3, 4]
    assert result["transitions"] == [
        {
            "from": ARC_COLOR_CHARS[0],
            "to": ARC_COLOR_CHARS[9],
            "count": 1,
            "cells": [[3, 4]],
            "cells_truncated": False,
        }
    ]
    assert "objects" not in result


def test_translated_object_is_reported_as_moved():
    before = _with_block(10, 2, 2, 2, 2, 9)
    after = _with_block(10, 2, 5, 2, 2, 9)

    result = _diff(before, after)
    moved = result["objects"]["moved"]

    assert len(moved) == 1
    assert moved[0]["delta"] == [0, 3]
    assert moved[0]["from"] == [2, 2]
    assert moved[0]["to"] == [2, 5]
    assert moved[0]["pixels"] == 4
    assert result["objects"]["appeared"] == []
    assert result["objects"]["disappeared"] == []


def test_overlapping_move_reports_vacated_and_filled_rows_separately():
    before = _with_block(10, 2, 2, 2, 2, 9)
    after = _with_block(10, 3, 2, 2, 2, 9)

    result = _diff(before, after)

    # The blocks overlap on row 3, which is unchanged and separates the two changed rows.
    assert result["regions_total"] == 2
    assert [region["bbox"] for region in result["regions"]] == [[2, 2, 2, 3], [4, 2, 4, 3]]
    assert result["objects"]["moved"][0]["delta"] == [1, 0]


def test_shape_change_reports_appeared_and_disappeared_pair():
    before = _with_block(10, 2, 2, 2, 2, 9)
    after = _with_block(10, 2, 2, 3, 2, 9)

    result = _diff(before, after)

    assert result["objects"]["moved"] == []
    assert [item["pixels"] for item in result["objects"]["appeared"]] == [6]
    assert [item["pixels"] for item in result["objects"]["disappeared"]] == [4]


def test_new_object_appears_without_a_counterpart():
    before = _blank(10)
    after = _with_block(10, 4, 4, 2, 3, 7)

    result = _diff(before, after)

    appeared = result["objects"]["appeared"]
    assert len(appeared) == 1
    assert appeared[0]["at"] == [4, 4]
    assert appeared[0]["pixels"] == 6
    assert result["objects"]["disappeared"] == []


def test_two_identical_objects_pair_with_their_nearest_match():
    before = _blank(12)
    after = _blank(12)
    for grid, positions in ((before, ((1, 1), (1, 9))), (after, ((2, 1), (1, 8)))):
        for row, col in positions:
            grid[row][col] = 9

    result = _diff(before, after)
    moved = sorted(result["objects"]["moved"], key=lambda item: item["from"])

    assert [item["delta"] for item in moved] == [[1, 0], [0, -1]]
    assert result["objects"]["appeared"] == []
    assert result["objects"]["disappeared"] == []


def test_caps_are_reported_with_totals():
    before = _blank(12)
    after = _blank(12)
    for index in range(5):
        after[0][index * 2] = index + 1

    result = _diff(before, after, objects=False)

    # Five isolated single-cell changes sit under the default caps, so nothing is dropped.
    assert result["transitions_total"] == 5
    assert len(result["transitions"]) == 5
    assert result["regions_total"] == 5
    assert len(result["regions"]) == 5

    capped = diff_grids(
        before,
        after,
        ARC_COLOR_CHARS,
        max_transitions=2,
        max_regions=3,
    )
    assert len(capped["transitions"]) == 2
    assert len(capped["regions"]) == 3


def test_cell_sample_truncation_is_flagged():
    before = _blank(8)
    after = _blank(8)
    for col in range(6):
        after[0][col] = 9

    result = diff_grids(
        before,
        after,
        ARC_COLOR_CHARS,
        max_cells_per_transition=2,
    )
    transition = result["transitions"][0]

    assert transition["count"] == 6
    assert transition["cells"] == [[0, 0], [0, 1]]
    assert transition["cells_truncated"] is True


def test_background_is_kept_out_of_the_object_lists():
    before = _with_block(10, 2, 2, 2, 2, 9)
    after = _with_block(10, 2, 5, 2, 2, 9)

    result = _diff(before, after)
    objects = result["objects"]

    # The surrounding background reshapes around the moved block, but reporting it as a
    # 90-cell appeared/disappeared pair would bury the one thing that actually moved.
    assert objects["appeared_total"] == 0
    assert objects["disappeared_total"] == 0
    assert objects["background_reshaped"] is True
    assert objects["background_colors"] == [ARC_COLOR_CHARS[0]]


def test_background_threshold_can_be_disabled():
    before = _with_block(10, 2, 2, 2, 2, 9)
    after = _with_block(10, 2, 5, 2, 2, 9)

    result = diff_grids(
        before,
        after,
        ARC_COLOR_CHARS,
        before_segmentation=segment_layer(before, ARC_COLOR_CHARS),
        after_segmentation=segment_layer(after, ARC_COLOR_CHARS),
        background_fraction=0,
    )

    assert result["objects"]["appeared_total"] == 1
    assert result["objects"]["background_reshaped"] is False


def test_mismatched_shapes_are_reported_not_raised():
    before = [[0, 0], [0, 0]]
    after = [[0, 0, 0], [0, 0, 0]]

    result = diff_grids(before, after, ARC_COLOR_CHARS)

    assert result["shape"] == {"before": [2, 2], "after": [2, 3]}
    assert result["changed_cells"] == 2
    assert result["transitions"][0]["from"] == "-"
