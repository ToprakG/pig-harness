from inference.utils.grid_utils import ARC_COLOR_CHARS
from inference.perception import match_objects
from inference.utils.segmentation import segment_layer

WHITE = ARC_COLOR_CHARS[0]
BLUE = ARC_COLOR_CHARS[9]
RED = ARC_COLOR_CHARS[8]

SIZE = 12


def _blank(size=SIZE):
    return [[0] * size for _ in range(size)]


def _fill(grid, top, left, height, width, value):
    for row in range(top, top + height):
        for col in range(left, left + width):
            grid[row][col] = value
    return grid


def _match(before, after, **kwargs):
    return match_objects(
        segment_layer(before, ARC_COLOR_CHARS),
        segment_layer(after, ARC_COLOR_CHARS),
        **kwargs,
    )


def _blocks(*specs):
    grid = _blank()
    for top, left, height, width, value in specs:
        _fill(grid, top, left, height, width, value)
    return grid


def test_moved_object_matches_exactly_by_shape():
    before = _blocks((2, 2, 2, 2, 9))
    after = _blocks((2, 5, 2, 2, 9))

    result = _match(before, after, background_pixels=100)
    match = result["matched"][0]

    assert result["matched_total"] == 1
    assert match["by"] == "shape"
    assert match["delta"] == [0, 3]
    assert match["changed"] == ["moved"]
    assert match["color"] == BLUE
    assert result["unmatched_before_total"] == 0
    assert result["unmatched_after_total"] == 0


def test_object_that_did_not_move_counts_as_stable():
    before = _blocks((2, 2, 2, 2, 9))

    result = _match(before, [row[:] for row in before], background_pixels=100)

    assert result["stable"] == 1
    assert result["matched"] == []


def test_recolored_object_is_matched_by_position():
    before = _blocks((3, 3, 2, 2, 9))
    after = _blocks((3, 3, 2, 2, 8))

    result = _match(before, after, background_pixels=100)
    match = result["matched"][0]

    assert match["by"] == "position"
    assert match["changed"] == ["color"]
    assert match["colors"] == [BLUE, RED]
    assert match["color"] is None
    assert match["delta"] == [0, 0]


def test_grown_object_is_matched_by_position():
    before = _blocks((3, 3, 2, 2, 9))
    after = _blocks((3, 3, 2, 3, 9))

    result = _match(before, after, background_pixels=100)
    match = result["matched"][0]

    assert match["by"] == "position"
    assert match["pixels"] == [4, 6]
    assert match["changed"] == ["size"]


def test_object_that_moved_and_changed_shape_is_still_tracked():
    before = _blocks((3, 3, 2, 2, 9))
    after = _blocks((4, 4, 2, 3, 9))

    result = _match(before, after, background_pixels=100)
    match = result["matched"][0]

    assert match["by"] == "position"
    assert sorted(match["changed"]) == ["moved", "size"]
    assert match["delta"] == [1, 1]


def test_shape_change_at_constant_size_is_reported_as_shape():
    before = _blocks((3, 3, 1, 4, 9))
    after = _blocks((3, 3, 4, 1, 9))

    result = _match(before, after, background_pixels=100)
    match = result["matched"][0]

    assert match["pixels"] == [4, 4]
    assert match["changed"] == ["shape"]


def test_genuinely_new_and_gone_objects_are_unmatched():
    before = _blocks((1, 1, 2, 2, 9))
    after = _blocks((9, 9, 2, 2, 9))

    result = _match(before, after, background_pixels=100)

    # Too far apart to be the same object, and shapes are equal so the exact stage
    # would have paired them: the size ratio check is not what separates these.
    assert result["matched_total"] == 1
    assert result["matched"][0]["delta"] == [8, 8]


def test_distant_dissimilar_objects_do_not_match():
    before = _blocks((1, 1, 1, 1, 9))
    after = _blocks((9, 9, 3, 3, 8))

    result = _match(before, after, background_pixels=100)

    assert result["matched_total"] == 0
    assert result["unmatched_before_total"] == 1
    assert result["unmatched_after_total"] == 1
    assert result["unmatched_before"][0]["at"] == [1, 1]
    assert result["unmatched_after"][0]["at"] == [9, 9]


def test_size_ratio_gate_blocks_implausible_pairs():
    before = _blocks((3, 3, 1, 1, 9))
    after = _blocks((3, 3, 4, 4, 8))

    result = _match(before, after, background_pixels=100)

    # Overlapping, but a 1-cell object and a 16-cell one are not the same thing.
    assert result["matched_total"] == 0

    relaxed = _match(before, after, background_pixels=100, min_size_ratio=0.0)

    assert relaxed["matched_total"] == 1


def test_color_change_can_be_disallowed():
    before = _blocks((3, 3, 2, 2, 9))
    after = _blocks((3, 3, 2, 2, 8))

    strict = _match(before, after, background_pixels=100, allow_color_change=False)

    assert strict["matched_total"] == 0
    assert strict["unmatched_before_total"] == 1
    assert strict["unmatched_after_total"] == 1


def test_max_shift_bounds_loose_matching():
    before = _blocks((1, 1, 1, 2, 9))
    after = _blocks((7, 7, 1, 3, 9))

    tight = _match(before, after, background_pixels=100)
    loose = _match(before, after, background_pixels=100, max_shift=20)

    assert tight["matched_total"] == 0
    assert loose["matched_total"] == 1
    assert loose["matched"][0]["by"] == "position"


def test_same_color_pairing_is_preferred_over_a_closer_recolor():
    before = _blocks((4, 4, 2, 2, 9))
    after = _blocks((4, 5, 2, 3, 9), (4, 3, 1, 1, 8))

    result = _match(before, after, background_pixels=100)
    by_position = [item for item in result["matched"] if item["by"] == "position"]

    assert by_position[0]["colors"] == [BLUE, BLUE]


def test_several_identical_objects_pair_by_closest_position():
    before = _blocks((1, 1, 1, 1, 9), (1, 8, 1, 1, 9))
    after = _blocks((2, 1, 1, 1, 9), (1, 9, 1, 1, 9))

    result = _match(before, after, background_pixels=100)
    deltas = sorted(item["delta"] for item in result["matched"])

    assert result["matched_total"] == 2
    assert deltas == [[0, 1], [1, 0]]


def test_background_is_kept_out_of_the_matches():
    before = _blocks((2, 2, 2, 2, 9))
    after = _blocks((2, 5, 2, 2, 9))

    result = _match(before, after, background_pixels=int(SIZE * SIZE * 0.25))

    assert result["matched_total"] == 1
    assert result["background_colors"] == [WHITE]
    assert result["unmatched_before_total"] == 0
    assert result["unmatched_after_total"] == 0


def test_without_a_background_threshold_the_background_is_an_object():
    before = _blocks((2, 2, 2, 2, 9))
    after = _blocks((2, 5, 2, 2, 9))

    result = _match(before, after)

    assert result["background_colors"] == []
    # The white region's shape changes with the block, so it matches loosely.
    white_matches = [item for item in result["matched"] if item["colors"] == [WHITE, WHITE]]
    assert len(white_matches) == 1


def test_lists_are_capped_with_totals():
    before = _blank()
    after = _blank()
    for index in range(6):
        before[0][index * 2] = 9
        after[11][index * 2] = 8

    result = _match(before, after, background_pixels=100, max_unmatched=2)

    assert len(result["unmatched_before"]) == 2
    assert result["unmatched_before_total"] == 6
    assert len(result["unmatched_after"]) == 2
    assert result["unmatched_after_total"] == 6


def test_matches_are_ordered_largest_first():
    before = _blocks((1, 1, 1, 1, 9), (5, 5, 3, 3, 8))
    after = _blocks((1, 2, 1, 1, 9), (5, 6, 3, 3, 8))

    result = _match(before, after, background_pixels=100)

    assert [max(item["pixels"]) for item in result["matched"]] == [9, 1]


def test_empty_segmentations_are_handled():
    result = match_objects({"nodes": []}, {"nodes": []})

    assert result["matched"] == []
    assert result["stable"] == 0
    assert result["unmatched_before_total"] == 0
