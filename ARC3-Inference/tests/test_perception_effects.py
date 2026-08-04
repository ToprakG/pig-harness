from inference.perception import summarize_action_effects
from inference.utils.grid_utils import ARC_COLOR_CHARS

BLUE = ARC_COLOR_CHARS[9]
RED = ARC_COLOR_CHARS[8]
SIZE = 12


def _blank(size=SIZE):
    return [[0] * size for _ in range(size)]


def _with_block(top, left, height=2, width=2, value=9):
    grid = _blank()
    for row in range(top, top + height):
        for col in range(left, left + width):
            grid[row][col] = value
    return grid


def _click(row, col):
    return f"MOUSE(row={row}, col={col})"


def _summarize(records, **kwargs):
    return summarize_action_effects(records, ARC_COLOR_CHARS, **kwargs)


def test_action_that_never_changes_anything_is_dead():
    still = _blank()
    records = [("LEFT", still, [row[:] for row in still]) for _ in range(3)]

    result = _summarize(records)

    assert result["actions"][0]["attempts"] == 3
    assert result["actions"][0]["changed"] == 0
    assert result["actions"][0]["dead"] is True
    assert result["dead_actions"] == ["LEFT"]
    assert result["wasted_actions"] == 3


def test_action_that_works_is_not_dead():
    records = [("LEFT", _with_block(2, 2), _with_block(2, 3)) for _ in range(3)]

    result = _summarize(records)

    assert result["actions"][0]["changed"] == 3
    assert result["actions"][0]["dead"] is False
    assert result["dead_actions"] == []
    assert result["wasted_actions"] == 0


def test_one_success_rescues_an_action_from_being_dead():
    still = _blank()
    records = [
        ("UP", still, [row[:] for row in still]),
        ("UP", still, [row[:] for row in still]),
        ("UP", _with_block(2, 2), _with_block(2, 3)),
    ]

    result = _summarize(records)

    assert result["actions"][0]["attempts"] == 3
    assert result["actions"][0]["changed"] == 1
    assert result["actions"][0]["dead"] is False


def test_single_attempt_is_not_enough_to_call_it_dead():
    still = _blank()

    result = _summarize([("RIGHT", still, [row[:] for row in still])])

    assert result["actions"][0]["dead"] is False
    assert result["dead_actions"] == []


def test_min_attempts_is_tunable():
    still = _blank()

    result = _summarize([("RIGHT", still, [row[:] for row in still])], min_attempts=1)

    assert result["dead_actions"] == ["RIGHT"]


def test_clicks_are_grouped_by_what_was_under_the_cursor():
    grid = _with_block(3, 3)
    records = [(_click(3, 3), grid, [row[:] for row in grid]) for _ in range(4)]

    result = _summarize(records)
    target = result["click_targets"][0]

    assert target["color"] == BLUE
    assert target["pixels"] == 4
    assert target["shape"] == [2, 2]
    assert target["target"] == f"{BLUE}:4px:2x2"
    assert target["attempts"] == 4
    assert target["dead"] is True
    assert result["dead_click_targets"] == [f"{BLUE}:4px:2x2"]


def test_two_instances_of_the_same_kind_share_one_verdict():
    grid = _blank()
    for row in range(2, 4):
        for col in range(2, 4):
            grid[row][col] = 9
    for row in range(8, 10):
        for col in range(8, 10):
            grid[row][col] = 9

    records = [
        (_click(2, 2), grid, [row[:] for row in grid]),
        (_click(8, 8), grid, [row[:] for row in grid]),
    ]

    result = _summarize(records)

    # Same color, size, and shape: one tally, so the second click is not a fresh start.
    assert result["click_targets_total"] == 1
    assert result["click_targets"][0]["attempts"] == 2
    assert result["click_targets"][0]["dead"] is True


def test_different_object_kinds_are_tracked_separately():
    grid = _blank()
    for row in range(2, 4):
        for col in range(2, 4):
            grid[row][col] = 9
    grid[8][8] = 8

    changed = [row[:] for row in grid]
    changed[0][0] = 3

    records = [
        (_click(2, 2), grid, [row[:] for row in grid]),
        (_click(2, 2), grid, [row[:] for row in grid]),
        (_click(8, 8), grid, changed),
    ]

    result = _summarize(records)
    by_target = {entry["target"]: entry for entry in result["click_targets"]}

    assert by_target[f"{BLUE}:4px:2x2"]["dead"] is True
    assert by_target[f"{RED}:1px:1x1"]["dead"] is False
    assert result["dead_click_targets"] == [f"{BLUE}:4px:2x2"]


def test_large_background_clicks_are_reported_as_a_region():
    grid = _blank()
    records = [(_click(6, 6), grid, [row[:] for row in grid]) for _ in range(2)]

    result = _summarize(records, max_region_cells=20)
    target = result["click_targets"][0]

    # The flood fill stops early instead of walking the whole board.
    assert target["target"].endswith(":region")
    assert target["pixels"] is None
    assert target["dead"] is True


def test_mouse_is_aggregated_as_one_action_entry():
    grid = _with_block(3, 3)
    records = [
        (_click(3, 3), grid, [row[:] for row in grid]),
        (_click(9, 9), grid, [row[:] for row in grid]),
    ]

    result = _summarize(records)

    assert [entry["action"] for entry in result["actions"]] == ["MOUSE"]
    assert result["actions"][0]["attempts"] == 2


def test_sample_cells_are_recorded_and_capped():
    grid = _with_block(3, 3)
    records = [(_click(3, 3), grid, [row[:] for row in grid]) for _ in range(9)]

    result = _summarize(records, max_sample_cells=3)

    assert result["click_targets"][0]["cells"] == [[3, 3], [3, 3], [3, 3]]
    assert result["click_targets"][0]["attempts"] == 9


def test_records_without_a_before_frame_are_skipped():
    grid = _blank()

    result = _summarize([("LEFT", None, grid), ("LEFT", grid, [row[:] for row in grid])])

    assert result["considered"] == 1
    assert result["actions"][0]["attempts"] == 1


def test_lists_are_capped_with_totals():
    grid = _blank()
    for index in range(6):
        grid[index * 2][0] = index + 3
    records = [(_click(index * 2, 0), grid, [row[:] for row in grid]) for index in range(6)]

    result = _summarize(records, max_targets=2)

    assert len(result["click_targets"]) == 2
    assert result["click_targets_total"] == 6


def test_wasted_actions_counts_dead_clicks_without_double_counting():
    grid = _with_block(3, 3)
    changed = [row[:] for row in grid]
    changed[0][0] = 3
    records = [
        # Clicking the block does nothing, three times.
        (_click(3, 3), grid, [row[:] for row in grid]),
        (_click(3, 3), grid, [row[:] for row in grid]),
        (_click(3, 3), grid, [row[:] for row in grid]),
        # A click elsewhere works, so MOUSE overall is alive.
        (_click(9, 9), grid, changed),
    ]

    result = _summarize(records)

    assert result["actions"][0]["dead"] is False
    # Only the three dead-target clicks count as waste, not all four MOUSE actions.
    assert result["wasted_actions"] == 3


def test_dead_mouse_overall_is_not_double_counted_with_its_targets():
    grid = _with_block(3, 3)
    records = [(_click(3, 3), grid, [row[:] for row in grid]) for _ in range(3)]

    result = _summarize(records)

    # MOUSE and its single target are both dead; the waste is 3 actions, not 6.
    assert result["actions"][0]["dead"] is True
    assert result["dead_click_targets"]
    assert result["wasted_actions"] == 3


def test_empty_history_is_handled():
    result = _summarize([])

    assert result["considered"] == 0
    assert result["actions"] == []
    assert result["dead_actions"] == []
    assert result["wasted_actions"] == 0


def test_malformed_mouse_display_still_counts_as_an_action():
    grid = _blank()

    result = _summarize([("MOUSE(garbled)", grid, [row[:] for row in grid])] * 2)

    assert result["actions"][0]["action"] == "MOUSE"
    assert result["actions"][0]["dead"] is True
    # No target could be identified, so nothing is attributed to one.
    assert result["click_targets"] == []
