"""End-to-end checks that perception helpers survive the splice into the tool sandbox.

The sandbox bootstrap is a source string that project modules are pasted into, so a
helper can be perfectly correct in isolation and still break the sandbox at import time.
Only running real code through a sandbox subprocess catches that.
"""
import pytest

pytest.importorskip(
    "inference.agent.python_tool_sandbox",
    reason="requires the project environment (analyzer dependencies installed)",
)

from inference.agent.python_tool_sandbox import run_sandboxed_python  # noqa: E402
from inference.utils.grid_utils import ARC_COLOR_CHARS, format_grid_ascii  # noqa: E402

SIZE = 16
WHITE = ARC_COLOR_CHARS[0]
BLUE = ARC_COLOR_CHARS[9]
RED = ARC_COLOR_CHARS[8]


def _blank(size=SIZE):
    return [[0] * size for _ in range(size)]


def _fill(grid, top, left, height, width, value):
    for row in range(top, top + height):
        for col in range(left, left + width):
            grid[row][col] = value
    return grid


def _grid(block_top, block_left):
    grid = _blank()
    _fill(grid, block_top, block_left, 2, 2, 9)
    grid[0][15] = 4
    return grid


def _frame_payload(grid, step, level=1):
    return {
        "ascii": format_grid_ascii(grid),
        "step": step,
        "level": level,
        "shape": [len(grid), len(grid[0])],
        "grid": grid,
    }


def _run_history(entries, code, last_batch_steps=0):
    """Run against a hand-built action history.

    ``entries`` is a chronological list of ``(action_display, grid, level)`` triples; the
    first is the seeded frame and carries no action.
    """
    history = [
        {"action": action, "frame": _frame_payload(grid, step + 1, level)}
        for step, (action, grid, level) in enumerate(entries)
    ]
    state = {
        "current_frame": history[-1]["frame"],
        "history": history,
        "valid_actions": ["UP", "DOWN", "LEFT", "RIGHT", "MOUSE"],
        "last_action_result": {"board_changed": False},
    }

    def _reject_actions(actions):
        raise AssertionError(f"unexpected action call: {actions}")

    return run_sandboxed_python(
        code=code,
        timeout_seconds=30,
        initial_state=state,
        action_handler=_reject_actions,
        last_batch_steps=last_batch_steps,
    )


def _run(code):
    return _run_frames(_grid(4, 4), _grid(4, 7), code)


def _run_frames(before, after, code):
    state = {
        "current_frame": _frame_payload(after, 2),
        "history": [
            {"action": "", "frame": _frame_payload(before, 1)},
            {"action": "RIGHT", "frame": _frame_payload(after, 2)},
        ],
        "valid_actions": ["UP", "DOWN", "LEFT", "RIGHT"],
        "last_action_result": {"board_changed": True},
    }

    def _reject_actions(actions):
        raise AssertionError(f"unexpected action call: {actions}")

    return run_sandboxed_python(
        code=code,
        timeout_seconds=30,
        initial_state=state,
        action_handler=_reject_actions,
    )


def _run_on(grid, code):
    """Run against a single frame, with the previous frame identical to it."""
    return _run_frames([row[:] for row in grid], grid, code)


BUTTON = (12, 2)
TRIGGER = (14, 2)


def _run_acting(code):
    """Run with a live action handler over a toy game with a gated trigger.

    Clicking the button toggles it; clicking the trigger moves the block, but only while
    the button is on. That reproduces the probe shape that made a real run stall: toggle
    on, trigger, toggle back off. The move is invisible in the final action's transition
    and only shows up when the whole batch is compared.
    """
    board = {"block_col": 4, "button_on": False}
    step = {"n": 1}

    def _render():
        grid = _blank()
        _fill(grid, 4, board["block_col"], 2, 2, 9)
        if board["button_on"]:
            grid[BUTTON[0]][BUTTON[1]] = 8
        grid[TRIGGER[0]][TRIGGER[1]] = 4
        return grid

    def _apply(entry):
        if str(entry.get("action")).upper() != "MOUSE":
            return
        target = (entry.get("row"), entry.get("col"))
        if target == BUTTON:
            board["button_on"] = not board["button_on"]
        elif target == TRIGGER and board["button_on"]:
            board["block_col"] += 3

    history = [{"action": "", "frame": _frame_payload(_render(), step["n"])}]

    def _handle(actions):
        changed = False
        for entry in actions:
            before = _render()
            _apply(entry)
            grid = _render()
            changed = grid != before
            step["n"] += 1
            display = (
                f"MOUSE(row={entry.get('row')}, col={entry.get('col')})"
                if str(entry.get("action")).upper() == "MOUSE"
                else str(entry.get("action"))
            )
            history.append(
                {"action": display, "frame": _frame_payload(grid, step["n"])}
            )
        result = {"board_changed": changed}
        return {
            "action_result": result,
            "state": {
                "current_frame": history[-1]["frame"],
                "history": [dict(item) for item in history],
                "valid_actions": ["MOUSE"],
                "last_action_result": result,
            },
        }

    return run_sandboxed_python(
        code=code,
        timeout_seconds=30,
        initial_state={
            "current_frame": history[-1]["frame"],
            "history": [dict(item) for item in history],
            "valid_actions": ["MOUSE"],
            "last_action_result": {},
        },
        action_handler=_handle,
    )


def _result(outcome):
    assert outcome["error"] == "", outcome["error"]
    return outcome["result"]


def test_diff_frames_is_available_and_reports_the_move():
    outcome = _run(
        "summary = diff_frames(previous_frame, current_frame)\n"
        "result = {\n"
        "    'moved': summary['objects']['moved'],\n"
        "    'appeared_total': summary['objects']['appeared_total'],\n"
        "    'changed_cells': summary['changed_cells'],\n"
        "    'bbox': summary['bbox'],\n"
        "}\n"
    )

    assert outcome["error"] == ""
    result = outcome["result"]
    assert result["changed_cells"] == 8
    assert result["bbox"] == [4, 4, 5, 8]
    assert len(result["moved"]) == 1
    assert result["moved"][0]["delta"] == [0, 3]
    # The background reshapes around the moved block but must not be reported as an object.
    assert result["appeared_total"] == 0


def test_transition_diff_matches_explicit_diff():
    outcome = _run(
        "result = {\n"
        "    'transition': last_transition.diff()['objects']['moved'],\n"
        "    'explicit': diff_frames(previous_frame, current_frame)['objects']['moved'],\n"
        "    'cheap_has_objects': 'objects' in diff_frames(\n"
        "        previous_frame, current_frame, objects=False\n"
        "    ),\n"
        "}\n"
    )

    assert outcome["error"] == ""
    result = outcome["result"]
    assert result["transition"] == result["explicit"]
    assert result["cheap_has_objects"] is False


def test_diff_frames_returns_none_when_a_frame_is_missing():
    outcome = _run("result = {'missing': diff_frames(None, current_frame)}\n")

    assert outcome["error"] == ""
    assert outcome["result"]["missing"] is None


def test_full_diff_dump_stays_within_the_tool_output_budget():
    outcome = _run(
        "import json\n"
        "summary = diff_frames(previous_frame, current_frame)\n"
        "result = {'chars': len(json.dumps(summary))}\n"
    )

    assert outcome["error"] == ""
    # Tool responses are capped near tool_output_tokens * 4 chars (4096 by default).
    assert outcome["result"]["chars"] < 4096


def test_segmentation_nodes_carry_geometry():
    result = _result(
        _run(
            "node = [n for n in current_frame.segmentation['nodes'] if n['pixels'] == 4][0]\n"
            "result = {\n"
            "    'bbox': node['bbox'],\n"
            "    'centroid': node['centroid'],\n"
            "    'has_shape_hash': isinstance(node.get('shape_hash'), str),\n"
            "}\n"
        )
    )

    assert result["bbox"] == [4, 7, 5, 8]
    assert result["centroid"] == [4.5, 7.5]
    assert result["has_shape_hash"] is True


def test_frame_crop_renders_a_labelled_region():
    result = _result(
        _run(
            "result = {\n"
            "    'labelled': current_frame.crop([4, 7, 5, 8]),\n"
            "    'plain': current_frame.crop([4, 7, 5, 8], labels=False),\n"
            "}\n"
        )
    )

    assert result["plain"] == "bb\nbb"
    assert result["labelled"] == "  78\n4 bb\n5 bb"


def test_frame_window_is_centred_and_clipped():
    result = _result(
        _run(
            "result = {\n"
            "    'centred': current_frame.window(4, 7, radius=1, labels=False),\n"
            "    'corner': current_frame.window(0, 0, radius=1, labels=False),\n"
            "}\n"
        )
    )

    assert result["centred"] == "WWW\nWbb\nWbb"
    assert result["corner"] == "WW\nWW"


def test_find_background_reports_area_and_stability():
    result = _result(
        _run(
            "result = find_background(current_frame, other_frames=[previous_frame])\n"
        )
    )

    assert result["color"] == WHITE
    assert result["fraction"] > 0.9
    assert result["components"] == 1
    # The four cells the block vacated were not background in the earlier frame.
    assert 0.9 < result["stable_fraction"] < 1.0


def test_find_hud_flags_a_bottom_bar_and_not_the_playfield_object():
    grid = _blank()
    _fill(grid, SIZE - 1, 0, 1, 10, 9)
    _fill(grid, 6, 6, 3, 3, 8)

    result = _result(_run_on(grid, "result = find_hud(current_frame)\n"))

    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert candidate["kind"] == "bar"
    assert candidate["edge"] == "bottom"
    assert candidate["colors"] == [BLUE]
    assert result["interior_bbox"] == [0, 0, SIZE - 2, SIZE - 1]


def test_find_hud_flags_a_segmented_strip():
    grid = _blank()
    for index in range(5):
        _fill(grid, 0, index * 3, 2, 2, 9)

    result = _result(_run_on(grid, "result = find_hud(current_frame)\n"))

    assert result["candidates"][0]["kind"] == "segmented_bar"
    assert result["candidates"][0]["segments"] == 5


def test_find_symmetry_on_a_region_and_on_the_whole_frame():
    grid = _blank()
    _fill(grid, 6, 7, 1, 3, 9)
    _fill(grid, 5, 8, 3, 1, 9)

    result = _result(
        _run_on(
            grid,
            "plus = find_symmetry(current_frame, bbox=[5, 7, 7, 9])\n"
            "board = find_symmetry(current_frame)\n"
            "result = {'plus': sorted(plus['symmetric']), 'board': board['symmetric']}\n",
        )
    )

    assert result["plus"] == sorted(
        ["left_right", "top_bottom", "rotation_180", "rotation_90", "transpose", "anti_transpose"]
    )
    assert result["board"] == []


def test_path_between_routes_around_a_wall():
    grid = _blank()
    _fill(grid, 8, 0, 1, SIZE - 2, 9)

    result = _result(
        _run_on(
            grid,
            f"result = path_between(current_frame, [0, 0], [15, 0], blocked_colors=[{BLUE!r}])\n",
        )
    )

    assert result["found"] is True
    # 15 rows down, plus a 14-column detour out to the gap and back.
    assert result["steps"] == 43
    assert sum(count for _direction, count in result["runs"]) == 43
    assert result["path"][0] == [0, 0]
    assert result["path"][-1] == [15, 0]
    # The wall spans every column but the last two, and the route never steps onto it.
    assert not [cell for cell in result["path"] if cell[0] == 8 and cell[1] < SIZE - 2]


def test_path_between_reports_an_unreachable_goal():
    grid = _blank()
    _fill(grid, 8, 0, 1, SIZE, 9)

    result = _result(
        _run_on(
            grid,
            f"result = path_between(current_frame, [0, 0], [15, 0], blocked_colors=[{BLUE!r}])\n",
        )
    )

    assert result["found"] is False
    assert result["reason"] == "unreachable"


def test_track_objects_follows_the_moved_block():
    result = _result(
        _run(
            "tracked = track_objects(previous_frame, current_frame)\n"
            "result = {\n"
            "    'matched': tracked['matched'],\n"
            "    'unmatched_after_total': tracked['unmatched_after_total'],\n"
            "    'background_colors': tracked['background_colors'],\n"
            "}\n"
        )
    )

    moved = [item for item in result["matched"] if item["pixels"] == [4, 4]]
    assert moved[0]["by"] == "shape"
    assert moved[0]["delta"] == [0, 3]
    assert moved[0]["changed"] == ["moved"]
    assert result["background_colors"] == [WHITE]
    assert result["unmatched_after_total"] == 0


def test_track_objects_follows_a_recolored_object():
    before = _blank()
    _fill(before, 4, 4, 2, 2, 9)
    after = _blank()
    _fill(after, 4, 4, 2, 2, 8)

    result = _result(
        _run_frames(before, after, "result = track_objects(previous_frame, current_frame)\n")
    )

    assert result["matched"][0]["by"] == "position"
    assert result["matched"][0]["changed"] == ["color"]
    assert result["matched"][0]["colors"] == [BLUE, RED]


def test_transition_track_matches_explicit_track():
    result = _result(
        _run(
            "result = {\n"
            "    'transition': last_transition.track()['matched'],\n"
            "    'explicit': track_objects(previous_frame, current_frame)['matched'],\n"
            "}\n"
        )
    )

    assert result["transition"] == result["explicit"]


def test_helpers_return_none_when_a_frame_is_missing():
    result = _result(
        _run(
            "result = {\n"
            "    'background': find_background(None),\n"
            "    'hud': find_hud(None),\n"
            "    'symmetry': find_symmetry(None),\n"
            "    'path': path_between(None, [0, 0], [1, 1]),\n"
            "    'track': track_objects(None, current_frame),\n"
            "}\n"
        )
    )

    assert set(result) == {"background", "hud", "symmetry", "path", "track"}
    assert all(value is None for value in result.values())


def test_hud_interior_composes_with_crop_and_pathfinding():
    grid = _blank()
    _fill(grid, SIZE - 1, 0, 1, 10, 9)
    _fill(grid, 4, 0, 1, SIZE - 2, 8)

    result = _result(
        _run_on(
            grid,
            "hud = find_hud(current_frame)\n"
            "interior = hud['interior_bbox']\n"
            "route = path_between(\n"
            "    current_frame,\n"
            "    [interior[0], interior[1]],\n"
            "    [interior[2], interior[1]],\n"
            f"    blocked_colors=[{RED!r}],\n"
            ")\n"
            "result = {\n"
            "    'interior': interior,\n"
            "    'crop_lines': len(current_frame.crop(interior, labels=False).splitlines()),\n"
            "    'steps': route['steps'],\n"
            "    'crosses_hud_row': any(cell[0] == 15 for cell in route['path']),\n"
            "}\n"
        )
    )

    assert result["interior"] == [0, 0, SIZE - 2, SIZE - 1]
    assert result["crop_lines"] == SIZE - 1
    # The route stays inside the interior instead of running along the HUD row.
    assert result["crosses_hud_row"] is False
    assert result["steps"] == 42


def test_action_effects_flags_a_repeated_no_op_click_as_dead():
    grid = _blank()
    _fill(grid, 4, 4, 2, 2, 9)
    still = [("", grid, 1)] + [(f"MOUSE(row=4, col=4)", grid, 1) for _ in range(3)]

    result = _result(_run_history(still, "result = action_effects()\n"))

    assert result["considered"] == 3
    assert result["dead_click_targets"] == [f"{BLUE}:4px:2x2"]
    assert result["click_targets"][0]["attempts"] == 3
    assert result["click_targets"][0]["cells"][0] == [4, 4]
    assert result["wasted_actions"] == 3


def test_action_effects_does_not_flag_actions_that_work():
    before = _blank()
    _fill(before, 4, 4, 2, 2, 9)
    after = _blank()
    _fill(after, 4, 7, 2, 2, 9)

    result = _result(
        _run_history(
            [("", before, 1), ("RIGHT", after, 1), ("RIGHT", before, 1)],
            "result = action_effects()\n",
        )
    )

    assert result["dead_actions"] == []
    assert result["actions"][0]["changed"] == 2
    assert result["wasted_actions"] == 0


def test_action_effects_ignores_earlier_levels_by_default():
    grid = _blank()
    _fill(grid, 4, 4, 2, 2, 9)
    entries = [
        ("", grid, 1),
        ("LEFT", grid, 1),
        ("LEFT", grid, 1),
        # Level 2 starts; the old verdicts should not carry over.
        ("DOWN", grid, 2),
        ("UP", grid, 2),
    ]

    result = _result(
        _run_history(
            entries,
            "result = {\n"
            "    'scoped': action_effects(),\n"
            "    'everything': action_effects(level_only=False),\n"
            "}\n",
        )
    )

    # Only the single within-level-2 transition counts, and one attempt is not enough.
    assert result["scoped"]["considered"] == 1
    assert result["scoped"]["dead_actions"] == []
    # Without scoping, the two level-1 LEFTs are visible and LEFT reads as dead.
    assert result["everything"]["considered"] == 4
    assert result["everything"]["dead_actions"] == ["LEFT"]


def test_action_effects_groups_two_instances_of_one_object_kind():
    grid = _blank()
    _fill(grid, 3, 3, 2, 2, 9)
    _fill(grid, 9, 9, 2, 2, 9)
    entries = [
        ("", grid, 1),
        ("MOUSE(row=3, col=3)", grid, 1),
        ("MOUSE(row=9, col=9)", grid, 1),
    ]

    result = _result(_run_history(entries, "result = action_effects()\n"))

    assert result["click_targets_total"] == 1
    assert result["click_targets"][0]["attempts"] == 2
    assert result["click_targets"][0]["dead"] is True


def test_action_effects_on_an_empty_history():
    result = _result(_run_history([("", _blank(), 1)], "result = action_effects()\n"))

    assert result["considered"] == 0
    assert result["dead_actions"] == []
    assert result["wasted_actions"] == 0


_GATED_PROBE = (
    "action([\n"
    f"    {{'action': 'MOUSE', 'row': {BUTTON[0]}, 'col': {BUTTON[1]}}},\n"
    f"    {{'action': 'MOUSE', 'row': {TRIGGER[0]}, 'col': {TRIGGER[1]}}},\n"
    f"    {{'action': 'MOUSE', 'row': {BUTTON[0]}, 'col': {BUTTON[1]}}},\n"
    "])\n"
)


def test_last_batch_sees_a_move_that_the_final_action_hides():
    result = _result(
        _run_acting(
            _GATED_PROBE + "result = {\n"
            "    'batch': [move['delta'] for move in last_batch.diff()['objects']['moved']],\n"
            "    'final_action_only': [\n"
            "        move['delta'] for move in last_transition.diff()['objects']['moved']\n"
            "    ],\n"
            "}\n"
        )
    )

    assert result["batch"] == [[0, 3]]
    assert result["final_action_only"] == []


def test_last_batch_steps_attribute_the_move_to_one_action():
    result = _result(
        _run_acting(
            _GATED_PROBE + "result = {\n"
            "    'actions': len(last_batch.actions),\n"
            "    'steps': len(last_batch.steps),\n"
            "    'movers': [\n"
            "        index\n"
            "        for index, step in enumerate(last_batch.steps)\n"
            "        if step.diff()['objects']['moved']\n"
            "    ],\n"
            "}\n"
        )
    )

    assert result["actions"] == 3
    assert result["steps"] == 3
    assert result["movers"] == [1]


def test_last_batch_tracks_and_carries_the_action_result():
    result = _result(
        _run_acting(
            _GATED_PROBE + "result = {\n"
            "    'tracked': [\n"
            "        match['delta'] for match in last_batch.track()['matched']\n"
            "        if match['delta'] != [0, 0]\n"
            "    ],\n"
            "    'result_keys': sorted(last_batch.result),\n"
            "    'frame_is_current': last_batch.after_frame.step == current_frame.step,\n"
            "}\n"
        )
    )

    assert result["tracked"] == [[0, 3]]
    assert result["result_keys"] == ["board_changed"]
    assert result["frame_is_current"] is True


def test_last_batch_reflects_only_the_most_recent_call():
    result = _result(
        _run_acting(
            _GATED_PROBE
            + f"action({{'action': 'MOUSE', 'row': {TRIGGER[0]}, 'col': {TRIGGER[1]}}})\n"
            "result = {\n"
            "    'actions': len(last_batch.actions),\n"
            "    'moved': last_batch.diff()['objects']['moved'],\n"
            "}\n"
        )
    )

    assert result["actions"] == 1
    assert result["moved"] == []


def test_last_batch_is_none_before_any_action_at_all():
    result = _result(_run_acting("result = {'batch': last_batch}\n"))

    assert result["batch"] is None


def test_last_batch_is_reported_so_the_next_call_can_restore_it():
    outcome = _run_acting(_GATED_PROBE)

    assert outcome["error"] == ""
    assert outcome["last_batch_steps"] == 3


def test_last_batch_is_restored_in_a_later_call_that_does_not_act():
    """A batch from an earlier turn is still readable while inspecting it."""
    moved = _grid(4, 7)
    entries = [
        ("", _grid(4, 4), 1),
        ("MOUSE(row=12, col=2)", _grid(4, 4), 1),
        ("RIGHT", moved, 1),
        ("MOUSE(row=12, col=2)", moved, 1),
    ]

    result = _result(
        _run_history(
            entries,
            "result = {\n"
            "    'steps': len(last_batch.steps),\n"
            "    'moved': [move['delta'] for move in last_batch.diff()['objects']['moved']],\n"
            "    'final_action_only': [\n"
            "        move['delta'] for move in last_transition.diff()['objects']['moved']\n"
            "    ],\n"
            "}\n",
            last_batch_steps=3,
        )
    )

    assert result["steps"] == 3
    assert result["moved"] == [[0, 3]]
    assert result["final_action_only"] == []


def test_every_helper_dump_stays_within_the_tool_output_budget():
    grid = _blank()
    _fill(grid, SIZE - 1, 0, 1, 10, 9)
    for index in range(5):
        _fill(grid, 0, index * 3, 2, 2, 8)

    result = _result(
        _run_on(
            grid,
            "import json\n"
            "sizes = {\n"
            "    'background': find_background(current_frame, other_frames=[previous_frame]),\n"
            "    'hud': find_hud(current_frame),\n"
            "    'symmetry': find_symmetry(current_frame),\n"
            "    'track': track_objects(previous_frame, current_frame),\n"
            f"    'path': path_between(current_frame, [3, 0], [14, 15], blocked_colors=[{BLUE!r}]),\n"
            "}\n"
            "result = {name: len(json.dumps(value)) for name, value in sizes.items()}\n",
        )
    )

    assert set(result) == {"background", "hud", "symmetry", "track", "path"}
    for name, chars in result.items():
        assert chars < 4096, f"{name} dump is {chars} chars"
