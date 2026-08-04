"""The `notes` dict is the only state that survives a `python` call.

Each call runs in a fresh subprocess, so persistence is entirely a matter of the dict being
carried out of one process and back into the next. These tests cover that round trip and the
host-side budget that guards it.
"""
import json

import pytest

pytest.importorskip(
    "inference.agent.python_tool_sandbox",
    reason="requires the project environment (analyzer dependencies installed)",
)

from inference.agent.python_tool_sandbox import run_sandboxed_python  # noqa: E402
from inference.utils.grid_utils import format_grid_ascii  # noqa: E402

SIZE = 8


def _frame_payload():
    grid = [[0] * SIZE for _ in range(SIZE)]
    grid[2][2] = 9
    return {
        "ascii": format_grid_ascii(grid),
        "step": 1,
        "level": 1,
        "shape": [SIZE, SIZE],
        "grid": grid,
    }


def _run(code, notes=None):
    frame = _frame_payload()

    def _reject_actions(actions):
        raise AssertionError(f"unexpected action call: {actions}")

    return run_sandboxed_python(
        code=code,
        timeout_seconds=30,
        initial_state={
            "current_frame": frame,
            "history": [{"action": "", "frame": frame}],
            "valid_actions": ["UP"],
            "last_action_result": {},
        },
        action_handler=_reject_actions,
        notes=notes,
    )


def test_notes_start_empty_and_come_back_out():
    outcome = _run(
        "result = {'incoming': dict(notes)}\n"
        "notes['buttons'] = [[42, 21], [42, 26]]\n"
    )

    assert outcome["error"] == ""
    assert outcome["result"] == {"incoming": {}}
    assert outcome["notes"] == {"buttons": [[42, 21], [42, 26]]}


def test_notes_are_restored_on_the_next_call():
    outcome = _run(
        "result = {'trigger': notes['trigger'], 'keys': sorted(notes)}\n",
        notes={"trigger": [55, 36], "dead": ["MOUSE(row=42, col=21)"]},
    )

    assert outcome["error"] == ""
    assert outcome["result"] == {"trigger": [55, 36], "keys": ["dead", "trigger"]}


def test_notes_survive_a_traceback_later_in_the_call():
    outcome = _run(
        "notes['confirmed'] = 'trigger moves the block while the button is on'\n"
        "raise ValueError('probe blew up')\n"
    )

    assert "probe blew up" in outcome["error"]
    assert outcome["notes"] == {
        "confirmed": "trigger moves the block while the button is on"
    }


def test_notes_do_not_leak_between_unrelated_calls():
    first = _run("notes['a'] = 1\n")
    second = _run("result = dict(notes)\n")

    assert first["notes"] == {"a": 1}
    assert second["result"] == {}


def test_storing_a_live_object_is_coerced_rather_than_breaking_the_store():
    outcome = _run("notes['frame'] = current_frame\nnotes['bbox'] = (1, 2)\n")

    assert outcome["error"] == ""
    assert json.dumps(outcome["notes"])
    assert outcome["notes"]["bbox"] == [1, 2]
    assert outcome["notes"]["frame"].startswith("AsciiFrameView")
