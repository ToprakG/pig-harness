"""Regression guard: the python tool sandbox must actually execute.

Phase 5 added history helpers to the sandbox bootstrap and registered them
BEFORE they were defined, so every sandbox process died with a NameError
("Sandbox process exited unexpectedly") and the agent could not act at all.
The mode-level unit tests missed it because they exercised the host-side
handler, never the sandbox itself. These tests run the real sandbox.
"""

from __future__ import annotations

from inference.agent.python_tool_sandbox import run_sandboxed_python

STATE = {
    "current_frame": {"ascii": "ab\ncd", "step": 1, "level": 1,
                       "shape": [2, 2], "segmentation": []},
    "history": [],
    "valid_actions": ["UP", "DOWN"],
    "last_action_result": {},
}


def run(code: str, history_handler=None) -> dict:
    return run_sandboxed_python(
        code=code,
        timeout_seconds=20,
        initial_state=STATE,
        action_handler=lambda actions: {"action_result": {"executed": True},
                                         "state": STATE},
        history_handler=history_handler,
    )


def test_sandbox_executes_plain_code():
    result = run("print('hello from sandbox')")
    assert not result.get("error"), result.get("error")
    assert "hello from sandbox" in result.get("stdout", "")


def test_sandbox_exposes_runtime_globals():
    result = run("print(current_frame.level, valid_actions)")
    assert not result.get("error"), result.get("error")
    assert "1" in result["stdout"] and "UP" in result["stdout"]


def test_sandbox_can_call_action():
    result = run('action({"action": "UP"})')
    assert not result.get("error"), result.get("error")
    assert result.get("action_results")


def test_history_helpers_exist_and_work_through_rpc():
    records = [{"action_num": 1, "action": "UP", "note": "MARKER"}]

    def handler(op, kwargs):
        if op == "tail":
            return records[-int(kwargs.get("n", 10)):]
        if op == "search":
            return [r for r in records
                    if kwargs.get("pattern", "") in str(r)]
        if op == "at":
            return records[0]
        if op == "stats":
            return {"records": len(records)}
        return None

    result = run(
        "print(history_stats()['records'], history_tail(1)[0]['action'], "
        "len(history_search('MARKER')), history_at(1)['note'])",
        history_handler=handler,
    )
    assert not result.get("error"), result.get("error")
    assert result["stdout"].strip() == "1 UP 1 MARKER"


def test_history_helpers_present_even_without_a_handler():
    """With no handler wired (all modes except external_history) the helpers
    still exist and return None instead of crashing the sandbox."""
    result = run("print(history_tail(3))")
    assert not result.get("error"), result.get("error")
    assert "None" in result["stdout"]
