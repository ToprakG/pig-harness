"""Host-side rules for the persisted `notes` store.

The sandbox hands notes back on every call; these are the decisions the host makes about them
before the next call sees them, and about what the model is shown between turns.
"""
import pytest

pytest.importorskip(
    "inference.agent.tool_agent",
    reason="requires the project environment (analyzer dependencies installed)",
)

from inference.agent.runtime_state import Frame, HistoryEntry  # noqa: E402
from inference.agent.tool_agent import _NOTES_CHAR_LIMIT, ToolAgent  # noqa: E402


def _agent():
    return ToolAgent(model="test-model", base_url="http://127.0.0.1:9/v1", provider="local")


def test_notes_round_trip_through_the_host():
    agent = _agent()

    assert agent._store_notes({"trigger": [55, 36]}) == ""
    assert agent._notes == {"trigger": [55, 36]}


def test_a_non_dict_return_leaves_notes_untouched():
    agent = _agent()
    agent._store_notes({"kept": 1})

    assert agent._store_notes(None) == ""
    assert agent._notes == {"kept": 1}


def test_an_oversized_update_is_rejected_whole_and_explained():
    agent = _agent()
    agent._store_notes({"kept": "a fact worth keeping"})

    error = agent._store_notes({"blob": "x" * (_NOTES_CHAR_LIMIT + 100)})

    assert "over the" in error and "not saved" in error
    assert agent._notes == {"kept": "a fact worth keeping"}


def test_notes_are_rendered_into_the_carried_world_model():
    agent = _agent()
    agent._store_notes({"buttons": [[42, 21]]})

    lines = agent._summarized_knowledge_lines()

    assert any("Stored notes" in line and "[[42, 21]]" in line for line in lines)


def test_carried_block_is_empty_when_there_is_nothing_to_carry():
    assert _agent()._summarized_knowledge_lines() == []


def test_notes_alone_are_enough_to_produce_a_carried_block():
    agent = _agent()
    agent._store_notes({"trigger": [55, 36]})

    lines = agent._summarized_knowledge_lines()

    assert lines and lines[0].startswith("Working world model")


def test_a_level_change_clears_level_specific_notes_but_keeps_cross_level():
    agent = _agent()
    agent._store_notes({"buttons": [[42, 21]], "cross_level": "clicks toggle state"})
    agent._summarized_knowledge["world_model"] = "five buttons and a trigger"
    agent._last_step_summary = {"level_transition": True}

    agent._update_summarized_knowledge_from_step_summary()

    assert agent._notes == {"cross_level": "clicks toggle state"}
    assert agent._summarized_knowledge["world_model"] == ""


def test_notes_persist_across_turns_that_do_not_change_level():
    agent = _agent()
    agent._store_notes({"buttons": [[42, 21]]})
    agent._last_step_summary = {"level_transition": False, "board_changed": True}

    agent._update_summarized_knowledge_from_step_summary()

    assert agent._notes == {"buttons": [[42, 21]]}


SIZE = 8


def _frame(grid, step, level=1):
    return Frame(grid=tuple(tuple(row) for row in grid), step=step, level=level)


def _blank():
    return [[0] * SIZE for _ in range(SIZE)]


def _with_dot():
    grid = _blank()
    grid[2][2] = 8
    return grid


def _repeated_noop_history(level=1):
    """Two clicks on the same target that changed nothing, then an action that did."""
    dot = _with_dot()
    return [
        HistoryEntry(action="", frame=_frame(dot, 1, level)),
        HistoryEntry(action="MOUSE(row=2, col=2)", frame=_frame(dot, 2, level)),
        HistoryEntry(action="MOUSE(row=2, col=2)", frame=_frame(dot, 3, level)),
        HistoryEntry(action="LEFT", frame=_frame(_blank(), 4, level)),
    ]


def test_dead_actions_are_stated_without_being_asked_for():
    lines = _agent()._dead_action_lines(_repeated_noop_history(), 1)

    assert lines and lines[0].startswith("Proven no-ops")
    body = " ".join(lines)
    assert "MOUSE" in body
    assert "2 actions have already gone into those" in body


def test_an_action_that_changed_the_board_is_not_called_dead():
    lines = _agent()._dead_action_lines(_repeated_noop_history(), 1)

    assert not any("LEFT" in line for line in lines)


def test_dead_actions_are_scoped_to_the_current_level():
    history = _repeated_noop_history(level=1)

    assert _agent()._dead_action_lines(history, 2) == []


def test_an_empty_store_is_advertised_once_actions_have_started():
    agent = _agent()

    prompt = agent._build_user_prompt(3, valid_actions=["LEFT"], history_entries=[])

    assert "Stored notes are empty" in prompt


def test_the_empty_store_nudge_is_absent_before_the_first_action():
    agent = _agent()

    prompt = agent._build_user_prompt(0, valid_actions=["LEFT"], history_entries=[])

    assert "Stored notes are empty" not in prompt


def test_a_filled_store_is_shown_instead_of_the_nudge():
    agent = _agent()
    agent._store_notes({"trigger": [55, 36]})

    prompt = agent._build_user_prompt(3, valid_actions=["LEFT"], history_entries=[])

    assert "Stored notes are empty" not in prompt
    assert "Stored notes" in prompt and "55" in prompt


def test_no_no_op_line_before_there_is_evidence():
    dot = _with_dot()
    history = [HistoryEntry(action="", frame=_frame(dot, 1))]

    assert _agent()._dead_action_lines(history, 1) == []
