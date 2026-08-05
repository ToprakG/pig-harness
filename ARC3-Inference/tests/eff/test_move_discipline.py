"""Phase 3: batch cap and commit gate.

Both gates only ever *reduce* how many actions one turn may execute, never
below 1 — they must not stall the agent.
"""

from __future__ import annotations

import os

import pytest

from inference.agent.tool_agent import ToolAgent

PLAN = ("Hypothesis: the key opens the matching door.\n"
        "Next test: moving onto the key should make the door tile change.")
NO_PLAN = "Let me just try moving right and see."


def make_agent(*, cap: str = "", require_plan: bool = False) -> ToolAgent:
    env = {
        "AGENT_MAX_ACTIONS_PER_TURN": cap,
        "AGENT_REQUIRE_PLAN_BEFORE_ACT": "1" if require_plan else "0",
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        agent = ToolAgent(model="t", base_url="http://stub.invalid/v1",
                          provider="vllm", api_key="t")
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    agent._begin_turn()
    return agent


# ---- batch cap -------------------------------------------------------------

def test_no_cap_by_default_means_unlimited():
    agent = make_agent()
    assert agent.turn_action_allowance() is None


def test_cap_limits_actions_per_turn():
    agent = make_agent(cap="2")
    assert agent.turn_action_allowance() == 2
    agent.note_turn_actions_executed(2)
    assert agent.turn_action_allowance() == 1   # never below 1: no stalling


def test_cap_resets_each_turn():
    agent = make_agent(cap="2")
    agent.note_turn_actions_executed(2)
    agent._begin_turn()
    assert agent.turn_action_allowance() == 2


# ---- commit gate -----------------------------------------------------------

def test_gate_allows_one_action_without_a_plan():
    agent = make_agent(require_plan=True)
    assert agent.turn_is_gated() is True
    assert agent.turn_action_allowance() == 1


def test_gate_opens_with_hypothesis_and_expectation():
    agent = make_agent(require_plan=True)
    agent._register_turn_plan(PLAN)
    assert agent.turn_is_gated() is False
    assert agent.turn_action_allowance() is None    # gate alone, no cap


def test_partial_note_does_not_open_the_gate():
    agent = make_agent(require_plan=True)
    agent._register_turn_plan("Hypothesis: the key matters.")   # no expectation
    assert agent.turn_is_gated() is True
    agent._register_turn_plan("Next test: watch the door.")     # no hypothesis
    assert agent.turn_is_gated() is True


def test_unstructured_content_does_not_open_the_gate():
    agent = make_agent(require_plan=True)
    agent._register_turn_plan(NO_PLAN)
    assert agent.turn_is_gated() is True


def test_gate_uses_the_existing_scientist_note_parser():
    """No second parser: the same World model / Plan labels also open it."""
    agent = make_agent(require_plan=True)
    agent._register_turn_plan("World model: walls block movement.\n"
                              "Plan: step left and expect no change.")
    assert agent.turn_is_gated() is False


# ---- composition -----------------------------------------------------------

def test_gates_compose_to_the_tighter_limit():
    agent = make_agent(cap="2", require_plan=True)
    assert agent.turn_action_allowance() == 1       # gate is tighter
    agent._register_turn_plan(PLAN)
    assert agent.turn_action_allowance() == 2       # cap is now the binder


def test_allowance_never_stalls_the_agent():
    agent = make_agent(cap="1", require_plan=True)
    for _ in range(5):
        allowance = agent.turn_action_allowance()
        assert allowance >= 1
        agent.note_turn_actions_executed(allowance)


def test_withheld_bookkeeping():
    agent = make_agent(cap="2")
    agent.note_turn_actions_executed(2, withheld=3)
    assert agent._turn_withheld == 3
