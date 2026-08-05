"""Phase 2: the score-awareness block.

Acceptance:
  * appears exactly once per turn, with a live, monotonically non-decreasing
    action count;
  * with the flag off, the constructed prompt is byte-identical to `main`.
"""

from __future__ import annotations

import os

import pytest

from inference.agent.runtime_state import Frame
from inference.agent.tool_agent import ToolAgent

MARKER = "SCORING: each completed level scores"


def make_agent(*, enabled: bool) -> ToolAgent:
    os.environ["AGENT_SCORE_AWARENESS_ENABLED"] = "1" if enabled else "0"
    try:
        return ToolAgent(model="t", base_url="http://stub.invalid/v1",
                         provider="vllm", api_key="t")
    finally:
        os.environ.pop("AGENT_SCORE_AWARENESS_ENABLED", None)


def build(agent: ToolAgent, *, action_num: int = 3, level: int = 1) -> str:
    frame = Frame(grid=((0, 1), (2, 3)), step=action_num, level=level)
    return agent._build_user_prompt(action_num, valid_actions=["UP", "DOWN"],
                                    current_frame=frame, history_entries=[])


def test_block_appears_exactly_once_when_enabled():
    agent = make_agent(enabled=True)
    agent.set_level_action_count(5)
    prompt = build(agent)
    assert prompt.count(MARKER) == 1


def test_block_absent_when_disabled():
    agent = make_agent(enabled=False)
    agent.set_level_action_count(5)
    assert MARKER not in build(agent)


def test_prompt_byte_identical_to_upstream_when_disabled():
    """The ONLY difference the flag may introduce is the block itself."""
    on, off = make_agent(enabled=True), make_agent(enabled=False)
    on.set_level_action_count(4)
    off.set_level_action_count(4)
    with_block, without = build(on), build(off)
    block = on._score_awareness_lines(1)[0]
    assert block in with_block and block not in without
    # removing the injected block AND its line separator restores the
    # upstream prompt exactly, byte for byte
    assert with_block.replace(block + "\n", "", 1) == without


def test_counter_is_live_and_monotonic_within_a_level():
    agent = make_agent(enabled=True)
    seen = []
    for count in (0, 1, 4, 9):
        agent.set_level_action_count(count)
        line = agent._score_awareness_lines(1)[0]
        seen.append(int(line.rsplit(":", 1)[1].strip().rstrip(".")))
    assert seen == [0, 1, 4, 9]
    assert all(b >= a for a, b in zip(seen, seen[1:]))


def test_block_reports_the_current_level():
    agent = make_agent(enabled=True)
    agent.set_level_action_count(2)
    assert "Level 3;" in agent._score_awareness_lines(3)[0]


def test_block_is_short_and_leaks_no_baseline():
    agent = make_agent(enabled=True)
    agent.set_level_action_count(11)
    line = agent._score_awareness_lines(1)[0]
    assert len(line) // 4 <= 80, "block must stay under ~80 tokens"
    # the formula names the baseline symbolically but never a numeric value
    assert "human_baseline" in line
    for forbidden in ("base_actions_per_level", "baseline=", "baseline is"):
        assert forbidden not in line


def test_block_is_generic():
    """No game identity, colours, shapes, or board content."""
    agent = make_agent(enabled=True)
    line = agent._score_awareness_lines(1)[0].lower()
    for forbidden in ("game_id", "colour", "color", "shape", "grid ="):
        assert forbidden not in line
