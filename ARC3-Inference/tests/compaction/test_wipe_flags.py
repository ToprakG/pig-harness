"""C3: the split wipe conditions honor the two wipe_* flags; defaults keep
upstream behavior exactly."""

from __future__ import annotations

from inference.agent.compaction import CompactionConfig
from inference.agent.tool_agent import ToolAgent


def make_agent(**cfg_overrides) -> ToolAgent:
    agent = ToolAgent(model="t", base_url="http://stub.invalid/v1",
                      provider="vllm", api_key="t")
    agent._compaction_config = CompactionConfig(**cfg_overrides)
    agent._summarized_knowledge["world_model"] = "knows things"
    return agent


def wipe_with(summary: dict, **cfg_overrides) -> bool:
    agent = make_agent(**cfg_overrides)
    agent._last_step_summary = summary
    agent._update_summarized_knowledge_from_step_summary()
    return agent._summarized_knowledge["world_model"] == ""


def test_defaults_reproduce_upstream_wipe_behavior():
    assert wipe_with({"level_transition": True}) is True
    assert wipe_with({"game_over": True}) is True
    assert wipe_with({"run_complete": True}) is True
    assert wipe_with({"board_changed": True}) is False


def test_level_transition_wipe_can_be_disabled():
    assert wipe_with({"level_transition": True},
                     wipe_knowledge_on_level_transition=False) is False
    # other triggers unaffected
    assert wipe_with({"game_over": True},
                     wipe_knowledge_on_level_transition=False) is True


def test_reset_wipe_can_be_disabled():
    assert wipe_with({"game_over": True}, wipe_knowledge_on_reset=False) is False
    assert wipe_with({"level_transition": True},
                     wipe_knowledge_on_reset=False) is True


def test_run_complete_always_wipes():
    assert wipe_with({"run_complete": True},
                     wipe_knowledge_on_level_transition=False,
                     wipe_knowledge_on_reset=False) is True
