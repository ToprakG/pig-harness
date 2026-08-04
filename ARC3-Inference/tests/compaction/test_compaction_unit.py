"""Phase 2 unit tests for the pure compaction logic (no agent wiring)."""

from __future__ import annotations

import pytest

from inference.agent.compaction import (
    CompactionConfig,
    CompactionStats,
    DIGEST_LABEL,
    HistoryDigest,
    build_compaction_prompt,
    compact,
    render_dropped_messages,
)


def make_config(**overrides) -> CompactionConfig:
    return CompactionConfig(enabled=True, min_dropped_tokens_to_compact=10,
                            **overrides)


def big_dropped_block(marker: str = "MARKER-42") -> list[dict]:
    return [
        {"role": "user", "content": f"observation with {marker} " + "x" * 400},
        {"role": "assistant", "content": "acted", "reasoning": "thought " * 50},
        {"role": "tool", "content": [{"type": "text", "text": "tool output " * 40}]},
    ]


def test_digest_clamps_to_token_budget():
    digest = HistoryDigest(max_tokens=60)
    digest.replace("line\n" * 500)
    assert digest.tokens() <= 60
    assert not digest.is_empty()


def test_digest_render_block_is_delimited_and_empty_when_blank():
    digest = HistoryDigest()
    assert digest.render_lines() == []
    digest.replace("some knowledge")
    lines = digest.render_lines()
    assert DIGEST_LABEL in lines[0]
    assert lines[-1].startswith("=====")


def test_compact_merges_via_llm_and_counts_event():
    digest = HistoryDigest(max_tokens=700)
    digest.replace("- old rule")
    stats = CompactionStats()
    seen_prompts = []

    def stub_llm(prompt, max_tokens, timeout_s):
        seen_prompts.append((prompt, max_tokens, timeout_s))
        return "- old rule\n- new rule from MARKER-42"

    result = compact(big_dropped_block(), digest, stub_llm,
                     config=make_config(), stats=stats)
    assert "MARKER-42" in result.text
    assert stats.compaction_events == 1
    assert stats.compaction_fallbacks == 0
    prompt, max_tokens, timeout_s = seen_prompts[0]
    assert "MARKER-42" in prompt          # dropped content reached the call
    assert "- old rule" in prompt         # existing digest reached the call
    assert max_tokens == 900 and timeout_s == 20.0


def test_failing_llm_falls_back_to_unmodified_digest():
    digest = HistoryDigest()
    digest.replace("- established fact")
    stats = CompactionStats()

    def broken_llm(prompt, max_tokens, timeout_s):
        raise TimeoutError("simulated compaction timeout")

    result = compact(big_dropped_block(), digest, broken_llm,
                     config=make_config(), stats=stats)
    assert result.text == "- established fact"
    assert stats.compaction_fallbacks == 1
    assert stats.compaction_events == 0


def test_empty_llm_response_counts_as_fallback():
    digest = HistoryDigest()
    digest.replace("- fact")
    stats = CompactionStats()
    result = compact(big_dropped_block(), digest, lambda *a: "   ",
                     config=make_config(), stats=stats)
    assert result.text == "- fact"
    assert stats.compaction_fallbacks == 1


def test_small_drops_skip_the_llm_entirely():
    digest = HistoryDigest()
    stats = CompactionStats()
    calls = []
    tiny = [{"role": "user", "content": "hi"}]
    compact(tiny, digest, lambda *a: calls.append(a) or "x",
            config=CompactionConfig(min_dropped_tokens_to_compact=800),
            stats=stats)
    assert calls == []
    assert stats.compaction_events == 0 and stats.compaction_fallbacks == 0


def test_render_dropped_messages_handles_parts_and_truncates():
    text = render_dropped_messages(
        [{"role": "tool", "content": [{"type": "text", "text": "y" * 5000}]}],
        max_chars_per_message=100,
    )
    assert "chars omitted" in text and text.startswith("[tool]")


def test_prompt_carries_sections_and_budget():
    prompt = build_compaction_prompt("dropped", "- digest", 700)
    assert "under 700 tokens" in prompt
    assert "Disproven hypotheses" in prompt


def test_config_from_dict_ignores_unknown_keys():
    cfg = CompactionConfig.from_dict({"enabled": True, "bogus": 1,
                                      "digest_max_tokens": 500})
    assert cfg.enabled is True and cfg.digest_max_tokens == 500
    assert CompactionConfig.from_dict(None).enabled is False


def test_stats_summary_line():
    stats = CompactionStats(compaction_events=2, compaction_fallbacks=1,
                            compaction_wallclock_s=3.5)
    line = stats.summary_line(digest_tokens=123)
    assert "events=2" in line and "fallbacks=1" in line and "digest_tokens=123" in line


def test_agent_compaction_call_handles_thinking_models(monkeypatch):
    """Empty content + populated reasoning_content must still yield a digest,
    and the prompt must carry the /no_think soft switch."""
    import json as json_module
    from types import SimpleNamespace

    from inference.agent.tool_agent import ToolAgent

    agent = ToolAgent(model="t", base_url="http://stub.invalid/v1",
                      provider="deepinfra", api_key="t")
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen["prompt"] = json["messages"][0]["content"]
        body = {"choices": [{"message": {"role": "assistant", "content": "",
                                          "reasoning_content": "- digest from reasoning"},
                              "finish_reason": "length"}]}
        return SimpleNamespace(status_code=200, text=json_module.dumps(body),
                               json=lambda: body, raise_for_status=lambda: None)

    monkeypatch.setattr("inference.agent.tool_agent.requests.post", fake_post)
    out = agent._compaction_llm_call("summarize this", 900, 20.0)
    assert out == "- digest from reasoning"
    assert seen["prompt"].endswith("/no_think")


def test_agent_batches_small_drops_into_one_compaction(monkeypatch):
    """Drops below the threshold accumulate; crossing it fires ONE call
    carrying all pending blocks."""
    from inference.agent.compaction import CompactionConfig
    from inference.agent.tool_agent import ToolAgent

    agent = ToolAgent(model="t", base_url="http://stub.invalid/v1",
                      provider="vllm", api_key="t")
    agent._compaction_config = CompactionConfig(
        enabled=True, min_dropped_tokens_to_compact=300)
    calls = []
    monkeypatch.setattr(
        agent, "_compaction_llm_call",
        lambda prompt, mt, ts: calls.append(prompt) or "- merged digest")

    small = [{"role": "user", "content": "EARLY-FACT " + "x" * 200}]
    agent._compact_dropped(small)          # ~70 tokens: buffered, no call
    assert calls == [] and len(agent._compaction_pending) == 1

    agent._compact_dropped(
        [{"role": "user", "content": "LATE-FACT " + "y" * 800}])
    assert len(calls) == 1                  # threshold crossed: one call
    assert "EARLY-FACT" in calls[0] and "LATE-FACT" in calls[0]
    assert agent._compaction_pending == []  # buffer flushed
    assert agent._history_digest.text == "- merged digest"
