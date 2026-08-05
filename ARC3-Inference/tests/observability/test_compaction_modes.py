"""Phase 5: async compaction and external-history modes."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from inference.agent.compaction import (
    AsyncCompactor,
    CompactionConfig,
    CompactionStats,
    HistoryDigest,
)
from inference.agent.external_history import ExternalHistory
from inference.agent.tool_agent import ToolAgent

BLOCK = [{"role": "user", "content": "EARLY-FACT " + "x" * 3000}]


# ---- Option A: async compaction -------------------------------------------

def make_compactor(llm_call, **cfg) -> tuple[AsyncCompactor, HistoryDigest,
                                              CompactionStats]:
    digest = HistoryDigest(max_tokens=700)
    stats = CompactionStats()
    config = CompactionConfig(enabled=True, mode="async",
                              min_dropped_tokens_to_compact=10,
                              compaction_timeout_s=60, **cfg)
    return (AsyncCompactor(digest, llm_call, config=config, stats=stats,
                           context="game-x"), digest, stats)


def test_async_submit_returns_immediately_and_swaps_in_later():
    def slow_llm(prompt, max_tokens, timeout_s):
        time.sleep(0.3)
        return "- rule learned from EARLY-FACT"

    compactor, digest, stats = make_compactor(slow_llm)
    started = time.monotonic()
    assert compactor.submit(BLOCK) is True
    assert time.monotonic() - started < 0.1      # did NOT block the caller
    assert digest.is_empty()                     # nothing swapped in yet
    assert compactor.poll() is False             # still running

    compactor._thread.join(timeout=5)
    assert compactor.poll() is True              # result swapped in
    assert "EARLY-FACT" in digest.text
    assert stats.compaction_events == 1 and stats.compaction_fallbacks == 0


def test_async_failure_keeps_previous_digest_and_never_raises():
    digest_text = "- previously established"

    def broken_llm(prompt, max_tokens, timeout_s):
        raise TimeoutError("simulated read timeout")

    compactor, digest, stats = make_compactor(broken_llm)
    digest.replace(digest_text)
    assert compactor.submit(BLOCK) is True
    compactor._thread.join(timeout=5)
    assert compactor.poll() is False
    assert digest.text == digest_text            # previous digest preserved
    assert stats.compaction_fallbacks == 1 and stats.compaction_events == 0


def test_async_single_flight():
    release = threading.Event()

    def blocking_llm(prompt, max_tokens, timeout_s):
        release.wait(timeout=5)
        return "- done"

    compactor, _digest, _stats = make_compactor(blocking_llm)
    assert compactor.submit(BLOCK) is True
    assert compactor.submit(BLOCK) is False      # one call in flight only
    release.set()
    compactor._thread.join(timeout=5)


def test_agent_async_mode_does_not_block_and_requeues(monkeypatch):
    agent = ToolAgent(model="t", base_url="http://stub.invalid/v1",
                      provider="vllm", api_key="t")
    agent._compaction_config = CompactionConfig(
        enabled=True, mode="async", min_dropped_tokens_to_compact=10,
        compaction_timeout_s=60)
    release = threading.Event()
    monkeypatch.setattr(agent, "_compaction_llm_call",
                        lambda *a: (release.wait(timeout=5), "- digest")[1])
    agent._compact_dropped(list(BLOCK))
    assert agent._compactor is not None and agent._compactor.busy
    agent._compact_dropped(list(BLOCK))          # busy: block is requeued
    assert agent._compaction_pending, "dropped block was lost while busy"
    release.set()
    agent._compactor._thread.join(timeout=5)


# ---- Option B: external history --------------------------------------------

def test_external_history_roundtrip(tmp_path: Path):
    store = ExternalHistory(tmp_path / "h.jsonl")
    for i in range(1, 26):
        store.append({"action_num": i, "level": 1, "action": "UP",
                      "board_changed": i % 2 == 0,
                      "board_ascii": ("KEY" if i == 7 else "plain") * 3})

    assert store.stats()["records"] == 25
    assert store.at(7)["action_num"] == 7
    assert store.at(999) is None
    tail = store.tail(5)
    assert [r["action_num"] for r in tail] == [21, 22, 23, 24, 25]
    hits = store.search("KEY")
    assert len(hits) == 1 and hits[0]["action_num"] == 7
    assert store.search("KEY", last_n=5) == []   # window excludes the hit


def test_external_history_clips_large_records(tmp_path: Path):
    store = ExternalHistory(tmp_path / "h.jsonl", max_chars_per_record=200)
    store.append({"action_num": 1, "board_ascii": "z" * 5000})
    record = store.at(1)
    assert "chars omitted" in record["board_ascii"]


def test_external_history_invalid_regex_falls_back_to_substring(tmp_path: Path):
    store = ExternalHistory(tmp_path / "h.jsonl")
    store.append({"action_num": 1, "note": "a[b unbalanced"})
    assert len(store.search("a[b")) == 1


def test_agent_external_history_mode_skips_summarisation(monkeypatch, tmp_path):
    agent = ToolAgent(model="t", base_url="http://stub.invalid/v1",
                      provider="vllm", api_key="t")
    agent._compaction_config = CompactionConfig(
        enabled=True, mode="external_history", min_dropped_tokens_to_compact=10)
    agent._session_runtime_dir = tmp_path
    agent._log_context = "game-y"
    calls = []
    monkeypatch.setattr(agent, "_compaction_llm_call",
                        lambda *a: calls.append(a) or "x")

    agent._compact_dropped(list(BLOCK))
    assert calls == []                            # no summarisation at all

    agent.record_history_event({"action_num": 3, "action": "LEFT",
                                "board_ascii": "MARKER"})
    assert agent._history_handler("stats", {})["records"] == 1
    assert agent._history_handler("at", {"action_num": 3})["action"] == "LEFT"
    assert len(agent._history_handler("search", {"pattern": "MARKER"})) == 1
    assert agent._history_handler("tail", {"n": 1})[0]["action_num"] == 3
    assert agent._history_handler("bogus_op", {}) is None


def test_history_helpers_are_generic():
    """Guard the mission constraint: no game/colour/shape conditioning in
    executable code (prose stating the constraint is fine)."""
    import ast
    import inspect

    from inference.agent import external_history

    tree = ast.parse(inspect.getsource(external_history))
    docstrings = {
        ast.get_docstring(n)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef))
    }
    docstrings.discard(None)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            haystack = node.id
        elif isinstance(node, ast.Attribute):
            haystack = node.attr
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in docstrings:
                continue  # prose stating the constraint, not behavior
            haystack = node.value
        else:
            continue
        for forbidden in ("game_id", "colour", "shape"):
            assert forbidden not in haystack.lower(), \
                f"{forbidden} in {haystack!r}"
