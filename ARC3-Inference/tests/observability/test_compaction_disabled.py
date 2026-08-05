"""Phase 1: with compaction.enabled=false, the compaction client is never
invoked on any of the three truncation paths."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference.agent.compaction import CompactionConfig
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state
from inference.agent.tool_agent import ToolAgent

BIG = "word " * 12000    # ~20k tokens (coarse blocks: trim undershoots)
SMALL = "word " * 1200   # ~2k tokens (fine blocks: overflow retry can shave)


def make_agent(monkeypatch) -> tuple[ToolAgent, list]:
    agent = ToolAgent(model="t", base_url="http://stub.invalid/v1",
                      provider="vllm", api_key="t")
    agent._compaction_config = CompactionConfig(enabled=False,
                                                min_dropped_tokens_to_compact=10)
    calls: list = []
    monkeypatch.setattr(
        agent, "_compaction_llm_call",
        lambda *a, **k: calls.append(a) or "should never happen")
    return agent, calls


def oversized_messages() -> list[dict]:
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(4):
        msgs.append({"role": "user", "content": f"u{i} " + BIG})
        msgs.append({"role": "assistant", "content": f"a{i} " + BIG})
    msgs.append({"role": "user", "content": "final"})
    return msgs


def test_trim_path_never_calls_compaction_client(monkeypatch):
    agent, calls = make_agent(monkeypatch)
    out = agent._trim_messages_for_context(oversized_messages())
    assert len(out) < 10          # trimming DID happen (upstream behavior)
    assert calls == []            # but the compaction client was never touched


def test_force_reduce_path_never_calls_compaction_client(monkeypatch):
    agent, calls = make_agent(monkeypatch)
    out = agent._force_reduce_messages(oversized_messages())
    assert len(out) < 10
    assert calls == []


def test_overflow_recovery_path_never_calls_compaction_client(tmp_path, monkeypatch):
    """The context_overflow_recovered path routes through the two functions
    above; drive it end-to-end via analyze() with a server that rejects the
    first request as too long."""
    agent, calls = make_agent(monkeypatch)
    frame = Frame(grid=((0, 1), (2, 3)), step=1, level=1)
    state_path = tmp_path / "runtime_state.json"
    write_runtime_state(state_path, current_frame=frame,
                        history=[HistoryEntry(action="", frame=frame)])
    # preload MANY small history blocks: the turn-start trim then lands just
    # under the local budget with several blocks left, so the server-side
    # rejection can still shave more and actually retry. Mark the session
    # dir first or _ensure_session wipes the preload.
    agent._session_runtime_dir = state_path.parent
    history = []
    for i in range(30):
        history.append({"role": "user", "content": f"u{i} " + SMALL})
        history.append({"role": "assistant", "content": f"a{i} " + SMALL})
    agent._history_messages = history

    responses = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        responses["n"] += 1
        if responses["n"] == 1:
            body = {"error": {"message": "maximum context length exceeded"}}
            text = __import__("json").dumps(body)
            resp = SimpleNamespace(status_code=400, text=text, json=lambda: body)
            def boom():
                import requests
                raise requests.HTTPError("400 maximum context length exceeded",
                                         response=resp)
            resp.raise_for_status = boom
            return resp
        body = {"choices": [{"message": {"role": "assistant",
                                          "content": "ok"},
                              "finish_reason": "stop"}]}
        text = __import__("json").dumps(body)
        return SimpleNamespace(status_code=200, text=text, json=lambda: body,
                               raise_for_status=lambda: None)

    monkeypatch.setattr("inference.agent.tool_agent.requests.post", fake_post)
    result = agent.analyze(state_path, 1)
    assert result is not None
    assert responses["n"] >= 2    # overflow retry actually happened
    assert calls == []            # and still no compaction call
