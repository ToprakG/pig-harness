"""C2 verification: retained reasoning already works — prove it, end to end.

Turn 1: the stubbed endpoint returns a message with ``reasoning_content``.
Turn 2: the next request payload must carry that reasoning inside an
assistant history message. If this fails, the design premise is wrong —
stop and report (see the mission brief).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state
from inference.agent.tool_agent import (
    ToolAgent,
    _extract_reasoning_text,
    _normalize_message_content,
)

REASONING_TEXT = "LEVEL RULE: the key opens only the matching-color door."


def _canned_response(content: str, reasoning: str) -> SimpleNamespace:
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }
    return SimpleNamespace(
        status_code=200,
        text=json.dumps(body),
        json=lambda: body,
        raise_for_status=lambda: None,
    )


def _write_state(tmp_path: Path) -> Path:
    frame = Frame(grid=((0, 1), (2, 3)), step=1, level=1)
    state_path = tmp_path / "runtime_state.json"
    write_runtime_state(
        state_path,
        current_frame=frame,
        history=[HistoryEntry(action="", frame=frame)],
    )
    return state_path


def test_reasoning_survives_into_next_request_payload(tmp_path, monkeypatch):
    state_path = _write_state(tmp_path)
    agent = ToolAgent(model="test-model", base_url="http://stub.invalid/v1",
                      provider="vllm", api_key="test")

    posted_payloads: list[dict] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        posted_payloads.append(json)
        return _canned_response("Exploring the grid now.", REASONING_TEXT)

    monkeypatch.setattr("inference.agent.tool_agent.requests.post", fake_post)

    for action_num in (1, 2):
        result = agent.analyze(state_path, action_num)
        assert result is not None

    assert len(posted_payloads) >= 2
    second_request = posted_payloads[-1]
    assistant_history = [
        m for m in second_request["messages"] if m.get("role") == "assistant"
    ]
    assert assistant_history, "no assistant history resent at all"
    retained = [m for m in assistant_history if m.get("reasoning")]
    assert retained, "assistant history lost the reasoning field"
    assert REASONING_TEXT in retained[0]["reasoning"]


def test_think_tags_stripped_but_reasoning_field_survives():
    # the tag markup is removed (the enclosed text is retained — the parser
    # separates reasoning via reasoning_content, not by dropping tag bodies)
    content = "<think>plan text</think>visible answer"
    normalized = _normalize_message_content(content)
    assert "<think>" not in normalized and "</think>" not in normalized
    assert "visible answer" in normalized

    message = {"reasoning_content": "<think>plan A</think>plan B"}
    extracted = _extract_reasoning_text(message)
    assert "plan B" in extracted
    assert "<think>" not in extracted

    # explicit `reasoning` takes precedence over `reasoning_content`
    message = {"reasoning": "primary", "reasoning_content": "secondary"}
    assert _extract_reasoning_text(message) == "primary"


def test_history_window_keeps_reasoning_field(tmp_path):
    agent = ToolAgent(model="test-model", base_url="http://stub.invalid/v1",
                      provider="vllm", api_key="test")
    history = [
        {"role": "user", "content": "step 1"},
        {"role": "assistant", "content": "act", "reasoning": REASONING_TEXT},
        {"role": "user", "content": "step 2"},
        {"role": "assistant", "content": "act again", "reasoning": "more"},
    ]
    kept = agent._keep_recent_history_turns(history, max_turns=30)
    reasonings = [m.get("reasoning") for m in kept if m.get("role") == "assistant"]
    assert REASONING_TEXT in reasonings and "more" in reasonings
