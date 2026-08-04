"""Phase 4 smoke test: with compaction enabled and a stubbed endpoint, a long
scripted session triggers >=2 compactions; dropped facts survive into the
digest; the digest stays clamped; every analyzer request stays under budget.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from inference.agent.compaction import DIGEST_LABEL, CompactionConfig
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state
from inference.agent.tool_agent import ToolAgent

MARKERS = ["MARKER-ALPHA-7", "MARKER-BETA-9"]
FILLER = "long observation detail " * 3000  # ~ 18k tokens per turn


def canned_body(content: str) -> dict:
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content,
                          "reasoning_content": "thinking"},
             "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def response(body: dict) -> SimpleNamespace:
    return SimpleNamespace(status_code=200, text=json.dumps(body),
                           json=lambda: body, raise_for_status=lambda: None)


def test_compaction_smoke(tmp_path: Path, monkeypatch):
    frame = Frame(grid=((0, 1), (2, 3)), step=1, level=1)
    state_path = tmp_path / "runtime_state.json"
    write_runtime_state(state_path, current_frame=frame,
                        history=[HistoryEntry(action="", frame=frame)])

    agent = ToolAgent(model="smoke-model", base_url="http://stub.invalid/v1",
                      provider="vllm", api_key="smoke")
    agent._compaction_config = CompactionConfig(
        enabled=True,
        digest_max_tokens=700,
        compaction_call_max_tokens=900,
        compaction_timeout_s=20,
        min_dropped_tokens_to_compact=200,
    )

    analyzer_payloads: list[dict] = []
    compaction_prompts: list[str] = []
    turn = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        import json as json_module

        payload = json
        messages = payload["messages"]
        is_compaction = (
            len(messages) == 1
            and messages[0]["role"] == "user"
            and "compressed memory digest" in messages[0]["content"]
        )
        if is_compaction:
            prompt = messages[0]["content"]
            compaction_prompts.append(prompt)
            # merge stub: carry forward any marker seen in the prompt
            kept = [m for m in MARKERS if m in prompt]
            digest = "Confirmed rules:\n" + "\n".join(f"- fact {m}" for m in kept)
            return response(canned_body(digest))
        # deep-copy: build_chat_payload embeds the live message list, which
        # the agent keeps mutating after the request is sent
        analyzer_payloads.append(json_module.loads(json_module.dumps(payload)))
        return response(canned_body(
            f"turn answer with {MARKERS[0] if turn['n'] == 0 else MARKERS[1]} "
            + FILLER))

    monkeypatch.setattr("inference.agent.tool_agent.requests.post", fake_post)

    digest_tokens_per_turn: list[int] = []
    turn_initial_indexes: list[int] = []
    for action_num in range(1, 9):
        turn_initial_indexes.append(len(analyzer_payloads))
        result = agent.analyze(state_path, action_num)
        assert result is not None
        digest_tokens_per_turn.append(agent._history_digest.tokens())
        turn["n"] += 1

    stats = agent._compaction_stats
    # >= 2 compaction events actually happened
    assert stats.compaction_events >= 2, stats
    assert stats.compaction_fallbacks == 0

    # dropped early facts survived into the digest via the stubbed merge
    assert MARKERS[0] in agent._history_digest.text

    # digest grew, then stabilized within its clamp
    assert any(t > 0 for t in digest_tokens_per_turn)
    assert max(digest_tokens_per_turn) <= 700

    # the digest block reaches the model: after the first compaction, at
    # least one analyzer request carries the labeled digest with the
    # surviving marker (each turn's fresh user prompt re-injects it; deep
    # mid-turn trims may drop it within a turn, same as scientist notes)
    def user_texts(payload: dict) -> list[str]:
        return [m["content"] for m in payload["messages"]
                if m.get("role") == "user" and isinstance(m.get("content"), str)]

    digest_payloads = [
        p for p in analyzer_payloads
        if any(DIGEST_LABEL in t for t in user_texts(p))
    ]
    assert digest_payloads, "digest never reached an analyzer request"
    assert any(
        any(DIGEST_LABEL in t and MARKERS[0] in t for t in user_texts(p))
        for p in digest_payloads
    ), "surviving marker never injected alongside the digest"

    # every analyzer request, as actually posted, stayed within the budget
    # (the agent re-trims before each request in the turn loop)
    assert turn_initial_indexes[-1] < len(analyzer_payloads)
    for payload in analyzer_payloads:
        estimate = agent._estimate_request_input_tokens(
            payload["messages"], tools=payload.get("tools"))
        assert estimate <= agent._context_budget_tokens, estimate
