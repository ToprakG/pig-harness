"""Phase 3 parity golden: with compaction disabled (default), the agent's
request payloads are byte-identical to upstream.

The golden fixture is generated from the pre-wiring agent (Phase 2 HEAD).
Regenerate only when intentionally changing upstream prompt semantics::

    COMPACTION_PARITY_REGEN=1 uv run pytest tests/compaction/test_noop_parity.py -q
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state
from inference.agent.tool_agent import ToolAgent

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "golden_disabled_payloads.json"

# large filler makes turn 3 overflow the context budget and exercise the
# trim path while compaction is disabled
FILLER = "grid observation detail " * 4000


def canned(content: str, reasoning: str) -> SimpleNamespace:
    body = {
        "choices": [
            {"message": {"role": "assistant", "content": content,
                          "reasoning_content": reasoning},
             "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    return SimpleNamespace(status_code=200, text=json.dumps(body),
                           json=lambda: body, raise_for_status=lambda: None)


def run_scripted_session(tmp_path: Path) -> list[dict]:
    frame = Frame(grid=((0, 1), (2, 3)), step=1, level=1)
    state_path = tmp_path / "runtime_state.json"
    write_runtime_state(state_path, current_frame=frame,
                        history=[HistoryEntry(action="", frame=frame)])
    agent = ToolAgent(model="parity-model", base_url="http://stub.invalid/v1",
                      provider="vllm", api_key="parity")

    responses = [
        canned(f"turn {i} answer. {FILLER[: 60000 if i >= 2 else 200]}",
               f"turn {i} reasoning")
        for i in range(1, 5)
    ]
    posted: list[dict] = []

    import inference.agent.tool_agent as ta

    original_post = ta.requests.post

    def fake_post(url, headers=None, json=None, timeout=None):
        posted.append(json)
        return responses[min(len(posted) - 1, len(responses) - 1)]

    ta.requests.post = fake_post
    try:
        for action_num in range(1, 5):
            agent.analyze(state_path, action_num)
    finally:
        ta.requests.post = original_post

    # canonicalize for comparison
    return json.loads(json.dumps(posted, sort_keys=True))


def fingerprint(payloads: list[dict]) -> dict:
    import hashlib

    canonical = json.dumps(payloads, sort_keys=True, ensure_ascii=True)
    return {
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "n_payloads": len(payloads),
        "message_counts": [len(p["messages"]) for p in payloads],
    }


def test_disabled_payloads_match_upstream_golden(tmp_path):
    payloads = run_scripted_session(tmp_path)
    assert len(payloads) >= 4
    fp = fingerprint(payloads)
    if os.environ.get("COMPACTION_PARITY_REGEN") == "1":
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(fp, indent=2, sort_keys=True))
    golden = json.loads(GOLDEN_PATH.read_text())
    assert fp == golden, "disabled-path request payloads drifted from upstream"
