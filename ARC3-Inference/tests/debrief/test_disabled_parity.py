"""Phase 0: with `debrief.enabled=false`, prompt construction is
byte-identical to `main`.

The golden is captured from the pristine upstream module (the file as it
exists on origin/main), not from a hand-written expectation, so it cannot
drift with the branch.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from inference.agent.runtime_state import Frame, HistoryEntry
from inference.agent.tool_agent import ToolAgent

REPO = Path(__file__).resolve().parents[2]
UPSTREAM_REF = "origin/main"
MODULE_PATH = "ARC3-Inference/inference/agent/tool_agent.py"


def _load_upstream_module(tmp_path: Path):
    """Import the upstream tool_agent.py under a private name."""
    source = subprocess.run(
        ["git", "show", f"{UPSTREAM_REF}:{MODULE_PATH}"],
        cwd=REPO.parent, capture_output=True, text=True, check=True).stdout
    path = tmp_path / "upstream_tool_agent.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("upstream_tool_agent", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["upstream_tool_agent"] = module
    spec.loader.exec_module(module)
    return module


def _prompt(agent, *, action_num: int = 3, level: int = 2) -> str:
    frame = Frame(grid=((0, 1), (2, 3)), step=action_num, level=level)
    history = [HistoryEntry(action="UP", frame=frame)]
    return agent._build_user_prompt(action_num, valid_actions=["UP", "DOWN"],
                                    current_frame=frame,
                                    history_entries=history)


def _agent(cls) -> object:
    return cls(model="t", base_url="http://stub.invalid/v1", provider="vllm",
               api_key="t")


def test_user_prompt_byte_identical_to_upstream_when_disabled(tmp_path):
    upstream = _load_upstream_module(tmp_path)
    ours = _prompt(_agent(ToolAgent))
    theirs = _prompt(_agent(upstream.ToolAgent))
    assert hashlib.sha256(ours.encode()).hexdigest() == \
        hashlib.sha256(theirs.encode()).hexdigest(), \
        "disabled debrief changed the user prompt"


def test_system_prompt_byte_identical_to_upstream_when_disabled(tmp_path):
    upstream = _load_upstream_module(tmp_path)
    assert _agent(ToolAgent)._system_prompt == \
        _agent(upstream.ToolAgent)._system_prompt


def test_config_ships_disabled():
    config = json.loads((REPO / "configs/inference.json").read_text())
    assert config["debrief"]["enabled"] is False
    assert config["debrief"]["include_baseline"] is False, \
        "baseline exposure is an open legal question — see NOTES.md"
