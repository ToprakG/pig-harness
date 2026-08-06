"""The level-transition rewrite call, its parser, and its validator.

One small LLM call per level transition (~1 per successful run), turning a
deterministic debrief packet into a standing instruction block for the next
level. Everything is defensive:

* temperature <= 0.3 — this is compression, not creativity;
* timeout 90 s by default, because the previous compaction feature failed
  1488/1488 attempts with a 20 s timeout against a saturated local vLLM;
* any failure (timeout, malformed output, oversize output, hallucinated
  action id) falls back to the caller's existing block — the agent never
  stalls and never silently accepts an unevidenced claim;
* the produced block is *appended to* the base prompt by the caller, never
  substituted for it, so a degenerate rewrite cannot erase core instructions.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Callable

from inference.debrief.packet import DebriefPacket, estimate_tokens

logger = logging.getLogger(__name__)

BLOCK_HEADER = "LEVEL DEBRIEF (auto-generated from the previous level)"

REQUIRED_SECTIONS = (
    "CONFIRMED MECHANICS",
    "DEAD ACTIONS",
    "WHAT CLEARED THE LAST LEVEL",
    "FIRST THINGS TO TEST ON THIS LEVEL",
)
# Only these two sections make factual claims about action ids, so only these
# are evidence-checked; the other two are plans, not assertions.
EVIDENCE_SECTIONS = ("CONFIRMED MECHANICS", "DEAD ACTIONS")

_INSTRUCTION = (
    "You are compressing a factual trace of the level a game-playing agent "
    "just cleared into standing instructions for its NEXT level of the same "
    "game. Use ONLY facts present in the trace; do not speculate and do not "
    "mention any action id that does not appear in it. Reply with exactly "
    "these four lines, nothing else:\n"
    "CONFIRMED MECHANICS: <what actions do, as tested facts>\n"
    "DEAD ACTIONS: <action ids that never changed the board, or 'none'>\n"
    "WHAT CLEARED THE LAST LEVEL: <one line>\n"
    "FIRST THINGS TO TEST ON THIS LEVEL: <2-3 specific probes>\n"
)


@dataclass
class DebriefConfig:
    enabled: bool = False
    max_tokens: int = 400
    timeout_s: float = 90.0
    temperature: float = 0.3
    include_baseline: bool = False
    min_actions_for_debrief: int = 5

    @classmethod
    def from_dict(cls, data: dict | None) -> "DebriefConfig":
        data = dict(data or {})
        known = {k: data[k] for k in data if k in cls.__dataclass_fields__}
        config = cls(**known)
        config.temperature = min(0.3, float(config.temperature))
        return config


@dataclass
class DebriefResult:
    block: str | None
    reason: str
    latency_ms: int
    dropped_lines: list[str]

    @property
    def ok(self) -> bool:
        return self.block is not None


def build_prompt(packet: DebriefPacket, current_block: str = "") -> str:
    standing = current_block.strip() or "(none yet)"
    return (f"{_INSTRUCTION}\nTRACE OF THE LEVEL JUST CLEARED:\n"
            f"{packet.render()}\n\nCURRENT STANDING INSTRUCTIONS:\n{standing}\n\n"
            "REPLACEMENT:")


def parse_block(text: str) -> dict[str, str] | None:
    """Pull the four required sections out of the model's reply.

    Returns ``None`` when a required section is missing — that is a malformed
    rewrite and must fall back, not be half-used.
    """
    if not text or not text.strip():
        return None
    sections: dict[str, str] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        matched = next((s for s in REQUIRED_SECTIONS
                        if line.upper().startswith(s + ":")), None)
        if matched:
            current = matched
            sections[current] = line.split(":", 1)[1].strip()
        elif current:
            sections[current] = f"{sections[current]} {line}".strip()
    if any(section not in sections for section in REQUIRED_SECTIONS):
        return None
    return sections


def validate_evidence(sections: dict[str, str],
                      packet: DebriefPacket) -> tuple[dict[str, str], list[str]]:
    """Monotone evidence rule: a factual section may only name action ids that
    appear in the packet. Offending lines are dropped, not silently kept."""
    known = {a.upper() for a in packet.action_ids()}
    # Candidate ids are tokens that are ALREADY uppercase in the model's
    # reply — matching against an uppercased copy would make every ordinary
    # word ("recolours") look like an action id.
    token_re = re.compile(r"\b(ACTION\d+|[A-Z][A-Z0-9_]{1,15})\b")
    stopwords = {"NONE", "AND", "OR", "THE", "NOT", "NO", "ALL", "ANY", "IT",
                 "IS", "WAS", "DID", "DO", "ON", "OFF", "TO", "OF", "IN",
                 "LEVEL", "BOARD", "ACTIONS", "ACTION", "N", "A"}
    cleaned = dict(sections)
    dropped: list[str] = []
    for section in EVIDENCE_SECTIONS:
        text = sections.get(section, "")
        mentioned = {t for t in token_re.findall(text) if t not in stopwords}
        unevidenced = mentioned - known
        if unevidenced:
            dropped.append(f"{section}: unevidenced action id(s) "
                           f"{sorted(unevidenced)}")
            cleaned[section] = ("(dropped: referenced action ids not present "
                                "in the trace)")
    return cleaned, dropped


def render_block(sections: dict[str, str]) -> str:
    lines = [BLOCK_HEADER]
    lines.extend(f"{section}: {sections.get(section, '').strip()}"
                 for section in REQUIRED_SECTIONS)
    return "\n".join(lines)


def rewrite(
    packet: DebriefPacket,
    current_block: str,
    llm_call: Callable[[str, int, float, float], str],
    *,
    config: DebriefConfig,
) -> DebriefResult:
    """One rewrite attempt. Never raises; a failure returns ``block=None``
    and the caller keeps whatever it had."""
    started = time.monotonic()
    prompt = build_prompt(packet, current_block)
    try:
        text = llm_call(prompt, config.max_tokens, config.timeout_s,
                        config.temperature)
    except Exception as exc:
        latency = int((time.monotonic() - started) * 1000)
        logger.warning("debrief rewrite failed: %s: %s", type(exc).__name__, exc)
        return DebriefResult(None, type(exc).__name__, latency, [])
    latency = int((time.monotonic() - started) * 1000)

    sections = parse_block(text or "")
    if sections is None:
        return DebriefResult(None, "malformed_output", latency, [])

    sections, dropped = validate_evidence(sections, packet)
    block = render_block(sections)
    if estimate_tokens(block) > config.max_tokens:
        return DebriefResult(None, "oversize_output", latency, dropped)
    return DebriefResult(block, "ok", latency, dropped)
