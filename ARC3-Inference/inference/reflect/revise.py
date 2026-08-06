"""Stall-triggered revise call, its validator, and the rollback ratchet.

Five loop-engineering constraints this implements (the literature references
— Reflexion, Self-Refine, Voyager, OPRO/TextGrad/DSPy — are **from memory and
unverified**; the constraints stand on their own):

1. *Write-gating on verified outcomes* — a revision may only name action ids
   that exist in the evidence packet / `valid_actions`.
2. *Grounded critique* — the input is a deterministic trace, never the
   model's recollection.
3. *Anchoring* — revisions append to an immutable base prompt and are capped
   per level, so the loop cannot drift toward generic text.
4. *Rollback ratchet* — a revision not followed by measurable improvement is
   reverted. Every earlier design in this project lacked this.
5. *Credit assignment* — the reflect block is textually separate from the
   debrief block and independently flagged.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from inference.debrief.packet import estimate_tokens
from inference.reflect.detector import StallConfig, StallState

logger = logging.getLogger(__name__)

BLOCK_HEADER = "STALL REVISION (auto-generated after a detected stall)"
REQUIRED_SECTIONS = ("WHY STUCK", "ABANDONED", "NEW HYPOTHESIS", "NEXT PROBES")
# only these two make checkable factual/actionable claims
EVIDENCE_SECTIONS = ("WHY STUCK", "NEXT PROBES")
MAX_STACK_DEPTH = 3

_INSTRUCTION = (
    "A game-playing agent has stalled: its recent actions stopped changing "
    "the board. Below is a deterministic trace of that stall. Diagnose it and "
    "propose a different line of attack.\n\n"
    "Reply with exactly four lines in this form — write your OWN content "
    "after each label, do not copy this example, do not restate the trace, "
    "and add no other text:\n\n"
    "WHY STUCK: SPACE has not changed the board in the last 12 actions.\n"
    "ABANDONED: that SPACE toggles the target.\n"
    "NEW HYPOTHESIS: the target only reacts after MOUSE selects it.\n"
    "NEXT PROBES: MOUSE, then SPACE; if unchanged, try UP.\n"
)

_PLACEHOLDER_RE = re.compile(r"^\s*<[^>]*>\s*$")
_TRACE_KEYS = ("stall_window", "repeated_states", "untried_actions",
               "last_progress")


@dataclass
class ReviseResult:
    block: str | None
    reason: str
    latency_ms: int
    dropped_lines: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.block is not None


def build_prompt(packet: dict, current_block: str = "") -> str:
    import json

    standing = current_block.strip() or "(none yet)"
    return (f"{_INSTRUCTION}\nSTALL TRACE:\n"
            f"{json.dumps(packet, sort_keys=True)}\n\n"
            f"CURRENT REVISION BLOCK:\n{standing}\n\nREPLACEMENT:")


def parse_block(text: str) -> dict[str, str] | None:
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
    if any(s not in sections for s in REQUIRED_SECTIONS):
        return None
    if any(not v.strip() or _PLACEHOLDER_RE.match(v) for v in sections.values()):
        return None          # template echoed instead of filled
    lowered = text.lower()
    if sum(key in lowered for key in _TRACE_KEYS) >= 2:
        return None          # the trace pasted back instead of compressed
    return sections


def validate_evidence(sections: dict[str, str], packet: dict,
                      valid_actions: Sequence[str]) -> tuple[dict[str, str], list[str]]:
    """Every action id named in an evidence section must be real: present in
    `valid_actions` or in the packet's own action list."""
    known = {str(a).upper() for a in valid_actions}
    known.update(str(a).upper() for a in packet.get("untried_actions", []))
    for entry in packet.get("stall_window", []):
        known.add(str(entry.get("action", "")).upper())
    known.discard("")
    token_re = re.compile(r"\b(ACTION\d+|[A-Z][A-Z0-9_]{1,15})\b")
    stopwords = {"NONE", "AND", "OR", "THE", "NOT", "NO", "IF", "THEN", "TRY",
                 "IT", "IS", "WAS", "HAS", "DID", "DO", "ON", "OFF", "TO",
                 "OF", "IN", "LEVEL", "BOARD", "ACTIONS", "ACTION", "A", "I"}
    cleaned = dict(sections)
    dropped: list[str] = []
    for section in EVIDENCE_SECTIONS:
        text = sections.get(section, "")
        mentioned = {t for t in token_re.findall(text) if t not in stopwords}
        unknown = mentioned - known
        if unknown:
            dropped.append(f"{section}: unknown action id(s) {sorted(unknown)}")
            cleaned[section] = "(dropped: named actions not in the trace)"
    return cleaned, dropped


def render_block(sections: dict[str, str]) -> str:
    lines = [BLOCK_HEADER]
    lines.extend(f"{s}: {sections.get(s, '').strip()}" for s in REQUIRED_SECTIONS)
    return "\n".join(lines)


def revise(packet: dict, current_block: str, valid_actions: Sequence[str],
           llm_call: Callable[[str, int, float, float], str], *,
           config: StallConfig) -> ReviseResult:
    """One revise attempt. Never raises."""
    started = time.monotonic()
    try:
        text = llm_call(build_prompt(packet, current_block), config.max_tokens,
                        config.timeout_s, config.temperature)
    except Exception as exc:
        return ReviseResult(None, type(exc).__name__,
                            int((time.monotonic() - started) * 1000))
    latency = int((time.monotonic() - started) * 1000)
    sections = parse_block(text or "")
    if sections is None:
        return ReviseResult(None, "malformed_output", latency)
    sections, dropped = validate_evidence(sections, packet, valid_actions)
    if all(v.startswith("(dropped") for k, v in sections.items()
           if k in EVIDENCE_SECTIONS):
        return ReviseResult(None, "no_evidenced_content", latency, dropped)
    block = render_block(sections)
    if estimate_tokens(block) > config.max_tokens:
        return ReviseResult(None, "oversize_output", latency, dropped)
    return ReviseResult(block, "ok", latency, dropped)


# --------------------------------------------------------------- ratchet

@dataclass
class PendingRevision:
    index: int
    fired_at: int
    previous_block: str
    hashes_at_fire: set[str]


class RevisionRatchet:
    """Per-level stack of revision blocks with rollback on no improvement.

    A revision that is not followed by measurable improvement inside
    ``eval_window`` actions is reverted, its text discarded, and the budget
    consumed. Stack depth is bounded by MAX_STACK_DEPTH so rollback is O(1).
    """

    def __init__(self, config: StallConfig):
        self.config = config
        self.stack: list[str] = [""]           # [0] is always the empty base
        self.pending: PendingRevision | None = None
        self.revisions_used = 0
        self.last_fire_action = -10**9
        self.rollbacks = 0

    # -- state ------------------------------------------------------------

    @property
    def block(self) -> str:
        return self.stack[-1]

    def reset_level(self) -> None:
        """A level transition clears the reflect block: debrief owns the
        transition, and overlapping writes destroy attribution."""
        self.stack = [""]
        self.pending = None
        self.revisions_used = 0
        self.last_fire_action = -10**9
        assert self.block == "", "reflect block leaked across a level"

    # -- firing -----------------------------------------------------------

    def may_fire(self, *, action_num: int, stalled: bool) -> bool:
        cfg = self.config
        if not cfg.enabled or not stalled:
            return False
        if self.pending is not None:
            return False                      # one open evaluation at a time
        if action_num < cfg.min_actions_before_first_revise:
            return False
        if self.revisions_used >= cfg.max_revisions_per_level:
            return False
        return (action_num - self.last_fire_action) >= cfg.cooldown_actions

    def accept(self, block: str, *, action_num: int,
               hashes: set[str]) -> None:
        self.pending = PendingRevision(index=self.revisions_used + 1,
                                       fired_at=action_num,
                                       previous_block=self.block,
                                       hashes_at_fire=set(hashes))
        self.stack.append(block)
        if len(self.stack) > MAX_STACK_DEPTH:
            # keep the base plus the most recent entries
            self.stack = [self.stack[0]] + self.stack[-(MAX_STACK_DEPTH - 1):]
        self.revisions_used += 1
        self.last_fire_action = action_num

    # -- evaluation -------------------------------------------------------

    def due(self, action_num: int) -> bool:
        return (self.pending is not None
                and action_num - self.pending.fired_at >= self.config.eval_window)

    def evaluate(self, *, action_num: int, stalled: bool,
                 hashes: set[str], level_cleared: bool) -> tuple[bool, int]:
        """Returns ``(improved, revision_index)`` and rolls back if not.

        Improvement = no longer stalled, OR a board state never seen at fire
        time has appeared, OR the level was cleared.
        """
        assert self.pending is not None
        pending, self.pending = self.pending, None
        improved = (level_cleared or not stalled
                    or bool(set(hashes) - pending.hashes_at_fire))
        if not improved:
            self.stack.pop()                  # discard the rolled-back text
            if not self.stack:
                self.stack = [""]
            self.rollbacks += 1
        return improved, pending.index
