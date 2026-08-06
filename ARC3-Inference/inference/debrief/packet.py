"""Deterministic level-debrief packet.

Assembled by the harness from its own event log at a level transition —
never from the model's memory. The existing `_summarized_knowledge`
mechanism relies on model self-report and degrades silently; this does not.

Everything here is a pure function over a list of action events. It contains
action ids, changed/unchanged bits and object-level frame deltas. It contains
no game identity, no colour semantics, and no hardcoded object meanings:
cell values are treated as opaque labels, and a "recolour" is simply "the
same cells now hold a different label".

Bool fields are normalised with ``v is True or v == "True"`` — `board_changed`
is a real bool in some artifacts and the string "True" in others, and reading
it naively has already corrupted one analysis.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

MAX_DEAD_ACTIONS = 8
MAX_STALL_SEGMENTS = 3
CLEAR_TRIGGER_ACTIONS = 3


def to_bool(v: Any) -> bool:
    return v is True or v == "True"


Grid = Sequence[Sequence[int]]


def describe_delta(before: Grid | None, after: Grid | None) -> str:
    """One-word-ish description of a frame delta, derived structurally.

    Categories: ``no_change``, ``appeared`` (more non-background cells),
    ``disappeared`` (fewer), ``moved`` (same multiset of labels, different
    positions), ``recoloured`` (same positions, different labels), ``mixed``.
    "Background" is the most common label in the *before* frame — an
    empirical majority, not a hardcoded colour.
    """
    if not before or not after:
        return "unknown"
    before_cells = {(r, c): v for r, row in enumerate(before)
                    for c, v in enumerate(row)}
    after_cells = {(r, c): v for r, row in enumerate(after)
                   for c, v in enumerate(row)}
    if before_cells == after_cells:
        return "no_change"

    background = Counter(before_cells.values()).most_common(1)[0][0]
    before_fg = {p: v for p, v in before_cells.items() if v != background}
    after_fg = {p: v for p, v in after_cells.items() if v != background}

    if len(after_fg) > len(before_fg):
        return "appeared"
    if len(after_fg) < len(before_fg):
        return "disappeared"
    if set(before_fg) == set(after_fg):
        return "recoloured"
    if Counter(before_fg.values()) == Counter(after_fg.values()):
        return "moved"
    return "mixed"


@dataclass
class ActionEffect:
    action: str
    changed: int = 0
    unchanged: int = 0
    deltas: Counter = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return self.changed + self.unchanged

    @property
    def common_delta(self) -> str:
        return self.deltas.most_common(1)[0][0] if self.deltas else "none"

    def as_dict(self) -> dict:
        return {"action": self.action, "changed": self.changed,
                "unchanged": self.unchanged, "effect": self.common_delta}


@dataclass
class DebriefPacket:
    level: int
    actions_used: int
    action_effects: list[dict]
    dead_actions: list[str]
    clear_trigger: list[dict]
    stall_segments: list[dict]
    human_multiple: float | None = None

    def as_dict(self) -> dict:
        payload = {
            "level": self.level,
            "actions_used": self.actions_used,
            "action_effects": self.action_effects,
            "dead_actions": self.dead_actions,
            "clear_trigger": self.clear_trigger,
            "stall_segments": self.stall_segments,
        }
        if self.human_multiple is not None:
            payload["human_multiple"] = round(self.human_multiple, 2)
        return payload

    def action_ids(self) -> set[str]:
        """Every action id the packet actually evidences — the validator
        refuses any rewrite that mentions an id outside this set."""
        ids = {e["action"] for e in self.action_effects}
        ids.update(self.dead_actions)
        ids.update(t["action"] for t in self.clear_trigger)
        return ids

    def render(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, indent=None)


def _action_label(event: dict) -> str:
    """Prefer the model-facing name (UP/MOUSE/...); fall back to the engine
    id. Never includes coordinates — those are board content."""
    display = str(event.get("action_display") or "").strip()
    if display:
        return display.split("(")[0].strip() or str(event.get("action_name"))
    return str(event.get("action_name") or "UNKNOWN")


def build_packet(
    events: Iterable[dict],
    *,
    level: int,
    baseline_actions: float | None = None,
    include_baseline: bool = False,
) -> DebriefPacket:
    """Build the packet for the level that has just been completed.

    ``events`` must be the action events of that level, in order, the last of
    which carries ``level_completed``. ``baseline_actions`` is only consulted
    when ``include_baseline`` is true (see NOTES.md — permission is an open
    question).
    """
    actions = list(events)
    effects: dict[str, ActionEffect] = {}
    stalls: list[dict] = []
    run_start: int | None = None
    run_len = 0
    run_preceded_by = "-"
    previous_action = "-"

    for idx, event in enumerate(actions):
        label = _action_label(event)
        changed = to_bool(event.get("board_changed"))
        effect = effects.setdefault(label, ActionEffect(action=label))
        if changed:
            effect.changed += 1
            before = actions[idx - 1].get("board") if idx else None
            effect.deltas[describe_delta(before, event.get("board"))] += 1
        else:
            effect.unchanged += 1

        # longest runs of consecutive no-change actions. ``preceded_by`` is
        # the action taken immediately BEFORE the stall began — that is the
        # context that matters; ``broken_by`` is what got the board moving.
        if not changed:
            if run_start is None:
                run_start = idx
                run_len = 0
                run_preceded_by = previous_action
            run_len += 1
        elif run_start is not None:
            stalls.append({"length": run_len, "start_action": run_start + 1,
                           "preceded_by": run_preceded_by, "broken_by": label})
            run_start, run_len = None, 0
        previous_action = label
    if run_start is not None:
        stalls.append({"length": run_len, "start_action": run_start + 1,
                       "preceded_by": run_preceded_by,
                       "broken_by": "(level end)"})

    clear_trigger = []
    for idx in range(max(0, len(actions) - CLEAR_TRIGGER_ACTIONS), len(actions)):
        event = actions[idx]
        before = actions[idx - 1].get("board") if idx else None
        clear_trigger.append({
            "action": _action_label(event),
            "effect": (describe_delta(before, event.get("board"))
                       if to_bool(event.get("board_changed")) else "no_change"),
        })

    dead = sorted(label for label, e in effects.items()
                  if e.changed == 0)[:MAX_DEAD_ACTIONS]
    ranked = sorted(effects.values(), key=lambda e: -e.total)
    human_multiple = None
    if include_baseline and baseline_actions:
        human_multiple = len(actions) / float(baseline_actions)

    return DebriefPacket(
        level=level,
        actions_used=len(actions),
        action_effects=[e.as_dict() for e in ranked],
        dead_actions=dead,
        clear_trigger=clear_trigger,
        stall_segments=sorted(stalls, key=lambda s: -s["length"])[:MAX_STALL_SEGMENTS],
        human_multiple=human_multiple,
    )


def split_levels(events: Iterable[dict]) -> list[list[dict]]:
    """Group a run's action events into per-level segments, each ending on
    the action that completed the level. A trailing, uncompleted segment is
    dropped: there is nothing to debrief about a level not cleared."""
    segments: list[list[dict]] = []
    current: list[dict] = []
    for event in events:
        if event.get("type") not in (None, "action"):
            continue
        current.append(event)
        if to_bool(event.get("level_completed")):
            segments.append(current)
            current = []
    return segments


def estimate_tokens(text: str) -> int:
    """Same coarse estimator the agent uses elsewhere (~3 chars/token)."""
    return max(1, (len(text) + 2) // 3)
