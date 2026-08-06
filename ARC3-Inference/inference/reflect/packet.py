"""Evidence packet for the stall critique (S2).

Built from the current level's history only, deterministically. Reuses the
debrief primitives (`describe_delta`, `to_bool`) rather than forking them.

Contains action ids, changed/unchanged bits, board *hashes* (never board
content), and the untried subset of `valid_actions` — no colours as
semantics, no game identity.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from inference.debrief.packet import to_bool
from inference.reflect.detector import NOOP_WINDOW, NOVELTY_WINDOW

MIN_REPEATS = 3
MAX_REPEATED_STATES = 3


def build_stall_packet(
    level_events: Sequence[dict[str, Any]],
    valid_actions: Sequence[str],
    *,
    level: int,
    current_plan: str = "",
) -> dict[str, Any]:
    events = list(level_events)
    window = events[-NOOP_WINDOW:]
    recent = events[-NOVELTY_WINDOW:]

    stall_window = [{"action": str(e.get("action") or "?"),
                     "changed": to_bool(e.get("board_changed"))}
                    for e in window]

    counts = Counter(str(e.get("board_hash") or "") for e in recent)
    repeated = []
    for state_hash, count in counts.most_common():
        if count < MIN_REPEATS or not state_hash:
            continue
        led_back = sorted({str(e.get("action") or "?") for e in recent
                           if str(e.get("board_hash") or "") == state_hash})
        repeated.append({"state": state_hash, "count": count,
                         "actions_leading_here": led_back})
        if len(repeated) >= MAX_REPEATED_STATES:
            break

    tried = {str(e.get("action") or "").upper() for e in events}
    untried = [a for a in valid_actions if str(a).upper() not in tried]

    last_progress: dict[str, Any] = {"action": None, "actions_ago": None}
    seen: set[str] = set()
    novel_at: list[int] = []
    for idx, event in enumerate(events, start=1):
        state_hash = str(event.get("board_hash") or "")
        if state_hash and state_hash not in seen:
            seen.add(state_hash)
            novel_at.append(idx)
    if novel_at:
        idx = novel_at[-1]
        last_progress = {"action": str(events[idx - 1].get("action") or "?"),
                         "actions_ago": len(events) - idx}

    return {
        "level": level,
        "actions_on_level": len(events),
        "stall_window": stall_window,
        "repeated_states": repeated,
        "untried_actions": untried,
        "last_progress": last_progress,
        "current_plan": (current_plan or "").strip()[:400],
    }
