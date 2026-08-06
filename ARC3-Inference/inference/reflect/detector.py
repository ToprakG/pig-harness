"""Deterministic stall detector.

Two signals over the live trajectory, no model involved:

    noop_20    = fraction of the last 20 actions that did not change the board
    novelty_30 = distinct board hashes among the last 30 frames, over the
                 window length
    stalled    = noop_20 >= noop_threshold or novelty_30 <= novelty_threshold

Both come from the same evidence as the previously fitted discrete-time
hazard model (`noop_20` coefficient -1.78, `novelty_30` +2.01, AUC 0.76), so
they are not an artefact of one parameterisation.

The detector sees only change bits, board hashes and action ids — never
colours as semantics, never board content, never game identity.

`board_changed` is a real bool in some artifacts and the string "True" in
others; `to_bool` is used everywhere.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

NOOP_WINDOW = 20
NOVELTY_WINDOW = 30


def to_bool(v) -> bool:
    return v is True or v == "True"


def board_hash(board) -> str:
    """Short stable digest of a frame. Board content is hashed, never kept."""
    return hashlib.md5(repr(board).encode("utf-8")).hexdigest()[:8]


@dataclass
class StallConfig:
    enabled: bool = False
    noop_threshold: float = 0.25
    novelty_threshold: float = 0.5
    min_actions_before_first_revise: int = 20
    cooldown_actions: int = 25
    max_revisions_per_level: int = 2
    eval_window: int = 20
    max_tokens: int = 250
    timeout_s: float = 90.0
    temperature: float = 0.3

    @classmethod
    def from_dict(cls, data: dict | None) -> "StallConfig":
        data = dict(data or {})
        known = {k: data[k] for k in data if k in cls.__dataclass_fields__}
        config = cls(**known)
        config.temperature = min(0.3, float(config.temperature))
        return config


@dataclass
class StallState:
    noop_20: float
    novelty_30: float
    stalled: bool
    window: int


def stall_state(ring_changed: Sequence[bool] | deque,
                ring_hashes: Sequence[str] | deque,
                cfg: StallConfig) -> StallState:
    """Pure function over the two trailing rings."""
    changed = list(ring_changed)[-NOOP_WINDOW:]
    hashes = list(ring_hashes)[-NOVELTY_WINDOW:]
    noop = (1.0 - sum(1 for c in changed if c) / len(changed)) if changed else 0.0
    novelty = (len(set(hashes)) / len(hashes)) if hashes else 1.0
    stalled = (noop >= cfg.noop_threshold) or (novelty <= cfg.novelty_threshold)
    return StallState(noop_20=noop, novelty_30=novelty, stalled=stalled,
                      window=len(changed))


def first_stall_index(
    events: Iterable[dict],
    cfg: StallConfig,
    *,
    min_window: int = NOOP_WINDOW,
) -> int | None:
    """Index (1-based action number) of the first stall in a trajectory, or
    ``None``. Evaluated only once both windows have enough history, so an
    opening burst of no-ops cannot trigger it spuriously."""
    changed: deque = deque(maxlen=NOOP_WINDOW)
    hashes: deque = deque(maxlen=NOVELTY_WINDOW)
    for idx, event in enumerate(events, start=1):
        changed.append(to_bool(event.get("board_changed")))
        hashes.append(event.get("board_hash")
                      or board_hash(event.get("board_ascii")
                                    or event.get("board")))
        if len(changed) < min_window:
            continue
        if stall_state(changed, hashes, cfg).stalled:
            return idx
    return None
