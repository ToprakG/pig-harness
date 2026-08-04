"""Lightweight per-session trajectory tracker for the live restart policy.

Stdlib-only (imported by the solver). Board CONTENT is never stored: each
action contributes one changed/unchanged bit to a 20-slot ring buffer and one
8-character hash to a 30-slot ring buffer; the raw grid is discarded
immediately after hashing.
"""

from __future__ import annotations

import hashlib
from collections import deque

from inference.meta.policy import MetaConfig, RestartController


def grid_digest(grid) -> str:
    """Short stable digest of a grid (nested int sequences)."""
    return hashlib.md5(repr(grid).encode("utf-8")).hexdigest()[:8]


class MetaRuntime:
    """Ring buffers + counters backing RestartController's inputs."""

    def __init__(self, config: MetaConfig):
        self.config = config
        self.controller = RestartController(config)
        self.changed_ring: deque[bool] = deque(maxlen=20)
        self.hash_ring: deque[str] = deque(maxlen=30)
        self.attempt_action_count = 0
        self.analysis_step_count = 0

    def note_action(self, board_changed: bool, grid) -> None:
        """Record one executed action. ``grid`` is hashed and discarded."""
        self.attempt_action_count += 1
        self.changed_ring.append(bool(board_changed))
        self.hash_ring.append(grid_digest(grid))

    def note_analysis_step(self) -> None:
        self.analysis_step_count += 1

    def noop_rate_20(self) -> float:
        if not self.changed_ring:
            return 0.0
        return 1.0 - sum(self.changed_ring) / len(self.changed_ring)

    def novelty_rate_30(self) -> float:
        if not self.hash_ring:
            return 1.0
        return len(set(self.hash_ring)) / len(self.hash_ring)

    def on_meta_restart(self) -> None:
        """Reset attempt-relative state after an issued meta restart."""
        self.controller.record_restart()
        self.attempt_action_count = 0
        self.changed_ring.clear()
        self.hash_ring.clear()

    def should_restart(self, *, level: int, budget_used_frac: float) -> bool:
        return self.controller.should_restart(
            t=self.attempt_action_count,
            level=level,
            fails=self.controller.failed_attempts,
            noop_rate_20=self.noop_rate_20(),
            novelty_rate_30=self.novelty_rate_30(),
            budget_used_frac=budget_used_frac,
        )
