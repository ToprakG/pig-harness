"""Session-scoped reflect runtime: detect the stall, revise, evaluate, roll
back. One instance per game; the block is cleared at every level transition
because the debrief branch owns transitions.

All events go to **stdout**: Kaggle captures stdout, the artifact bundle is
~167 MB and impractical, and the earlier meta feature left no stdout trace at
all and was therefore unverifiable after the fact.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from typing import Any, Callable, Sequence

from inference.debrief.packet import estimate_tokens
from inference.reflect.detector import (
    NOOP_WINDOW,
    NOVELTY_WINDOW,
    StallConfig,
    board_hash,
    stall_state,
    to_bool,
)
from inference.reflect.packet import build_stall_packet
from inference.reflect.revise import RevisionRatchet, revise

logger = logging.getLogger(__name__)


def load_config() -> StallConfig:
    raw = os.environ.get("REFLECT_CONFIG", "").strip()
    if not raw:
        return StallConfig()
    try:
        return StallConfig.from_dict(json.loads(raw))
    except (ValueError, TypeError):
        logger.warning("invalid REFLECT_CONFIG json; reflect disabled")
        return StallConfig()


def _print(line: str) -> None:
    print(line, flush=True)


class ReflectRuntime:
    def __init__(self, config: StallConfig, game_id: str = "?"):
        self.config = config
        self.game_id = game_id
        self.ratchet = RevisionRatchet(config)
        self.changed: deque = deque(maxlen=NOOP_WINDOW)
        self.hashes: deque = deque(maxlen=NOVELTY_WINDOW)
        self.level_events: list[dict[str, Any]] = []
        self.level = 1
        self.action_num = 0
        self.fired = self.ok = self.fallback = 0

    # ---- lifecycle -------------------------------------------------------

    @property
    def block(self) -> str:
        return self.ratchet.block

    def on_level_transition(self, level: int) -> None:
        self.ratchet.reset_level()      # asserts the block does not leak
        self.changed.clear()
        self.hashes.clear()
        self.level_events = []
        self.level = level

    # ---- the loop --------------------------------------------------------

    def note_action(self, *, action: str, board_changed: bool, board: Any,
                    level: int, level_cleared: bool,
                    valid_actions: Sequence[str],
                    llm_call: Callable[[str, int, float, float], str],
                    current_plan: str = "") -> str:
        """Record one executed action and run the loop. Returns the standing
        reflect block (possibly unchanged). Never raises."""
        if not self.config.enabled:
            return ""
        try:
            self.action_num += 1
            digest = board_hash(board)
            self.changed.append(to_bool(board_changed))
            self.hashes.append(digest)
            self.level_events.append({"action": action,
                                      "board_changed": to_bool(board_changed),
                                      "board_hash": digest})

            state = stall_state(self.changed, self.hashes, self.config)

            # 1. an open revision may be due for evaluation
            if self.ratchet.due(self.action_num):
                improved, index = self.ratchet.evaluate(
                    action_num=self.action_num, stalled=state.stalled,
                    hashes=set(self.hashes), level_cleared=level_cleared)
                if not improved:
                    _print(f"[REFLECT] rollback game={self.game_id} "
                           f"level={self.level} revision={index} "
                           f"reason=no_improvement")

            # 2. level transitions belong to the debrief loop
            if level_cleared:
                self.on_level_transition(level + 1)
                return self.block

            # 3. fire?
            if not self.ratchet.may_fire(action_num=self.action_num,
                                         stalled=state.stalled):
                return self.block

            self.fired += 1
            _print(f"[REFLECT] fire game={self.game_id} level={self.level} "
                   f"t={self.action_num} noop20={state.noop_20:.2f} "
                   f"nov30={state.novelty_30:.2f} "
                   f"revision={self.ratchet.revisions_used + 1}")
            packet = build_stall_packet(self.level_events, valid_actions,
                                        level=self.level,
                                        current_plan=current_plan)
            result = revise(packet, self.block, valid_actions, llm_call,
                            config=self.config)
            for dropped in result.dropped_lines:
                _print(f"[REFLECT] dropped game={self.game_id} detail={dropped}")
            if not result.ok:
                self.fallback += 1
                _print(f"[REFLECT] fallback game={self.game_id} "
                       f"level={self.level} reason={result.reason} "
                       f"latency_ms={result.latency_ms}")
                # a failed call must not consume the cooldown-free path
                self.ratchet.last_fire_action = self.action_num
                return self.block
            self.ratchet.accept(result.block, action_num=self.action_num,
                                hashes=set(self.hashes))
            self.ok += 1
            _print(f"[REFLECT] ok game={self.game_id} level={self.level} "
                   f"tokens={estimate_tokens(result.block)} "
                   f"latency_ms={result.latency_ms}")
            return self.block
        except Exception as exc:      # the loop must never break the run
            logger.warning("reflect loop failed: %s", exc, exc_info=True)
            _print(f"[REFLECT] fallback game={self.game_id} "
                   f"reason={type(exc).__name__}")
            self.fallback += 1
            return self.block

    def summary_line(self) -> str:
        return (f"[REFLECT] fired={self.fired} ok={self.ok} "
                f"rollback={self.ratchet.rollbacks} fallback={self.fallback}")
