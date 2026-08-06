"""Session-scoped debrief runtime: collect the level's events, fire the
rewrite at a level transition, hand the block to the agent.

Scope is deliberately one game. A cross-game rewriting loop would let a bad
rewrite at game 10 degrade the remaining 100 in a single Kaggle session, with
no way to detect it mid-run and no way to ablate it. `reset()` is called when
a new game starts and asserts the block does not survive.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

from inference.debrief.packet import build_packet, to_bool
from inference.debrief.rewrite import DebriefConfig, rewrite

logger = logging.getLogger(__name__)


def load_config() -> DebriefConfig:
    """Read the `debrief` block passed via the DEBRIEF_CONFIG env var (JSON,
    exported by the Makefile from configs/inference.json)."""
    raw = os.environ.get("DEBRIEF_CONFIG", "").strip()
    if not raw:
        return DebriefConfig()
    try:
        return DebriefConfig.from_dict(json.loads(raw))
    except (ValueError, TypeError):
        logger.warning("invalid DEBRIEF_CONFIG json; debrief disabled")
        return DebriefConfig()


def _print(line: str) -> None:
    # stdout: Kaggle captures it, and the 167 MB artifact bundle is
    # impractical to inspect. The previous meta feature left no stdout trace
    # at all and was therefore unverifiable after the fact.
    print(line, flush=True)


class DebriefRuntime:
    """One instance per game session."""

    def __init__(self, config: DebriefConfig, game_id: str = "?"):
        self.config = config
        self.game_id = game_id
        self.level_events: list[dict[str, Any]] = []
        self.block = ""
        self.levels_debriefed = 0
        self.ok_count = 0
        self.fallback_count = 0

    # ---- lifecycle -------------------------------------------------------

    def reset(self, game_id: str = "?") -> None:
        """New game: the block must not survive (D3)."""
        self.game_id = game_id
        self.level_events = []
        self.block = ""
        assert not self.block, "debrief block leaked across games"

    def note_action(self, event: dict[str, Any]) -> None:
        if self.config.enabled:
            self.level_events.append(event)

    # ---- the transition --------------------------------------------------

    def on_level_completed(
        self,
        *,
        level: int,
        llm_call: Callable[[str, int, float, float], str],
        baseline_actions: float | None = None,
    ) -> str:
        """Called right after an action reported `level_completed`.

        Returns the (possibly unchanged) standing block. Never raises.
        """
        if not self.config.enabled:
            return self.block
        events, self.level_events = self.level_events, []
        if len(events) < self.config.min_actions_for_debrief:
            _print(f"[DEBRIEF] fallback game={self.game_id} level={level} "
                   f"reason=too_few_actions actions={len(events)}")
            self.fallback_count += 1
            return self.block
        try:
            packet = build_packet(
                events, level=level, baseline_actions=baseline_actions,
                include_baseline=self.config.include_baseline)
            result = rewrite(packet, self.block, llm_call, config=self.config)
        except Exception as exc:  # a debrief must never break the run
            logger.warning("debrief failed: %s", exc, exc_info=True)
            _print(f"[DEBRIEF] fallback game={self.game_id} level={level} "
                   f"reason={type(exc).__name__}")
            self.fallback_count += 1
            return self.block

        for dropped in result.dropped_lines:
            _print(f"[DEBRIEF] dropped game={self.game_id} level={level} "
                   f"detail={dropped}")
        if not result.ok:
            _print(f"[DEBRIEF] fallback game={self.game_id} level={level} "
                   f"reason={result.reason} latency_ms={result.latency_ms}")
            self.fallback_count += 1
            return self.block

        self.block = result.block
        self.levels_debriefed += 1
        self.ok_count += 1
        from inference.debrief.packet import estimate_tokens
        _print(f"[DEBRIEF] ok game={self.game_id} level={level} "
               f"tokens={estimate_tokens(result.block)} "
               f"latency_ms={result.latency_ms}")
        return self.block

    def summary_line(self) -> str:
        return (f"[DEBRIEF] summary game={self.game_id} "
                f"ok={self.ok_count} fallback={self.fallback_count} "
                f"block_tokens={len(self.block) // 3 if self.block else 0}")
