"""Action-efficiency telemetry on stdout (Kaggle captures stdout; the
artifact bundle is ~167 MB and impractical to inspect).

Line grammar, one line per event, all key=value for grepping:

    [ACT] game=<id> t=<n> level=<l> lvl_actions=<n> changed=<bool>
          batch=<i>/<size> analysis_step=<n> [gated=1]
    [EFF] game=<id> pass=<n> levels=<k>/<n> a_per_level=... base_per_level=...
          human_mult_per_level=... cleared_human_mult=... score=...
          score_if_1x=... headroom=...
    [BUDGET] game=<id> planned=<s> used=<s> util=<pct>

Actions are the only thing that costs score, so the action counter is the
primary metric of this branch.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from inference.eff.scoring import (
    cleared_human_mult,
    final_score,
    human_multipliers,
    score_if_1x,
)


def _p(line: str) -> None:
    print(line, flush=True)


def _join(values: Sequence[Any] | None, digits: int | None = None) -> str:
    if values is None:
        return "hidden"
    if digits is None:
        return ",".join(str(v) for v in values)
    return ",".join(f"{float(v):.{digits}f}" for v in values)


def action_line(*, game_id: str, t: int, level: int, lvl_actions: int,
                changed: bool, batch_index: int, batch_size: int,
                analysis_step: int, gated: bool = False) -> None:
    suffix = " gated=1" if gated else ""
    _p(f"[ACT] game={game_id} t={t} level={level} lvl_actions={lvl_actions} "
       f"changed={bool(changed)} batch={batch_index}/{batch_size} "
       f"analysis_step={analysis_step}{suffix}")


def efficiency_line(*, game_id: str, pass_index: int, levels_completed: int,
                    number_of_levels: int, actions_per_level: Sequence[int],
                    base_actions_per_level: Sequence[float] | None,
                    score: float | None) -> dict:
    """Emit [EFF] and return the same payload for run-level aggregation.

    When the engine hides baselines (``base_actions_per_level is None``) the
    derived fields are skipped rather than guessed.
    """
    payload: dict[str, Any] = {
        "game": game_id,
        "pass_index": pass_index,
        "levels_completed": levels_completed,
        "number_of_levels": number_of_levels,
        "actions_per_level": list(actions_per_level),
        "score": score,
        "baselines_hidden": base_actions_per_level is None,
    }
    head = (f"[EFF] game={game_id} pass={pass_index} "
            f"levels={levels_completed}/{number_of_levels} "
            f"a_per_level={_join(actions_per_level)}")
    if base_actions_per_level is None:
        _p(f"{head} base_per_level=hidden score={score}")
        return payload

    mults = human_multipliers(actions_per_level, base_actions_per_level)
    cleared = cleared_human_mult(actions_per_level, base_actions_per_level,
                                 levels_completed)
    ideal = score_if_1x(base_actions_per_level=base_actions_per_level,
                        levels_completed=levels_completed,
                        number_of_levels=number_of_levels)
    actual = score if score is not None else final_score(
        actions_per_level=actions_per_level,
        base_actions_per_level=base_actions_per_level,
        levels_completed=levels_completed,
        number_of_levels=number_of_levels)
    headroom = ideal - actual
    payload.update({"cleared_human_mult": cleared, "score_if_1x": ideal,
                    "headroom": headroom, "score": actual})
    _p(f"{head} base_per_level={_join(base_actions_per_level)} "
       f"human_mult_per_level={_join(mults, 2)} "
       f"cleared_human_mult={'n/a' if cleared is None else f'{cleared:.2f}'} "
       f"score={actual:.4f} score_if_1x={ideal:.4f} headroom={headroom:.4f}")
    return payload


def budget_line(*, game_id: str, planned_s: float, used_s: float) -> None:
    util = (used_s / planned_s * 100.0) if planned_s > 0 else 0.0
    _p(f"[BUDGET] game={game_id} planned={planned_s:.0f} used={used_s:.0f} "
       f"util={util:.0f}%")


class RunEfficiency:
    """Run-level aggregation; prints the end-of-run block."""

    def __init__(self, granted_seconds: float | None = None):
        self.started_at = time.monotonic()
        self.granted_seconds = granted_seconds
        self.games: list[dict] = []
        self.total_actions = 0
        self.total_analysis_steps = 0
        self.gated_turns = 0
        self.withheld_actions = 0

    def record_game(self, payload: dict, *, actions: int,
                    analysis_steps: int) -> None:
        payload = dict(payload)
        payload["actions"] = actions
        payload["analysis_steps"] = analysis_steps
        self.games.append(payload)
        self.total_actions += actions
        self.total_analysis_steps += analysis_steps

    def print_block(self) -> None:
        used = time.monotonic() - self.started_at
        lines = ["================= ACTION EFFICIENCY ================="]
        apc = (self.total_actions / self.total_analysis_steps
               if self.total_analysis_steps else 0.0)
        lines.append(f"total_actions={self.total_actions} "
                     f"analyzer_calls={self.total_analysis_steps} "
                     f"actions_per_call={apc:.2f}")
        cleared = [g["cleared_human_mult"] for g in self.games
                   if g.get("cleared_human_mult") is not None]
        if cleared:
            lines.append(f"mean_cleared_human_mult="
                         f"{sum(cleared) / len(cleared):.2f} "
                         f"(games_with_a_clear={len(cleared)})")
        else:
            lines.append("mean_cleared_human_mult=n/a (no cleared level with "
                         "a visible baseline)")
        headrooms = [g["headroom"] for g in self.games
                     if g.get("headroom") is not None]
        if headrooms:
            lines.append(f"sum_headroom={sum(headrooms):.3f} "
                         f"sum_score={sum(g.get('score') or 0 for g in self.games):.3f}")
        if any(g.get("baselines_hidden") for g in self.games):
            hidden = sum(1 for g in self.games if g.get("baselines_hidden"))
            lines.append(f"WARNING baselines_hidden_for={hidden}/{len(self.games)} "
                         "games — efficiency cannot be measured directly there")
        lines.append(f"gated_turns={self.gated_turns} "
                     f"withheld_actions={self.withheld_actions}")
        if self.granted_seconds:
            lines.append(f"wallclock used={used:.0f}s granted="
                         f"{self.granted_seconds:.0f}s "
                         f"utilisation={used / self.granted_seconds * 100:.1f}%")
        else:
            lines.append(f"wallclock used={used:.0f}s granted=unknown")
        lines.append("=====================================================")
        _p("\n".join(lines))
