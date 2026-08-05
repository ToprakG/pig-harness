"""Budget planner: turn a wall-clock grant into per-game time allocation.

The 0.91 submission used 6010 s of a 32400 s grant (18.5%): 25 games ran in
ONE concurrent wave (concurrency 28) capped at 90 min each, so the grant was
never touchable by per-game caps sized for sequential thinking. The planner
therefore allocates per WAVE:

    waves            = ceil(games * passes / concurrency)
    usable_seconds   = (granted - safety_margin) * target_utilisation
    per_game_seconds = usable_seconds / waves

Scoring semantics (NOTES.md, Phase 3): the competition score is a mean over
games of a single-pass score that grows with levels completed, so extra
budget goes to per-game time. Extra passes cannot raise the scored-run
result (the rerun forces n_passes=1); locally they only reduce measurement
variance.

CLI dry-run::

    uv run python -m inference.framework.budget --games 25 --passes 1 --concurrency 28
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass


@dataclass
class BudgetConfig:
    enabled: bool = False
    granted_seconds: float = 32400.0
    target_utilisation: float = 0.85
    safety_margin_seconds: float = 1800.0
    allocation: str = "uniform"
    # Upper bound on any single game's wall clock. With one wave, uniform
    # allocation hands the whole usable grant to every game in parallel;
    # this caps how long a single stuck game may hold the run open.
    max_seconds_per_game: float = 10800.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "BudgetConfig":
        d = dict(d or {})
        known = {k: d[k] for k in d if k in cls.__dataclass_fields__}
        return cls(**known)


def plan_budget(config: BudgetConfig, *, n_games: int, n_passes: int,
                concurrency: int) -> dict:
    """Uniform allocation of the usable grant across concurrent waves."""
    if config.allocation != "uniform":
        raise ValueError(f"unknown allocation strategy: {config.allocation}")
    jobs = max(1, n_games * max(1, n_passes))
    waves = math.ceil(jobs / max(1, concurrency))
    usable = max(0.0, (config.granted_seconds - config.safety_margin_seconds)
                 ) * config.target_utilisation
    uncapped = usable / waves
    per_game = min(uncapped, config.max_seconds_per_game)
    return {
        "granted_seconds": config.granted_seconds,
        "safety_margin_seconds": config.safety_margin_seconds,
        "target_utilisation": config.target_utilisation,
        "max_seconds_per_game": config.max_seconds_per_game,
        "n_games": n_games,
        "n_passes": n_passes,
        "concurrency": concurrency,
        "waves": waves,
        "usable_seconds": round(usable, 1),
        "per_game_seconds": round(per_game, 1),
        "per_game_seconds_uncapped": round(uncapped, 1),
        "capped": per_game < uncapped,
        "planned_wallclock_seconds": round(per_game * waves, 1),
    }


def format_plan(plan: dict) -> str:
    return (
        "==================== BUDGET PLAN ====================\n"
        f"granted={plan['granted_seconds']:.0f}s "
        f"margin={plan['safety_margin_seconds']:.0f}s "
        f"target_utilisation={plan['target_utilisation']:.0%}\n"
        f"jobs={plan['n_games']}x{plan['n_passes']} "
        f"concurrency={plan['concurrency']} waves={plan['waves']}\n"
        f"per_game_seconds={plan['per_game_seconds']:.0f}"
        + (f" (capped from {plan['per_game_seconds_uncapped']:.0f}) " if plan.get("capped") else " ") +
        f"planned_wallclock={plan['planned_wallclock_seconds']:.0f}s "
        f"({plan['planned_wallclock_seconds'] / plan['granted_seconds']:.0%} of grant)\n"
        "====================================================="
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=28)
    ap.add_argument("--granted-seconds", type=float, default=32400.0)
    ap.add_argument("--target-utilisation", type=float, default=0.85)
    ap.add_argument("--safety-margin-seconds", type=float, default=1800.0)
    ap.add_argument("--max-seconds-per-game", type=float, default=10800.0)
    args = ap.parse_args(argv)
    config = BudgetConfig(
        enabled=True,
        granted_seconds=args.granted_seconds,
        target_utilisation=args.target_utilisation,
        safety_margin_seconds=args.safety_margin_seconds,
        max_seconds_per_game=args.max_seconds_per_game,
    )
    plan = plan_budget(config, n_games=args.games, n_passes=args.passes,
                       concurrency=args.concurrency)
    print(format_plan(plan))
    print(json.dumps(plan, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
