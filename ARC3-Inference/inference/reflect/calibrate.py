"""Calibrate the stall detector on our own trajectories.

Two jobs:

1. **Reproduce** the reference figures the branch is premised on (498 runs
   with >=25 actions; 161 stall before the first clear with a 0.13 clear
   rate; 337 never stall with 0.68). If they do not reproduce, either the
   detector or the earlier analysis is wrong and both are reportable — the
   caller stops rather than proceeding on an unverified premise.

2. **Select thresholds by leave-one-game-out**, never by taking the best cell
   on the full set: the restart policy in this project was overfitted exactly
   that way (in-sample optimum T=40 versus LOGO-selected T=60).

Usage::

    uv run python -m inference.reflect.calibrate reference
    uv run python -m inference.reflect.calibrate sweep
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from pathlib import Path

from inference.reflect.detector import StallConfig, first_stall_index, to_bool

ARTIFACTS = "../example-run/artifacts/*_events.jsonl"
MIN_ACTIONS = 25
REFERENCE = {"runs": 498, "stalled": 161, "stalled_clear_rate": 0.13,
             "never_clear_rate": 0.68, "median_first_stall": 25}
NOOP_GRID = (0.20, 0.25, 0.30)
NOVELTY_GRID = (0.4, 0.5, 0.6)


def load_runs(pattern: str = ARTIFACTS) -> list[dict]:
    """One record per run: its pre-first-clear window and whether it cleared."""
    runs = []
    for path in sorted(glob.glob(pattern)):
        match = re.match(r"(?P<game>.+)_p(?P<pass>\d+)_events\.jsonl$",
                         Path(path).name)
        if not match:
            continue
        actions = []
        cleared_at = None
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                if event.get("type") != "action":
                    continue
                actions.append(event)
                if cleared_at is None and to_bool(event.get("level_completed")):
                    cleared_at = len(actions)
        if len(actions) < MIN_ACTIONS:
            continue
        # the window the detector would actually have seen: everything before
        # the first clear (or the whole run when it never cleared)
        window = actions[:cleared_at] if cleared_at else actions
        runs.append({"game": match.group("game"), "pass": int(match.group("pass")),
                     "window": window, "cleared": cleared_at is not None,
                     "n_actions": len(actions)})
    return runs


def evaluate(runs: list[dict], cfg: StallConfig) -> dict:
    stalled, never = [], []
    fire_times = []
    for run in runs:
        idx = first_stall_index(run["window"], cfg)
        if idx is None:
            never.append(run)
        else:
            stalled.append(run)
            fire_times.append(idx)

    def rate(group):
        return (sum(1 for r in group if r["cleared"]) / len(group)) if group else 0.0

    stalled_rate, never_rate = rate(stalled), rate(never)
    return {
        "n_runs": len(runs),
        "n_stalled": len(stalled),
        "fire_rate": len(stalled) / len(runs) if runs else 0.0,
        "clear_rate_stalled": stalled_rate,
        "clear_rate_never": never_rate,
        "discrimination": (never_rate / stalled_rate) if stalled_rate else float("inf"),
        "median_first_stall": statistics.median(fire_times) if fire_times else None,
    }


def logo_select(runs: list[dict]) -> dict:
    """Leave-one-game-out threshold selection.

    For each held-out game, pick the cell maximising discrimination on the
    remaining games, then score that choice on the held-out game. The chosen
    configuration is the cell selected in the most folds; the reported
    discrimination is the held-out one, not the in-sample one.
    """
    games = sorted({r["game"] for r in runs})
    picks: dict[tuple[float, float], int] = {}
    heldout_disc: list[float] = []
    for holdout in games:
        train = [r for r in runs if r["game"] != holdout]
        test = [r for r in runs if r["game"] == holdout]
        best, best_score = None, -1.0
        for noop in NOOP_GRID:
            for novelty in NOVELTY_GRID:
                cfg = StallConfig(noop_threshold=noop, novelty_threshold=novelty)
                result = evaluate(train, cfg)
                # require the trigger to fire on a usable share of runs;
                # a cell that never fires has infinite discrimination and is
                # useless
                if not 0.05 <= result["fire_rate"] <= 0.60:
                    continue
                score = result["discrimination"]
                if score > best_score:
                    best, best_score = (noop, novelty), score
        if best is None:
            continue
        picks[best] = picks.get(best, 0) + 1
        cfg = StallConfig(noop_threshold=best[0], novelty_threshold=best[1])
        heldout = evaluate(test, cfg)
        if heldout["clear_rate_stalled"]:
            heldout_disc.append(heldout["discrimination"])
    modal = max(picks.items(), key=lambda kv: kv[1]) if picks else (None, 0)
    return {"picks": {f"{k[0]}/{k[1]}": v for k, v in sorted(picks.items())},
            "modal_choice": modal[0], "modal_folds": modal[1],
            "n_folds": len(games),
            "median_heldout_discrimination":
                statistics.median(heldout_disc) if heldout_disc else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["reference", "sweep"])
    ap.add_argument("--pattern", default=ARTIFACTS)
    ap.add_argument("--tolerance", type=float, default=0.03)
    args = ap.parse_args(argv)

    runs = load_runs(args.pattern)
    if args.command == "reference":
        result = evaluate(runs, StallConfig())
        print(f"runs(>= {MIN_ACTIONS} actions) = {result['n_runs']} "
              f"(reference {REFERENCE['runs']})")
        print(f"stalled = {result['n_stalled']} "
              f"({result['fire_rate']:.0%})  (reference {REFERENCE['stalled']})")
        print(f"clear rate | stalled     = {result['clear_rate_stalled']:.2f} "
              f"(reference {REFERENCE['stalled_clear_rate']})")
        print(f"clear rate | never stall = {result['clear_rate_never']:.2f} "
              f"(reference {REFERENCE['never_clear_rate']})")
        print(f"discrimination = {result['discrimination']:.1f}x")
        print(f"median first stall = {result['median_first_stall']} "
              f"(reference {REFERENCE['median_first_stall']})")
        ok = (abs(result["clear_rate_stalled"] - REFERENCE["stalled_clear_rate"])
              <= args.tolerance
              and abs(result["clear_rate_never"] - REFERENCE["never_clear_rate"])
              <= args.tolerance)
        print("\nREFERENCE:", "REPRODUCED" if ok else "NOT REPRODUCED")
        return 0 if ok else 1

    print(f"{'noop':>5} {'nov':>5} {'fire%':>6} {'clr|fire':>9} "
          f"{'clr|no':>7} {'disc':>6} {'med t':>6}")
    for noop in NOOP_GRID:
        for novelty in NOVELTY_GRID:
            r = evaluate(runs, StallConfig(noop_threshold=noop,
                                           novelty_threshold=novelty))
            print(f"{noop:5.2f} {novelty:5.2f} {r['fire_rate']:6.0%} "
                  f"{r['clear_rate_stalled']:9.2f} {r['clear_rate_never']:7.2f} "
                  f"{r['discrimination']:6.1f} {str(r['median_first_stall']):>6}")
    print("\nleave-one-game-out selection:")
    print(json.dumps(logo_select(runs), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
