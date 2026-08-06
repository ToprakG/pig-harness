"""Powered 2x2 ablation over the two loops.

    arm A: debrief off, reflect off      arm C: debrief off, reflect on
    arm B: debrief on,  reflect off      arm D: debrief on,  reflect on

Primary endpoint: **level-1 clear rate among runs that stall** — the
population the reflect loop targets. The stall flag is recomputed offline
from each run's own event log with the same detector the harness uses, so a
run's arm cannot influence whether it counts as stalled through anything but
its actual behaviour.

Secondary endpoints: level-2 clear rate given level-1 (the debrief endpoint),
mean score.

Sizing note that matters: the population figure (0.13 vs 0.68, 5.2x) is
inflated by a between-game confound; the honest within-game gap is ~0.26
(see NOTES.md, Phase 1). Power is therefore reported against a 0.26-point
effect as well as the optimistic one.

Runs that never opened (engine/network failure — `observation_space is None`,
0 actions) are excluded from the denominator rather than counted as failures
to clear level 1; that was observed live on `vc33`.

Sub-commands::

    power    — required n for a given effect
    analyse  — endpoints, Fisher, bootstrap CI, achieved power over runs
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import re
import statistics
from pathlib import Path

from inference.reflect.calibrate import MIN_ACTIONS
from inference.reflect.detector import StallConfig, first_stall_index, to_bool

BOOTSTRAP_RESAMPLES = 10000
N_POWER_SIMS = 20000
ARMS = ("A", "B", "C", "D")
# three comparisons against A when all four arms run
N_COMPARISONS = 3


# ------------------------------------------------------------------ loading

def load_run_records(run_dir: str, cfg: StallConfig) -> list[dict]:
    """One record per pass in a finished run directory."""
    records = []
    for path in sorted(glob.glob(f"{run_dir}/artifacts/*_events.jsonl")):
        match = re.match(r"(?P<game>.+)_p(?P<pass>\d+)_events\.jsonl$",
                         Path(path).name)
        if not match:
            continue
        actions, cleared_at, levels = [], None, 0
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                if event.get("type") != "action":
                    continue
                actions.append(event)
                if to_bool(event.get("level_completed")):
                    levels += 1
                    if cleared_at is None:
                        cleared_at = len(actions)
        if len(actions) < MIN_ACTIONS:
            # never opened, or died before it could play: excluded, not a
            # failure to clear (observed live on vc33)
            records.append({"game": match.group("game"),
                            "pass": int(match.group("pass")),
                            "excluded": True, "n_actions": len(actions)})
            continue
        window = actions[:cleared_at] if cleared_at else actions
        records.append({
            "game": match.group("game"), "pass": int(match.group("pass")),
            "excluded": False, "n_actions": len(actions),
            "stalled": first_stall_index(window, cfg) is not None,
            "cleared_l1": cleared_at is not None,
            "cleared_l2": levels >= 2,
        })
    return records


def load_arm(patterns: list[str], cfg: StallConfig) -> list[dict]:
    out: list[dict] = []
    for pattern in patterns:
        for run_dir in sorted(glob.glob(pattern)) or [pattern]:
            if Path(run_dir).is_dir():
                out.extend(load_run_records(run_dir, cfg))
    return [r for r in out if not r["excluded"]]


# ------------------------------------------------------------------- stats

def fisher_one_sided(a: int, b: int, c: int, d: int) -> float:
    n1, n2 = a + b, c + d
    successes, total = a + c, a + b + c + d
    if not n1 or not n2:
        return float("nan")

    def logcomb(n: int, k: int) -> float:
        if k < 0 or k > n:
            return float("-inf")
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

    denom = logcomb(total, successes)
    terms = [logcomb(n1, i) + logcomb(n2, successes - i) - denom
             for i in range(a, min(n1, successes) + 1)]
    terms = [t for t in terms if t > float("-inf")]
    if not terms:
        return 1.0
    peak = max(terms)
    return min(1.0, math.exp(peak + math.log(sum(math.exp(t - peak) for t in terms))))


def simulate_power(*, n_per_arm: int, baseline: float, effect: float,
                   alpha: float = 0.05, comparisons: int = 1,
                   sims: int = N_POWER_SIMS, seed: int = 0) -> float:
    rng = random.Random(seed)
    threshold = alpha / max(1, comparisons)
    hits = 0
    for _ in range(sims):
        a = sum(rng.random() < effect for _ in range(n_per_arm))
        c = sum(rng.random() < baseline for _ in range(n_per_arm))
        if fisher_one_sided(a, n_per_arm - a, c, n_per_arm - c) < threshold:
            hits += 1
    return hits / sims


def required_n(*, baseline: float, effect: float, target_power: float = 0.8,
               comparisons: int = 1, seed: int = 0) -> int:
    for n in (10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800):
        if simulate_power(n_per_arm=n, baseline=baseline, effect=effect,
                          comparisons=comparisons, sims=4000, seed=seed) >= target_power:
            return n
    return -1


def bootstrap_ci(a: list[bool], b: list[bool], rng: random.Random,
                 resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float]:
    if not a or not b:
        return (float("nan"), float("nan"))
    diffs = sorted(
        sum(rng.choice(b) for _ in b) / len(b) - sum(rng.choice(a) for _ in a) / len(a)
        for _ in range(resamples))
    return diffs[int(0.025 * resamples)], diffs[min(resamples - 1, int(0.975 * resamples))]


def endpoints(records: list[dict]) -> dict:
    stalled = [r for r in records if r["stalled"]]
    cleared_l1 = [r for r in records if r["cleared_l1"]]
    return {
        "n_runs": len(records),
        "n_stalled": len(stalled),
        "primary_l1_given_stall": (sum(r["cleared_l1"] for r in stalled) / len(stalled)
                                   if stalled else float("nan")),
        "secondary_l2_given_l1": (sum(r["cleared_l2"] for r in cleared_l1) / len(cleared_l1)
                                  if cleared_l1 else float("nan")),
        "l1_rate_overall": (sum(r["cleared_l1"] for r in records) / len(records)
                            if records else float("nan")),
    }


def compare(base: list[dict], cand: list[dict], *, seed: int = 0,
            comparisons: int = N_COMPARISONS) -> dict:
    bs = [r["cleared_l1"] for r in base if r["stalled"]]
    cs = [r["cleared_l1"] for r in cand if r["stalled"]]
    a, c = sum(cs), sum(bs)
    p_raw = fisher_one_sided(a, len(cs) - a, c, len(bs) - c)
    lo, hi = bootstrap_ci(bs, cs, random.Random(seed))
    base_rate = (c / len(bs)) if bs else float("nan")
    cand_rate = (a / len(cs)) if cs else float("nan")
    n_min = min(len(bs), len(cs))
    return {
        "n_stalled_baseline": len(bs), "n_stalled_candidate": len(cs),
        "baseline_rate": base_rate, "candidate_rate": cand_rate,
        "difference": cand_rate - base_rate,
        "p_one_sided_raw": p_raw,
        "p_corrected": min(1.0, p_raw * comparisons) if p_raw == p_raw else float("nan"),
        "ci95_difference": [lo, hi],
        "achieved_power": (simulate_power(n_per_arm=n_min, baseline=base_rate,
                                          effect=cand_rate, comparisons=comparisons,
                                          seed=seed, sims=4000)
                           if n_min and base_rate == base_rate else float("nan")),
        "meets_n_requirement": n_min >= 50,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("power", help="required n for an effect size")
    p.add_argument("--baseline", type=float, default=0.13)
    p.add_argument("--effects", type=float, nargs="+", default=[0.26, 0.39, 0.60])
    p.add_argument("--comparisons", type=int, default=N_COMPARISONS)
    a = sub.add_parser("analyse", help="endpoints + Fisher + CI over run dirs")
    a.add_argument("--baseline", nargs="+", required=True)
    a.add_argument("--candidate", nargs="+", required=True)
    args = ap.parse_args(argv)

    if args.command == "power":
        print(f"baseline={args.baseline} comparisons={args.comparisons}")
        for effect in args.effects:
            n = required_n(baseline=args.baseline, effect=effect,
                           comparisons=args.comparisons)
            runs = math.ceil(n / 0.32) if n > 0 else -1
            print(f"  effect {args.baseline:.2f} -> {effect:.2f}: "
                  f"n={n} stalled runs/arm  (~{runs} runs/arm at a 32% stall rate)")
        return 0

    cfg = StallConfig()
    base = load_arm(args.baseline, cfg)
    cand = load_arm(args.candidate, cfg)
    result = {"baseline": endpoints(base), "candidate": endpoints(cand),
              "comparison": compare(base, cand)}
    print(json.dumps(result, indent=2))
    if not result["comparison"]["meets_n_requirement"]:
        print("\nINSUFFICIENT: fewer than 50 stalled runs in an arm; "
              "no configuration decision may be made from this data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
