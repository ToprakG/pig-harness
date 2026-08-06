"""Powered ablation for the level-debrief feature.

Primary endpoint: **level-2-clear rate among runs that cleared level 1** —
it measures the targeted bottleneck directly and is far less heavy-tailed
than score.

Two arms only (baseline vs `debrief.enabled=1`): fewer arms means a lighter
multiplicity penalty and more runs per arm for the same compute.

Game selection is a *rule*, never an eyeball choice:
  R1  level-1 clear rate >= 0.5 in example-run/benchmark.json
      (so level-2 transitions are actually sampled)
  R2  0.1 <= P(level 2 | level 1) <= 0.9
      (the endpoint must be able to move in both directions; a game at 0.00
      or 1.00 contributes samples but no resolvable signal)

Sub-commands::

    select   — print the games the rule admits, with their historical rates
    power    — simulate achieved power for a given n and effect size
    analyse  — Fisher + bootstrap CI + achieved power over finished runs
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
from collections import defaultdict
from pathlib import Path

BOOTSTRAP_RESAMPLES = 10000
N_POWER_SIMS = 20000
N_ARMS_FOR_CORRECTION = 2   # baseline vs debrief


# ---------------------------------------------------------------- selection

def historical_rates(benchmark_path: Path) -> list[dict]:
    runs = json.loads(benchmark_path.read_text(encoding="utf-8"))["game_runs"]
    by_game: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_game[run["game_id"]].append(run)
    rows = []
    for game_id, game_runs in sorted(by_game.items()):
        cleared_1 = [r for r in game_runs if int(r["levels_completed"]) >= 1]
        cleared_2 = [r for r in game_runs if int(r["levels_completed"]) >= 2]
        rows.append({
            "game": game_id,
            "n": len(game_runs),
            "l1_rate": len(cleared_1) / len(game_runs),
            "cond_l2": (len(cleared_2) / len(cleared_1)) if cleared_1 else 0.0,
        })
    return rows


def select_games(rows: list[dict], *, min_l1: float = 0.5,
                 cond_low: float = 0.1, cond_high: float = 0.9) -> list[dict]:
    return [r for r in rows
            if r["l1_rate"] >= min_l1 and cond_low <= r["cond_l2"] <= cond_high]


# ------------------------------------------------------------------- power

def _fisher_one_sided(a: int, b: int, c: int, d: int) -> float:
    """P(observing >= a successes in arm 1 | margins fixed), hypergeometric."""
    n1, n2 = a + b, c + d
    successes, total = a + c, a + b + c + d

    def logcomb(n: int, k: int) -> float:
        if k < 0 or k > n:
            return float("-inf")
        return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))

    denom = logcomb(total, successes)
    upper = min(n1, successes)
    terms = []
    for i in range(a, upper + 1):
        value = logcomb(n1, i) + logcomb(n2, successes - i) - denom
        if value > float("-inf"):
            terms.append(value)
    if not terms:
        return 1.0
    peak = max(terms)
    return min(1.0, math.exp(peak + math.log(sum(math.exp(t - peak) for t in terms))))


def simulate_power(*, n_per_arm: int, baseline_rate: float, effect_rate: float,
                   alpha: float = 0.05, corrections: int = 1,
                   sims: int = N_POWER_SIMS, seed: int = 0) -> float:
    rng = random.Random(seed)
    threshold = alpha / max(1, corrections)
    hits = 0
    for _ in range(sims):
        a = sum(rng.random() < effect_rate for _ in range(n_per_arm))
        c = sum(rng.random() < baseline_rate for _ in range(n_per_arm))
        if _fisher_one_sided(a, n_per_arm - a, c, n_per_arm - c) < threshold:
            hits += 1
    return hits / sims


# ----------------------------------------------------------------- analysis

def load_arm(paths: list[str]) -> list[dict]:
    """Read finished runs; keep only those that cleared level 1 (the
    conditional denominator)."""
    out = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for run in data.get("game_runs", []):
            levels = int(run.get("levels_completed") or 0)
            if levels >= 1:
                out.append({"game": run.get("game_id"), "levels": levels,
                            "cleared_l2": levels >= 2})
    return out


def bootstrap_diff_ci(arm_a: list[bool], arm_b: list[bool], *, rng: random.Random,
                      resamples: int = BOOTSTRAP_RESAMPLES,
                      alpha: float = 0.05) -> tuple[float, float]:
    if not arm_a or not arm_b:
        return (float("nan"), float("nan"))
    diffs = []
    for _ in range(resamples):
        sa = [rng.choice(arm_a) for _ in arm_a]
        sb = [rng.choice(arm_b) for _ in arm_b]
        diffs.append(sum(sb) / len(sb) - sum(sa) / len(sa))
    diffs.sort()
    return (diffs[int(alpha / 2 * resamples)],
            diffs[min(resamples - 1, int((1 - alpha / 2) * resamples))])


def analyse(baseline_paths: list[str], candidate_paths: list[str], *,
            seed: int = 0) -> dict:
    base = load_arm(baseline_paths)
    cand = load_arm(candidate_paths)
    base_flags = [r["cleared_l2"] for r in base]
    cand_flags = [r["cleared_l2"] for r in cand]
    a, c = sum(cand_flags), sum(base_flags)
    b, d = len(cand_flags) - a, len(base_flags) - c
    p_raw = _fisher_one_sided(a, b, c, d) if (a + b and c + d) else float("nan")
    rng = random.Random(seed)
    lo, hi = bootstrap_diff_ci(base_flags, cand_flags, rng=rng)
    base_rate = (c / len(base_flags)) if base_flags else float("nan")
    cand_rate = (a / len(cand_flags)) if cand_flags else float("nan")
    achieved = (simulate_power(n_per_arm=min(len(base_flags), len(cand_flags)),
                               baseline_rate=base_rate,
                               effect_rate=cand_rate,
                               corrections=N_ARMS_FOR_CORRECTION, seed=seed)
                if base_flags and cand_flags else float("nan"))
    return {
        "n_baseline": len(base_flags), "n_candidate": len(cand_flags),
        "baseline_l2_rate": base_rate, "candidate_l2_rate": cand_rate,
        "difference": cand_rate - base_rate,
        "p_one_sided_raw": p_raw,
        "p_bonferroni": min(1.0, p_raw * N_ARMS_FOR_CORRECTION),
        "ci95_difference": [lo, hi],
        "achieved_power_at_observed_effect": achieved,
        "meets_n_requirement": min(len(base_flags), len(cand_flags)) >= 50,
    }


# --------------------------------------------------------------------- cli

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("select", help="apply the game selection rule")
    s.add_argument("--benchmark", default="../example-run/benchmark.json")

    p = sub.add_parser("power", help="simulate achieved power")
    p.add_argument("--n", type=int, nargs="+", default=[10, 25, 50, 75])
    p.add_argument("--baseline-rate", type=float, default=0.30)
    p.add_argument("--effect-rate", type=float, default=0.60)
    p.add_argument("--corrections", type=int, default=1)

    a = sub.add_parser("analyse", help="Fisher + bootstrap CI over runs")
    a.add_argument("--baseline", nargs="+", required=True)
    a.add_argument("--candidate", nargs="+", required=True)

    args = ap.parse_args(argv)

    if args.command == "select":
        rows = historical_rates(Path(args.benchmark))
        chosen = select_games(rows)
        print(f"{'game':18} {'n':>4} {'L1 rate':>8} {'P(L2|L1)':>9}")
        for row in chosen:
            print(f"{row['game']:18} {row['n']:>4} {row['l1_rate']:8.2f} "
                  f"{row['cond_l2']:9.2f}")
        pooled_l1 = sum(r["l1_rate"] for r in chosen) / len(chosen)
        pooled_cond = sum(r["cond_l2"] for r in chosen) / len(chosen)
        print(f"\nselected={len(chosen)}/{len(rows)} pooled_l1={pooled_l1:.2f} "
              f"pooled_cond_l2={pooled_cond:.2f}")
        print(f"runs needed for n=50 level-1 clears per arm: "
              f"{50 / pooled_l1:.0f} per arm, {2 * 50 / pooled_l1:.0f} total")
        return 0

    if args.command == "power":
        print(f"baseline={args.baseline_rate} effect={args.effect_rate} "
              f"corrections={args.corrections}")
        for n in args.n:
            power = simulate_power(n_per_arm=n, baseline_rate=args.baseline_rate,
                                   effect_rate=args.effect_rate,
                                   corrections=args.corrections)
            print(f"  n={n:>4} per arm -> power={power:.2f}")
        return 0

    result = analyse(sorted(sum((glob.glob(p) or [p] for p in args.baseline), [])),
                     sorted(sum((glob.glob(p) or [p] for p in args.candidate), [])))
    print(json.dumps(result, indent=2))
    if not result["meets_n_requirement"]:
        print("\nINSUFFICIENT: fewer than 50 level-1 clears in an arm; "
              "no configuration decision may be made from this data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
