"""Replicate protocol: aggregate repeated runs of one config, or compare two.

Tufa report sigma ~ 0.45 on the public game set, and the *same* Kaggle
submission has scored between 0.77 and 1.30. A single run therefore carries
no information about a change smaller than ~0.5. This tool makes the
replicate count explicit and refuses to support a decision below
``--min-replicates`` (default 4).

It aggregates finished runs; it does not launch them. Produce runs with
``make fast-eval`` / ``make run`` and point this at their ``score.json``.

Usage::

    uv run python -m inference.eff.replicate summarise --runs runs/A*/score.json
    uv run python -m inference.eff.replicate compare \\
        --baseline runs/A*/score.json --candidate runs/B*/score.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import statistics
from pathlib import Path

BOOTSTRAP_RESAMPLES = 10000
DEFAULT_MIN_REPLICATES = 4


def load_scores(path: str) -> dict[str, float]:
    """Per-game score from a score.json (mean over passes when several)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for game_id, payload in data.get("games", {}).items():
        seeds = payload.get("seed_scores") or {}
        out[game_id] = (statistics.fmean(float(v) for v in seeds.values())
                        if seeds else float(payload.get("score", 0.0)))
    return out


def expand(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or ([pattern] if Path(pattern).exists() else []))
    return paths


def bootstrap_ci(values: list[float], rng: random.Random,
                 resamples: int = BOOTSTRAP_RESAMPLES,
                 alpha: float = 0.05) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    n = len(values)
    means = sorted(statistics.fmean(rng.choice(values) for _ in range(n))
                   for _ in range(resamples))
    return (means[int(alpha / 2 * resamples)],
            means[min(resamples - 1, int((1 - alpha / 2) * resamples))])


def summarise(paths: list[str], *, min_replicates: int,
              rng: random.Random) -> dict:
    run_means = []
    for path in paths:
        scores = load_scores(path)
        run_means.append(statistics.fmean(scores.values()) if scores else 0.0)
    n = len(run_means)
    sd = statistics.stdev(run_means) if n > 1 else float("nan")
    lo, hi = bootstrap_ci(run_means, rng)
    return {"replicates": n, "run_means": run_means,
            "mean": statistics.fmean(run_means) if run_means else 0.0,
            "stdev": sd, "stderr": sd / math.sqrt(n) if n > 1 else float("nan"),
            "ci95": [lo, hi], "sufficient": n >= min_replicates,
            "min_replicates": min_replicates, "paths": paths}


def compare(baseline: list[str], candidate: list[str], *, min_replicates: int,
            rng: random.Random) -> dict:
    """Paired per-game comparison, bootstrapped over games."""
    def per_game(paths: list[str]) -> dict[str, list[float]]:
        acc: dict[str, list[float]] = {}
        for path in paths:
            for game, score in load_scores(path).items():
                acc.setdefault(game, []).append(score)
        return acc

    base, cand = per_game(baseline), per_game(candidate)
    games = sorted(set(base) & set(cand))
    diffs = [statistics.fmean(cand[g]) - statistics.fmean(base[g]) for g in games]
    lo, hi = bootstrap_ci(diffs, rng)
    reps = min(len(baseline), len(candidate))
    return {"games": len(games), "baseline_replicates": len(baseline),
            "candidate_replicates": len(candidate),
            "mean_paired_diff": statistics.fmean(diffs) if diffs else 0.0,
            "ci95": [lo, hi],
            "significant": (not math.isnan(lo)) and (lo > 0 or hi < 0),
            "sufficient": reps >= min_replicates,
            "min_replicates": min_replicates,
            "per_game_diff": dict(zip(games, diffs))}


def verdict(result: dict) -> str:
    if not result.get("sufficient"):
        return (f"INSUFFICIENT: {result['min_replicates']} replicates required; "
                "no configuration decision may be made from this data.")
    if "mean_paired_diff" in result:
        return ("SIGNAL: the 95% CI on the paired difference excludes zero."
                if result["significant"] else
                "NO SIGNAL: the 95% CI on the paired difference includes zero.")
    return "OK"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-replicates", type=int, default=DEFAULT_MIN_REPLICATES)
    ap.add_argument("--seed", type=int, default=0)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("summarise", help="mean/SE/bootstrap CI over replicates")
    s.add_argument("--runs", nargs="+", required=True)
    c = sub.add_parser("compare", help="paired difference between two configs")
    c.add_argument("--baseline", nargs="+", required=True)
    c.add_argument("--candidate", nargs="+", required=True)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    if args.command == "summarise":
        paths = expand(args.runs)
        if not paths:
            raise SystemExit("no score.json files matched")
        result = summarise(paths, min_replicates=args.min_replicates, rng=rng)
        print(f"replicates={result['replicates']} mean={result['mean']:.3f} "
              f"sd={result['stdev']:.3f} se={result['stderr']:.3f} "
              f"ci95=[{result['ci95'][0]:.3f}, {result['ci95'][1]:.3f}]")
    else:
        base, cand = expand(args.baseline), expand(args.candidate)
        if not base or not cand:
            raise SystemExit("no score.json files matched for one side")
        result = compare(base, cand, min_replicates=args.min_replicates, rng=rng)
        print(f"games={result['games']} "
              f"baseline_replicates={result['baseline_replicates']} "
              f"candidate_replicates={result['candidate_replicates']}")
        print(f"mean_paired_diff={result['mean_paired_diff']:+.3f} "
              f"ci95=[{result['ci95'][0]:+.3f}, {result['ci95'][1]:+.3f}]")
    print(verdict(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
