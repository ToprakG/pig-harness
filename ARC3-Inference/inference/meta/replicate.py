"""Replicate protocol: run a config N times and report mean ± SE with a
bootstrap CI, or compare two configs on paired per-game differences.

Motivation (see inference/meta/REPORT.md): the public game set has
sigma ~ 0.45 and the *same* Kaggle submission has scored between 0.77 and
1.30. A single run therefore carries no information about a configuration
change smaller than ~0.5. This tool makes the replicate count explicit and
refuses to summarise fewer than ``--min-replicates`` (default 4).

It does not launch runs itself — runs are produced by ``make fast-eval`` /
``make run`` and land in ``runs/<stamp>/score.json``. This tool aggregates
those score files.

Usage::

    # summarise one config's replicates
    uv run python -m inference.meta.replicate summarise \\
        --runs runs/A1/score.json runs/A2/score.json ...

    # paired comparison of two configs (same games, replicate-matched)
    uv run python -m inference.meta.replicate compare \\
        --baseline runs/A*/score.json --candidate runs/B*/score.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from pathlib import Path

BOOTSTRAP_RESAMPLES = 10000
DEFAULT_MIN_REPLICATES = 4


def load_scores(path: str) -> dict[str, float]:
    """Per-game scores from a score.json (mean over passes when several)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for game_id, payload in data.get("games", {}).items():
        seed_scores = payload.get("seed_scores") or {}
        if seed_scores:
            out[game_id] = statistics.fmean(float(v) for v in seed_scores.values())
        else:
            out[game_id] = float(payload.get("score", 0.0))
    return out


def expand(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or ([pattern] if Path(pattern).exists() else []))
    return paths


def _bootstrap_ci(values: list[float], rng, resamples: int = BOOTSTRAP_RESAMPLES,
                  alpha: float = 0.05) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    means = []
    n = len(values)
    for _ in range(resamples):
        means.append(statistics.fmean(rng.choice(values) for _ in range(n)))
    means.sort()
    lo = means[int(alpha / 2 * resamples)]
    hi = means[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return (lo, hi)


def summarise(paths: list[str], *, min_replicates: int, rng) -> dict:
    per_run = [(p, load_scores(p)) for p in paths]
    run_means = [statistics.fmean(scores.values()) if scores else 0.0
                 for _p, scores in per_run]
    n = len(run_means)
    mean = statistics.fmean(run_means) if run_means else 0.0
    sd = statistics.stdev(run_means) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    lo, hi = _bootstrap_ci(run_means, rng)
    return {
        "replicates": n,
        "run_means": run_means,
        "mean": mean,
        "stdev": sd,
        "stderr": se,
        "ci95": [lo, hi],
        "sufficient": n >= min_replicates,
        "min_replicates": min_replicates,
        "paths": paths,
    }


def compare(baseline_paths: list[str], candidate_paths: list[str], *,
            min_replicates: int, rng) -> dict:
    """Paired per-game comparison: for each game, mean over baseline
    replicates vs mean over candidate replicates; bootstrap over games."""
    def per_game_mean(paths: list[str]) -> dict[str, list[float]]:
        acc: dict[str, list[float]] = {}
        for p in paths:
            for game, score in load_scores(p).items():
                acc.setdefault(game, []).append(score)
        return acc

    base = per_game_mean(baseline_paths)
    cand = per_game_mean(candidate_paths)
    games = sorted(set(base) & set(cand))
    diffs = [statistics.fmean(cand[g]) - statistics.fmean(base[g]) for g in games]
    mean_diff = statistics.fmean(diffs) if diffs else 0.0
    lo, hi = _bootstrap_ci(diffs, rng)
    n_reps = min(len(baseline_paths), len(candidate_paths))
    return {
        "games": len(games),
        "baseline_replicates": len(baseline_paths),
        "candidate_replicates": len(candidate_paths),
        "mean_paired_diff": mean_diff,
        "ci95": [lo, hi],
        "significant": (not math.isnan(lo)) and (lo > 0 or hi < 0),
        "sufficient": n_reps >= min_replicates,
        "min_replicates": min_replicates,
        "per_game_diff": dict(zip(games, diffs)),
    }


def _verdict(result: dict) -> str:
    if not result.get("sufficient"):
        return (f"INSUFFICIENT: {result['min_replicates']} replicates required; "
                "no configuration decision may be made from this data.")
    if "mean_paired_diff" in result:
        if not result["significant"]:
            return "NO SIGNAL: the 95% CI on the paired difference includes zero."
        return "SIGNAL: the 95% CI on the paired difference excludes zero."
    return "OK"


def main(argv: list[str] | None = None) -> int:
    import random

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-replicates", type=int, default=DEFAULT_MIN_REPLICATES)
    ap.add_argument("--seed", type=int, default=0)
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("summarise", help="mean/SE/bootstrap CI over replicates")
    s.add_argument("--runs", nargs="+", required=True,
                   help="score.json paths or globs")

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
        print(_verdict(result))
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
        print(_verdict(result))
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
