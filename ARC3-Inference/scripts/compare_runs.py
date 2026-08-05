#!/usr/bin/env python3
"""Compare harness runs arm-by-arm, with the noise context needed to read them.

    ./scripts/compare_runs.py runs/*abl-*

The first argument is treated as the baseline. For each arm this reports mean
score, the spread across seeds, mean scored actions, and how deep runs got.

Actions are reported next to score on purpose: the ARC-AGI-3 level score divides
by actions *squared*, so an arm that reaches the goal but spends more actions can
score worse while looking like progress. `make significance` is the gate for
merging; this is the quick read that tells you whether it is worth running.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def collect_runs(run_dir: Path) -> list[dict[str, Any]]:
    """Pull every per-game run record out of a saved benchmark."""
    benchmark_path = run_dir / "benchmark.json"
    if not benchmark_path.is_file():
        return []
    found: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if {"levels_completed", "number_of_levels", "actions_per_level"} <= node.keys():
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(benchmark_path.read_text()))
    return found


def score_of(run: dict[str, Any]) -> float:
    """Mirror taaf.game.GameRun._compute_final_score for runs saved without one."""
    persisted = run.get("final_score")
    if isinstance(persisted, (int, float)):
        return float(persisted)
    baselines = run.get("base_actions_per_level")
    levels = int(run.get("number_of_levels") or 0)
    if not baselines or not levels:
        return 0.0
    per_level = run.get("actions_per_level") or []
    completed = int(run.get("levels_completed") or 0)
    total_score = 0.0
    total_weight = 0
    scoring_weight = 0
    for index in range(levels):
        weight = index + 1
        total_weight += weight
        actions = per_level[index] if index < len(per_level) else 0
        level_score = (
            min(115.0, (baselines[index] / actions) ** 2 * 100)
            if index < completed and actions > 0
            else 0.0
        )
        if level_score > 0:
            scoring_weight += weight
        total_score += level_score * weight
    if not total_weight:
        return 0.0
    return min(total_score / total_weight, scoring_weight / total_weight * 100)


def summarize(run_dir: Path) -> dict[str, Any]:
    runs = collect_runs(run_dir)
    by_game: dict[str, list[tuple[float, int, int]]] = defaultdict(list)
    for run in runs:
        by_game[str(run.get("game_id", "?"))].append(
            (score_of(run), sum(run.get("actions_per_level") or []), int(run.get("levels_completed") or 0))
        )
    games = {
        game: {
            "score": statistics.mean(s for s, _, _ in rows),
            "spread": statistics.pstdev([s for s, _, _ in rows]) if len(rows) > 1 else 0.0,
            "actions": statistics.mean(a for _, a, _ in rows),
            "levels": statistics.mean(lv for _, _, lv in rows),
            "zero_rate": sum(1 for s, _, _ in rows if s == 0) / len(rows),
            "n": len(rows),
        }
        for game, rows in sorted(by_game.items())
    }
    overall = statistics.mean(v["score"] for v in games.values()) if games else 0.0
    return {"name": run_dir.name, "games": games, "overall": overall}


def main(argv: list[str]) -> int:
    dirs = [Path(arg) for arg in argv[1:]]
    arms = [summarize(d) for d in dirs if d.is_dir()]
    arms = [arm for arm in arms if arm["games"]]
    if not arms:
        print("no runs with a readable benchmark.json", file=sys.stderr)
        return 1

    header = f"{'arm':<26}{'game':<16}{'n':>3}{'score':>9}{'±':>8}{'actions':>9}{'levels':>8}{'zero':>7}"
    print(header)
    print("-" * len(header))
    for arm in arms:
        for game, stats in arm["games"].items():
            print(
                f"{arm['name'][:25]:<26}{game[:15]:<16}{stats['n']:>3}"
                f"{stats['score']:>9.2f}{stats['spread']:>8.2f}"
                f"{stats['actions']:>9.1f}{stats['levels']:>8.2f}{stats['zero_rate']:>6.0%}"
            )
        print(f"{'':<26}{'MEAN':<16}{'':>3}{arm['overall']:>9.2f}")
        print()

    baseline = arms[0]
    if len(arms) > 1:
        print(f"vs baseline ({baseline['name']}):")
        for arm in arms[1:]:
            shared = sorted(set(arm["games"]) & set(baseline["games"]))
            if not shared:
                continue
            deltas = [arm["games"][g]["score"] - baseline["games"][g]["score"] for g in shared]
            act_deltas = [arm["games"][g]["actions"] - baseline["games"][g]["actions"] for g in shared]
            worst_spread = max(baseline["games"][g]["spread"] for g in shared)
            delta = statistics.mean(deltas)
            verdict = (
                "inside noise — proves nothing"
                if abs(delta) <= worst_spread
                else "outside this run's spread, still needs make significance"
            )
            print(
                f"  {arm['name'][:32]:<34}score {delta:+6.2f}   actions {statistics.mean(act_deltas):+7.1f}"
                f"   ({verdict})"
            )
        print()
        print(
            "Reference: a single 25-game pass has std 0.448. Per-arm n here is small;\n"
            "treat every delta as a hypothesis until it survives ~20 seeds on held-out games."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
