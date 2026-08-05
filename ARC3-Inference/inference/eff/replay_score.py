"""Recompute the official score from a finished run, and price efficiency.

Reads a run directory (its ``benchmark.json`` carries per-run
``actions_per_level`` / ``base_actions_per_level`` / ``levels_completed``;
``*_events.jsonl`` is used to cross-check action counts) and reports:

* the recomputed official score, validated against ``score.json``
* ``score_if_1x`` and ``headroom`` (what perfect efficiency would be worth)
* an alpha sweep: the score if every CLEARED level had used ``alpha x`` its
  actual actions, for alpha in {0.5, 0.7, 1.0}

A mismatch against ``score.json`` means our understanding of the formula is
wrong — the tool exits non-zero and says so rather than papering over it.

Usage::

    uv run python -m inference.eff.replay_score --run runs/<stamp>
    uv run python -m inference.eff.replay_score --run ../example-run
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from inference.eff.scoring import (
    cleared_human_mult,
    final_score,
    score_if_1x,
    score_with_alpha,
    to_bool,
)

ALPHAS = (0.5, 0.7, 1.0)


def load_runs(run_dir: Path) -> list[dict]:
    benchmark = run_dir / "benchmark.json"
    if not benchmark.is_file():
        raise SystemExit(f"no benchmark.json in {run_dir}")
    return json.loads(benchmark.read_text(encoding="utf-8")).get("game_runs", [])


def load_expected_scores(run_dir: Path) -> dict[str, list[float]]:
    """Per-game scores from score.json (one entry per pass)."""
    path = run_dir / "score.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[float]] = {}
    for game_id, payload in data.get("games", {}).items():
        seeds = payload.get("seed_scores") or {}
        out[game_id] = [float(v) for v in seeds.values()] or [
            float(payload.get("score", 0.0))]
    return out


def count_events_actions(run_dir: Path) -> dict[str, int]:
    """Actions per (game, pass) straight from the event stream — an
    independent check on ``actions_per_level`` sums."""
    out: dict[str, int] = {}
    for path in sorted((run_dir / "artifacts").glob("*_events.jsonl")):
        match = re.match(r"(?P<game>.+)_p(?P<pass>\d+)_events\.jsonl$", path.name)
        if not match:
            continue
        actions = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "action":
                    actions += 1
        out[f"{match.group('game')}_p{match.group('pass')}"] = actions
    return out


def replay(run_dir: Path) -> dict:
    runs = load_runs(run_dir)
    expected = load_expected_scores(run_dir)
    event_actions = count_events_actions(run_dir)

    rows, mismatches = [], []
    totals = {"score": 0.0, "if_1x": 0.0, **{f"alpha_{a}": 0.0 for a in ALPHAS}}
    hidden = 0
    pass_counter: dict[str, int] = {}

    for run in runs:
        game_id = run.get("game_id", "?")
        idx = pass_counter.get(game_id, 0)
        pass_counter[game_id] = idx + 1
        apl = list(run.get("actions_per_level") or [])
        bapl = run.get("base_actions_per_level")
        levels = int(run.get("levels_completed") or 0)
        n_levels = int(run.get("number_of_levels") or 0)
        if bapl is None:
            hidden += 1

        recomputed = final_score(actions_per_level=apl, base_actions_per_level=bapl,
                                 levels_completed=levels, number_of_levels=n_levels)
        stored = run.get("final_score")
        if stored is not None and abs(recomputed - float(stored)) > 1e-9:
            mismatches.append({"game": game_id, "recomputed": recomputed,
                               "stored": float(stored)})

        ideal = score_if_1x(base_actions_per_level=bapl, levels_completed=levels,
                            number_of_levels=n_levels)
        row = {
            "game": game_id, "pass": idx, "levels": levels,
            "actions_per_level": apl,
            "actions_total": sum(apl),
            "actions_from_events": event_actions.get(f"{game_id}_p{idx}"),
            "score": recomputed, "score_if_1x": ideal,
            "headroom": ideal - recomputed,
            "cleared_human_mult": cleared_human_mult(apl, bapl, levels),
            "baselines_hidden": bapl is None,
        }
        for alpha in ALPHAS:
            value = score_with_alpha(actions_per_level=apl,
                                     base_actions_per_level=bapl,
                                     levels_completed=levels,
                                     number_of_levels=n_levels, alpha=alpha)
            row[f"alpha_{alpha}"] = value
            totals[f"alpha_{alpha}"] += value
        totals["score"] += recomputed
        totals["if_1x"] += ideal
        rows.append(row)

    # cross-check against score.json where both sides exist
    score_json_mismatches = []
    by_game: dict[str, list[float]] = {}
    for row in rows:
        by_game.setdefault(row["game"], []).append(row["score"])
    for game_id, scores in by_game.items():
        want = expected.get(game_id)
        if not want:
            continue
        got_mean = sum(scores) / len(scores)
        want_mean = sum(want) / len(want)
        if abs(got_mean - want_mean) > 1e-6:
            score_json_mismatches.append({"game": game_id, "recomputed": got_mean,
                                          "score_json": want_mean})

    return {"run_dir": str(run_dir), "n_runs": len(rows), "rows": rows,
            "totals": totals, "baselines_hidden_runs": hidden,
            "benchmark_mismatches": mismatches,
            "score_json_mismatches": score_json_mismatches}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory")
    ap.add_argument("--json", action="store_true", help="dump the full payload")
    ap.add_argument("--top", type=int, default=10,
                    help="show the N runs with the most headroom")
    args = ap.parse_args(argv)

    result = replay(Path(args.run))
    totals, n = result["totals"], result["n_runs"]
    print(f"runs={n} baselines_hidden={result['baselines_hidden_runs']}")
    print(f"score={totals['score']:.4f}  score_if_1x={totals['if_1x']:.4f}  "
          f"headroom={totals['if_1x'] - totals['score']:.4f}")
    for alpha in ALPHAS:
        value = totals[f"alpha_{alpha}"]
        ratio = value / totals["score"] if totals["score"] else float("nan")
        print(f"  alpha={alpha}: score={value:.4f} ({ratio:.2f}x actual)")

    worst = sorted((r for r in result["rows"] if r["headroom"]),
                   key=lambda r: -r["headroom"])[: args.top]
    if worst:
        print("\nmost headroom (cleared levels that were expensive):")
        for row in worst:
            mult = row["cleared_human_mult"]
            print(f"  {row['game']} p{row['pass']} levels={row['levels']} "
                  f"actions={row['actions_total']} "
                  f"human_mult={'n/a' if mult is None else f'{mult:.2f}'} "
                  f"score={row['score']:.3f} -> if_1x={row['score_if_1x']:.3f}")

    bad = result["benchmark_mismatches"] + result["score_json_mismatches"]
    if bad:
        print(f"\nFORMULA MISMATCH on {len(bad)} run(s) — our understanding of "
              "the scoring formula is wrong; stopping.")
        for item in bad[:5]:
            print("  ", item)
        return 1
    print("\nformula check: recomputed scores match benchmark.json and "
          "score.json exactly")
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
