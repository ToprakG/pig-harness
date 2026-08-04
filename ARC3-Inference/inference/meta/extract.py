"""Extract per-action features and run summaries from example-run event logs.

Reads every ``*_events.jsonl`` under the example-run artifacts directory and
emits two parquet files under ``inference/meta/artifacts/``:

* ``features.parquet``  – one row per (game, pass, action_num). Raw board
  content is never written; only a short md5 hash of ``board_ascii``.
* ``runs_summary.parquet`` – one row per run with total actions, the list of
  clear actions, the final score joined from ``score.json``, and the final
  ``run_status``.

``--verify`` recomputes the survival-block reference statistics and asserts
them (counts exact, continuous values within ±2%), printing PASS/FAIL per
statistic.

Usage::

    uv run python -m inference.meta.extract [--events-dir DIR] [--score-json F]
    uv run python -m inference.meta.extract --verify
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys

from inference.meta.stats import km_curve, km_survival_at, median, to_bool, windowed_hazard

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_EVENTS_DIR = os.path.join(_REPO_ROOT, "example-run", "artifacts")
DEFAULT_SCORE_JSON = os.path.join(_REPO_ROOT, "example-run", "score.json")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

_EVENTS_RE = re.compile(r"(?P<game>.+)_p(?P<pass>\d+)_events\.jsonl$")

# Survival-block reference statistics (see REPORT.md). Counts are exact;
# continuous values are checked within ±2%.
REFERENCE = {
    "n_runs": 500,
    "n_clearers": 252,
    "n_never": 248,
    "clear_count_dist": {0: 248, 1: 198, 2: 51, 3: 3},
    "median_first_clear": 25.5,
    "km": {40: 0.672, 100: 0.526, 200: 0.462},
    "hazard": {0: 0.198, 60: 0.062, 140: 0.008},
    "mean_actions": 137.4,
    "ended_game_over": 12,
    "p_second_given_first": 0.214,
    "gap_median": 22.5,
    "gap_max": 89,
}

REL_TOL = 0.02


def board_hash(board_ascii: str) -> str:
    """Short content hash of the ascii board. Raw board text is never stored."""
    return hashlib.md5(board_ascii.encode("utf-8")).hexdigest()[:8]


def parse_events_file(path: str) -> list[dict]:
    """Parse one events file into a list of per-action feature dicts."""
    rows = []
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if e.get("type") != "action":
                continue
            rows.append(
                {
                    "action_num": e["action_num"],
                    "level": e["level"],
                    "board_changed": to_bool(e.get("board_changed")),
                    "analysis_step": e.get("analysis_step"),
                    "board_hash": board_hash(e.get("board_ascii", "")),
                    "level_completed": to_bool(e.get("level_completed")),
                    "game_over": to_bool(e.get("game_over")),
                    "run_status": e.get("run_status"),
                }
            )
    return rows


def load_scores(score_json: str) -> dict[tuple[str, int], float]:
    """Map (game, pass) -> final score from score.json seed_scores."""
    with open(score_json) as f:
        data = json.load(f)
    out: dict[tuple[str, int], float] = {}
    for game, gdata in data["games"].items():
        for seed_key, score in gdata.get("seed_scores", {}).items():
            m = re.search(r"pass-(\d+)$", seed_key)
            if m:
                out[(game, int(m.group(1)))] = float(score)
    return out


def extract(events_dir: str, score_json: str):
    """Parse all runs. Returns (feature_rows, run_rows)."""
    scores = load_scores(score_json)
    files = sorted(glob.glob(os.path.join(events_dir, "*_events.jsonl")))
    if not files:
        raise SystemExit(f"no *_events.jsonl files found under {events_dir}")
    feature_rows: list[dict] = []
    run_rows: list[dict] = []
    for fp in files:
        m = _EVENTS_RE.match(os.path.basename(fp))
        if not m:
            continue
        game, pass_num = m.group("game"), int(m.group("pass"))
        actions = parse_events_file(fp)
        for a in actions:
            feature_rows.append({"game": game, "pass_num": pass_num, **{
                k: a[k] for k in ("action_num", "level", "board_changed",
                                  "analysis_step", "board_hash",
                                  "level_completed", "game_over")}})
        clear_actions = [a["action_num"] for a in actions if a["level_completed"]]
        run_rows.append(
            {
                "game": game,
                "pass_num": pass_num,
                "total_actions": len(actions),
                "clear_actions": clear_actions,
                "final_score": scores.get((game, pass_num)),
                "final_run_status": actions[-1]["run_status"] if actions else None,
                "ended_game_over": bool(actions) and actions[-1]["game_over"],
            }
        )
    return feature_rows, run_rows


def write_parquet(feature_rows: list[dict], run_rows: list[dict], out_dir: str) -> None:
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    features = pd.DataFrame(feature_rows)
    assert "board_ascii" not in features.columns  # hash only, never raw boards
    features.to_parquet(os.path.join(out_dir, "features.parquet"), index=False)
    pd.DataFrame(run_rows).to_parquet(os.path.join(out_dir, "runs_summary.parquet"), index=False)
    print(f"wrote {len(features)} feature rows, {len(run_rows)} run rows -> {out_dir}")


def _check(name: str, got, want, exact: bool) -> bool:
    if exact:
        ok = got == want
    else:
        ok = abs(got - want) <= REL_TOL * abs(want)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {got!r}, reference {want!r}"
          + ("" if exact else f" (±{REL_TOL:.0%})"))
    return ok


def verify(run_rows: list[dict]) -> bool:
    """Recompute the survival-block reference statistics and assert them."""
    ref = REFERENCE
    ok = True
    fc = [r["clear_actions"][0] if r["clear_actions"] else None for r in run_rows]
    lengths = [r["total_actions"] for r in run_rows]
    clearers = [r for r in run_rows if r["clear_actions"]]

    ok &= _check("runs", len(run_rows), ref["n_runs"], exact=True)
    ok &= _check("runs with >=1 clear", len(clearers), ref["n_clearers"], exact=True)
    ok &= _check("never cleared", len(run_rows) - len(clearers), ref["n_never"], exact=True)

    dist: dict[int, int] = {}
    for r in run_rows:
        dist[len(r["clear_actions"])] = dist.get(len(r["clear_actions"]), 0) + 1
    ok &= _check("clear-count distribution", dict(sorted(dist.items())),
                 ref["clear_count_dist"], exact=True)

    ok &= _check("median first-clear action",
                 median([r["clear_actions"][0] for r in clearers]),
                 ref["median_first_clear"], exact=False)

    curve = km_curve(fc, lengths)
    for t, want in ref["km"].items():
        ok &= _check(f"KM S({t})", round(km_survival_at(curve, t), 3), want, exact=False)

    hz = {h["t_lo"]: h["hazard"] for h in windowed_hazard(fc, lengths, bin_width=20)}
    for a, want in ref["hazard"].items():
        ok &= _check(f"hazard [{a},{a + 20})", round(hz.get(a, 0.0), 3), want, exact=False)

    ok &= _check("mean actions/run", round(sum(lengths) / len(lengths), 1),
                 ref["mean_actions"], exact=False)
    ok &= _check("runs ended by engine game_over",
                 sum(1 for r in run_rows if r["ended_game_over"]),
                 ref["ended_game_over"], exact=True)

    second = [r for r in clearers if len(r["clear_actions"]) >= 2]
    ok &= _check("P(2nd clear | 1st)", round(len(second) / len(clearers), 3),
                 ref["p_second_given_first"], exact=False)
    gaps = [r["clear_actions"][1] - r["clear_actions"][0] for r in second]
    ok &= _check("inter-clear gap median", median(gaps), ref["gap_median"], exact=False)
    ok &= _check("inter-clear gap max", max(gaps), ref["gap_max"], exact=True)
    return bool(ok)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events-dir", default=DEFAULT_EVENTS_DIR)
    ap.add_argument("--score-json", default=DEFAULT_SCORE_JSON)
    ap.add_argument("--out-dir", default=ARTIFACTS_DIR)
    ap.add_argument("--verify", action="store_true",
                    help="recompute and assert the reference statistics")
    args = ap.parse_args(argv)

    feature_rows, run_rows = extract(args.events_dir, args.score_json)
    write_parquet(feature_rows, run_rows, args.out_dir)
    if args.verify:
        if not verify(run_rows):
            print("VERIFY: FAIL", file=sys.stderr)
            return 1
        print("VERIFY: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
