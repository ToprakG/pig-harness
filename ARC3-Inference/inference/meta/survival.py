"""M1 survival layer: Kaplan-Meier, windowed hazard, and t=40 covariates.

Reads the parquet artifacts produced by ``inference.meta.extract`` and writes

* ``km_table.json``        – full KM step function + reference checkpoints
* ``hazard_windows.json``  – 20-action-bin conditional hazard
* ``covariates_t40.json``  – covariate comparison at t=40 between runs that
  eventually clear and runs that never clear (both alive & uncleared at 40)

plus human-readable stdout tables with PASS/FAIL against the reference
statistics (continuous tolerance ±2%, counts exact).

Usage::

    uv run python -m inference.meta.survival
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from inference.meta.stats import km_curve, km_survival_at, windowed_hazard

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

# Reference values for the t=40 covariate comparison (see REPORT.md).
REF_T40 = {
    "n_eventual_clear": 88,
    "n_never": 248,
    "noop_rate_40": {"eventual_clear": 0.042, "never": 0.122},
    "actions_per_llm_call_40": {"eventual_clear": 5.13, "never": 2.85},
}
REF_KM = {40: 0.672, 100: 0.526, 200: 0.462}
REF_HAZARD = {0: 0.198, 60: 0.062, 140: 0.008}
REL_TOL = 0.02


def _check(name: str, got, want, exact: bool = False) -> bool:
    ok = got == want if exact else abs(got - want) <= REL_TOL * abs(want)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {got!r}, reference {want!r}")
    return ok


def load_runs(artifacts_dir: str):
    import pandas as pd

    runs = pd.read_parquet(os.path.join(artifacts_dir, "runs_summary.parquet"))
    features = pd.read_parquet(os.path.join(artifacts_dir, "features.parquet"))
    return runs, features


def first_clear(clear_actions) -> int | None:
    return int(clear_actions[0]) if len(clear_actions) else None


def covariates_t40(runs, features, t: int = 40) -> dict:
    """Compare first-``t``-action covariates between runs alive & uncleared at
    ``t`` that eventually clear vs runs that never clear."""
    groups = {"eventual_clear": [], "never": []}
    for r in runs.itertuples():
        fc = first_clear(r.clear_actions)
        if r.total_actions < t or (fc is not None and fc <= t):
            continue  # not alive, or already cleared, at t
        groups["eventual_clear" if fc is not None else "never"].append(
            (r.game, r.pass_num))

    out: dict = {"t": t, "n": {k: len(v) for k, v in groups.items()}}
    stats: dict = {"noop_rate": {}, "actions_per_llm_call": {}}
    for name, keys in groups.items():
        noop_rates = []
        apc = []
        for game, pass_num in keys:
            f = features[(features.game == game) & (features.pass_num == pass_num)
                         & (features.action_num <= t)]
            noop_rates.append(1.0 - f.board_changed.mean())
            apc.append(len(f) / f.analysis_step.nunique())
        stats["noop_rate"][name] = sum(noop_rates) / len(noop_rates)
        stats["actions_per_llm_call"][name] = sum(apc) / len(apc)
    out.update(stats)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts-dir", default=ARTIFACTS_DIR)
    args = ap.parse_args(argv)

    runs, features = load_runs(args.artifacts_dir)
    fc = [first_clear(c) for c in runs.clear_actions]
    lengths = [int(x) for x in runs.total_actions]

    ok = True

    curve = km_curve(fc, lengths)
    print("\n== Kaplan-Meier first-clear survival ==")
    print(f"{'t':>6} {'S(t)':>8}")
    for t in (20, 40, 60, 80, 100, 140, 200, 300):
        print(f"{t:>6} {km_survival_at(curve, t):>8.3f}")
    for t, want in REF_KM.items():
        ok &= _check(f"S({t})", round(km_survival_at(curve, t), 3), want)

    windows = windowed_hazard(fc, lengths, bin_width=20)
    print("\n== Windowed conditional hazard (20-action bins) ==")
    print(f"{'window':>12} {'at_risk':>8} {'events':>7} {'hazard':>8}")
    for w in windows:
        print(f"[{w['t_lo']:>4},{w['t_hi']:>4}) {w['at_risk']:>8} {w['events']:>7} "
              f"{w['hazard']:>8.3f}")
    hz = {w["t_lo"]: w["hazard"] for w in windows}
    for a, want in REF_HAZARD.items():
        ok &= _check(f"hazard [{a},{a + 20})", round(hz.get(a, 0.0), 3), want)

    cov = covariates_t40(runs, features)
    print("\n== Covariates at t=40 (alive & uncleared) ==")
    print(f"{'covariate':>24} {'eventual_clear':>15} {'never':>8}")
    for key in ("noop_rate", "actions_per_llm_call"):
        print(f"{key:>24} {cov[key]['eventual_clear']:>15.3f} {cov[key]['never']:>8.3f}")
    print("(actions_per_llm_call is diagnostic only — excluded from the policy;"
          " it washes out in the multivariate fit)")
    ok &= _check("n eventual-clear at t=40", cov["n"]["eventual_clear"],
                 REF_T40["n_eventual_clear"], exact=True)
    ok &= _check("n never at t=40", cov["n"]["never"], REF_T40["n_never"], exact=True)
    for grp in ("eventual_clear", "never"):
        ok &= _check(f"noop_rate_40 {grp}", round(cov["noop_rate"][grp], 3),
                     REF_T40["noop_rate_40"][grp])
        ok &= _check(f"actions_per_llm_call_40 {grp}",
                     round(cov["actions_per_llm_call"][grp], 2),
                     REF_T40["actions_per_llm_call_40"][grp])

    os.makedirs(args.artifacts_dir, exist_ok=True)
    with open(os.path.join(args.artifacts_dir, "km_table.json"), "w") as f:
        json.dump({"curve": [{"t": t, "S": s} for t, s in curve],
                   "checkpoints": {str(t): km_survival_at(curve, t) for t in REF_KM}}, f, indent=2)
    with open(os.path.join(args.artifacts_dir, "hazard_windows.json"), "w") as f:
        json.dump(windows, f, indent=2)
    with open(os.path.join(args.artifacts_dir, "covariates_t40.json"), "w") as f:
        json.dump(cov, f, indent=2)

    print("\nSURVIVAL:", "ALL PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
