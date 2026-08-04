"""M2 covariate hazard layer: discrete-time logistic hazard model.

Person-period construction: one row per (run, 10-action period) while the run
is alive and has not yet had its first clear. Periods start at t=10 so that
the trailing covariate windows have history. Features (whitelisted only):

* ``log_t``       – log of the period start action number
* ``noop_20``     – no-op rate (board unchanged) over the trailing 20 actions
* ``novelty_30``  – fraction of distinct board hashes among the trailing 30
  actions (computable live from a 30-slot hash ring buffer)

The event is the first level clear falling inside the period. Fit is a plain
logistic regression (sklearn, effectively unregularized). Outputs
``hazard_fit.json`` with coefficients, AUC, and a quantile calibration table.

Usage::

    uv run python -m inference.meta.hazard
"""

from __future__ import annotations

import argparse
import json
import os

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

PERIOD = 10
NOOP_WINDOW = 20
NOVELTY_WINDOW = 30

# Expected fit region (see REPORT.md): coef signs are the acceptance
# criterion; magnitudes are indicative.
EXPECTED_SIGNS = {"log_t": -1, "noop_20": -1, "novelty_30": +1}
MIN_AUC = 0.72


def build_person_periods(runs, features):
    """Return (X rows as dicts, y events) for the discrete-time hazard fit."""
    import numpy as np

    rows = []
    feats_by_run = {k: g.sort_values("action_num")
                    for k, g in features.groupby(["game", "pass_num"])}
    for r in runs.itertuples():
        fc = int(r.clear_actions[0]) if len(r.clear_actions) else None
        length = int(r.total_actions)
        g = feats_by_run[(r.game, r.pass_num)]
        changed = g.board_changed.to_numpy()
        hashes = list(g.board_hash.to_numpy())
        end = min(fc, length) if fc is not None else length
        t = PERIOD
        while t < end:
            # at risk at period [t, t+PERIOD): uncleared and alive at t
            lo_noop = max(0, t - NOOP_WINDOW)
            lo_nov = max(0, t - NOVELTY_WINDOW)
            window = hashes[lo_nov:t]
            rows.append(
                {
                    "game": r.game,
                    "pass_num": int(r.pass_num),
                    "t": t,
                    "log_t": float(np.log(t)),
                    "noop_20": float(1.0 - changed[lo_noop:t].mean()),
                    "novelty_30": len(set(window)) / len(window),
                    "event": int(fc is not None and t < fc <= t + PERIOD),
                }
            )
            t += PERIOD
    return rows


def fit(rows):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    feats = ["log_t", "noop_20", "novelty_30"]
    X = np.array([[r[f] for f in feats] for r in rows])
    y = np.array([r["event"] for r in rows])
    clf = LogisticRegression(C=1e6, max_iter=5000)
    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    auc = float(roc_auc_score(y, p))

    # Quantile calibration: deciles of predicted hazard vs observed event rate
    order = np.argsort(p)
    calib = []
    for chunk in np.array_split(order, 10):
        calib.append(
            {
                "n": int(len(chunk)),
                "pred_mean": float(p[chunk].mean()),
                "obs_rate": float(y[chunk].mean()),
            }
        )
    coefs = {f: float(c) for f, c in zip(feats, clf.coef_[0])}
    return {
        "n_rows": int(len(rows)),
        "n_events": int(y.sum()),
        "intercept": float(clf.intercept_[0]),
        "coefficients": coefs,
        "auc": auc,
        "calibration": calib,
    }


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts-dir", default=ARTIFACTS_DIR)
    args = ap.parse_args(argv)

    runs = pd.read_parquet(os.path.join(args.artifacts_dir, "runs_summary.parquet"))
    features = pd.read_parquet(os.path.join(args.artifacts_dir, "features.parquet"))
    rows = build_person_periods(runs, features)
    result = fit(rows)

    print(f"person-period rows: {result['n_rows']}  events: {result['n_events']}")
    print("\n== Coefficients ==")
    print(f"{'feature':>12} {'coef':>8}")
    print(f"{'(intercept)':>12} {result['intercept']:>8.3f}")
    for f, c in result["coefficients"].items():
        print(f"{f:>12} {c:>8.3f}")
    print(f"\nAUC: {result['auc']:.3f}")
    print("\n== Quantile calibration (deciles of predicted hazard) ==")
    print(f"{'decile':>7} {'n':>6} {'pred':>8} {'obs':>8}")
    for i, c in enumerate(result["calibration"], 1):
        print(f"{i:>7} {c['n']:>6} {c['pred_mean']:>8.4f} {c['obs_rate']:>8.4f}")

    ok = True
    for f, sign in EXPECTED_SIGNS.items():
        got = result["coefficients"][f]
        sign_ok = got * sign > 0
        print(f"[{'PASS' if sign_ok else 'FAIL'}] sign({f}) expected "
              f"{'+' if sign > 0 else '-'}, got {got:.3f}")
        ok &= sign_ok
    auc_ok = result["auc"] >= MIN_AUC
    print(f"[{'PASS' if auc_ok else 'FAIL'}] AUC {result['auc']:.3f} >= {MIN_AUC}")
    ok &= auc_ok

    with open(os.path.join(args.artifacts_dir, "hazard_fit.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\nHAZARD:", "ALL PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
