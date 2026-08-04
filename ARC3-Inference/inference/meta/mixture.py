"""M3 mixture layer: two-component model of per-attempt clear probability.

The game population is modeled as a mixture of *tractable* games (per-attempt
probability ``p_E`` of a first clear by deadline ``T``) and *resistant* games
(``p_H``), with prior ``pi`` = P(tractable).

Two estimators, cross-checking each other:

* **rate-0.5 split** – classify each game by its empirical clear-by-T rate
  (>= 0.5 -> tractable); ``p_E``/``p_H`` are the mean rates within each class,
  ``pi`` the tractable fraction.
* **binomial EM** – 2-component binomial mixture over per-game clear counts
  (20 passes per game), fit by EM.

Also emits the structural bimodality summary (per-game any-clear rates and a
Beta moment fit) used in REPORT.md. Output: ``mixture_params.json``.

Usage::

    uv run python -m inference.meta.mixture
"""

from __future__ import annotations

import argparse
import json
import math
import os

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
DEADLINES = (50, 60)

# Reference split-based parameters at T=60; acceptance tolerance ±0.05.
REF_T60 = {"pi": 0.36, "p_E": 0.83, "p_H": 0.15}
ABS_TOL = 0.05


def per_game_clear_rates(runs, deadline: int) -> dict[str, tuple[int, int]]:
    """Map game -> (clears_by_deadline, attempts)."""
    out: dict[str, list[int]] = {}
    for r in runs.itertuples():
        fc = int(r.clear_actions[0]) if len(r.clear_actions) else None
        k, n = out.setdefault(r.game, [0, 0])
        out[r.game][1] += 1
        if fc is not None and fc <= deadline:
            out[r.game][0] += 1
    return {g: (k, n) for g, (k, n) in out.items()}


def split_estimate(rates: dict[str, tuple[int, int]]) -> dict:
    """Rate-0.5 split estimator."""
    tract = {g: k / n for g, (k, n) in rates.items() if k / n >= 0.5}
    resist = {g: k / n for g, (k, n) in rates.items() if k / n < 0.5}
    return {
        "pi": len(tract) / len(rates),
        "p_E": sum(tract.values()) / len(tract) if tract else float("nan"),
        "p_H": sum(resist.values()) / len(resist) if resist else float("nan"),
        "n_tractable": len(tract),
        "n_resistant": len(resist),
    }


def _log_binom_pmf(k: int, n: int, p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log(1 - p))


def em_estimate(rates: dict[str, tuple[int, int]], max_iter: int = 500,
                tol: float = 1e-8) -> dict:
    """2-component binomial mixture via EM over per-game (k, n) counts."""
    data = list(rates.values())
    pi, p_e, p_h = 0.5, 0.8, 0.2  # init: separated components
    ll_prev = -float("inf")
    converged = False
    for it in range(max_iter):
        # E-step
        resp = []
        ll = 0.0
        for k, n in data:
            a = math.log(pi) + _log_binom_pmf(k, n, p_e)
            b = math.log(1 - pi) + _log_binom_pmf(k, n, p_h)
            m = max(a, b)
            za = math.exp(a - m)
            zb = math.exp(b - m)
            resp.append(za / (za + zb))
            ll += m + math.log(za + zb)
        # M-step
        w = sum(resp)
        pi = w / len(data)
        p_e = sum(r * k for r, (k, n) in zip(resp, data)) / max(
            sum(r * n for r, (k, n) in zip(resp, data)), 1e-12)
        p_h = sum((1 - r) * k for r, (k, n) in zip(resp, data)) / max(
            sum((1 - r) * n for r, (k, n) in zip(resp, data)), 1e-12)
        if abs(ll - ll_prev) < tol:
            converged = True
            break
        ll_prev = ll
    # canonical order: p_E is the larger component
    if p_h > p_e:
        p_e, p_h, pi = p_h, p_e, 1 - pi
    return {"pi": pi, "p_E": p_e, "p_H": p_h, "log_likelihood": ll,
            "iterations": it + 1, "converged": converged}


def bimodality_summary(runs) -> dict:
    """Per-game P(any clear in a pass) + Beta moment fit (structural insight)."""
    by_game: dict[str, list[int]] = {}
    for r in runs.itertuples():
        by_game.setdefault(r.game, []).append(int(len(r.clear_actions) > 0))
    rates = {g: sum(v) / len(v) for g, v in by_game.items()}
    vals = list(rates.values())
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    # Beta method-of-moments fit
    common = mean * (1 - mean) / var - 1 if var > 0 else float("nan")
    return {
        "per_game_any_clear_rate": dict(sorted(rates.items())),
        "n_low_le_0.20": sum(1 for v in vals if v <= 0.20),
        "n_high_ge_0.80": sum(1 for v in vals if v >= 0.80),
        "beta_moment_fit": {"alpha": mean * common, "beta": (1 - mean) * common},
    }


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts-dir", default=ARTIFACTS_DIR)
    args = ap.parse_args(argv)

    runs = pd.read_parquet(os.path.join(args.artifacts_dir, "runs_summary.parquet"))

    result: dict = {"deadlines": {}}
    ok = True
    for T in DEADLINES:
        rates = per_game_clear_rates(runs, T)
        split = split_estimate(rates)
        em = em_estimate(rates)
        result["deadlines"][str(T)] = {"split": split, "em": em,
                                       "per_game_rate": {g: k / n for g, (k, n) in sorted(rates.items())}}
        print(f"\n== Deadline T={T} ==")
        print(f"  split: pi={split['pi']:.3f}  p_E={split['p_E']:.3f}  "
              f"p_H={split['p_H']:.3f}  (n_tract={split['n_tractable']}, "
              f"n_resist={split['n_resistant']})")
        print(f"  EM   : pi={em['pi']:.3f}  p_E={em['p_E']:.3f}  p_H={em['p_H']:.3f}  "
              f"(converged={em['converged']} in {em['iterations']} iters)")

    split60 = result["deadlines"]["60"]["split"]
    for key, want in REF_T60.items():
        got = split60[key]
        good = abs(got - want) <= ABS_TOL
        print(f"[{'PASS' if good else 'FAIL'}] split {key} @T=60: got {got:.3f}, "
              f"reference {want} (±{ABS_TOL})")
        ok &= good
    em60 = result["deadlines"]["60"]["em"]
    em_ok = em60["converged"] and all(
        abs(em60[k] - REF_T60[k]) <= 3 * ABS_TOL for k in ("pi", "p_E", "p_H"))
    print(f"[{'PASS' if em_ok else 'FAIL'}] EM @T=60 converged in same region: "
          f"pi={em60['pi']:.3f} p_E={em60['p_E']:.3f} p_H={em60['p_H']:.3f}")
    ok &= em_ok

    bimodal = bimodality_summary(runs)
    result["bimodality"] = bimodal
    print(f"\nbimodality: {bimodal['n_low_le_0.20']} games <=0.20, "
          f"{bimodal['n_high_ge_0.80']} games >=0.80, Beta moment fit "
          f"alpha={bimodal['beta_moment_fit']['alpha']:.2f} "
          f"beta={bimodal['beta_moment_fit']['beta']:.2f}")

    with open(os.path.join(args.artifacts_dir, "mixture_params.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\nMIXTURE:", "ALL PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
