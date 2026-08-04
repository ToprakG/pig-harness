"""Phase 4: renewal simulator + leave-one-game-out policy tournament.

Model: a pass is a renewal process over attempts drawn (with replacement)
from the game's observed run pool. Each pooled attempt is
``(first_clear | None, length, score)`` plus trailing-window covariates at
checkpoint times. Action count is the time currency (the real budget is
wall-clock; mean actions/run ~= 137 is used as the pass budget).

Per-attempt timeline under a policy: the policy may schedule a restart at
deadline ``R`` (only if the attempt has not cleared by ``R``). The attempt
then ends at the earliest of natural end ``L``, restart ``R``, or budget
exhaustion. Credit: full score on natural end; ``partial_credit * score``
when budget-truncated after a first clear; zero otherwise (including
policy restarts, which by construction abandoned an uncleared attempt).

Policies: ``no_restart`` (baseline), ``fixed(T)``, ``gated(T1,T2,noop,nov)``,
``luby(base)``, and the deploy candidate ``mixture_adaptive(T, phi)`` which
carries the hard deploy caps (min actions before restart, max restarts per
pass, budget reserve) and a sequential posterior over game type.

``--logo``: per fold, parameters are selected on the full 24-game train set
(mixture parameters are re-estimated from train games only) and evaluated on
the held-out game. Both selection and evaluation use **exact expectations**
(the pass process over a 20-attempt pool has a tiny state space), so point
estimates carry zero Monte-Carlo noise; the >=5 seeds drive the bootstrap
CI over per-game diffs. ``--mc-check`` cross-validates the exact values
against the MC sampler. ``--sensitivity``: partial-credit x budget grid.

Usage::

    uv run python -m inference.meta.simulate --logo
    uv run python -m inference.meta.simulate --logo --sensitivity
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from inference.meta.mixture import per_game_clear_rates, split_estimate

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

BUDGET_DEFAULT = 137
PARTIAL_CREDIT_DEFAULT = 0.5
SEEDS_DEFAULT = 5
BOOTSTRAP_RESAMPLES = 3000

# Deploy hard caps (mixture_adaptive only — they ship with the policy).
MIN_ACTIONS_BEFORE_RESTART = 30
MAX_RESTARTS_PER_PASS = 2
BUDGET_RESERVE_FRAC = 0.25

# Parameter grids. The mixture grid is intentionally small (overfitting risk
# on 25 games) — do not expand it.
GRID_FIXED = [30, 40, 50, 60, 80]
GRID_GATED = [(t1, t2, noop, nov) for t1 in (30, 40) for t2 in (70, 90)
              for noop in (0.10, 0.20) for nov in (0.60,)]
GRID_LUBY = [20, 30, 40]
GRID_MIXTURE = [(T, phi) for T in (50, 60) for phi in (0.15, 0.25, 0.35)]

CHECKPOINTS = (30, 40, 50, 60, 70, 80, 90)  # times where covariates are needed


def luby_seq(i: int) -> int:
    """i-th element (1-based) of the Luby sequence: 1,1,2,1,1,2,4,..."""
    if i == 1:
        return 1
    k = i.bit_length()  # 2^(k-1) <= i < 2^k
    if i == (1 << k) - 1:
        return 1 << (k - 1)
    return luby_seq(i - (1 << (k - 1)) + 1)


class Attempt:
    """One pooled trajectory: outcome summary + covariates at checkpoints."""

    __slots__ = ("fc", "length", "score", "cov")

    def __init__(self, fc, length, score, cov):
        self.fc = fc          # first-clear action or None
        self.length = length  # total actions (natural end)
        self.score = score    # observed final score of the run
        self.cov = cov        # {t: (noop_20, novelty_30)} at checkpoints


def load_pools(artifacts_dir: str) -> dict[str, list[Attempt]]:
    import pandas as pd

    runs = pd.read_parquet(os.path.join(artifacts_dir, "runs_summary.parquet"))
    features = pd.read_parquet(os.path.join(artifacts_dir, "features.parquet"))
    feats_by_run = {k: g.sort_values("action_num")
                    for k, g in features.groupby(["game", "pass_num"])}
    pools: dict[str, list[Attempt]] = {}
    for r in runs.itertuples():
        g = feats_by_run[(r.game, r.pass_num)]
        changed = g.board_changed.to_numpy()
        hashes = list(g.board_hash.to_numpy())
        cov = {}
        for t in CHECKPOINTS:
            if t <= len(hashes):
                noop = 1.0 - float(changed[max(0, t - 20):t].mean())
                win = hashes[max(0, t - 30):t]
                cov[t] = (noop, len(set(win)) / len(win))
        fc = int(r.clear_actions[0]) if len(r.clear_actions) else None
        pools.setdefault(r.game, []).append(
            Attempt(fc, int(r.total_actions), float(r.final_score or 0.0), cov))
    return pools


def posterior_tractable(f: int, pi: float, p_e: float, p_h: float) -> float:
    """P(tractable | f failed attempts with no clear by T)."""
    a = pi * (1 - p_e) ** f
    b = (1 - pi) * (1 - p_h) ** f
    return a / (a + b) if a + b > 0 else 0.0


def planned_restart(policy: dict, attempt: Attempt, attempt_idx: int, fails: int,
                    spent: int, budget: int, restarts_used: int) -> int | None:
    """Return the deadline R at which this attempt would be restarted if it
    has not cleared by then, or None if the policy commits to the attempt."""
    kind = policy["kind"]
    if kind == "no_restart":
        return None
    if kind == "fixed":
        return policy["T"]
    if kind == "luby":
        return policy["base"] * luby_seq(attempt_idx + 1)
    if kind == "gated":
        t1, t2, noop_th, nov_th = policy["params"]
        c = attempt.cov.get(t1)
        # bad early signals (high no-op rate or low novelty) -> restart at T1
        if c is not None and (c[0] >= noop_th or c[1] <= nov_th):
            return t1
        return t2
    if kind == "mixture":
        T, phi = policy["T"], policy["phi"]
        if restarts_used >= MAX_RESTARTS_PER_PASS:
            return None
        if posterior_tractable(fails, *policy["mixture_params"]) < phi:
            return None  # commit the remaining budget to one long attempt
        if T < MIN_ACTIONS_BEFORE_RESTART:
            return None
        if (spent + T) > (1 - BUDGET_RESERVE_FRAC) * budget:
            return None  # no restarts in the final quarter of the budget
        return T
    raise ValueError(kind)


def simulate_pass(pool: list[Attempt], policy: dict, rng: np.random.Generator,
                  budget: int, partial_credit: float) -> float:
    """Simulate one pass; returns the credited pass score.

    An observed run ends when its wall-clock budget ends (mid-run engine
    game-overs and their auto-resets are already inside the observed
    trajectories), so a natural attempt end — or budget truncation — ends
    the pass. A policy restart is the only way a pass gets another attempt.
    """
    spent = 0
    fails = 0
    restarts_used = 0
    attempt_idx = 0
    while True:
        a = pool[rng.integers(len(pool))]
        rem = budget - spent
        r = planned_restart(policy, a, attempt_idx, fails, spent, budget,
                            restarts_used)
        attempt_idx += 1
        if (r is not None and (a.fc is None or a.fc > r)
                and r < min(a.length, rem)):
            # policy restart: abandon an uncleared attempt at its deadline
            spent += r
            fails += 1
            restarts_used += 1
            continue
        if a.length <= rem:
            return a.score  # natural end within remaining budget: full credit
        # budget truncation: partial credit only if it had already cleared
        if a.fc is not None and a.fc <= rem:
            return partial_credit * a.score
        return 0.0


def expected_score(pool, policy, budget, partial_credit) -> float:
    """Exact expected pass score under the policy.

    The pass process is a Markov chain over a 20-attempt pool with a tiny
    state space (spent, fails, restarts, attempt index), so the expectation
    is computed exactly by memoized recursion — zero Monte-Carlo noise.
    ``simulate_pass`` (the MC sampler) is kept as a cross-check.
    """
    memo: dict = {}

    def rec(spent, fails, restarts, idx):
        # attempt index only matters for luby's schedule
        key = (spent, fails, restarts, idx if policy["kind"] == "luby" else 0)
        if key in memo:
            return memo[key]
        total = 0.0
        for a in pool:
            rem = budget - spent
            r = planned_restart(policy, a, idx, fails, spent, budget, restarts)
            if (r is not None and (a.fc is None or a.fc > r)
                    and r < min(a.length, rem)):
                total += rec(spent + r, fails + 1, restarts + 1, idx + 1)
            elif a.length <= rem:
                total += a.score
            elif a.fc is not None and a.fc <= rem:
                total += partial_credit * a.score
        memo[key] = total / len(pool)
        return memo[key]

    return rec(0, 0, 0, 0)


def make_policies(train_runs_df) -> dict[str, list[dict]]:
    """Instantiate every policy/parameter combo. Mixture parameters are
    estimated from the training games only."""
    combos: dict[str, list[dict]] = {
        "no_restart": [{"kind": "no_restart", "label": "baseline"}],
        "fixed": [{"kind": "fixed", "T": T, "label": f"fixed(T={T})"}
                  for T in GRID_FIXED],
        "gated": [{"kind": "gated", "params": p, "fail_T": p[1],
                   "label": f"gated(T1={p[0]},T2={p[1]},noop={p[2]},nov={p[3]})"}
                  for p in GRID_GATED],
        "luby": [{"kind": "luby", "base": b, "fail_T": 60,
                  "label": f"luby(base={b})"} for b in GRID_LUBY],
    }
    mix = []
    for T, phi in GRID_MIXTURE:
        est = split_estimate(per_game_clear_rates(train_runs_df, T))
        mix.append({"kind": "mixture", "T": T, "phi": phi,
                    "mixture_params": (est["pi"], est["p_E"], est["p_H"]),
                    "label": f"mixture(T={T},phi={phi})"})
    combos["mixture_adaptive"] = mix
    return combos


def run_logo(pools, runs_df, budget, partial_credit) -> dict:
    """The LOGO tournament (deterministic — selection and evaluation both use
    exact expectations). Returns per-policy per-game eval scores plus
    per-fold parameter choices."""
    games = sorted(pools)
    results: dict[str, dict] = {}
    families = list(make_policies(runs_df).keys())
    for fam in families:
        results[fam] = {"per_game_policy": {}, "per_game_baseline": {},
                        "chosen": {}}
    for holdout in games:
        train_games = [g for g in games if g != holdout]
        train_df = runs_df[runs_df.game != holdout]
        combos = make_policies(train_df)
        baseline = combos["no_restart"][0]
        base_eval = expected_score(pools[holdout], baseline, budget,
                                   partial_credit)
        for fam, cands in combos.items():
            if fam == "no_restart":
                best = baseline
            else:
                # full-train-set selection: maximize summed expected score
                scores = [sum(expected_score(pools[g], cand, budget,
                                             partial_credit)
                              for g in train_games)
                          for cand in cands]
                best = cands[int(np.argmax(scores))]
            ev = expected_score(pools[holdout], best, budget, partial_credit)
            results[fam]["per_game_policy"][holdout] = ev
            results[fam]["per_game_baseline"][holdout] = base_eval
            results[fam]["chosen"][holdout] = best["label"]
    return results


def summarize(fold_result: dict, rng: np.random.Generator) -> dict:
    games = sorted(fold_result["per_game_policy"])
    pol = np.array([fold_result["per_game_policy"][g] for g in games])
    base = np.array([fold_result["per_game_baseline"][g] for g in games])
    uplift = pol.sum() / base.sum() - 1.0
    idx = np.arange(len(games))
    boots = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        s = rng.choice(idx, size=len(idx), replace=True)
        if base[s].sum() > 0:
            boots.append(pol[s].sum() / base[s].sum() - 1.0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "uplift": float(uplift),
        "ci95": [float(lo), float(hi)],
        # tolerance guards against float-epsilon artifacts of the exact
        # recursion (a policy identical to baseline can differ by ~1e-18)
        "harmed_games": int((pol < base - 1e-9).sum()),
        "chosen": fold_result["chosen"],
    }


def tournament(pools, runs_df, budget, partial_credit, seeds,
               quiet=False) -> dict:
    """Run the (deterministic) LOGO once; seeds drive the bootstrap RNG.

    Point estimates carry zero simulator noise (exact expectations), so the
    seed spread shows up only in the bootstrap CI bounds; the min-max range
    of the uplift itself is degenerate by construction.
    """
    fold_by_fam = run_logo(pools, runs_df, budget, partial_credit)
    per_seed: dict[str, list[dict]] = {}
    for seed in seeds:
        boot_rng = np.random.default_rng(seed + 10_000)
        for fam, res in fold_by_fam.items():
            per_seed.setdefault(fam, []).append(summarize(res, boot_rng))
        if not quiet:
            print(f"  bootstrap seed {seed} done")
    out = {}
    for fam, summaries in per_seed.items():
        ups = [s["uplift"] for s in summaries]
        out[fam] = {
            "uplift_median": float(np.median(ups)),
            "uplift_min": float(np.min(ups)),
            "uplift_max": float(np.max(ups)),
            "ci95_median": [float(np.median([s["ci95"][0] for s in summaries])),
                            float(np.median([s["ci95"][1] for s in summaries]))],
            "harmed_median": float(np.median([s["harmed_games"] for s in summaries])),
            "per_seed": summaries,
        }
    return out


def selection_stability(result: dict) -> dict:
    """How often the modal parameter choice was selected across folds
    (median across seeds)."""
    stab = {}
    for fam, r in result.items():
        if fam == "no_restart":
            continue
        counts = []
        for s in r["per_seed"]:
            chosen = list(s["chosen"].values())
            modal = max(set(chosen), key=chosen.count)
            counts.append(chosen.count(modal))
        stab[fam] = {"modal_folds_median": float(np.median(counts)),
                     "n_folds": len(s["chosen"])}
    return stab


def print_table(result: dict) -> None:
    print(f"\n{'policy':>18} {'uplift(med)':>12} {'range':>18} "
          f"{'95% CI (med)':>18} {'harmed':>7}")
    for fam in ("fixed", "gated", "luby", "mixture_adaptive"):
        r = result[fam]
        print(f"{fam:>18} {r['uplift_median']:>+11.1%} "
              f"[{r['uplift_min']:>+6.1%},{r['uplift_max']:>+6.1%}] "
              f"[{r['ci95_median'][0]:>+6.1%},{r['ci95_median'][1]:>+6.1%}] "
              f"{r['harmed_median']:>7.0f}")


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts-dir", default=ARTIFACTS_DIR)
    ap.add_argument("--logo", action="store_true", help="run the LOGO tournament")
    ap.add_argument("--sensitivity", action="store_true",
                    help="partial-credit x budget sensitivity grid")
    ap.add_argument("--seeds", type=int, default=SEEDS_DEFAULT)
    ap.add_argument("--budget", type=int, default=BUDGET_DEFAULT)
    ap.add_argument("--partial-credit", type=float, default=PARTIAL_CREDIT_DEFAULT)
    ap.add_argument("--mc-check", action="store_true",
                    help="cross-check exact expectations against MC sampling")
    args = ap.parse_args(argv)

    runs_df = pd.read_parquet(os.path.join(args.artifacts_dir, "runs_summary.parquet"))
    pools = load_pools(args.artifacts_dir)
    seeds = list(range(1, args.seeds + 1))
    report: dict = {"config": {"budget": args.budget,
                               "partial_credit": args.partial_credit,
                               "seeds": seeds,
                               "evaluation": "exact-expectation"}}

    if args.mc_check:
        rng = np.random.default_rng(0)
        combos = make_policies(runs_df)
        print("== MC cross-check (20k passes vs exact) ==")
        for fam in ("fixed", "mixture_adaptive"):
            cand = combos[fam][0]
            for g in sorted(pools)[:3]:
                exact = expected_score(pools[g], cand, args.budget,
                                       args.partial_credit)
                mc = float(np.mean([simulate_pass(pools[g], cand, rng,
                                                  args.budget,
                                                  args.partial_credit)
                                    for _ in range(20_000)]))
                print(f"  {cand['label']:>24} {g}: exact {exact:.4f}  mc {mc:.4f}")

    if args.logo:
        print(f"LOGO tournament: budget={args.budget} "
              f"partial_credit={args.partial_credit} seeds={seeds}")
        result = tournament(pools, runs_df, args.budget, args.partial_credit,
                            seeds)
        print_table(result)
        stab = selection_stability(result)
        print("\nparameter-selection stability (modal choice folds, median over seeds):")
        for fam, s in stab.items():
            print(f"  {fam}: {s['modal_folds_median']:.0f}/{s['n_folds']}")
        report["tournament"] = result
        report["stability"] = stab

        mix = result["mixture_adaptive"]
        ok = mix["uplift_median"] > 0 and mix["ci95_median"][0] > 0
        print(f"\n[{'PASS' if ok else 'FAIL'}] mixture_adaptive median uplift "
              f"{mix['uplift_median']:+.1%} > 0 with CI lower bound "
              f"{mix['ci95_median'][0]:+.1%} > 0")
        harmed = {f: result[f]["harmed_median"] for f in
                  ("fixed", "gated", "luby", "mixture_adaptive")}
        least = min(harmed.values())
        ord_ok = harmed["mixture_adaptive"] <= least
        print(f"[{'PASS' if ord_ok else 'FAIL'}] mixture_adaptive lowest "
              f"harmed-game count: {harmed}")
        ok &= ord_ok
    else:
        ok = True

    if args.sensitivity:
        print("\n== Sensitivity grid (mixture_adaptive vs baseline) ==")
        grid = {}
        for pc in (0.0, 0.5, 1.0):
            for b in (110, 137, 165):
                res = tournament(pools, runs_df, b, pc, seeds, quiet=True)
                m = res["mixture_adaptive"]
                grid[f"pc={pc},B={b}"] = {
                    "uplift_median": m["uplift_median"],
                    "ci95_median": m["ci95_median"],
                    "harmed_median": m["harmed_median"],
                }
                print(f"  pc={pc} B={b}: uplift {m['uplift_median']:+.1%} "
                      f"CI [{m['ci95_median'][0]:+.1%},{m['ci95_median'][1]:+.1%}] "
                      f"harmed {m['harmed_median']:.0f}")
        report["sensitivity"] = grid

    with open(os.path.join(args.artifacts_dir, "logo_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nSIMULATE:", "ALL PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
