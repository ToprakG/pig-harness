# Probabilistic-restart meta-strategy — analysis & deploy report

Branch: `feat/meta-calibration`. All numbers below are reproducible from the
committed pipeline:

```
uv sync --extra meta --extra dev
uv run python -m inference.meta.extract --verify   # Phase 1: 16/16 PASS
uv run python -m inference.meta.survival           # Phase 2 (M1): all PASS
uv run python -m inference.meta.hazard             # Phase 2 (M2): all PASS
uv run python -m inference.meta.mixture            # Phase 3 (M3): all PASS
uv run python -m inference.meta.simulate --logo --sensitivity   # Phase 4
uv run python -m inference.meta.guard              # genericity guard
uv run pytest tests/meta -q                        # 17 passed
```

Data: `example-run/` — 25 games x 20 passes = 500 full trajectories,
68,682 action events, joined with `score.json` per-pass scores.

## 1. Survival layer (M1)

500 runs; 252 with >=1 level clear, 248 never cleared. Clear-count
distribution {0: 248, 1: 198, 2: 51, 3: 3}; median first-clear action 25.5
among clearers; mean 137.4 actions/run; only 12 runs *end* on an engine
game-over (mid-run game-overs + auto-resets occur inside 329 runs).
P(2nd clear | 1st) = 0.214; inter-clear gap median 22.5, max 89.

Kaplan-Meier survival of first-clear time (right-censored at run length):

| t   | S(t)  |
|-----|-------|
| 40  | 0.672 |
| 100 | 0.526 |
| 200 | 0.462 |

Windowed conditional hazard (20-action bins, events / at-risk):

| window     | at risk | events | hazard |
|------------|---------|--------|--------|
| [0, 20)    | 500     | 99     | 0.198  |
| [20, 40)   | 401     | 64     | 0.160  |
| [40, 60)   | 337     | 35     | 0.104  |
| [60, 80)   | 289     | 18     | 0.062  |
| [80, 100)  | 241     | 16     | 0.066  |
| [100, 120) | 207     | 9      | 0.043  |
| [120, 140) | 171     | 7      | 0.041  |
| [140, 160) | 131     | 1      | 0.008  |
| [160, 180) | 99      | 1      | 0.010  |
| [180, 200) | 75      | 1      | 0.013  |

The hazard is strongly **decreasing**: an attempt that has not cleared early
is progressively less likely to clear per unit time.

Covariates at t=40 (runs alive & uncleared at 40; eventual clearers n=88 vs
never-clearers n=248):

| covariate                  | eventual clear | never |
|----------------------------|----------------|-------|
| first-40 no-op rate        | 0.042          | 0.122 |
| actions per LLM call       | 5.13           | 2.85  |

Actions-per-LLM-call separates the groups univariately but washes out in the
multivariate hazard fit; it is **diagnostic only** and excluded from the
policy inputs.

## 2. Covariate hazard layer (M2)

Discrete-time logistic hazard on person-period data: 10-action periods while
uncleared, periods starting at t=10; 4,510 rows, 205 events. Features:
`log(t)`, `noop_20` (trailing-20 no-op rate), `novelty_30` (fraction of
distinct board hashes in the trailing 30 actions — the definition a live
30-slot ring buffer computes; the alternative "globally-new-hash rate" is
confounded with time and fits with the wrong sign).

| term        | coefficient |
|-------------|-------------|
| (intercept) | -1.376      |
| log(t)      | -0.795      |
| noop_20     | -3.156      |
| novelty_30  | +1.706      |

AUC 0.769. Quantile calibration (deciles of predicted hazard):

| decile | n   | predicted | observed |
|--------|-----|-----------|----------|
| 1      | 451 | 0.0021    | 0.0067   |
| 2      | 451 | 0.0075    | 0.0044   |
| 3      | 451 | 0.0143    | 0.0155   |
| 4      | 451 | 0.0211    | 0.0155   |
| 5      | 451 | 0.0276    | 0.0244   |
| 6      | 451 | 0.0345    | 0.0266   |
| 7      | 451 | 0.0449    | 0.0421   |
| 8      | 451 | 0.0599    | 0.0776   |
| 9      | 451 | 0.0871    | 0.0953   |
| 10     | 451 | 0.1560    | 0.1463   |

**Ship decision:** M2 stays diagnostic. In the LOGO tournament the
hazard-gated policy beat M3-alone by +2.8pp median — below the 3pp
threshold required to ship the gating layer.

## 3. Mixture over game type (M3 — the deployed policy)

Per-game P(any clear in a pass) is bimodal: 9 games <=0.20, 7 games >=0.80;
Beta moment fit Beta(0.45, 0.44) (U-shaped). Two-component parameters for
P(first clear <= T per attempt), split estimator vs binomial EM:

| T  | estimator | pi    | p_E   | p_H   |
|----|-----------|-------|-------|-------|
| 50 | split     | 0.320 | 0.838 | 0.141 |
| 50 | EM        | 0.319 | 0.838 | 0.142 |
| 60 | split     | 0.360 | 0.833 | 0.150 |
| 60 | EM        | 0.366 | 0.827 | 0.147 |

Sequential posterior: P(tractable | f failed attempts) =
pi(1-p_E)^f / (pi(1-p_E)^f + (1-pi)(1-p_H)^f). With the T=60 parameters the
posterior collapses fast: 0.36 (f=0) -> 0.10 (f=1) -> 0.02 (f=2). The
deployed policy therefore restarts an uncleared attempt at the deadline at
most **once** on posterior grounds, then commits the remaining budget to one
long attempt — which is exactly what the resistant games' late-clear tail
needs.

## 4. Renewal-reward justification (M4)

Restarting targets the renewal-reward rate
`psi(T) = E[S * 1{tau1 <= T}] / (E[L * 1{tau1 <= T}] + T * S(T))` — expected
score per unit budget when every attempt is abandoned at deadline T. The
classical result: with a **decreasing hazard** (Section 1's windowed table;
Section 2's negative log-t coefficient), truncating attempts at a deadline
raises the long-run reward rate, because the expected residual time to a
clear grows as an attempt ages. The mixture (Section 3) supplies the second
half of the argument: the population is a mix of tractable and resistant
games, so repeated failures are evidence of a resistant game, and the
optimal behavior switches from "restart at T" to "never restart" once the
posterior crosses phi — restarting forever would starve resistant games of
the long attempts they occasionally convert.

## 5. Policy tournament (leave-one-game-out)

Simulator: renewal over per-game pools of observed attempts
`(first_clear | None, length, score)`; budget 137 actions (the mean run
length — wall-clock is the real budget and action count its proxy); a
policy restart is the only way a pass gets a second attempt (natural end or
budget truncation ends the pass — mid-run engine game-overs are already
inside observed trajectories); budget-truncated attempts credit
`partial_credit x score` if they had cleared by the cutoff. Selection on the
full 24-game train set per fold; mixture parameters re-estimated from train
games only. Evaluation uses **exact expectations** (the pass process over a
20-attempt pool has a tiny state space), so point estimates carry zero
Monte-Carlo noise; the 5 seeds drive the 3,000-resample bootstrap CI over
per-game diffs. `--mc-check` cross-validates exact vs sampled values.

Default configuration (budget 137, partial credit 0.5):

| policy            | uplift (median) | seed range        | 95% CI (median)  | harmed games |
|-------------------|-----------------|-------------------|------------------|--------------|
| fixed(T)          | +16.8%          | [+16.8%, +16.8%]  | [+1.1%, +28.4%]  | 8            |
| gated(T1,T2)      | +15.3%          | [+15.3%, +15.3%]  | [+4.2%, +24.1%]  | 5            |
| luby(base)        | +17.8%          | [+17.8%, +17.8%]  | [+4.5%, +29.2%]  | 8            |
| **mixture_adaptive** | **+12.5%**   | [+12.5%, +12.5%]  | [+4.7%, +19.0%]  | **5**        |

(The seed ranges are degenerate by construction — exact evaluation has no
simulator noise; the reference analysis's ±3–4pp seed movement was
Monte-Carlo variance, eliminated here.)

* `mixture_adaptive` CI excludes zero and has the (joint-)lowest
  harmed-game count — the qualitative reference ordering reproduces.
* Parameter selection is stable in 25/25 folds for every family
  (`mixture(T=50, phi=0.15)` selected in all folds; the phi grid values are
  behaviorally identical because the posterior collapses after one failure).
* A harmed game is one whose exact expected score drops by more than 1e-9;
  the deployed policy harms 5 games with worst-case per-game expected-score
  drops of 0.07–0.40 points against baseline expected scores of ~1–7.

Sensitivity grid (mixture_adaptive vs baseline; uplift median, CI, harmed):

| partial credit | B=110                       | B=137                        | B=165                        |
|---------------|------------------------------|------------------------------|------------------------------|
| 0.0           | -2.5% [-8.4%, -0.1%], 7      | -0.2% [-10.5%, +7.1%], 9     | +4.2% [-9.4%, +12.5%], 9     |
| 0.5           | +10.2% [+1.9%, +17.6%], 6    | +12.5% [+4.7%, +19.0%], 5    | +12.4% [+4.2%, +19.4%], 7    |
| 1.0           | +15.7% [+4.4%, +25.3%], 6    | +19.7% [+11.7%, +27.9%], 4   | +19.9% [+12.0%, +28.7%], 3   |

The uplift is robust across budgets for partial credit 0.5–1.0 and vanishes
only under the pessimistic partial-credit-0 assumption (a truncated-but-
cleared attempt scores nothing) — in the real harness cleared levels do
score, so the operative row is 0.5–1.0.

## 6. D1 — RESET semantics (blocking question, resolved)

From `arcengine/base_game.py` (v0.9.3): `RESET` calls `handle_reset()`,
which performs a **full reset** (levels re-cloned, score to 0, level 0) only
when `_action_count == 0` or the game is in `WIN`; any mid-run RESET
performs a **level reset** — the current level is restored to its clean
state and score/level index are retained (`ONLY_RESET_LEVELS=true` forces
level-reset outside WIN). Consequences:

* Mid-run RESET does **not** wipe completed-level progress.
* v1 is doubly safe: restarts are issued only at level 1 (where the two
  semantics coincide and score is 0) and `post_clear_restart="forbidden"`
  blocks everything after a clear.
* Policy v2 could exploit cheap per-level retries; v1 must not, and the
  controller structurally cannot (any `level > 1` returns False).

## 7. D2 — restart cost c_reset

Measured on the 735 RESET actions inside example-run trajectories: the next
board-changing action follows a RESET after 0.13 actions on average
(median 0); the 10-action window after a RESET runs at 4.01 actions per LLM
call vs 3.57 run-wide — the agent replays known moves *faster* after a
reset. **c_reset ≈ 0 action-equivalents**, far below the 20-action threshold
that would have required re-running the Phase-4 tournament with restart
costs.

## 8. Live integration

* Config: `configs/inference.json` `meta` block (shipped `enabled: false`);
  Makefile passes it as `--meta-config`; `HarnessSolver.meta_config`.
* `_HarnessGameSession` keeps a 20-slot board-changed ring and a 30-slot
  board-hash ring (8-char md5 — board content is never stored), an
  attempt-relative action counter, and restart counters.
* The hook runs in `should_stop()` after all hard stop conditions and never
  fires mid-batch (`step_env`) or inside `analyzer.analyze` (which polls
  `should_stop` as a callback). A policy restart executes the existing
  `_execute_auto_reset()` and tags its viewer event `"meta_restart": true`
  for A/B attribution.
* Genericity guard (`inference.meta.guard`): policy inputs are whitelisted
  ({action_num, level, level_progress, t_since_level, noop_rate_w,
  novelty_rate_w, budget_frac, reward_seen, attempt_index,
  posterior_tractable}) and `inference/meta/` is source-scanned for board
  content / game identity tokens; a deliberately-dirty fixture proves the
  guard fails when it should. No behavior anywhere conditions on game
  identity or board content.
* No-op parity (D4): with meta disabled, a scripted mock session (single
  action, batch, no-op, engine game-over -> auto-reset, level clear)
  produces a viewer event sequence identical to a golden fixture generated
  from the pre-integration solver (`tests/meta/test_noop_parity.py`).

## 9. Live A/B instructions

1. Run the baseline: `meta.enabled: false` (as shipped), full 25-game set,
   n_passes >= 20 -> produces `<baseline_run>/score.json`.
2. Run the candidate: identical config except `meta.enabled: true` ->
   `<candidate_run>/score.json`.
3. Compare with the existing significance tooling (paired per-game
   bootstrap + permutation tests — do not reimplement):

   ```
   uv run inference-significance \
       --baseline <baseline_run> \
       --candidate <candidate_run> \
       --bootstrap-samples 10000
   ```

4. Attribution: candidate viewer events with `"meta_restart": true` identify
   policy-issued resets; per-pass restart counts should match the
   simulation's prediction (<= 1 posterior-driven restart per pass in the
   common case, hard cap 2).
5. Scoring caveat from the sensitivity grid: confirm the harness scoring
   credits levels cleared before the wall-clock cutoff (it does — scores
   accrue per level); the pc=0.0 row is the only configuration where the
   policy is not clearly positive.

## 10. Do not merge unless

* [ ] The live A/B 95% CI on the total-score uplift **excludes zero**
      (paired per-game bootstrap via `inference-significance`).
* [x] D1 resolved — RESET semantics documented (Section 6; level-reset
      mid-run; v1 restricted to level 1 where it is equivalent to a full
      restart).
* [x] No-op parity test green — `tests/meta/test_noop_parity.py` passes
      against the pre-integration golden fixture with meta disabled.

Additional standing gates, all currently green: `extract --verify` 16/16,
`survival`/`hazard`/`mixture` acceptance PASS, LOGO acceptance PASS
(uplift CI > 0, lowest harmed count), `guard` clean (and failing on the
dirty fixture), `pytest tests/meta` 17/17.
