# Meta-calibration implementation notes

Findings that differ from (or refine) the task spec. Per the working rules,
these are reported, not silently skipped.

## Phase 1 — data findings

* **String-bool trap did not reproduce.** A full scan of all 500
  `*_events.jsonl` files (68,682 action events) found `board_changed`,
  `level_completed`, and `game_over` to be real JSON booleans in every file —
  zero `"True"`/`"False"` strings. `to_bool()` normalization is nonetheless
  applied everywhere as specified (it is harmless and defends against future
  logs that do stringify).
* **`run_status` at end of run.** The spec says 488/500 runs ended with
  `run_status="playing"`. Observed: **500/500** final action events have
  `run_status="playing"`. The "runs ended by engine game_over: 12" reference
  statistic reproduces exactly when defined as *the final action event of the
  run has `game_over=true`*. The binding-budget conclusion (wall-clock time,
  not actions) is unchanged.
* **Mid-run game overs are common.** `game_over=true` appears somewhere in
  329/500 runs — these are engine game-overs followed by the existing
  auto-reset (`_execute_auto_reset`), after which the run continues. Only 12
  runs *terminate* on a game-over event.
* All other survival-block reference statistics reproduce exactly
  (`extract.py --verify`: 16/16 PASS).

## Phase 2 — model fitting notes

* `survival.py`: all reference numbers reproduce (KM checkpoints, hazard
  windows, t=40 groups n=88/248, noop rates 0.042/0.122, actions-per-LLM-call
  5.13/2.85). Group membership at t=40 is defined as `total_actions >= 40`
  and `first_clear > 40`.
* `hazard.py` novelty definition: `novelty_30` = fraction of **distinct**
  board hashes among the trailing 30 actions. The alternative
  "globally-new-to-run hash rate" produces the *wrong sign* (−0.70) because
  global novelty mechanically decays with time; the distinct-hash-window
  definition matches the spec's expected fit region and is directly
  computable online from a 30-slot ring buffer.
* Fit vs expected region: rows 4510 (~4.5k ✓), events 205 (spec ~186),
  coef(log_t) −0.795 (≈ −0.7 ✓), coef(novelty_30) +1.71 (≈ +2.0 ✓),
  coef(noop_20) −3.16 (spec ≈ −1.8: same sign, larger magnitude — sensitive
  to the exact person-period construction; the sign + AUC acceptance
  criteria pass, AUC 0.769 ≥ 0.72). Calibration is close to the diagonal.

## Phase 4 — simulator design decisions

* **Renewal semantics.** A natural attempt end (or budget truncation) ends
  the pass; a policy restart is the *only* way a pass gets another attempt.
  Rationale: an observed run ends when its wall-clock budget ends, and
  mid-run engine game-overs + auto-resets are already inside the observed
  trajectories (329/500 runs contain them). An earlier draft that "renewed"
  for free after natural ends inflated the baseline and halved every
  policy's uplift.
* **Exact evaluation.** Selection and evaluation both use exact expected
  scores (memoized recursion over the 20-attempt pool; state =
  spent/fails/restarts/idx) instead of Monte-Carlo passes. This removes all
  simulator noise: the ±3–4pp seed movement the spec warns about collapses
  to zero, seeds only drive the bootstrap CIs, and parameter selection is
  25/25-fold stable (reference: 24/25). `--mc-check` cross-validates the
  exact values against the MC sampler (agreement to ~3 decimals at 20k
  passes).
* **Harm counting.** A game counts as harmed only if the exact expected
  score drops by more than 1e-9. Without the tolerance, a policy that is
  *numerically identical* to baseline on a game (posterior blocks all
  restarts) can register as "harmed" by −3e-18 of float error.
* **Hard caps** (min 30 actions, max 2 restarts, 25% budget reserve) are
  applied to `mixture_adaptive` only — they ship with the deploy policy.
  Comparators run uncapped, as analysis baselines.
* **Posterior arithmetic check.** With (π=0.36, p_E=0.83, p_H=0.15),
  P(tractable | 1 failure) = 0.10 — below every φ in the grid, so the
  deployed policy restarts **at most once** on posterior grounds; the
  max-restarts cap is a belt-and-braces backstop. φ=0.35 with T=50 is inert
  (π(T=50)=0.32 < 0.35): the posterior gate blocks even the first restart.
* **Tournament result (default config: B=137, pc=0.5).** fixed +16.8%
  (harmed 8), gated +15.3% (harmed 5), luby +17.8% (harmed 8),
  mixture_adaptive +12.5%, CI [+4.7%, +19.0%], harmed 5 — joint-lowest harm,
  CI excludes zero, qualitative ordering reproduced (reference: mixture
  lowest harm at 5/25). Gated beats mixture by +2.8pp median < the 3pp
  threshold required to ship the M2 gating layer, so the deploy candidate
  remains M3-only, matching the spec's decision rule.
* **Sensitivity.** Uplift is robust for partial-credit ∈ {0.5, 1.0} across
  budgets {110, 137, 165}; at partial-credit 0.0 the uplift vanishes
  (−2.5%..+4.2%) because a restarted pass usually ends budget-truncated, and
  pc=0 refuses credit for truncated-but-cleared attempts. The live A/B must
  therefore treat truncated clears fairly (they do score in the real
  harness).
