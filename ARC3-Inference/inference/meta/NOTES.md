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
