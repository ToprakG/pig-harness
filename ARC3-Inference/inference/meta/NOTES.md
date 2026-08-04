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
