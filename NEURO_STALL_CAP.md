# Göktürk V6: hazard-calibrated level-stall cap

## The finding

V5's local preview run: `wa30-ee6fef47` burned **2547 actions on one level,
zero clears** -- 44% of that game's entire 5818-action budget, for a score
of exactly 0. `wa30` is one of the four keyboard-only games already flagged
(NOTES.md, strategy doc) as structurally weak; this shows the failure mode
concretely: nothing in the solver stops a level that will never clear.

## Why: `max_actions_per_level` is accepted but dead

`HarnessSolver` has a `max_actions_per_level` field (visible in its own repr
in production logs) but grep confirms it is **never read anywhere** in
`solver.py`. There is no cap on a single level's action count -- only a
whole-game cap (`max_actions_per_game`) and a wall-clock cap
(`max_runtime_s_per_game`), neither of which stops one bad level from eating
both budgets while contributing nothing.

## The fix, and where the number comes from

`feat/meta-calibration`'s `inference/meta/survival.py` +
`inference/meta/stats.py` already fit exactly the right model for this:
Kaplan-Meier on first-level-clear time (event = level cleared, censored =
run ended uncleared) plus 20-action-window conditional hazard. Their fitted
reference numbers (`REPORT.md`):

```
S(40)=0.672  S(100)=0.526  S(200)=0.462
hazard[0,20)=0.198  hazard[60,80)=0.062  hazard[140,160)=0.008
```

By t=140 with no clear, the per-window chance of clearing is already ~0.8%.
`wa30`'s 2547 actions is 18x past that checkpoint.

Added a hard cap in the main `play()` loop, at the exact point that already
handles the analogous GAME_OVER-auto-reset case (`_execute_auto_reset()`
already existed for that): if `action_count - level_start_action >= 250`
(≈1.8x the t=140 hazard-collapse checkpoint -- deliberately generous so a
merely-hard level isn't cut off, only a genuine dead loop), force a level
RESET (the only legal recovery move in competition mode) and restart the
counter. Logs a warning on trigger so it's auditable in transcripts.

Verified in isolation before touching the real file (see commit): replaying
wa30's exact shape (2547 actions, zero progress) against the new logic
triggers exactly `2547 // 250 = 10` resets; replaying a normal 40-action
clear triggers zero.

## What this does NOT claim

- Not yet run against a real game (no local gateway/arcengine harness here
  to execute `_HarnessGameSession` end-to-end); verified by isolated replay
  of the exact counter logic only, syntax-checked against the real file.
- 250 is a judgment call with a stated rationale (1.8x their t=140
  checkpoint), not itself independently calibrated -- `feat/meta-calibration`
  or `feat/stall-reflect`'s own detector may already have a more principled
  answer; this is deliberately the minimal, isolated version of the same
  idea, not a replacement for either of those branches.
- Applied the same patch to the live Kaggle dataset bundle separately
  (mirrors the V4/V5 process) -- not yet pushed to a new kernel version or
  submitted, pending confirmation.
