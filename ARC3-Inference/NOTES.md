# feat/action-efficiency — working notes

## Phase 0 — branch and shutdown

Branch cut from `origin/main` @ `3e2906a` ("Add DeepInfra provider ...").

### Gating trace: both features are absent, not merely disabled

On this branch neither feature exists in code — they were only ever added on
`feat/meta-calibration` and `feat/context-compaction`:

    $ ls inference/meta                 -> No such file or directory
    $ ls inference/agent/compaction.py  -> No such file or directory
    $ grep -c "meta_config|_maybe_meta_restart|should_restart" \
          inference/framework/solver.py -> 0
    $ grep -c "compaction|_compact_dropped|HistoryDigest" \
          inference/agent/tool_agent.py -> 0

`configs/inference.json` nevertheless now carries explicit
`meta.enabled=false` and `compaction.enabled=false` so the decision is
visible where an operator looks, and so any future reintroduction has to
delete a deliberate "false" rather than add a missing key.

The only RESET path left is the pre-existing engine-recovery one
(`solver.py:281`): it fires solely when the engine reports GAME_OVER and the
previous action was not itself a RESET. That is not a policy — it is how the
harness continues after the engine ends a game.

### Why these stay off (evidence)

* **Restart is provably harmful.** `taaf/diagnostics.py:724` increments
  `actions_per_level[levels_completed]` on *every* action, and the comment at
  :726 confirms a full reset does not clear it (it only stops
  `levels_completed` from decreasing — verified in source on this checkout).
  Actions spent in an abandoned attempt therefore stay permanently in the
  denominator of the level eventually cleared. Two restarts at T=60 plus a
  30-action clear yields `actions_per_level[0]=150`, i.e. `(30/150)^2 = 4%`
  of a clean 30-action clear. Re-simulated over the 500 trajectories with the
  correct reward model: T=30 -> -4.9%, T=40 -> -3.5%, T=60 -> -1.2%,
  T=90 -> -0.6%, T=inf -> optimum. Monotone: **`meta.enabled` stays false
  permanently.**
* **Compaction never worked in production.** The 0.91 submission log contains
  1488 `compaction_fallback: ReadTimeout` lines against `127.0.0.1:1234`
  (20 s timeout) and zero successes: the summarisation call was issued
  synchronously to the same local vLLM the agent was saturating.

### Scoring formula (source of truth)

`tufa-arc-agi-framework/src/taaf/game.py:381` `GameRun._compute_final_score`
(mirrors `arc_agi.scorecard.EnvironmentScoreCalculator` v0.9.8): per cleared
level `min(115, (baseline/actions)^2 * 100)`, weighted by `level_index+1`,
normalised by total weights, capped at `max_weights/total_weights*100`.
Score elasticity w.r.t. actions is **-2**.
