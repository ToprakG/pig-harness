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

## Phase 1 — action accounting

### Formula validation (done first: everything downstream depends on it)

`inference/eff/scoring.py` reimplements `GameRun._compute_final_score`.
Checked against **all 500 runs** in `example-run/benchmark.json` (which carry
real `actions_per_level`, `base_actions_per_level`, `levels_completed`,
`final_score`):

    VALIDATION: 500 runs, mismatches=0

Quadratic law confirmed independently (`score * a^2` constant per game for
single-level clears), matching the mission's numbers exactly:

    ar25-0c556536: score*a^2 = 2844.4  (n=11)
    bp35-0a0ad940: score*a^2 = 980.0   (n=18)
    cn04-2fe56bfb: score*a^2 = 4004.8  (n=3)

Headroom on that historical data (sum over 500 runs):

    actual=800.1  if-1x=1207.1  headroom=407.0
    alpha=0.5 -> 1071.9 (1.34x actual)
    alpha=0.7 ->  962.2 (1.20x actual)

Reading: halving actions on already-cleared levels is worth **+34%** total
score, not +300% — the per-level cap (115) and the `max_weights` cap bound
the payoff. Efficiency is the dominant *available* lever, but the honest
ceiling from efficiency alone on these trajectories is ~1.3-1.5x.

### Telemetry

`inference/eff/telemetry.py` + solver wiring emits `[ACT]` per action,
`[EFF]` + `[BUDGET]` per game, and an end-of-run block.

**Key finding: baselines are NOT hidden.** Every game in the smoke run
reported real `base_per_level`, so efficiency is directly measurable:

    [EFF] game=ar25-0c556536 pass=0 levels=0/8 a_per_level=146,0,0,0,0,0,0,0
          base_per_level=32,50,75,37,89,159,233,73
          human_mult_per_level=4.56,0.00,... cleared_human_mult=n/a
          score=0.0000 score_if_1x=0.0000 headroom=0.0000
    [EFF] game=sb26-7fbdac44 ... a_per_level=72,... base_per_level=18,28,...
          human_mult_per_level=4.00,... score=0.0000 score_if_1x=0.0000
    [BUDGET] game=sb26-7fbdac44 planned=120 used=122 util=102%

    ================= ACTION EFFICIENCY =================
    total_actions=218 analyzer_calls=216 actions_per_call=1.01
    mean_cleared_human_mult=n/a (no cleared level with a visible baseline)
    sum_headroom=0.000 sum_score=0.000
    gated_turns=0 withheld_actions=0
    wallclock used=369s granted=120s utilisation=307.9%
    =====================================================

`cleared_human_mult` / `headroom` are `n/a` / 0 because the stub cleared no
level — by construction, not a bug: with `levels_completed=0` there is no
cleared level whose actions could have been cheaper.

**Smoke used a local stub LLM** (canned python-tool actions) because both
external API credits were exhausted; the harness, arcade, engine baselines
and scoring are entirely real. `actions_per_call=1.01` is therefore the
stub's shape, not the agent's — the real baseline to beat is 3.57 from the
0.91 submission.

Two artifacts of the stub, recorded so they are not misread later: lp85
executed 0 actions (its canned moves are invalid there), and wallclock
utilisation >100% is the per-game soft cap measured against a 3-game
sequential run.

## Phase 2 — score awareness

`ToolAgent._score_awareness_lines()` injects one block next to
`_summarized_knowledge_lines()`, gated on `agent.score_awareness_enabled`
(env `AGENT_SCORE_AWARENESS_ENABLED`, default true on this branch):

    SCORING: each completed level scores min(115, (human_baseline/your_actions)^2
    * 100). Actions are the ONLY cost; thinking, inspecting variables and
    re-reading history are FREE. An action that does not advance you is a
    permanent quadratic loss. Level 2; actions spent on it: 7.

276 chars ~= 69 tokens (budget was ~80). The live counter is pushed by the
solver each turn via `set_level_action_count(actions_per_level[level])`
before `analyze()`, so it can never go stale. No baseline VALUE is shown
(only the symbol `human_baseline`): baselines can be hidden by the engine,
and showing a target invites gaming it rather than genuine efficiency.

Tests (`tests/eff/test_score_awareness.py`, 7 passed): block appears exactly
once; absent when disabled; **prompt byte-identical to upstream when the flag
is off** (removing the block and its separator reproduces it exactly); the
counter is live and monotonic within a level; the block reports the current
level; stays under the token budget; leaks no baseline value; contains no
game identity, colour, shape, or board content.
