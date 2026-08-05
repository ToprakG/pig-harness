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

## Phase 3 — move discipline

Two gates, both env-flagged and ablatable, composing by taking the tighter
limit and never dropping below 1 action (they must not stall the agent):

* `agent.max_actions_per_turn` (`AGENT_MAX_ACTIONS_PER_TURN`, 2 on this
  branch) — the batch cap.
* `agent.require_plan_before_act` (`AGENT_REQUIRE_PLAN_BEFORE_ACT`, true) —
  a turn may execute more than one action only if the assistant stated, in
  that same turn, a hypothesis AND the observation expected if it holds.
  Parsed with the **existing** `_extract_scientist_note` (its `Hypothesis` /
  `Next test` and `World model` / `Plan` labels) — no second parser.

Withheld actions are returned to the model with an explicit note
("N queued action(s) were withheld pending re-observation") rather than
silently dropped, and counted in the end-of-run block.

### Bug found by the first smoke run

The truncation block was inserted *before* `requested_displays` was defined,
so `step_env` raised `UnboundLocalError` on every gated turn and the agent
executed **zero** actions (visible as `total_actions=0` with the game's whole
60 s budget consumed). Fixed by moving the block after the displays are
built, and covered by `tests/eff/test_step_env_gates.py`, which drives the
real `step_env` with a fake game — no clock, no server, so it cannot regress
silently again.

### Smoke result (stub requesting 5 actions per turn, 2 games, 3 min each)

    gates OFF: total_actions=261 analyzer_calls=54  actions_per_call=4.83
    gates ON : actions=197       analyzer_calls=197 actions_per_call=1.00
               197/197 [ACT] lines carry gated=1, all with batch=1/5

i.e. the model requested 5-action batches in both runs; with the gates on
exactly one executed per turn and four were withheld each time.
**Actions per analyzer call fell 4.83 -> 1.00.**

This is a mechanism check only. Whether spending fewer actions per turn
raises the SCORE requires the Phase 4 replicate protocol (>= 4 replicates);
a stub agent cannot answer it, because the stub has no reasoning to
re-observe with.

Tests: `tests/eff` 22 passed (7 score-awareness, 11 gate logic, 4 step_env).

## Phase 4 — simulator and replicate protocol

`inference/eff/replay_score.py --run ../example-run`:

    runs=500 baselines_hidden=0
    score=800.1018  score_if_1x=1207.1140  headroom=407.0122
      alpha=0.5: score=1071.9107 (1.34x actual)
      alpha=0.7: score=962.2464 (1.20x actual)
      alpha=1.0: score=800.1018 (1.00x actual)
    formula check: recomputed scores match benchmark.json and score.json exactly

It also ranks the runs with the most headroom — i.e. the cleared levels that
were most expensive. Top of that list on the historical data: sc25 p10
(2 levels, 200 actions, human_mult 2.50, score 2.29 vs if_1x 14.29) and four
vc33 passes (human_mult 2.4–4.0, score 1.2–2.1 vs if_1x 10.71). These are the
concrete targets efficiency work should move.

`inference/eff/replicate.py`: `summarise` (mean/sd/SE + bootstrap 95% CI over
replicate run means) and `compare` (paired per-game differences, bootstrap
CI), both printing `INSUFFICIENT` below `--min-replicates` (default 4).
Verified on synthetic score files: 4 replicates -> summarised;
paired diff +0.750 flagged significant; 1 replicate -> `sufficient=False`.

Acceptance for all phases met. `pytest tests/eff -q` -> 22 passed.

## Real-model test (Arm E) — and the bug it exposed

First run of this branch against a real model (DeepInfra Qwen3.6-27B,
text-only, 20 min/game, score-awareness + cap 2 + commit gate all ON):

    [finished] ar25-0c556536 level=1/8 score=1.05 actions=113
    [EFF] game=ar25-0c556536 levels=1/8 a_per_level=52,61,...
          base_per_level=32,50,... human_mult_per_level=1.62,1.22,...
          cleared_human_mult=1.62 score=1.0519 score_if_1x=2.7778
          headroom=1.7258
    total_actions=113 analyzer_calls=17 actions_per_call=6.65
    gated_turns=53 withheld_actions=0

Good: a level was actually cleared, so the efficiency fields are real for the
first time — 52 actions against a 32-action human baseline (1.62x), worth
1.05 where perfect efficiency would have scored 2.78.

**Bug the telemetry caught: the batch cap did not bind.** 6.65 actions per
analyzer call with `max_actions_per_turn=2`. Cause: the agent may call
`action()` several times inside one Python snippet (the prompt explicitly
permits it) and each call is a separate `step_env`; my
`turn_action_allowance()` had a `max(1, ...)` floor, so after the turn budget
was spent every further call still executed one more action. The cap was
therefore per-call, not per-turn — the opposite of the spec ("execute at most
one action that turn").

Fix: the floor of 1 now applies only while the turn has executed nothing (so
a turn can always make progress); afterwards the allowance goes to 0 and
`step_env` returns a *non-error* withheld payload telling the agent to
re-observe. An error payload was deliberately avoided: it would invite a
retry storm.

Covered by `tests/eff/test_move_discipline.py::test_cap_is_per_turn_not_per_action_call`
and `tests/eff/test_step_env_gates.py::test_budget_spent_withholds_everything_without_erroring`.
`pytest tests/eff -q` -> 23 passed.

Also worth recording: `gated_turns=53` means the commit gate was the binding
constraint on 53 turns — with a real model the gate DOES open (Qwen emits a
hypothesis plus an expected observation), it just did not restrict much while
the cap was broken.
