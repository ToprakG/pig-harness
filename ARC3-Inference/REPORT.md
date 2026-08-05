# feat/action-efficiency — report

Branch cut from `origin/main` @ `3e2906a`. One feature per branch: this one
is about spending fewer actions per cleared level, and nothing else.

    uv sync --locked --extra dev
    uv run pytest tests/eff -q                              # 22 passed
    uv run python -m inference.eff.replay_score --run ../example-run
    uv run python -m inference.eff.replicate --help

## 1. The scoring formula, and why it changes everything

`tufa-arc-agi-framework/src/taaf/game.py:381`
(`GameRun._compute_final_score`, mirroring
`arc_agi.scorecard.EnvironmentScoreCalculator` v0.9.8):

    per cleared level: min(115, (baseline / actions)^2 * 100)
    weight            = level_index + 1
    score             = sum(level_score * weight) / sum(weight)
    capped at           max_weights / total_weights * 100

**Empirically validated, not assumed.** `inference/eff/scoring.py`
reimplements it and reproduces **all 500** `example-run` trajectories
exactly (`replay_score --run ../example-run`: "recomputed scores match
benchmark.json and score.json exactly", 0 mismatches). The quadratic law is
independently confirmed by `score x a^2` being constant per game for
single-level clears: ar25 2844.4 (n=11), bp35 980.0 (n=18), cn04 4004.8
(n=3).

Score elasticity with respect to actions is **-2**:

| Actions vs human | Level score |
|---|---|
| 1x | 100% |
| 1.5x | 44% |
| 2x | 25% |
| 3x | 11% |
| 4x | 6% |

### Consequence 1 — restart is provably harmful; it is not coming back

`taaf/diagnostics.py:724` increments `actions_per_level[levels_completed]` on
every action and a full reset does **not** clear it (:726). Actions burned in
an abandoned attempt stay permanently in the denominator of the level you
eventually clear: two restarts at T=60 plus a 30-action clear scores
`(30/150)^2 = 4%` of a clean 30-action clear. Re-simulated over the 500
trajectories: T=30 -> -4.9%, T=40 -> -3.5%, T=60 -> -1.2%, T=90 -> -0.6%,
T=inf -> optimum. Monotone. `meta.enabled` stays false permanently, and no
restart logic exists on this branch (`inference/meta/` is absent).

### Consequence 2 — action efficiency is the lever, with an honest ceiling

Priced on the historical data by `replay_score`:

    score=800.10   score_if_1x=1207.11   headroom=407.01
      alpha=0.5: 1071.91 (1.34x actual)
      alpha=0.7:  962.25 (1.20x actual)

Halving actions on already-cleared levels is worth **+34%**, not +300%: the
per-level cap (115) and the `max_weights` cap bound the payoff. Efficiency is
the dominant *available* lever, but the ceiling from efficiency alone on
these trajectories is ~1.3–1.5x. Claiming more would be dishonest.

### Consequence 3 — the two decision layers are independent

`actions_per_level` is per level and frozen once a level is cleared, so
actions burned on a level you never clear cannot damage levels already
banked. Layer A (this branch): within a level you will clear, every extra
action costs quadratically. Layer B (not this branch): time allocation
across games is a budget problem with no quadratic penalty. Conflating them
is exactly the error that produced the discarded restart policy.

## 2. What this branch adds

| Phase | Change |
|---|---|
| 0 | Explicit `meta.enabled=false` / `compaction.enabled=false`; gating trace showing neither feature exists in code here |
| 1 | `inference/eff/scoring.py` (validated); `[ACT]` per action, `[EFF]` + `[BUDGET]` per game, end-of-run efficiency block |
| 2 | Score-awareness prompt block (69 tokens) with a live per-level action count, behind `agent.score_awareness_enabled` |
| 3 | Batch cap (`agent.max_actions_per_turn`) and commit gate (`agent.require_plan_before_act`), both ablatable, never stalling |
| 4 | `replay_score.py`, `replicate.py`, this report |

Measured mechanism effect (stub agent requesting 5-action batches, 2 games):
**actions per analyzer call 4.83 -> 1.00**, with 197/197 gated turns
executing exactly one of the five requested actions and returning the rest
with an explicit "withheld pending re-observation" note.

Rationale for the gates: under a quadratic penalty an agent that spends 10
exploratory actions plus 20 executed ones (30 total) scores 7.1x an agent
that flails for 80 — with identical underlying reasoning ability. LLM calls,
Python inspection of the observation and re-reading history are free; only
actions cost score. This is the mechanism behind the RGB Agent's
queued-action pattern and Symbolica's bounded-action sub-agents.

**Baselines are visible.** Every game in the smoke runs reported real
`base_actions_per_level`, so `human_mult`, `score_if_1x` and `headroom` are
directly measurable rather than inferred. Where the engine does hide them,
the telemetry prints `base_per_level=hidden` and skips the derived fields
instead of guessing.

## 3. Ablation plan (one variable per run)

| Arm | Config |
|---|---|
| **A** | baseline: score-awareness off, no gates |
| **B** | score-awareness only |
| **C** | batch cap only |
| **D** | commit gate only |
| **E** | B + C + D |

Each arm: >= 4 replicates over the 25 public games, identical budget and
seeds, compared to A with `replicate.py compare`. Primary metric is score;
`cleared_human_mult` and actions-per-analyzer-call are the mechanism metrics
that explain a score move (or its absence).

## 4. Measurement rule (non-negotiable)

* Tufa report sigma ~ 0.45 on the public set, and the *same* Kaggle
  submission has scored 0.77–1.30. **No configuration decision on fewer than
  4 replicates**, and **no single-submission delta under ~0.5 is signal.**
  `replicate.py` prints `INSUFFICIENT` below the threshold rather than a
  number that invites over-reading.
* Public-leaderboard rank is inflatable by resubmission: with a noisy metric
  the maximum over k submissions rises with k (order-statistic effect) even
  when nothing changed. Private score is not. **Resubmission is
  presentation, never measurement.**

## 5. Status

* Phases 0–4 complete; `pytest tests/eff -q` -> 22 passed;
  `replay_score --run ../example-run` -> exact match on 500 runs.
* No arm has been run with a real model: both external API credits were
  exhausted during this work, so the smoke runs used a local stub LLM. They
  validate the mechanism (telemetry, gates, formula), not the score.
* Under §4 this branch therefore makes **no claim** about score. The next
  step is arms A and E with >= 4 replicates each.
* Bug found and fixed by the branch's own instrumentation: the Phase 3
  truncation block was inserted before `requested_displays` existed, so every
  gated turn raised `UnboundLocalError` and executed zero actions. It is now
  covered by `tests/eff/test_step_env_gates.py`, which drives the real
  `step_env` with a fake game (no clock, no server).
