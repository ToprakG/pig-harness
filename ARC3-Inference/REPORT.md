# fix/observability-and-budget — report

Branch cut from `feat/meta-plus-compaction` @ `a600bcd` (the commit behind
the 0.91 public-leaderboard submission, ref 55255753).

    uv sync --locked --extra dev --extra meta
    uv run pytest tests -q                       # 50 passed
    uv run python -m inference.framework.budget --games 25 --passes 1 --concurrency 28
    uv run python -m inference.meta.replicate --help

## 1. Motivation: seven verified facts about the 0.91 run

1. **Compaction never once succeeded.** 1488 `compaction_fallback:
   ReadTimeout` lines against `127.0.0.1:1234` (read timeout 20 s), zero
   successes. The summarisation call was issued synchronously to the same
   local vLLM server the agent was saturating; on one GPU it cannot return
   inside 20 s. The feature burned wall-clock and preserved nothing.
2. **No evidence meta-restart ever fired.** Zero `meta_restart` occurrences
   and no echo of the meta config in stdout — the run could not distinguish
   "disabled" from "silently active". Root cause found in this branch: the
   policy had **no stdout logging at all**; restarts were tagged only into
   viewer artifacts.
3. **Both features shipped on one branch**, destroying attribution.
4. **81% of the compute budget was unused**: 6010 s of a 32400 s grant.
5. **Score is carried by ~3 games of 25**: mean 0.94, median 0.00; non-zero
   ar25=8.33, r11l=4.76, lp85=2.78, ls20=0.30, re86=0.14, bp35=0.08.
6. Per-game 29–178 actions, ~37–40k tokens, 176.78 generated tokens/sec.
   The binding constraint is generation throughput, not reasoning quality.
7. **Measurement noise is large**: sigma ≈ 0.45 on the public set (Tufa
   Labs), and the *same* submission has scored 0.77–1.30 on Kaggle.

## 2. What this branch changes

| Phase | Change |
|-------|--------|
| 1 | `compaction.enabled=false` shipped; all three truncation paths proven gated (`tests/observability/test_compaction_disabled.py`) |
| 2 | Startup banner (resolved meta/compaction config, explicit `enabled` booleans, model, budget), `[META] restart/suppressed`, `[COMPACT] ok/fallback`, `[BUDGET]` stdout lines, end-of-run accounting block |
| 3 | Scoring semantics resolved (below) |
| 4 | Budget planner: wave-aware allocation of the wall-clock grant, per-game cap, plan printed in the banner, `[BUDGET]` per game |
| 5 | Compaction `mode`: `async` (off critical path) and `external_history` (RGB-style, no summarisation). Neither enabled by default |
| 6 | `inference/meta/replicate.py` + this report |

Nothing in this branch touches prompts' task content, model selection,
temperatures, or tool semantics. No behaviour is conditioned on game
identity, colours, or board content (AST-guarded for the new helpers).

## 3. Scoring semantics (Phase 3, blocking question — resolved)

**The competition score is a mean over games of a single-pass, per-game
score that rewards both levels completed and per-level action efficiency.
It is not a raw count of levels.**

* `tufa-arc-agi-framework/src/taaf/game.py:381` `GameRun._compute_final_score`
  (mirrors `arc_agi.scorecard.EnvironmentScoreCalculator` v0.9.8): per level
  `min(115, (baseline_actions / actions_used)^2 × 100)` when completed, else
  0; weighted by `level_index + 1`; normalised by total weights.
* `taaf/kaggle/taaf_kaggle_run_share.ipynb` forces `bm.n_passes = 1` before
  `bm.run(...)` — the scored rerun is one pass per game.
* The 0.91 run's own summary (mean 0.94 over per-game scores of 8.33/4.76/…)
  is consistent with a mean, not a sum.

**Consequence for budget:** spare wall-clock must go to **longer per-game
time**, not more passes. Extra passes cannot raise the scored result; they
only narrow local confidence intervals.

**Why 81% went unused (Phase 4 finding):** with concurrency 28 all 25 games
run in a single wave, so utilisation is bounded by the per-game cap. Raising
it means raising per-game seconds and/or lowering concurrency to create more
waves — the planner now makes this explicit:

    jobs=25x1 concurrency=28 waves=1 → per_game=10800s (capped) → 33% of grant
    jobs=25x1 concurrency=6  waves=5 → per_game=5202s          → 80% of grant

## 4. Measurement rule (non-negotiable)

* **No configuration decision is made on fewer than 4 replicates.**
  `replicate.py` refuses to summarise below `--min-replicates` (default 4)
  and prints `INSUFFICIENT`.
* **No single-submission delta below ~0.5 is treated as signal.** With
  sigma ≈ 0.45 and an observed 0.77–1.30 spread for one fixed submission,
  a 0.74 → 0.91 difference is *within* noise. The earlier framing of that
  delta as a +23% improvement was not warranted by the evidence.
* Comparisons use paired per-game differences with a bootstrap 95% CI:

      uv run python -m inference.meta.replicate compare \
          --baseline runs/A*/score.json --candidate runs/B*/score.json

## 5. Ablation plan (one variable per run)

| Arm | Config | Question |
|-----|--------|----------|
| **A** | both features off | baseline distribution + sigma estimate |
| **B** | meta only (`meta.enabled=true`) | does restarting help, given it now logs? |
| **C** | compaction `mode: async` only | does off-path summarisation pay for itself? |
| **D** | compaction `mode: external_history` only | does searchable history beat summarisation? |

Each arm: ≥4 replicates, identical games/seeds/budget, compared to A with
`replicate.py compare`. Only an arm whose CI excludes zero is adopted, and
arms are never combined before each is measured alone — the mistake that
produced fact #3.

## 6. Public vs private leaderboard

Public-leaderboard rank can be inflated simply by resubmitting: with a noisy
metric, the maximum over k submissions rises with k (an order-statistic
effect) even when the underlying configuration is unchanged. The private
leaderboard score cannot be improved this way. **Resubmission is therefore a
presentation tactic, never a measurement.** Any claim that a change helped
must come from the replicate protocol in §4, not from a leaderboard move.

## 7. Status

* Phases 0–6 complete; `pytest tests -q` → 50 passed.
* Nothing in this branch has been submitted; no ablation arm has been run
  yet. Under §4 the branch currently makes **no** claim about score.
* Known limitation: Phase 2/4 smoke evidence was produced against a local
  stub LLM (both external API credits were exhausted), so it validates the
  plumbing — banner, event lines, accounting, budget allocation — not model
  behaviour.
* Post-merge regression (found by the new accounting, fixed, guarded): the
  Phase 5 sandbox helpers were registered before being defined, killing
  every python-tool sandbox. Any change to the sandbox bootstrap must be
  covered by a test that executes the sandbox
  (`tests/observability/test_sandbox_smoke.py`), not only its host-side
  handler — the mode unit tests passed while the agent could not act at all.
  This is also the first thing the new observability layer paid for: a
  silent zero-action run was previously indistinguishable from a bad model.
