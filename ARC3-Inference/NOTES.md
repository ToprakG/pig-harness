# fix/observability-and-budget — working notes

## Phase 0 — baseline

The 0.91 public-leaderboard submission (`taaf-duck-harness-kaggle-share` v1,
submission ref 55255753, 2026-08-05) was built from branch
`feat/meta-plus-compaction` at commit:

    a600bcd  combined(fix): quote SERVER_DEFAULT_CHAT_TEMPLATE_KWARGS ...

plus two deploy-time-only settings that are NOT in the tree (recorded here
for reproducibility):
- `KAGGLE_MODEL_DATASET_SOURCE=driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`
- `KAGGLE_SERVED_MODEL_NAME=vrfai/Qwen3.6-27B-FP8`
- env `META_CONFIG_JSON` / `COMPACTION_CONFIG` with `enabled: true` for both
  features (the tree defaults ship both disabled)
- an uncommitted working-tree patch (`MULTIMODAL_MAX_IMAGES`) that is a
  DeepInfra-only request-image cap; inactive on Kaggle (env var unset).

This branch (`fix/observability-and-budget`) is cut from that commit.

## Phase 1 — compaction gating proof

`configs/inference.json` ships `compaction.enabled: false` (verified). All
three truncation paths route through the single gate:

- `_trim_messages_for_context` → `self._compact_dropped(dropped_blocks)`
  (tool_agent.py:1752)
- `_force_reduce_messages` → `self._compact_dropped(dropped)`
  (tool_agent.py:1830)
- the `context_overflow_recovered` path calls only the two functions above
  (tool_agent.py ~1900), so it is transitively gated.

`_compact_dropped` (tool_agent.py:1755) begins with
`if not cfg.enabled or not dropped: return` — the compaction client
(`_compaction_llm_call`) is reachable ONLY through `compact()` which is
called ONLY after that gate. Digest prompt injection is separately gated on
`self._compaction_config.enabled` in `_build_user_prompt`.

Unit proof: `tests/observability/test_compaction_disabled.py` (3 tests) —
drives all three paths (including a real server-rejection overflow retry)
with the compaction client monkeypatched to record calls; zero invocations
with the flag false.

## Scoring semantics (Phase 3)

**Definitive answer: the competition score is a MEAN over games of a
single-pass per-game score; it is NOT a count of total levels cleared —
but the per-game score grows with both levels completed and per-level
action efficiency.** Evidence:

1. **Per-run score formula** — `tufa-arc-agi-framework/src/taaf/game.py:381`
   (`GameRun._compute_final_score`, documented as mirroring
   `arc_agi.scorecard.EnvironmentScoreCalculator` v0.9.8):
   per level `min(115, (baseline_actions / actions_used)^2 × 100)` if the
   level was completed (else 0), weighted by `level_index + 1`, normalized
   by total weights, capped by `max_weights / total_weights × 100`.
   ⇒ each additional completed level adds a positively weighted term
   (deeper levels weigh more); completing a level slowly still adds score
   (the term is positive whenever completed), so more per-game time is
   weakly monotone in expected score.

2. **The competition rerun is one pass per game** — the share notebook
   (`tufa-arc-agi-framework/src/taaf/kaggle/taaf_kaggle_run_share.ipynb`)
   forces `bm.n_passes = 1` before `bm.run(...)`. Extra passes are not part
   of the scored run at all; locally they only reduce measurement variance.

3. **Aggregation across games is a mean** — the 0.91 run's own final
   summary reports `mean 0.94, median 0.00` over per-game scores, and the
   leaderboard value (0.91) sits at that scale, while individual games score
   e.g. 8.33/4.76/2.78. A sum/total-levels metric would be orders of
   magnitude larger. Consistency check: the listed non-zero games sum to
   16.39; 16.39 / 0.94 ≈ 17.4 ⇒ the public rerun set appears to contain
   ~17–18 games (a public split), not our local 25. (Inference, flagged as
   such.)

4. Local tooling relation: `inference/tools/eval.py` writes per-game
   `score` = mean over `seed_scores` (one entry per pass) — the local mirror
   of the same mean-over-passes convention (with the rerun fixing passes=1).

**Budget consequence (feeds Phase 4):** unused wall-clock should go to
**longer per-game time** (more levels within the single scored pass).
Adding passes cannot raise the competition score (n_passes is forced to 1
in the rerun); locally, extra passes only narrow confidence intervals.

## Phase 2 — acceptance evidence (3-game smoke, stub LLM)

Both external providers (DeepInfra, Cerebras) exhausted their credit during
this phase (402s), so the smoke uses a local OpenAI-compatible stub LLM
(canned python-tool action calls) — harness, live arcade, meta policy, and
accounting are fully real; only the model is canned. Config:
`first_clear_deadline=10, min_actions_before_restart=10, phi=0.25`,
3 games × 3 min.

Banner (verbatim):

    ==================== RUN CONFIG BANNER ====================
    meta.enabled=True compaction.enabled=False
    meta.config={"budget_reserve_frac": 0.25, "enabled": true, "first_clear_deadline": 10, ...}
    compaction.config={"compaction_call_max_tokens": 900, ..., "enabled": false, ...}
    model_id=stub-model
    max_runtime_s_per_game=180.0
    ...

Sample event lines:

    [META] restart game=ar25-0c556536 pass=0 t=10 level=1 fails=0 posterior=0.360 reason=deadline
    [META] suppressed game=ar25-0c556536 ... reason=posterior
    [BUDGET] game=ar25-0c556536 planned=180 used=181 util=101%

Accounting block (verbatim):

    ==================== RUN ACCOUNTING ====================
    wallclock used=555s granted=540s utilisation=102.9%
    game=ar25-0c556536 pass=0 actions=206 tokens=4060 levels=0 score=0.0 meta_restarts=1 compaction_ok=0 compaction_fallbacks=0 elapsed_s=181
    game=lp85-305b61c3 pass=0 actions=0 tokens=215580 levels=0 score=0.0 meta_restarts=0 compaction_ok=0 compaction_fallbacks=0 elapsed_s=181
    game=sb26-7fbdac44 pass=0 actions=95 tokens=16740 levels=0 score=0.0 meta_restarts=1 compaction_ok=0 compaction_fallbacks=0 elapsed_s=182
    [META] fired=2 suppressed=4
    [COMPACT] ok=0 fallback=0
    generated_tokens_per_sec=425.55 mean_seconds_per_action=1.85
    ========================================================

Side finding the accounting immediately surfaced: the stub's canned moves
are invalid for lp85 (0 actions, 216k tokens of retries) — exactly the class
of silent failure this branch exists to expose. (Stub artifact, not a
harness bug; ar25/sb26 played normally.)

Answering the mission's fact #2 (why no meta evidence in the 0.91 run):
the policy had NO stdout logging at all before this phase — restarts were
tagged only into viewer_data artifacts. With this phase, fired AND
suppressed decisions are stdout lines.
