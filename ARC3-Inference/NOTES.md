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
