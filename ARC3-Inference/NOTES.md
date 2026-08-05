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
