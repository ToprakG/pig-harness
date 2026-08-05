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
