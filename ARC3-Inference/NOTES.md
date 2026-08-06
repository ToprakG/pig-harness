# feat/level-debrief — working notes

## Phase 0 — branch and config

Branch cut from `origin/main` @ `3e2906a`. Config block shipped **disabled**:

    "debrief": {"enabled": false, "max_tokens": 400, "timeout_s": 90,
                "temperature": 0.3, "include_baseline": false,
                "min_actions_for_debrief": 5}

`timeout_s` is 90, not 20, deliberately: the previous compaction feature
failed **1488 of 1488** attempts with a 20 s `ReadTimeout` against the local
vLLM the main agent was already saturating.

Parity is enforced against the real upstream module rather than a
hand-written expectation: `tests/debrief/test_disabled_parity.py` loads
`origin/main:.../tool_agent.py`, builds a prompt with both classes, and
compares SHA-256. 3 passed.

## Open question — `base_actions_per_level` (D4)

`inference/tools/traces.py:34` already reads `baseline_actions`, and a
competitor reports (Kaggle discussion/687655) it appears exposed for the 110
hidden games. **The organisers have not answered whether using it is
permitted.** Implemented behind `debrief.include_baseline`, default
**false**. Do not enable without a recorded team decision here; prize-winning
solutions must be open-sourced, so any use will be visible.

DECISION: (unrecorded — flag stays false)
