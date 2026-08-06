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

## Phase 1 — packet builder (offline, no model, no network)

`inference/debrief/packet.py` is a pure function over a run's action events.
Acceptance run over every real level transition in `example-run/artifacts`:

    transitions=309 failures=0 oversized(>400tok)=0
    packet tokens: min=105 median=151 p95=301 max=339

**Count correction:** the brief said 306 transitions; the true figure is
**309**, and the arithmetic confirms it — the clear-count distribution over
the 500 runs is {0: 248, 1: 198, 2: 51, 3: 3}, so
198x1 + 51x2 + 3x3 = 309 cleared levels. Reported rather than silently
adopted.

Design points:
* deltas are derived structurally, never semantically — "background" is the
  majority label of the *previous* frame, so `appeared / disappeared / moved
  / recoloured / mixed` carry no colour meaning and no game-specific
  knowledge;
* `MOUSE(row=3, col=4)` is recorded as `MOUSE` — the action id is generic,
  the coordinates are board content and are dropped;
* `preceded_by` in a stall segment is the action taken *before the stall
  began* (a first implementation recorded the last no-change action instead;
  caught by a unit test);
* `human_multiple` only appears when `include_baseline` is explicitly on.

Three real packets:

### ar25-0c556536_p0_events.jsonl — level 1 (303 tokens)
{
 "level": 1,
 "actions_used": 82,
 "action_effects": [
  {
   "action": "DOWN",
   "changed": 30,
   "unchanged": 0,
   "effect": "mixed"
  },
  {
   "action": "RIGHT",
   "changed": 18,
   "unchanged": 0,
   "effect": "appeared"
  },
  {
   "action": "LEFT",
   "changed": 17,
   "unchanged": 0,
   "effect": "mixed"
  },
  {
   "action": "UP",
   "changed": 10,
   "unchanged": 0,
   "effect": "mixed"
  },
  {
   "action": "SPACE",
   "changed": 4,
   "unchanged": 0,
   "effect": "recoloured"
  },
  {
   "action": "MOUSE",
   "changed": 0,
   "unchanged": 2,
   "effect": "none"
  },
  {
   "action": "RESET",
   "changed": 1,
   "unchanged": 0,
   "effect": "mixed"
  }
 ],
 "dead_actions": [
  "MOUSE"
 ],
 "clear_trigger": [
  {
   "action": "DOWN",
   "effect": "disappeared"
  },
  {
   "action": "DOWN",
   "effect": "disappeared"
  },
  {
   "action": "DOWN",
   "effect": "appeared"
  }
 ],
 "stall_segments": [
  {
   "length": 1,
   "start_action": 34,
   "preceded_by": "SPACE",
   "broken_by": "RIGHT"
  },
  {
   "length": 1,
   "start_action": 57,
   "preceded_by": "SPACE",
   "broken_by": "LEFT"
  }
 ]
}

### ar25-0c556536_p0_events.jsonl — level 2 (232 tokens)
{
 "level": 2,
 "actions_used": 38,
 "action_effects": [
  {
   "action": "DOWN",
   "changed": 8,
   "unchanged": 17,
   "effect": "mixed"
  },
  {
   "action": "LEFT",
   "changed": 7,
   "unchanged": 0,
   "effect": "mixed"
  },
  {
   "action": "RIGHT",
   "changed": 5,
   "unchanged": 0,
   "effect": "disappeared"
  },
  {
   "action": "SPACE",
   "changed": 1,
   "unchanged": 0,
   "effect": "disappeared"
  }
 ],
 "dead_actions": [],
 "clear_trigger": [
  {
   "action": "DOWN",
   "effect": "disappeared"
  },
  {
   "action": "DOWN",
   "effect": "disappeared"
  },
  {
   "action": "DOWN",
   "effect": "appeared"
  }
 ],
 "stall_segments": [
  {
   "length": 9,
   "start_action": 12,
   "preceded_by": "LEFT",
   "broken_by": "RIGHT"
  },
  {
   "length": 8,
   "start_action": 22,
   "preceded_by": "RIGHT",
   "broken_by": "SPACE"
  }
 ]
}

### ar25-0c556536_p10_events.jsonl — level 1 (181 tokens)
{
 "level": 1,
 "actions_used": 44,
 "action_effects": [
  {
   "action": "LEFT",
   "changed": 19,
   "unchanged": 0,
   "effect": "disappeared"
  },
  {
   "action": "RIGHT",
   "changed": 14,
   "unchanged": 1,
   "effect": "appeared"
  },
  {
   "action": "DOWN",
   "changed": 10,
   "unchanged": 0,
   "effect": "mixed"
  }
 ],
 "dead_actions": [],
 "clear_trigger": [
  {
   "action": "LEFT",
   "effect": "disappeared"
  },
  {
   "action": "LEFT",
   "effect": "disappeared"
  },
  {
   "action": "LEFT",
   "effect": "appeared"
  }
 ],
 "stall_segments": [
  {
   "length": 1,
   "start_action": 25,
   "preceded_by": "RIGHT",
   "broken_by": "LEFT"
  }
 ]
}


## Phase 2 — rewrite call, parser, evidence validator

`inference/debrief/rewrite.py`. Four failure paths, all falling back to the
caller's existing block and none of them raising:

| stub case | outcome |
|---|---|
| valid output | block rendered, `reason=ok`, no dropped lines |
| timeout | `reason=TimeoutError`, `block=None` — caller keeps its block |
| malformed (empty / prose / missing sections) | `reason=malformed_output` |
| hallucinated action id | offending SECTION dropped, rest of block kept |

Design decisions worth recording:
* only `CONFIRMED MECHANICS` and `DEAD ACTIONS` are evidence-checked. The
  other two sections are plans, not assertions, so naming an untried probe
  there is legitimate.
* a hallucination drops one section rather than the whole block — killing the
  entire rewrite for one bad clause would throw away good evidence.
* **validator bug caught by its own tests:** the first implementation matched
  action-id candidates against an uppercased copy of the text, so ordinary
  words ("recolours") were flagged as unevidenced ids. It now matches tokens
  that are already uppercase in the model's reply.
* temperature is clamped to <= 0.3 in `DebriefConfig.from_dict`; the call
  cannot be configured into creativity.

`pytest tests/debrief -q` -> 31 passed.

## Phase 3 — wiring and smoke

`inference/debrief/runtime.py` holds one `DebriefRuntime` per game session:
it records each executed action, and on `level_completed` builds the packet,
fires the rewrite and hands the block to the agent
(`ToolAgent.set_debrief_block`). The block is appended next to the existing
knowledge block, never substituted for the base prompt, and
`DebriefRuntime.reset()` asserts it does not survive into a new game (D3).

End-to-end smoke through the real solver hook (fake game that actually
completes a level, stubbed rewrite reply):

    [DEBRIEF] ok game=fake-1111 level=1 tokens=89 latency_ms=0

    --- next level's prompt contains: ---
    LEVEL DEBRIEF (auto-generated from the previous level)
    CONFIRMED MECHANICS: UP moves the object; SPACE does nothing.
    DEAD ACTIONS: SPACE
    WHAT CLEARED THE LAST LEVEL: LEFT made the target disappear.
    FIRST THINGS TO TEST ON THIS LEVEL: probe UP; then LEFT; avoid SPACE.
    end of world model.

The rewrite call reuses the analyzer's own endpoint, headers and client, and
reads `reasoning_content` when a thinking model spends its whole budget
before emitting `content` (that failure mode has bitten a previous feature).

**No conclusions are drawn from this smoke run** — it shows the mechanism
fires and reaches the next level's prompt, nothing about whether it helps.

## Phase 4 — powered ablation: plan, and the compute wall

### Game selection (rule, not eyeball)

Applied to `example-run/benchmark.json` by
`inference.debrief.ablation select`:

  R1  level-1 clear rate >= 0.5 (so level-2 transitions are sampled)
  R2  0.1 <= P(level 2 | level 1) <= 0.9 (the endpoint must be able to move
      in both directions; a game stuck at 0.00 or 1.00 contributes samples
      but no resolvable signal)

    game                  n  L1 rate  P(L2|L1)
    ar25-0c556536        20     0.75      0.27
    ft09-0d8bbf25        20     0.75      0.87
    re86-8af5384d        20     0.65      0.38
    sb26-7fbdac44        20     1.00      0.10
    vc33-5430563c        20     0.90      0.72

    selected=5/25 pooled_l1=0.81 pooled_cond_l2=0.47

R2 is an addition to the brief's rule and is recorded as such: R1 alone
admits 14 games, 9 of which have P(L2|L1) = 0.00 and would spend compute on
an endpoint that cannot move.

### Power (reproduced independently)

`inference.debrief.ablation power --baseline-rate 0.30 --effect-rate 0.60
--corrections 2` (one-sided Fisher, 20k sims):

    n=10 -> 0.13    n=25 -> 0.51    n=50 -> 0.83    n=75 -> 0.96

This matches the brief's figures and confirms n>=50 per arm.

### The wall: what n=50 actually costs

n is 50 runs that CLEAR LEVEL 1 per arm. At the pooled level-1 rate of 0.81
that is **62 runs per arm, 123 runs total**.

| route | wall-clock | money |
|---|---|---|
| local, DeepInfra, 20 min/game | ~41 h sequential | ~$50-68 |
| local, DeepInfra, 10 min/game | ~21 h sequential | ~$25-35 |
| Kaggle save-runs (25 games/session, ~1.8 h, local vLLM) | ~11 h over 6 sessions | **$0** |

**STATUS: not run.** Acceptance for this phase requires n>=50 per arm; a
smaller pilot would produce exactly the uninformative result the brief warns
about (the previous goal-hints ablation used n=10 and its 8/10 was
uninformative in both directions). Per the standing rules the criterion is
not being loosened and no partial run is being reported as evidence. The
decision to spend the compute is the operator's.
