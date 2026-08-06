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

## Live 3-game mechanism check — and the failure it exposed

Real model (DeepInfra Qwen3.6-27B), 20 min/game, debrief enabled, games
`sb26`, `vc33`, `ft09` (highest level-1 clear rate inside the selection set,
so the trigger is likely). **Not an ablation** — n=3 has no power.

| game | outcome |
|---|---|
| sb26 | level 1 cleared, score 0.10, 127 actions (93 on level 1) — **one `[DEBRIEF] ok`, latency 6769 ms, block 328 tokens** |
| vc33 | **invalid** — the game never opened: TLS handshake timeout to `three.arcprize.org`, then `observation_space is None after make()`. Infrastructure, not measurement |
| ft09 | 0 levels, score 0.00, 56 actions — no transition, so no debrief |

### What worked

The rewrite call itself is sound where compaction was not: **6.8 s against a
90 s budget** on a live endpoint, versus compaction's 1488/1488 failures at
20 s. No timeout, no fallback, no oversize output.

### What did not — and my earlier claim was wrong

I reported this run as "worked cleanly". Inspecting the block that actually
reached the prompt shows it was **garbage**: the model restated the
instruction's angle-bracket placeholders verbatim and then pasted the trace
back:

    CONFIRMED MECHANICS: <what actions do, as tested facts>
    DEAD ACTIONS: <action ids that never changed the board, or 'none'>
    WHAT CLEARED THE LAST LEVEL: <one line>
    FIRST THINGS TO TEST ON THIS LEVEL: <2-3 specific probes>
      - **Trace Data:** - `action_effects`: - SPACE: changed 66 ...

The parser accepted it because it only checked that the four headers were
*present*, never that they were *filled*. That block was then injected into
**17 subsequent prompts**. This is precisely the silent degradation the
design was meant to prevent, arriving through my own prompt and parser.

Fixes (35 tests now, incl. the verbatim live reply as a regression case):
* the instruction shows a **filled example** instead of echoable placeholders
  and explicitly says "write your OWN content ... do not restate the trace";
* `parse_block` rejects any section whose content is an unfilled `<...>`
  slot or empty;
* a reply containing >= 2 machine-readable packet keys (`action_effects`,
  `clear_trigger`, `stall_segments`, `actions_used`) is treated as a trace
  dump and rejected.

### Hypothesis worth tracking, not a conclusion

`sb26` spent 93 actions on level 1 against a historical range of 9-26 for
that game, with a polluted prompt for 17 turns. That is consistent with the
garbage block hurting, but n=1 and the prompt pollution is now fixed, so it
is a hypothesis for the ablation's secondary metrics, not a finding.

### Ablation runner requirement added

`vc33` shows runs can die before the game opens (10 s timeout inside the
`arc_agi` client, not ours). The Phase 4 runner must retry such runs and
exclude them from the denominator rather than counting them as failures to
clear level 1.

# feat/stall-reflect

## Phase 0 — branch and config

Cut from `feat/level-debrief` @ `6a0c3c1`. `reflect.*` config added
(disabled); `debrief.*` untouched.

## Phase 1 — detector calibration (BLOCKING gate)

### Reference reproduced exactly

`uv run python -m inference.reflect.calibrate reference`:

    runs(>= 25 actions) = 498   (reference 498)
    stalled = 161 (32%)         (reference 161)
    clear rate | stalled     = 0.13   (reference 0.13)
    clear rate | never stall = 0.68   (reference 0.68)
    discrimination = 5.2x
    median first stall = 25     (reference 25)
    REFERENCE: REPRODUCED

All four figures land exactly. The premise of the branch holds at the
population level.

### Sweep — and two findings that qualify it

    noop   nov  fire%  clr|fire  clr|no   disc  med t
     0.20  0.40    32%      0.13    0.68    5.2     25
     0.20  0.50    32%      0.13    0.68    5.2     25
     0.20  0.60    32%      0.13    0.68    5.2     25
     0.25  0.40    32%      0.13    0.68    5.2     25
     0.25  0.50    32%      0.13    0.68    5.2     25
     0.25  0.60    32%      0.13    0.68    5.2     25
     0.30  0.40    28%      0.14    0.64    4.6     35
     0.30  0.50    28%      0.14    0.64    4.6     35
     0.30  0.60    28%      0.14    0.64    4.6     35

**Finding 1 — the novelty signal never fires on its own.** Every column is
identical across `novelty_threshold`, and the direct count confirms it:
noop-only 117 runs, **novelty-only 0**, both 44. The median `novelty_30` is
0.93 (1st percentile 0.40): board states are nearly always distinct, so the
repetition signal has essentially no independent mass in this data. The
detector is, empirically, a no-op-rate detector. Keeping the novelty clause
costs nothing and may matter on other games, but it must not be described as
a second signal here.

**Finding 2 — most of the 5.2x is between games, not within.** LOGO selection
picks 0.20/0.40 in 25/25 folds, but that unanimity is an artefact of ties
(the whole 0.20–0.25 block is identical in-sample, and my tie-break took the
first cell). The number that matters: **median held-out discrimination 1.33**,
versus 5.2 in-sample. Per game, among the 8 games with >=3 runs on each side:

    ar25 stalled 0.50 | never 0.92      cd82 stalled 0.19 | never 0.50
    cn04 stalled 0.00 | never 0.19      ft09 stalled 0.62 | never 0.83
    g50t stalled 0.06 | never 0.00      ka59 stalled 0.22 | never 0.82
    ls20 stalled 0.00 | never 0.31      sc25 stalled 0.12 | never 0.00
    median within-game gap = +0.26 (never − stalled)

So stalling does predict failure within a game, but the effect is roughly a
26-point rate gap, not a 5x ratio. The population 5.2x is inflated by a
confound: hard games both stall more and clear less. **The realistic target
for this intervention is the within-game gap, and the ablation must be
powered against that, not against 0.13 vs 0.68.**

### Threshold choice

The 0.20–0.25 block is tied on every metric, so LOGO cannot discriminate
inside it. Rather than take an arbitrary corner, the **pre-registered
defaults `noop_threshold=0.25`, `novelty_threshold=0.5` are kept**: they sit
on the tie plateau, and picking 0.20/0.40 purely because it was evaluated
first would be exactly the overfitting the LOGO procedure exists to prevent.
Recorded as a deliberate choice, not an oversight.

## Phases 2-3 — revise call and the rollback ratchet

`inference/reflect/revise.py`. The five stub cases pass (valid, timeout,
malformed incl. template echo, hallucinated id, oversize), plus a trace-dump
rejection carried over from the debrief branch's live failure — the same
mistake would otherwise have been repeated here.

The ratchet is the part every earlier design in this project lacked:

* `may_fire` gates on warm-up (>=20 actions), cooldown (25), an open
  evaluation, and the per-level budget (2);
* `accept` pushes the block and remembers the pre-revision text plus the set
  of board hashes seen at fire time;
* `evaluate` at fire+`eval_window` counts improvement as *no longer stalled*
  OR *a board state unseen at fire time* OR *level cleared*; anything else
  reverts the block, discards the text, counts a rollback — **and still
  consumes budget**, so a thrashing loop cannot retry indefinitely;
* the stack is bounded at 3 (asserted over a 10-revision replay);
* `reset_level()` clears the block and asserts it, because the debrief branch
  owns level transitions and overlapping writes destroy attribution.

`pytest tests/reflect -q` -> 20 passed.

## Phase 4 — wiring and smoke

Wired: `ReflectRuntime` per game session, driven from the solver's action
path; the block is injected as a **separate labelled region** after the
debrief block (credit assignment), and cleared at level transitions.

End-to-end smoke through the real solver hook (fake game whose board freezes
after action 5, stubbed revise reply):

    [REFLECT] fire game=fake-stall level=1 t=20 noop20=0.75 nov30=0.25 revision=1
    [REFLECT] dropped game=fake-stall detail=NEXT PROBES: unknown action id(s) ['MOUSE']
    [REFLECT] ok game=fake-stall level=1 tokens=94 latency_ms=0
    [REFLECT] rollback game=fake-stall level=1 revision=1 reason=no_improvement
    [REFLECT] fire game=fake-stall level=1 t=45 noop20=1.00 nov30=0.03 revision=2
    [REFLECT] ok game=fake-stall level=1 tokens=94 latency_ms=0
    [REFLECT] fired=2 ok=2 rollback=1 fallback=0

    --- next turn's prompt contains: ---
    STALL REVISION (auto-generated after a detected stall)
    WHY STUCK: SPACE has not changed the board in the last 12 actions.
    ABANDONED: that SPACE toggles the target.
    NEW HYPOTHESIS: the target only reacts after MOUSE selects it.
    NEXT PROBES: (dropped: named actions not in the trace)

Every designed behaviour is visible in that trace:
* the detector fires at the warm-up boundary with a real no-op rate (0.75);
* **the ratchet rolled the first revision back** — the board stayed frozen
  through the evaluation window, so the block was reverted and the text
  discarded, while still consuming budget;
* the cooldown held exactly (45 − 20 = 25);
* the budget stopped the loop at 2 revisions;
* the write-gate dropped `MOUSE` from `NEXT PROBES` — correctly: the fake
  game's `available_actions` do not include ACTION6, so that probe was not
  executable. The validator is refusing an unexecutable instruction, not
  malfunctioning.

**No conclusions from this smoke run.** It shows the loop fires, revises,
evaluates, reverts and injects; it says nothing about whether any of that
helps a real agent.
