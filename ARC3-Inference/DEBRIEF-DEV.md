# Level debriefing — development notes

Branch `feat/level-debrief`, cut from `origin/main` @ `3e2906a`.
One feature, one branch.

    uv sync --locked --extra dev
    uv run pytest tests/debrief -q                       # 31 passed
    uv run python -m inference.debrief.ablation select
    uv run python -m inference.debrief.ablation power --corrections 2

## 1. Why this target

Measured on `example-run/benchmark.json` (500 runs, real baselines):

* 252/500 runs clear level 1; only 54 go on to clear level 2 →
  **P(2nd | 1st) = 0.214**
* median actions between the first and second clear: **22.5** (max 89)

So the failure is not running out of time after level 1 — the agent is
re-deriving mechanics it had already established, on a new board.

Projected score if that conditional rate moves:

| P(2nd \| 1st) | mean score | change |
|---|---|---|
| 0.214 (current) | 1.600 | — |
| 0.30 | 2.077 | +30% |
| 0.40 | 2.255 | +41% |
| 0.50 | 2.434 | +52% |
| 1.00 (ceiling) | 3.977 | +149% |

**Why not efficiency.** Across 309 cleared levels the median human-multiple
is already 1.07, and 140/309 are better than human. Perfect efficiency on all
cleared levels is worth 1.51x; one extra level is worth 3.96x. Depth is the
lever; efficiency is close to exhausted. (Independently confirmed here: the
`feat/action-efficiency` branch measured a cleared level at 0.94x human with
`headroom=0.0000` — already at the scoring ceiling for a one-level clear.)

**Why level scope and not game scope.** On Kaggle all hidden games run in one
session. A cross-game rewriting loop lets a bad rewrite at game 10 degrade
the remaining 100, undetectably mid-run, and it cannot be ablated at n>=50
because each replicate is a whole session. Level scope caps the blast radius
at one game and stays ablatable. Cross-game rewriting is explicitly out of
scope until level scope passes a powered ablation.

## 2. Design

**Deterministic packet, not model recall** (`inference/debrief/packet.py`).
At a level transition the harness builds a packet from its own event log:
per-action changed/unchanged counts with a structural delta label
(`appeared / disappeared / moved / recoloured / mixed`), dead actions, the
last three actions before the clear, and the longest stall segments with what
preceded and broke them. The model's self-report is never trusted — that is
precisely how the existing `_summarized_knowledge` mechanism degrades
silently.

Genericity: "background" is the majority label of the previous frame, not a
hardcoded colour; `MOUSE(row=3, col=4)` is recorded as `MOUSE` because
coordinates are board content; no game id ever enters the packet.

**One small rewrite call** (`inference/debrief/rewrite.py`) at the
transition, temperature clamped to <= 0.3, `timeout_s` **90**. The 20 s
timeout used by the previous compaction feature failed **1488 of 1488**
attempts against the local vLLM the agent was already saturating; that
mistake is not repeated.

**Failure is always a fallback, never a stall.** Timeout, malformed output,
oversize output → keep the previous block and log
`[DEBRIEF] fallback reason=<...>` to stdout. A hallucinated action id drops
only the offending section (`CONFIRMED MECHANICS` / `DEAD ACTIONS` are
evidence-checked against the packet; the two planning sections are not,
because naming an untried probe there is legitimate).

**No cross-game persistence.** `DebriefRuntime.reset()` clears the block on a
new game and asserts it.

**`base_actions_per_level` is off.** `inference/tools/traces.py:34` already
reads `baseline_actions` and a competitor reports it is exposed for the
hidden games, but the organisers have not answered whether using it is
permitted. Implemented behind `debrief.include_baseline`, default **false**,
and it stays false until a team decision is recorded in `NOTES.md`.
Prize-winning solutions must be open-sourced, so any use would be visible.

## 3. What is verified so far

| claim | evidence |
|---|---|
| disabled ⇒ no behaviour change | prompts SHA-256-identical to `origin/main`'s own module, loaded via `git show` |
| packet builds on real data | 309/309 real level transitions, 0 failures, max 339 tokens (cap 400) |
| the 309 figure | clear-count distribution {0:248, 1:198, 2:51, 3:3} ⇒ 198+102+9 = 309 (the brief said 306) |
| all four failure paths | stubbed valid / timeout / malformed / hallucination — 31 tests |
| the block reaches the next level | end-to-end smoke through the real solver hook: `[DEBRIEF] ok game=fake-1111 level=1 tokens=89` and the block present in the following prompt |

Two bugs were caught by these tests before any run: a stall segment recorded
the wrong `preceded_by`, and the evidence validator flagged ordinary words as
action ids because it matched against an uppercased copy of the text.

## 4. Ablation — designed, costed, **not yet run**

Endpoint: level-2-clear rate among runs that cleared level 1. Two arms only
(baseline vs `debrief.enabled=1`).

Selection rule (`ablation select`): level-1 clear rate >= 0.5, and
0.1 <= P(L2|L1) <= 0.9 so the endpoint can move in both directions →
ar25, ft09, re86, sb26, vc33 (5 of 25 games), pooled L1 rate 0.81.

Power (`ablation power`, one-sided Fisher, Bonferroni for 2 arms, 20k sims):

    n=10 -> 0.13    n=25 -> 0.51    n=50 -> 0.83    n=75 -> 0.96

n = 50 **level-1 clears** per arm ⇒ 62 runs/arm ⇒ **123 runs**:

| route | wall-clock | money |
|---|---|---|
| local, 20 min/game | ~41 h | ~$50–68 |
| local, 10 min/game | ~21 h | ~$25–35 |
| Kaggle save-runs, 25 games/session | ~11 h (6 sessions) | $0 |

**Status: not run.** A smaller pilot is not reported as evidence: at n=10 the
power is 0.13, which is how the earlier goal-hints ablation produced an 8/10
result that was uninformative in both directions. A null result here will be
published with the same prominence as a positive one.

## 5. Standing rules

* **No Kaggle submission from an arm with n < 50 or corrected p > 0.05.**
* One feature per branch, one variable per arm. The 1.04 submission moved two
  toggles at once and can be attributed to neither.
* Restart/meta logic is not reintroduced: `actions_per_level` is not cleared
  by RESET (`diagnostics.py:724`, comment at `:726`), so restarts permanently
  inflate the score denominator — re-simulated over 500 trajectories,
  T=30 → −4.9%, T=60 → −1.2%, T=∞ optimal, monotone harmful.
* Public-leaderboard rank is inflatable by resubmission: with σ ≈ 0.45 (the
  same submission has scored 0.77–1.30) the maximum over k submissions rises
  with k even when nothing changed. Private score is not. **Resubmission is
  presentation, never measurement.**
