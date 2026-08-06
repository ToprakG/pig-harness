# submission/v1 — what is in it, and why so little

Base: `fix/observability-and-budget` (`d07a451`), which is downstream of the
commit actually submitted as 0.91. On top of that, exactly three changes:

1. the request-timeout fix (cherry-picked, `soft_end_time` no longer drives
   the per-request HTTP budget) + its regression test,
2. `max_runtime_minutes` 45 → 90,
3. this file.

**No behavioural feature is enabled.** `meta.enabled`, `compaction.enabled`
and `budget.enabled` all ship `false`, exactly as on the base branch.
`pytest tests -q` → 60 passed.

## Why not merge the feature branches

Every measured deviation from stock duck has scored **below** stock duck:

| submission | score | vs duck baseline (1.21 official / 1.60 public-25) |
|---|---:|---|
| 0.91 run (meta + compaction) | 0.91 | below |
| goal-hints (carryover + static-region) | 1.04 | below |
| prolong-memory | 1.06 | below |

All three are inside the noise band (σ ≈ 0.45; the *same* submission has
scored 0.77–1.30), so none of them proves harm either. That is the point:
**nothing here is measured.** Every feature branch says so in its own words —
`feat/action-efficiency`: "this branch makes no claim about score";
`fix/observability-and-budget`: "the branch currently makes no claim about
score"; `feat/keyboard-nav-hints`: "untested, do not submit";
`feat/level-debrief`: "No Kaggle submission from an arm with n < 50".

Combining unmeasured features is precisely what produced 1.04 (two toggles at
once, attributable to neither). This submission does not repeat that.

### Specifically excluded, with reasons

| feature | why it stays off |
|---|---|
| `meta` (probabilistic restart) | **Provably harmful.** `actions_per_level` is never cleared by RESET (`taaf/game.py:571`, `diagnostics.py:724`), so an abandoned attempt's actions stay in the denominator of the level you later clear. Re-simulated over the 500 reference trajectories: T=30 → −4.9%, T=60 → −1.2%, T=∞ optimal, **monotone**. Two independent branches reached this conclusion. The `feat/meta-calibration` tournament that reported +12.5% uplift does not model this carry-over: `inference/meta/simulate.py` credits each restarted attempt its own independently-recorded `final_score`, i.e. it silently simulates best-of-k, which the competition does not allow. |
| `compaction` (LLM mode) | Never once succeeded in the 0.91 run: 1488 `compaction_fallback: ReadTimeout`, zero successes — a synchronous summarisation call to the same local vLLM the agent was already saturating, on a 20 s timeout. The code ships; the toggle stays off. |
| `budget.enabled` | Its planner models `waves = ceil(jobs / concurrency)`, but the scheduler is a pool (`asyncio.Semaphore`, `solver.py:965`), so fast games free their slot immediately and wall clock is not `cap × waves`. Activating a new code path on an untestable submission, on a wrong scheduler model, is worse than editing one config value. |
| score-awareness / batch cap / commit gate | Unmeasured behavioural changes. The batch gate had a bug that made the agent execute **zero** actions (caught by that branch's own instrumentation) — the exact failure class behind the low scores. |
| keyboard-nav hints, level-debrief, goal hints | Unmeasured, or measured negative (1.04). |

## The one change that is not a bug fix: 45 → 90 min per game

This is the only lever here that can move the score, and it is arithmetic
rather than behavioural.

**Verified:** the 0.91 Kaggle run used **6010 s of a 32400 s grant** (18.5%).
With 25 games at concurrency 28 they all run in one pool wave, so gameplay is
bounded by the per-game cap — 2700 s at 45 min — and the remaining ~3300 s is
fixed overhead (vLLM boot, weights, wheelhouse). 2700 + 3300 = 6010 fits the
observed total exactly.

**Why more time helps, from the scoring formula:**

* Depth dominates. `level_score = min(115, (baseline/actions)² × 100)`, weighted
  by `level_index + 1`; one more level is worth ~3× on a 6-level game, perfect
  efficiency on already-cleared levels ~+23%.
* An uncleared level scores **0** either way, and `actions_per_level` is frozen
  once a level is cleared, so actions spent on a level you never clear cannot
  damage levels already banked. Extra time is therefore near-monotone positive:
  the only cost is quadratic efficiency on levels that clear *later* than they
  would have — and those would not have cleared at all.
* In the 500-run reference set, **87% of runs ran to their time limit** rather
  than ending naturally. Time is binding.

**Why 90 and not more.** The first-clear hazard is strongly decreasing
(Kaplan-Meier S(40)=0.672, S(100)=0.526, S(200)=0.462), so the marginal value
of extra time falls steeply — doubling is worth something, quadrupling much
less. And the 9-hour wall is hard: with ~110 hidden games at concurrency 32
that is ~3.4 pool rounds, so 90 min/game ≈ 3300 + 3.4 × 5400 ≈ 21700 s of
32400 s (67%), leaving a third of the grant as margin for boot variance and
slow games. 45 min leaves ~62% of the grant unused; 120+ min risks the wall.

**Open risk, not resolved here:** whether a hard kill at 9 h preserves the
scores of already-completed games. In `TRUE_SUBMISSION` the gateway records
results as games finish, so it should — but this was not confirmed, and it is
what sets how much margin the cap needs. Confirm before raising past 90.

## What this submission is for

It is a **clean baseline with observability**, not a bid for a big jump. The
banner, `[BUDGET]`/`[COMPACT]`/`[META]` lines and end-of-run accounting from
the base branch mean the next run can be diagnosed instead of guessed at — the
0.91 run could not even distinguish "meta disabled" from "meta silently
active".

The honest expectation is a score in the same noise band as duck, with the
per-game time change as the only directional bet. Any real gain has to come
from a powered ablation afterwards, not from stacking more untested toggles.

**Do not read a leaderboard move as evidence.** With σ ≈ 0.45, the maximum
over k submissions rises with k even when nothing changed. Resubmission is
presentation, never measurement.
