# Keyboard-Nav Hints — branch notes

## Why

Reference run (25 games x 20 passes, stock duck) split by ARC API game tag:

| tag | games | mean score | zero-score |
|---|---:|---:|---:|
| click | 7 | 2.20 | 0 |
| keyboard_click | 13 | 1.06 | 0 |
| **keyboard** | **4** | **0.12** | **2** |

`keyboard` games (only UP/DOWN/LEFT/RIGHT/SPACE, no MOUSE) score 18x worse than
`click` games. Both zero-score games (`tr87`, `wa30`) are `keyboard`. Confirmed
again in the 2026-08-05 Kaggle Phase A rerun (submission-config, not this
branch): `tr87`=0.00, `wa30`=0.00, still the two worst games.

Hypothesis: the system prompt's generic anti-hallucination guidance --
"do not assume a player exists", "do not frame the objective as reaching a
specific row/col" -- is right for click/logic puzzles but likely
counter-productive for games whose only actions are grid steps, where an
avatar + BFS-to-target framing is usually correct.

## What this branch adds

`DUCK_KEYBOARD_NAV_HINTS` (default off). When the current `valid_actions` are
directional-only (no MOUSE), inserts a counter-hint: test the avatar
hypothesis via frame-diff after 1-2 moves, allow explicit row/col reasoning
once an avatar is confirmed, prefer BFS/shortest-path over trial-and-error
stepping.

Silent on every other game (click, keyboard_click with MOUSE present, or no
directional actions at all) -- checked against `valid_actions` on every turn,
not a static per-game flag.

## Status: untested, do not submit

Built off `main` (`dedbd6e`), independent of `feat/deepinfra-and-goal-hints`
(carryover + static-region), which scored 1.04 on the hidden set --
**below the known duck baseline (1.21 official / 1.60 public mean)**. That
branch's lesson applies here too: run a proper ablation (>=10 seeds on
`tr87`+`wa30`+`ls20`+`g50t`, the 4 keyboard games) before considering a
submission with this toggle on. A 2-3 sample smoke test is not evidence.

## Quick local test

```bash
DUCK_KEYBOARD_NAV_HINTS=1 make interactive GAME=tr87,wa30,ls20,g50t N_PASSES=4 CONCURRENT_JOBS=4
```
