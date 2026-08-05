# Harness Dev — Goal-Inference Toggles

Three experiment toggles for the duck harness, each behind an env var (default off)
so a branch can be A/B'd against main without editing code.

## Toggles

| Toggle | What it does |
|---|---|
| `DUCK_DEAD_ACTION_HINTS` | Tells the model which actions already left *this exact board* unchanged. Derived from history each turn, so it cannot drift. |
| `DUCK_LEVEL_CARRYOVER` | On level transition, forces the model to bank confirmed mechanics into the world model before re-grounding on the new board. Targets **depth**, the score formula's dominant lever. |
| `DUCK_STATIC_REGION_HINTS` | Splits the board into cells that have ever changed this level vs never have, names the colors unique to each. Targets **goal inference** — observed failure mode: model correctly infers action mechanics but stalls on "Goal model: Unknown". |

All three carry through to Kaggle via `kaggle.py`'s setup-env dicts.

## First ablation (2026-08-05)

10 runs/arm on `lp85` + `su15`, DeepInfra BF16, 30 min/game. Compared on
level-clearing rate (binary, less heavy-tailed than score at this n):

| arm | cleared ≥1 level | one-sided Fisher vs baseline |
|---|---:|---:|
| baseline | 3/10 | — |
| `DUCK_DEAD_ACTION_HINTS` | 4/10 | 0.500 |
| `DUCK_STATIC_REGION_HINTS` | 6/10 | 0.185 |
| `DUCK_LEVEL_CARRYOVER` | 8/10 | 0.035 |

## Kaggle submission result (2026-08-05)

Submitted with `DUCK_LEVEL_CARRYOVER=1` + `DUCK_STATIC_REGION_HINTS=1` on the
official hidden set: **public score 1.04**, below the known duck baseline
(1.21 official Kaggle score / 1.60 public-25-games mean). The two-game BF16
ablation signal above did not hold at the 25-game hidden-set scale — expected,
since it was never Bonferroni-significant (3 arms tested, raw p=0.035).

**Do not resubmit with these toggles on without re-testing on more games /
held-out set first.** `DUCK_DEAD_ACTION_HINTS` in particular looks negative
(p=0.500, and on `lp85` it cleared 0/5 while cutting mean actions to 3.6 —
looks like it makes the model stop acting rather than act better).

## Score formula, for context

```
level_score = min(115, (human_baseline_actions / your_actions)^2 * 100)   # 0 if not completed
game_score  = sum(level_score * level_index) / sum(all level_indices)
              capped at sum(completed level_indices) / sum(all) * 100
```

Depth dominates: clearing one more level is worth roughly 3x on a 6-level
game; perfect efficiency on already-cleared levels is worth roughly +23%.
`DUCK_LEVEL_CARRYOVER` targets this; the other two target the actions-wasted
side, which the reference run shows is a much smaller lever (0-6.3% of
actions are repeated no-ops across the 25 public games).

## compare_runs.py

`scripts/compare_runs.py` scores saved run directories directly from
`benchmark.json` (mirrors `taaf.game.GameRun._compute_final_score`), reports
score/actions/levels/spread per game, and flags deltas that fall inside a
single run's noise band. Validated against `example-run/` (reproduces the
reference mean of 1.60 exactly).

```bash
python scripts/compare_runs.py runs/*baseline runs/*candidate
```
