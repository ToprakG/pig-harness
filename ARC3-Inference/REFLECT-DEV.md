# Stall-triggered reflect-and-revise — development notes

Branch `feat/stall-reflect`, cut from `feat/level-debrief` @ `6a0c3c1`.
Additive: the debrief code is untouched, the two loops are independently
flagged (`debrief.*` / `reflect.*`) and write to separate, labelled prompt
regions so they stay attributable.

    uv sync --locked --extra dev
    uv run pytest tests/reflect tests/debrief -q
    uv run python -m inference.reflect.calibrate reference
    uv run python -m inference.reflect.calibrate sweep

## 1. Why a second loop at all

`feat/level-debrief` fires at a level transition and carries verified
mechanics forward. It has **no feedback signal**: it writes instructions for
a level it has never seen. That is transfer, not learning.

This branch closes the loop *inside* a level, where a signal exists (is the
board changing?) and a revision can be evaluated and reverted.

## 2. The trigger — reproduced, then qualified

Stall = over the trailing 20 actions `no-op rate >= 0.25`, or over the
trailing 30 frames `distinct-hash ratio <= 0.5`. On our own 498 runs
(>= 25 actions), measured before the first level clear:

| population | n | level-1 clear rate |
|---|---|---|
| stalled at least once | 161 (32%) | **0.13** |
| never stalled | 337 (68%) | **0.68** |

median first stall: action **25**. All four reference figures reproduce
exactly (`calibrate reference` -> `REFERENCE: REPRODUCED`).

**Two qualifications the calibration forced, both material:**

*The novelty signal never fires on its own.* Sweeping
`novelty_threshold ∈ {0.4, 0.5, 0.6}` changes nothing at any `noop_threshold`,
and the direct count is noop-only 117 runs, **novelty-only 0**, both 44
(median `novelty_30` = 0.93). In this data the detector is a no-op-rate
detector. The clause is kept — it may matter on other games — but it must not
be presented as a second independent signal.

*Most of the 5.2x is between games, not within.* Held-out (leave-one-game-out)
discrimination is **1.33**, versus 5.2 in-sample. Per game, among the 8 games
with >=3 runs on each side, the median gap is **+0.26** (never-stalled minus
stalled clear rate):

    ar25 0.50 / 0.92   cd82 0.19 / 0.50   cn04 0.00 / 0.19   ft09 0.62 / 0.83
    g50t 0.06 / 0.00   ka59 0.22 / 0.82   ls20 0.00 / 0.31   sc25 0.12 / 0.00

Hard games both stall more and clear less; the population ratio is inflated
by that confound. Stalling still predicts failure within a game, but as a
~26-point rate gap. **The ablation must be powered against the within-game
gap, not against 0.13 vs 0.68.** Sizing against the inflated number is how a
study ends up underpowered while believing it is not.

Thresholds: the whole 0.20–0.25 block is tied on every metric, so LOGO
cannot discriminate inside it (its "unanimous" 25/25 pick of 0.20/0.40 is a
tie-break artefact). The pre-registered defaults **0.25 / 0.5** are kept
deliberately, rather than taking an arbitrary corner — the restart policy in
this project was overfitted exactly by picking an in-sample optimum.

## 3. Loop-engineering constraints

The published sources for these (Reflexion, Self-Refine, Voyager, and the
prompt-optimisation line: OPRO / TextGrad / DSPy) were **not read for this
work — the references come from memory and are unverified. Verify before
repeating them externally.** The constraints stand on their own.

| constraint | implementation |
|---|---|
| write-gating on verified outcomes | `validate_evidence` drops any section naming an action id absent from `valid_actions`/the packet; if every factual section is dropped the whole revision falls back |
| grounded critique | the input is a deterministic stall packet, never the model's recollection |
| anchoring against degeneration | revisions append to an immutable base prompt, capped at 2 per level, stack bounded at 3 |
| rollback ratchet | no measurable improvement within `eval_window` -> revert, discard the text, still consume budget |
| credit assignment | separate config namespaces, separate labelled blocks, reflect block cleared at level transitions (asserted) |

Improvement is defined mechanically: no longer stalled, or a board state
unseen at fire time appeared, or the level cleared.

## 4. Ablation — designed, not yet run

2x2 over the two loops: A (both off), B (debrief), C (reflect), D (both).

Primary endpoint: **level-1 clear rate among runs that stall** — the
population the mechanism targets. Secondary: level-2 clear rate given level-1
(the debrief endpoint), mean score, mean human-multiple.

n >= 50 per arm. At the real (within-game) effect size the required n is
*larger* than the brief's sizing implies, not smaller — sizing off 0.13 vs
0.68 would overstate power. Concretely, 50 stalled runs per arm at a 32%
stall rate needs ~156 runs per arm; four arms is ~625 runs. Costed:

| route | wall-clock | money |
|---|---|---|
| local, 20 min/game | ~200 h | ~$250-350 |
| Kaggle save-runs (free, 25 games ≈ 1.8 h) | ~45 h over 25 sessions | $0 |

If compute forces a choice: drop arm D and test the interaction later —
**never reduce n**. The earlier goal-hints ablation ran n=10 (power 0.13
after correction) and its 8/10 was uninformative in both directions.

Game selection rule: level-1 clear rate >= 0.5 in
`example-run/benchmark.json` (recorded in `NOTES.md`; the debrief branch adds
`0.1 <= P(L2|L1) <= 0.9` for its own endpoint).

## 5. Status and standing rules

* Phases 0-3 complete: reference reproduced, revise call and ratchet
  implemented, `pytest tests/reflect -q` -> 20 passed.
* Phase 4 (wiring + live smoke) and Phase 5 (ablation) are **not done**. No
  claim is made about whether this helps.
* **No Kaggle submission from an arm with n < 50 or corrected p > 0.05.**
* A null result is published with the same prominence as a positive one.
* Public-leaderboard rank is inflatable by resubmission (order statistic over
  noise: σ ≈ 0.45, the same submission has scored 0.77–1.30); private score is
  not. Resubmission is presentation, never measurement.
* Restart/meta logic stays out: `actions_per_level` is not cleared by RESET
  (`diagnostics.py:724`, comment `:726`) — monotone harmful, re-simulated
  T=30 -> -4.9%, T=60 -> -1.2%, T=inf optimal.
