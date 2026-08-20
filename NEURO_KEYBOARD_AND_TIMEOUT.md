# Göktürk V3: keyboard modality prompt hints + runtime-cap root cause

## Keyboard modality (avatar probe + occupancy + BFS)

Grounding: place/grid/head-direction cells -- navigation runs on a dedicated
allocentric-map substrate, not a trial-and-error policy. Duck's own 500-run
analysis: keyboard-only games mean 0.12 vs click 2.20; two score zero across
20 attempts.

IMPORTANT OVERLAP: `feat/keyboard-nav-hints` already ships a feature-flagged
`_keyboard_nav_lines()` (`DUCK_KEYBOARD_NAV_HINTS`) doing essentially this.
Checked before writing -- found after, not before, this draft was written
locally. **Prefer that branch.** This commit is kept for the independent
derivation / comparison value, not as a merge candidate over it.

## Runtime-cap root cause (Kaggle submission artifact, not this repo's code)

On a Kaggle run of a duck-derived submission notebook, all 25/25 games were
cut off within 30 seconds of each other (7920-7950s), not by natural
completion. Root cause: `benchmark_initial.pkl` (a build-time artifact, not
tracked in this source repo) had `max_runtime_s_per_game=7920.0` (132 min)
baked in regardless of the 9h total budget, with `concurrency=28 >= 25 games`
so all games run in one round and all hit the same wall. ~76% of the 9h
budget went unused.

IMPORTANT OVERLAP: `fix/request-timeout-soft-deadline` ("drop session soft
deadline from per-request budget") is very likely the same class of bug.
Check that branch's mechanism before assuming this note's byte-patch method
is needed -- it may already be fixed upstream of wherever `benchmark_initial.pkl`
gets built.

Fix applied downstream (not in this repo): patched the pickled float in place
(`struct.pack_into('>d', ...)`, offset located via `pickletools.dis`) from
7920.0 to 30900.0 (515 min = 9h minus ~25min setup/teardown budget). Verified
the pickle still parses (`pickletools.dis` succeeds) but not yet verified
against a real re-run.

## NEURO.md

Full neuroscience-grounding writeup (CLS, perseveration, predictive coding,
place/grid cells, successor representation, and where the analogy stops) is
in this same commit for reference.
