# Göktürk V4: request-timeout collapse fix + keyboard modality hints (correction)

## Correction to earlier claim in this file

Earlier version of this doc claimed the runtime-cap bug was already fixed
downstream via a pickle byte-patch, based on a Kaggle run finishing in
8h35m (matching a proposed 30900s value). **That inference was wrong.**
Fetched the actual run log afterward: `max_runtime_s_per_game=7920.0` was
still in effect, unchanged, for that run. The longer duration and 2.87->8.25
mean-score jump between two byte-identical notebook versions (Kaggle UI
diff: +0/-0) is explained by the SAME unfixed bug behaving differently
run-to-run -- not a fix. Score is currently a coin flip on this bug.

## Actual root cause (confirmed against `fix/request-timeout-soft-deadline`)

`solver.py: request_timeout_seconds()` took the min of three candidates:
configured timeout, per-game remaining time, and session-wide
`soft_time_remaining_seconds()`. The third one is a *pacing hint shared
across all games*, not a per-request budget. Once the session soft deadline
passes, it reads 0.0 for every game that starts afterward -- flooring the
per-request timeout at `max(0.1, min(...))` = 0.1s. No request can complete
in 0.1s, `should_stop()` has no soft-deadline check, so the game spins
failing requests (`Read timed out. (read timeout=0.1)`, thousands of them
in one observed run) until the unrelated 7920s per-game hard cap finally
kills it -- burning ~76% of the wall-clock budget on nothing.

## Fix applied (this commit)

Removed `soft_remaining` from the candidate list in
`request_timeout_seconds()` (`ARC3-Inference/inference/framework/solver.py`).
Matches `fix/request-timeout-soft-deadline` exactly -- same diff, verified
independently against my own diagnosis before copying. That branch should
still be preferred for merge if it has more testing behind it; this commit
exists so the Kaggle bundle (a separate deployment artifact, not wired to
this repo) can carry the identical fix.

## Keyboard modality prompt hints

Unchanged from V3 note: grounding in place/grid/head-direction cells,
overlaps `feat/keyboard-nav-hints` (prefer that branch), kept here for the
independent-derivation comparison value.
