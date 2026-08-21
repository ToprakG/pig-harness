# Göktürk V5: gateway 429 retry/backoff (correction to prior concurrency framing)

## What I got wrong before

Suggested lowering `concurrency` to test whether the platform limits parallel
games. That framing was wrong -- checked official docs, no such platform
limit exists:

- `docs.arcprize.org/toolkit/competition_mode.md`: no concurrent-game limit
  stated. Concurrency is entirely `bm.solver.concurrency` (a HarnessSolver
  field we set ourselves, seen at 28 in production logs).
- `docs.arcprize.org/rate_limits.md`: the real constraint is **600 requests
  per minute (RPM), account-wide** -- shared across every concurrently
  running game, not a per-game or per-concurrency-slot limit.

## What the code actually shows (verified, not inferred)

Traced the real submission path (`taaf-duck-sub-20260805-share` notebook,
cell 14): `TRUE_SUBMISSION` uses `arc_agi.OperationMode.COMPETITION` against
`http://gateway:8001/` -- every `.step()`/`.reset()` call during a real
scored run is a network request to that gateway, subject to the 600 RPM
ceiling.

`arc_agi==0.9.9`'s `RemoteEnvironmentWrapper` calls
`response.raise_for_status()` unconditionally on every gateway request
(`remote_wrapper.py:108,199`) -- confirmed by grep, zero retry/backoff/429
handling anywhere in that package.

`taaf/game_api.py::_execute_action` called `self.env.step(...)` directly,
so an unhandled `HTTPError` (429 or otherwise) propagates up into
`solver.py`'s broad `except Exception` (line ~625), which converts it into
`stop_reason="action_error"` -- silently costing the game a wasted LLM turn,
with no signal anywhere that a rate limit (not a real environment problem)
was the cause. With `concurrency=28` threads each capable of firing a
gateway request the moment its LLM turn returns, request arrival across
threads is effectively uncorrelated -- bursts above 10 req/s (600 RPM) are
plausible, especially right after the gateway comes up and many games issue
their first `reset()` near-simultaneously.

## Fix (this commit)

Added `_step_with_backoff` in `taaf/game_api.py`: wraps `self.env.step(...)`
with up to 5 retries, exponential backoff (0.5s base, doubling), honoring
`Retry-After` if the gateway sends one. Only retries on 429 and on
connection-level errors (`ConnectionError`, `Timeout`) that a saturated
gateway plausibly produces; any other `HTTPError` re-raises immediately,
unchanged from prior behavior.

## What this does NOT claim

- Not verified end-to-end against a real competition rerun yet (can't
  trigger one without spending the scarce daily submission slot).
- Doesn't touch `concurrency=28` itself -- the retry is the targeted fix;
  lowering concurrency remains an option if retries still aren't enough,
  but there's no evidence yet that's needed.
- No visibility into how OFTEN 429s actually occur in a real run (no gateway
  access outside official reruns), so I can't size the expected effect.
  The `logging.warning` added on every retry gives that visibility the next
  time this notebook is actually submitted.
