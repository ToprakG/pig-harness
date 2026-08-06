"""`request_timeout_seconds()` must never be driven by the session soft deadline.

`soft_end_time` is a session-wide pacing hint (`taaf/solver.py`: "solvers never
need to check the clock themselves"), not a per-request budget. Folding it into
the `min()` floored every request at 0.1s once the deadline passed, and
`should_stop()` has no soft-deadline check to end the resulting spin.
"""

from __future__ import annotations

from types import SimpleNamespace

from inference.framework.solver import _HarnessGameSession


def make_session(*, configured, per_game_remaining, soft_remaining):
    """A session stub exposing only what request_timeout_seconds() reads."""
    session = _HarnessGameSession.__new__(_HarnessGameSession)
    session.analyzer = SimpleNamespace(_timeout=configured)
    session.solver = SimpleNamespace(
        max_runtime_s_per_game=None if per_game_remaining is None else 1800.0,
        soft_time_remaining_seconds=lambda: soft_remaining,
    )
    session.timing_payload = lambda: {
        "run_elapsed_seconds": 0.0,
        "time_remaining_seconds": per_game_remaining,
    }
    return session


def test_expired_soft_deadline_does_not_collapse_the_timeout():
    """The regression: soft deadline passed, plenty of per-game time left."""
    session = make_session(configured=120.0, per_game_remaining=1500.0,
                           soft_remaining=0.0)
    assert session.request_timeout_seconds() == 120.0


def test_soft_deadline_is_ignored_even_when_it_is_the_smallest():
    session = make_session(configured=120.0, per_game_remaining=1500.0,
                           soft_remaining=3.0)
    assert session.request_timeout_seconds() == 120.0


def test_per_game_remaining_still_bounds_the_request():
    """The per-game term is kept: it is self-limiting (it reaches 0 exactly
    when runtime_limit_reached() makes should_stop() true)."""
    session = make_session(configured=120.0, per_game_remaining=30.0,
                           soft_remaining=None)
    assert session.request_timeout_seconds() == 30.0


def test_configured_timeout_used_when_no_per_game_limit():
    session = make_session(configured=120.0, per_game_remaining=None,
                           soft_remaining=None)
    assert session.request_timeout_seconds() == 120.0


def test_returns_none_when_nothing_is_configured():
    session = make_session(configured=None, per_game_remaining=None,
                           soft_remaining=0.0)
    assert session.request_timeout_seconds() is None
