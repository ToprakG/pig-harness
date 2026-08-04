"""Integration test: an enabled meta policy fires exactly one tagged restart
on a stalled run, then the collapsed posterior commits to the long attempt."""

from __future__ import annotations

import threading
from pathlib import Path

from inference.framework.solver import HarnessSolver, _HarnessGameSession

from .test_noop_parity import GRID_A, GRID_B, GRID_C, FakeGame, FakeState, ScriptedAnalyzer


def test_enabled_meta_fires_one_tagged_restart(tmp_path: Path):
    # 8 uneventful actions, alternating boards, never clearing a level
    states = [FakeState(GRID_B if i % 2 else GRID_C) for i in range(8)]
    calls = [{"action": "UP"} for _ in range(8)]
    stop_event = threading.Event()
    game = FakeGame(states, reset_state_factory=lambda _prev: FakeState(GRID_A))
    analyzer = ScriptedAnalyzer(calls, stop_event)
    solver = HarnessSolver(
        label="meta-test",
        meta_config={
            "enabled": True,
            "first_clear_deadline": 3,
            "min_actions_before_restart": 0,
            "phi": 0.25,
            "max_restarts_per_pass": 2,
        },
    )
    session = _HarnessGameSession(
        solver=solver,
        game=game,
        analyzer=analyzer,
        game_index=0,
        pass_index=0,
        state_path=tmp_path / "state.json",
        transcript_path=tmp_path / "transcript.log",
        analysis_html_relpath="analysis.html",
        stop_event=stop_event,
        viewer_data_path=tmp_path / "viewer.json",
    )
    session.play()

    # exactly one restart: fired at t=3, then P(tractable|1 fail)=0.10 < phi
    assert game.game_run.history.count("RESET") == 1
    assert game.game_run.history.index("RESET") == 3  # after the 3rd action

    tagged = [e for e in session.viewer_events if e.get("meta_restart")]
    assert len(tagged) == 1
    assert tagged[0]["action_name"] == "RESET"

    meta = session._meta()
    assert meta.controller.counters == {"restarts_used": 1,
                                        "failed_attempts": 1}
    # attempt-relative counter was reset at the restart: 8 - 3 actions since
    assert meta.attempt_action_count == 5
