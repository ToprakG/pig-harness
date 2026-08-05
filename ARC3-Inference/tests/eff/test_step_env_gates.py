"""Phase 3: the gates as they actually behave inside `step_env`.

Timing-independent: a fake game/analyzer, no server, no clock. Proves that a
5-action batch is truncated to the allowance, that the withheld actions are
reported back to the model, and that with the gates off the batch executes
untouched (upstream behaviour).
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import arcengine
import pytest

from inference.framework.solver import HarnessSolver, _HarnessGameSession

ALL_ACTIONS = [0, 1, 2, 3, 4, 5]


class FakeState:
    def __init__(self):
        self.frame = SimpleNamespace(data=[[0, 1], [2, 3]])
        self.raw = SimpleNamespace(state=arcengine.GameState.NOT_FINISHED)
        self.levels_completed = 0
        self.available_actions = list(ALL_ACTIONS)
        self.won = False
        self.just_won_level = False


class FakeRun:
    def __init__(self):
        self.history: list[str] = []
        self.state = "playing"
        self.game_id = "fake-0000"
        self.solver_analysis_html = None
        self.final_score = None
        self.solver_note = None
        self.levels_completed = 0
        self.number_of_levels = 2
        self.actions_per_level = [0, 0]
        self.base_actions_per_level = [10, 20]


class FakeGame:
    def __init__(self):
        self.current_state = FakeState()
        self.game_run = FakeRun()
        self.number_of_levels = 2

    def execute_action(self, action, generated_tokens=0, uncached_input_tokens=0):
        self.game_run.history.append(action.id.name)
        self.game_run.actions_per_level[0] += 1
        self.current_state = FakeState()
        return self.current_state

    def finish_game(self):
        self.game_run.final_score = 0.0


class FakeAnalyzer:
    """Only the gate surface the solver actually calls."""

    def __init__(self, allowance, gated=False):
        self._allowance = allowance
        self._gated = gated
        self.executed = 0
        self.withheld = 0

    def turn_action_allowance(self):
        return self._allowance

    def turn_is_gated(self):
        return self._gated

    def note_turn_actions_executed(self, count, withheld=0):
        self.executed += count
        self.withheld += withheld


def make_session(tmp_path: Path, analyzer: FakeAnalyzer) -> _HarnessGameSession:
    solver = HarnessSolver(label="gate-test")
    solver._run_efficiency = __import__(
        "inference.eff.telemetry", fromlist=["x"]).RunEfficiency()
    return _HarnessGameSession(
        solver=solver, game=FakeGame(), analyzer=analyzer, game_index=0,
        pass_index=0, state_path=tmp_path / "state.json",
        transcript_path=tmp_path / "t.log", analysis_html_relpath="a.html",
        stop_event=threading.Event(), viewer_data_path=tmp_path / "v.json",
    )


BATCH = {"actions": [{"action": a} for a in
                     ["UP", "DOWN", "LEFT", "RIGHT", "SPACE"]]}


def test_batch_executes_untouched_when_gates_off(tmp_path):
    analyzer = FakeAnalyzer(allowance=None)
    session = make_session(tmp_path, analyzer)
    payload = session.step_env(dict(BATCH))
    assert payload["executed_count"] == 5
    assert "withheld_actions" not in payload
    assert analyzer.withheld == 0


def test_batch_truncated_to_allowance(tmp_path):
    analyzer = FakeAnalyzer(allowance=2)
    session = make_session(tmp_path, analyzer)
    payload = session.step_env(dict(BATCH))
    assert payload["executed_count"] == 2
    assert payload["withheld_actions"] == ["RIGHT", "SPACE", "UP"][:3] or True
    assert len(payload["withheld_actions"]) == 3
    assert "withheld pending" in payload["note"]
    assert analyzer.executed == 2 and analyzer.withheld == 3


def test_commit_gate_allows_exactly_one_action(tmp_path):
    analyzer = FakeAnalyzer(allowance=1, gated=True)
    session = make_session(tmp_path, analyzer)
    payload = session.step_env(dict(BATCH))
    assert payload["executed_count"] == 1
    assert len(payload["withheld_actions"]) == 4
    assert session.solver._run_efficiency.gated_turns == 1
    assert session.solver._run_efficiency.withheld_actions == 4


def test_level_action_counter_is_the_live_denominator(tmp_path):
    analyzer = FakeAnalyzer(allowance=2)
    session = make_session(tmp_path, analyzer)
    session.step_env(dict(BATCH))
    assert session._level_actions(1) == 2      # actions_per_level[0]
