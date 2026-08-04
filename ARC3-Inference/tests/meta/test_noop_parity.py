"""D4 no-op parity golden test.

With ``meta.enabled=false`` a scripted mock session must produce a viewer
event sequence byte-identical to upstream. The golden fixture was generated
from the pre-integration solver (commit before the meta hook landed); any
drift in the disabled path fails this test.

Regenerate (only when intentionally changing upstream event semantics)::

    META_PARITY_REGEN=1 uv run pytest tests/meta/test_noop_parity.py -q
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import arcengine

from inference.framework.solver import HarnessSolver, _HarnessGameSession

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "golden_viewer_events.json"

GRID_A = [[0, 1], [2, 3]]
GRID_B = [[0, 1], [2, 4]]
GRID_C = [[5, 1], [2, 4]]
GRID_D = [[5, 6], [2, 4]]

ALL_ACTIONS = [0, 1, 2, 3, 4, 5]  # RESET + ACTION1..5


class FakeState:
    def __init__(self, grid, *, state=arcengine.GameState.NOT_FINISHED,
                 levels_completed=0, just_won_level=False):
        self.frame = SimpleNamespace(data=[list(row) for row in grid])
        self.raw = SimpleNamespace(state=state)
        self.levels_completed = levels_completed
        self.available_actions = list(ALL_ACTIONS)
        self.won = state == arcengine.GameState.WIN
        self.just_won_level = just_won_level


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
        self.actions_per_level: list[int] = []


class FakeGame:
    """Scripted game: non-RESET actions pop the next scripted state; RESET
    always returns the designated reset state."""

    def __init__(self, script: list[FakeState], reset_state_factory):
        self.script = list(script)
        self.reset_state_factory = reset_state_factory
        self.current_state = FakeState(GRID_A)
        self.game_run = FakeRun()
        self.number_of_levels = 2

    def execute_action(self, action, generated_tokens=0, uncached_input_tokens=0):
        self.game_run.history.append(action.id.name)
        if action.id == arcengine.GameAction.RESET:
            new_state = self.reset_state_factory(self.current_state)
        else:
            new_state = self.script.pop(0)
        self.current_state = new_state
        self.game_run.levels_completed = int(new_state.levels_completed)
        return new_state

    def finish_game(self):
        if self.game_run.final_score is None:
            self.game_run.final_score = float(self.game_run.levels_completed)


class ScriptedAnalyzer:
    """Executes one scripted step_env call per analysis step, then stops."""

    generated_tokens = 0

    def __init__(self, calls: list[dict], stop_event: threading.Event):
        self.calls = list(calls)
        self.stop_event = stop_event
        self.payloads: list[dict] = []

    def analyze(self, state_path, action_count, *, valid_actions, step_env,
                transcript_path, analysis_step, request_timeout_seconds,
                should_stop):
        if not self.calls:
            self.stop_event.set()
            return SimpleNamespace(retryable_failure=False, step_executed=False,
                                   yielded_control=False)
        self.payloads.append(step_env(self.calls.pop(0)))
        return SimpleNamespace(retryable_failure=False, step_executed=True,
                               yielded_control=False)


def scripted_states() -> list[FakeState]:
    return [
        FakeState(GRID_B),                                     # UP: board changes
        FakeState(GRID_B),                                     # DOWN: no-op
        FakeState(GRID_C),                                     # LEFT: board changes
        FakeState(GRID_C, state=arcengine.GameState.GAME_OVER),  # SPACE: game over
        # after auto-RESET:
        FakeState(GRID_D, levels_completed=1, just_won_level=True),  # RIGHT: clear
    ]


def scripted_calls() -> list[dict]:
    return [
        {"action": "UP"},
        {"actions": [{"action": "DOWN"}, {"action": "LEFT"}]},
        {"action": "SPACE"},
        {"action": "RIGHT"},
    ]


def build_session(tmp_path: Path, solver: HarnessSolver):
    stop_event = threading.Event()
    game = FakeGame(scripted_states(),
                    reset_state_factory=lambda _prev: FakeState(GRID_A))
    analyzer = ScriptedAnalyzer(scripted_calls(), stop_event)
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
    return session, game, analyzer


def run_scripted_session(tmp_path: Path, solver: HarnessSolver) -> list[dict]:
    session, _game, _analyzer = build_session(tmp_path, solver)
    session.play()
    return session.viewer_events


def test_noop_parity_matches_upstream_golden(tmp_path):
    solver = HarnessSolver(label="parity-test")
    events = run_scripted_session(tmp_path, solver)
    if os.environ.get("META_PARITY_REGEN") == "1":
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(events, indent=2, sort_keys=True))
    golden = json.loads(GOLDEN_PATH.read_text())
    assert json.loads(json.dumps(events, sort_keys=True)) == golden


def test_noop_parity_with_meta_disabled_config(tmp_path):
    """The explicit disabled meta config must also be a strict no-op."""
    solver = HarnessSolver(label="parity-test")
    if hasattr(solver, "meta_config"):
        solver.meta_config = {"enabled": False}
    events = run_scripted_session(tmp_path, solver)
    golden = json.loads(GOLDEN_PATH.read_text())
    assert json.loads(json.dumps(events, sort_keys=True)) == golden
