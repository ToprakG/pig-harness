"""Every helper exposed in the sandbox carries its own contract.

The prompt names the helpers but does not describe their return shapes; the model is told to
read `helper.__doc__` instead. That only works if the docstring survives the splice into the
sandbox and the wrapper indirection, so it is worth asserting rather than assuming.
"""
import pytest

pytest.importorskip(
    "inference.agent.python_tool_sandbox",
    reason="requires the project environment (analyzer dependencies installed)",
)

from inference.agent.python_tool_sandbox import run_sandboxed_python  # noqa: E402
from inference.utils.grid_utils import format_grid_ascii  # noqa: E402

SIZE = 8

HELPERS = (
    "diff_frames",
    "track_objects",
    "find_background",
    "find_hud",
    "find_symmetry",
    "path_between",
    "action_effects",
)


def _run(code):
    grid = [[0] * SIZE for _ in range(SIZE)]
    grid[2][2] = 9
    frame = {
        "ascii": format_grid_ascii(grid),
        "step": 1,
        "level": 1,
        "shape": [SIZE, SIZE],
        "grid": grid,
    }

    def _reject_actions(actions):
        raise AssertionError(f"unexpected action call: {actions}")

    outcome = run_sandboxed_python(
        code=code,
        timeout_seconds=30,
        initial_state={
            "current_frame": frame,
            "history": [{"action": "", "frame": frame}],
            "valid_actions": ["UP"],
            "last_action_result": {},
        },
        action_handler=_reject_actions,
    )
    assert outcome["error"] == "", outcome["error"]
    return outcome["result"]


def test_every_exposed_helper_has_a_readable_docstring():
    listing = ", ".join(f"'{name}': {name}" for name in HELPERS)
    result = _run(
        f"helpers = {{{listing}}}\n"
        "result = {name: len((fn.__doc__ or '').strip()) for name, fn in helpers.items()}\n"
    )

    assert set(result) == set(HELPERS)
    for name, length in result.items():
        assert length > 40, f"{name} has no usable docstring in the sandbox"


def test_frame_view_crop_and_window_document_themselves():
    result = _run(
        "result = {\n"
        "    'crop': len((current_frame.crop.__doc__ or '').strip()),\n"
        "    'window': len((current_frame.window.__doc__ or '').strip()),\n"
        "}\n"
    )

    assert result["crop"] > 40
    assert result["window"] > 40


def test_a_docstring_states_the_return_shape_the_prompt_no_longer_lists():
    result = _run("result = {'doc': diff_frames.__doc__}\n")

    doc = result["doc"]
    for key in ("changed", "regions", "objects"):
        assert key in doc, f"diff_frames docstring does not mention {key}"
