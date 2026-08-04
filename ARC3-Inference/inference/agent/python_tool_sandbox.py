"""Lightweight isolated runner for analyzer Python tool calls."""
from __future__ import annotations

import inspect
import json
import os
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import textwrap
import time
from typing import Any, Callable

from inference.perception import background as _background
from inference.perception import crop as _crop
from inference.perception import diffing as _diffing
from inference.perception import effects as _effects
from inference.perception import grid as _grid
from inference.perception import hud as _hud
from inference.perception import objects as _objects
from inference.perception import pathfinding as _pathfinding
from inference.perception import symmetry as _symmetry
from inference.utils import segmentation as _segmentation
from inference.utils.grid_utils import ARC_COLOR_CHARS

# Definition order does not matter -- names resolve at call time -- but dependencies come
# first so the assembled bootstrap reads top-down.
_PERCEPTION_MODULES = (
    _grid,
    _objects,
    _crop,
    _background,
    _hud,
    _symmetry,
    _pathfinding,
    _diffing,
    _effects,
)

_PROJECT_IMPORT = re.compile(r"^(?:from|import)\s+inference\b")


def _sandbox_source(module) -> str:
    """Module source with its project imports removed.

    The perception modules are pasted into a single flat namespace in the sandbox, where
    project packages cannot be imported, so imports between them are both unnecessary and
    unresolvable there. Parenthesized multi-line imports are dropped in full.
    """
    kept = []
    skipping = False
    for line in inspect.getsource(module).splitlines(keepends=True):
        if skipping:
            skipping = ")" not in line
            continue
        if _PROJECT_IMPORT.match(line):
            skipping = "(" in line and ")" not in line
            continue
        kept.append(line)
    return "".join(kept)


_SANDBOX_BOOTSTRAP = textwrap.dedent(
    r"""
    import builtins
    import contextlib
    import io
    import json
    import os
    import sys
    import traceback

    try:
        import resource
    except ImportError:  # pragma: no cover
        resource = None

    COLOR_CHARS = ""

    __SEGMENTATION_SOURCE__

    __PERCEPTION_SOURCE__

    HOST_STDOUT = sys.stdout

    SAFE_MODULES = {
        "bisect",
        "collections",
        "copy",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
    }
    SAFE_BUILTINS = {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "complex",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "Exception",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "hasattr",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "TypeError",
        "type",
        "ValueError",
        "RuntimeError",
        "zip",
    }


    def _send(payload):
        HOST_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
        HOST_STDOUT.flush()


    def _recv():
        line = sys.stdin.readline()
        if not line:
            raise EOFError("sandbox input closed")
        return json.loads(line)


    class FrameView:
        def __init__(self, *, ascii, step, level, shape, grid):
            self.ascii = ascii
            self.step = step
            self.level = level
            self.shape = tuple(shape)
            self._grid = grid
            self._segmentation = None

        @property
        def segmentation(self):
            if self._segmentation is None:
                self._segmentation = segment_layer(self._grid, COLOR_CHARS)
            return self._segmentation

        # ASCII for one region, clipped to the frame; labels put the coordinates on it.
        def crop(self, bbox, labels=True):
            return crop_ascii(self._grid, COLOR_CHARS, bbox, labels=labels)

        # ASCII for the square within `radius` cells of (row, col).
        def window(self, row, col, radius=3, labels=True):
            return window_ascii(self._grid, COLOR_CHARS, row, col, radius=radius, labels=labels)

        def __str__(self):
            rows, cols = self.shape
            return f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"

        __repr__ = __str__


    class HistoryEntryView:
        def __init__(self, *, action, frame):
            self.action = action
            self.frame = frame

        def __str__(self):
            return f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame})"

        __repr__ = __str__


    class TransitionView:
        def __init__(self, *, action, before_frame, after_frame, result):
            self.action = action
            self.before_frame = before_frame
            self.after_frame = after_frame
            self.frame = after_frame
            self.result = dict(result) if isinstance(result, dict) else {}

        def diff(self, objects=True):
            return diff_frames(self.before_frame, self.after_frame, objects=objects)

        def track(self, **kwargs):
            return track_objects(self.before_frame, self.after_frame, **kwargs)

        def __str__(self):
            return (
                "ActionTransitionView("
                f"action={self.action!r}, "
                f"before_frame={self.before_frame}, "
                f"after_frame={self.after_frame})"
            )

        __repr__ = __str__


    # One whole `action(...)` call: the frame before the first action through the frame after
    # the last. `TransitionView` covers a single action, so reading it after a batch shows
    # only the final step and hides what the earlier actions did.
    class BatchView:
        def __init__(self, *, actions, before_frame, after_frame, result, steps):
            self.actions = list(actions)
            self.before_frame = before_frame
            self.after_frame = after_frame
            self.frame = after_frame
            self.result = dict(result) if isinstance(result, dict) else {}
            self.steps = list(steps)

        def diff(self, objects=True):
            return diff_frames(self.before_frame, self.after_frame, objects=objects)

        def track(self, **kwargs):
            return track_objects(self.before_frame, self.after_frame, **kwargs)

        def __str__(self):
            return (
                "ActionBatchView("
                f"actions={len(self.actions)}, "
                f"steps={len(self.steps)}, "
                f"before_frame={self.before_frame}, "
                f"after_frame={self.after_frame})"
            )

        __repr__ = __str__


    # Summarize the change from `before` to `after`; None if either frame is missing.
    # objects=False skips the object-level diff, avoiding segmentation of both frames
    # when only cell-level changes are needed.
    def diff_frames(before, after, objects=True):
        if before is None or after is None:
            return None
        return diff_grids(
            before._grid,
            after._grid,
            COLOR_CHARS,
            before_segmentation=before.segmentation if objects else None,
            after_segmentation=after.segmentation if objects else None,
        )


    def _background_pixels(frame, background_fraction):
        rows, cols = frame.shape
        return int(rows * cols * background_fraction)


    # Most likely background color of `frame`, with the evidence behind the call. Pass
    # other frames to also get how static that color is across them.
    def find_background(frame, other_frames=(), max_colors=4):
        if frame is None:
            return None
        return identify_background(
            frame._grid,
            COLOR_CHARS,
            segmentation=frame.segmentation,
            other_grids=[other._grid for other in other_frames if other is not None],
            max_colors=max_colors,
        )


    # Candidate HUD bars and segmented strips along the frame's edges, plus the interior
    # region left once they are trimmed off.
    def find_hud(frame, **kwargs):
        if frame is None:
            return None
        return detect_hud(frame._grid, COLOR_CHARS, frame.segmentation, **kwargs)


    # Mirror, rotation, and diagonal symmetry of the frame, or of one region of it.
    def find_symmetry(frame, bbox=None, ignore_colors=()):
        if frame is None:
            return None
        return detect_symmetry(frame._grid, COLOR_CHARS, bbox=bbox, ignore_colors=ignore_colors)


    # Shortest path between two cells of the frame, as directions to act on.
    def path_between(frame, start, goal, **kwargs):
        if frame is None:
            return None
        return find_path(frame._grid, COLOR_CHARS, start, goal, **kwargs)


    # Tally which actions have changed the board and which are proven no-ops. Restricted to
    # the current level by default, since mechanics change between levels.
    def _action_effects(transitions, current_frame, level_only=True, **kwargs):
        level = current_frame.level if current_frame is not None else None
        records = []
        for transition in transitions:
            before = transition.before_frame
            after = transition.after_frame
            if level_only and level is not None:
                if after is not None and after.level != level:
                    continue
                if before is not None and before.level != level:
                    continue
            records.append(
                (
                    transition.action,
                    before._grid if before is not None else None,
                    after._grid if after is not None else None,
                )
            )
        return summarize_action_effects(records, COLOR_CHARS, **kwargs)


    # Track objects from `before` to `after`, following shape and color changes.
    def track_objects(before, after, background_fraction=0.25, **kwargs):
        if before is None or after is None:
            return None
        return match_objects(
            before.segmentation,
            after.segmentation,
            background_pixels=_background_pixels(after, background_fraction),
            **kwargs,
        )


    # Hand each wrapper the contract of the function it delegates to, so a helper can be looked
    # up on demand with `print(helper.__doc__)` rather than described in full in every prompt.
    FrameView.crop.__doc__ = crop_ascii.__doc__
    FrameView.window.__doc__ = window_ascii.__doc__
    diff_frames.__doc__ = diff_grids.__doc__
    track_objects.__doc__ = match_objects.__doc__
    find_background.__doc__ = identify_background.__doc__
    find_hud.__doc__ = detect_hud.__doc__
    find_symmetry.__doc__ = detect_symmetry.__doc__
    path_between.__doc__ = find_path.__doc__
    _action_effects.__doc__ = summarize_action_effects.__doc__


    def _frame_from_payload(payload):
        if not isinstance(payload, dict):
            return None
        return FrameView(
            ascii=str(payload.get("ascii", "")),
            step=int(payload.get("step", 0)),
            level=int(payload.get("level", 0)),
            shape=payload.get("shape", [0, 0]),
            grid=payload.get("grid", []),
        )


    def _history_from_payload(payload):
        items = []
        for entry in payload or []:
            if not isinstance(entry, dict):
                continue
            items.append(
                HistoryEntryView(
                    action=str(entry.get("action", "")),
                    frame=_frame_from_payload(entry.get("frame")),
                )
            )
        return items


    def _transitions_from_history(history, last_action_result):
        transitions = []
        for index, entry in enumerate(history):
            action = str(getattr(entry, "action", "") or "").strip()
            if not action:
                continue
            before_frame = history[index - 1].frame if index > 0 else None
            transitions.append(
                TransitionView(
                    action=action,
                    before_frame=before_frame,
                    after_frame=entry.frame,
                    result={},
                )
            )
        if transitions and isinstance(last_action_result, dict):
            transitions[-1].result = dict(last_action_result)
        return transitions


    def _json_safe(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        return str(value)


    def _sanitize_exception(exc):
        extracted = traceback.extract_tb(exc.__traceback__)
        user_frames = [frame for frame in extracted if frame.filename == "<python_tool>"]
        lines = ["Traceback (most recent call last):"]
        for frame in user_frames or extracted[-1:]:
            lines.append(f'  File "<python_tool>", line {frame.lineno}, in {frame.name}')
        lines.append(f"{exc.__class__.__name__}: {exc}")
        return "\n".join(lines)


    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name or "").split(".", 1)[0]
        if root not in SAFE_MODULES:
            raise ImportError(f"Module '{name}' is not allowed in the sandbox.")
        return builtins.__import__(name, globals, locals, fromlist, level)


    def _set_limits(timeout_seconds):
        if resource is None:
            return
        cpu_limit = max(1, int(timeout_seconds)) + 1
        for limit, value in (
            (getattr(resource, "RLIMIT_CPU", None), cpu_limit),
            (getattr(resource, "RLIMIT_FSIZE", None), 1_000_000),
            (getattr(resource, "RLIMIT_NOFILE", None), 32),
        ):
            if limit is None:
                continue
            try:
                resource.setrlimit(limit, (value, value))
            except (OSError, ValueError):
                pass


    def _normalize_actions(actions):
        if isinstance(actions, str):
            items = [actions]
        elif isinstance(actions, dict):
            items = [actions]
        elif isinstance(actions, (list, tuple)):
            items = list(actions)
        else:
            raise TypeError(
                "action(actions) expects a string, an action object, or a list of action strings/objects."
            )
        if not items:
            raise ValueError("action(actions) requires at least one action.")

        normalized = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, str):
                action_name = item.strip()
                if not action_name:
                    raise ValueError(f"Action {index} is empty.")
                normalized.append({"action": action_name})
                continue
            if isinstance(item, dict):
                action_name = str(item.get("action", "")).strip()
                if not action_name:
                    raise ValueError(f"Action {index} is missing an `action` field.")
                entry = {"action": action_name}
                if action_name.upper() == "MOUSE" and ("x" in item or "y" in item):
                    raise ValueError(
                        f"Action {index} uses legacy MOUSE x/y fields; use row and col."
                    )
                if "row" in item:
                    entry["row"] = item.get("row")
                if "col" in item:
                    entry["col"] = item.get("col")
                normalized.append(entry)
                continue
            raise TypeError(f"Action {index} must be a string or a dict.")
        return normalized


    def _batch_step_count(runtime_globals):
        batch = runtime_globals.get("last_batch")
        return len(batch.steps) if batch is not None else 0


    def main():
        initial = _recv()
        global COLOR_CHARS
        COLOR_CHARS = str(initial.get("color_chars") or "")
        timeout_seconds = max(1, int(initial.get("timeout_seconds", 30)))
        sandbox_cwd = str(initial.get("sandbox_cwd", "")).strip()
        if sandbox_cwd:
            os.chdir(sandbox_cwd)
        _set_limits(timeout_seconds)

        action_results = []
        stdout = io.StringIO()
        runtime_globals = {
            "__builtins__": {
                name: getattr(builtins, name)
                for name in SAFE_BUILTINS
            },
            "result": None,
        }
        runtime_globals["__builtins__"]["__import__"] = _safe_import

        def _refresh_state(state_payload):
            current_frame = _frame_from_payload(state_payload.get("current_frame"))
            history = _history_from_payload(state_payload.get("history"))
            last_action_result = state_payload.get("last_action_result")
            action_result = (
                dict(last_action_result) if isinstance(last_action_result, dict) else {}
            )
            transitions = _transitions_from_history(history, action_result)
            last_transition = transitions[-1] if transitions else None

            runtime_globals["current_frame"] = current_frame
            runtime_globals["latest_frame"] = current_frame
            runtime_globals["history"] = history
            runtime_globals["transitions"] = transitions
            runtime_globals["last_transition"] = last_transition
            runtime_globals["previous_frame"] = (
                last_transition.before_frame if last_transition is not None else None
            )
            runtime_globals["last_action_frame"] = (
                last_transition.after_frame if last_transition is not None else None
            )
            def action_effects(level_only=True, **kwargs):
                return _action_effects(
                    transitions, current_frame, level_only=level_only, **kwargs
                )

            action_effects.__doc__ = summarize_action_effects.__doc__
            runtime_globals["action_effects"] = action_effects
            runtime_globals["last_action"] = last_transition.action if last_transition is not None else None
            runtime_globals["valid_actions"] = [str(item) for item in state_payload.get("valid_actions", [])]
            runtime_globals["last_action_result"] = action_result

        def action(actions):
            normalized_actions = _normalize_actions(actions)
            before_frame = runtime_globals.get("current_frame")
            steps_before = len(runtime_globals.get("transitions") or ())
            _send({"type": "action", "actions": normalized_actions})
            reply = _recv()
            if reply.get("type") == "action_error":
                raise RuntimeError(str(reply.get("error", "action failed")))
            if reply.get("type") != "action_result":
                raise RuntimeError("Invalid action response from sandbox host.")
            action_result = reply.get("action_result") or {}
            action_results.append(action_result)
            _refresh_state(reply.get("state") or {})
            transitions = runtime_globals.get("transitions") or []
            # History is append-only in practice, but if the host ever trims it, fall back to
            # the frames this batch is known to have produced rather than slicing nonsense.
            steps = transitions[steps_before:] if len(transitions) >= steps_before else []
            if not steps:
                steps = transitions[-len(normalized_actions):] if transitions else []
            runtime_globals["last_batch"] = BatchView(
                actions=normalized_actions,
                before_frame=before_frame,
                after_frame=runtime_globals.get("current_frame"),
                result=action_result,
                steps=steps,
            )
            return action_result

        runtime_globals["action"] = action
        runtime_globals["diff_frames"] = diff_frames
        runtime_globals["track_objects"] = track_objects
        runtime_globals["find_background"] = find_background
        runtime_globals["find_hud"] = find_hud
        runtime_globals["find_symmetry"] = find_symmetry
        runtime_globals["path_between"] = path_between
        # Only set by `action(...)`, so it always describes a batch from this run of the code.
        runtime_globals["last_batch"] = None
        # The one object that outlives this process: whatever is left in it is handed back to
        # the host and restored on the next call.
        incoming_notes = initial.get("notes")
        runtime_globals["notes"] = (
            dict(incoming_notes) if isinstance(incoming_notes, dict) else {}
        )
        _refresh_state(initial.get("state") or {})

        # Rebuild the previous call's batch from history so that inspecting without acting still
        # has it. Without this, `last_batch` is mysteriously None on exactly the turns spent
        # working out what the last batch did.
        try:
            carried_steps = int(initial.get("last_batch_steps") or 0)
        except (TypeError, ValueError):
            carried_steps = 0
        carried_transitions = runtime_globals.get("transitions") or []
        if carried_steps > 0 and len(carried_transitions) >= carried_steps:
            batch_steps = carried_transitions[-carried_steps:]
            runtime_globals["last_batch"] = BatchView(
                actions=[step.action for step in batch_steps],
                before_frame=batch_steps[0].before_frame,
                after_frame=runtime_globals.get("current_frame"),
                result=runtime_globals.get("last_action_result") or {},
                steps=batch_steps,
            )

        try:
            compiled = compile(str(initial.get("code", "")), "<python_tool>", "exec")
            with contextlib.redirect_stdout(stdout):
                exec(compiled, runtime_globals, runtime_globals)
            _send(
                {
                    "type": "final",
                    "stdout": stdout.getvalue(),
                    "result": _json_safe(runtime_globals.get("result")),
                    "action_results": _json_safe(action_results),
                    "notes": _json_safe(runtime_globals.get("notes")),
                    "last_batch_steps": _batch_step_count(runtime_globals),
                }
            )
        except Exception as exc:
            # Notes are returned even on failure: a fact stored before the traceback is still a
            # fact, and losing it would send the model back to re-deriving it.
            _send(
                {
                    "type": "error",
                    "error": _sanitize_exception(exc),
                    "stdout": stdout.getvalue(),
                    "action_results": _json_safe(action_results),
                    "notes": _json_safe(runtime_globals.get("notes")),
                    "last_batch_steps": _batch_step_count(runtime_globals),
                }
            )


    if __name__ == "__main__":
        main()
    """
).replace("__SEGMENTATION_SOURCE__\n", inspect.getsource(_segmentation)).replace(
    "__PERCEPTION_SOURCE__\n",
    "\n\n".join(_sandbox_source(module) for module in _PERCEPTION_MODULES),
)


def _sanitize_host_error_text(text: str) -> str:
    if not str(text or "").strip():
        return "Sandbox process exited unexpectedly."
    return "Sandbox process exited unexpectedly."


def _sandbox_env() -> dict[str, str]:
    return {
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PATH": os.environ.get("PATH", ""),
    }


def _send_json_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        try:
            process.kill()
        except OSError:
            pass


def _wait_for_process_exit(process: subprocess.Popen[str], *, timeout: float = 1.0) -> None:
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
    except OSError:
        return

    try:
        process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        pass


def run_sandboxed_python(
    *,
    code: str,
    timeout_seconds: int,
    initial_state: dict[str, Any],
    action_handler: Callable[[list[dict[str, Any]]], dict[str, Any]],
    notes: dict[str, Any] | None = None,
    last_batch_steps: int = 0,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rgb_python_tool_") as sandbox_dir:
        host_action_results: list[dict[str, Any]] = []
        try:
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", _SANDBOX_BOOTSTRAP],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                cwd=sandbox_dir,
                env=_sandbox_env(),
                start_new_session=True,
            )
        except OSError:
            return {
                "error": "Sandbox process could not start.",
                "stdout": "",
                "action_results": [],
            }
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None

        stdout_queue: queue.Queue[str | None] = queue.Queue()

        def _stdout_reader() -> None:
            for raw_line in process.stdout:
                stdout_queue.put(raw_line)
            stdout_queue.put(None)

        threading.Thread(target=_stdout_reader, daemon=True).start()

        _send_json_line(
            process.stdin,
            {
                "code": code,
                "timeout_seconds": timeout_seconds,
                "sandbox_cwd": sandbox_dir,
                "state": initial_state,
                "color_chars": ARC_COLOR_CHARS,
                "notes": dict(notes) if isinstance(notes, dict) else {},
                "last_batch_steps": max(0, int(last_batch_steps or 0)),
            },
        )

        deadline = time.monotonic() + max(1, int(timeout_seconds))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": f"Tool timed out after {timeout_seconds}s",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                line = stdout_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                stderr = process.stderr.read()
                _wait_for_process_exit(process)
                return {
                    "error": _sanitize_host_error_text(stderr),
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                stderr = process.stderr.read()
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": "Sandbox process returned an invalid response.",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            msg_type = str(message.get("type", "")).strip()
            if msg_type == "action":
                try:
                    action_result_payload = action_handler(list(message.get("actions") or []))
                except Exception:  # noqa: BLE001
                    _send_json_line(
                        process.stdin,
                        {
                            "type": "action_error",
                            "error": "action failed in sandbox host.",
                        },
                    )
                    continue
                raw_action_result = action_result_payload.get("action_result") or {}
                if isinstance(raw_action_result, dict):
                    host_action_results.append(dict(raw_action_result))
                _send_json_line(
                    process.stdin,
                    {
                        "type": "action_result",
                        "action_result": raw_action_result,
                        "state": action_result_payload.get("state") or {},
                    },
                )
                continue

            if msg_type in {"final", "error"}:
                _wait_for_process_exit(process)
                returned_notes = message.get("notes")
                try:
                    returned_batch_steps = max(0, int(message.get("last_batch_steps") or 0))
                except (TypeError, ValueError):
                    returned_batch_steps = 0
                return {
                    "stdout": str(message.get("stdout", "") or ""),
                    "result": message.get("result"),
                    "error": str(message.get("error", "") or ""),
                    "action_results": list(message.get("action_results") or host_action_results),
                    "notes": returned_notes if isinstance(returned_notes, dict) else None,
                    "last_batch_steps": returned_batch_steps,
                }

            _wait_for_process_exit(process)
            return {
                "error": "Sandbox process returned an unknown message type.",
                "stdout": "",
                "action_results": list(host_action_results),
            }
