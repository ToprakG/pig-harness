"""PRO-LONG-style programmatic memory: append-all write, code-based read."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from inference.agent.runtime_state import Frame, RUNTIME_STATE_FILENAME
from inference.utils.grid_utils import format_grid_ascii

PROGRAMMATIC_LOG_FILENAME = "logs.txt"
_ACTION_HEADER_RE = re.compile(
    r"^Action\s+(\d+)\s*\|\s*Level\s+(\d+)",
    flags=re.IGNORECASE,
)


def resolve_programmatic_log_path(state_path: Path) -> Path:
    """Place the append-only game log next to other per-game artifacts."""
    parent = state_path.parent
    if parent.name == "artifacts" and parent.parent != parent:
        runtime_state_stem = Path(RUNTIME_STATE_FILENAME).stem
        suffix = f"_{runtime_state_stem}"
        state_stem = state_path.stem
        game_stem = state_stem[: -len(suffix)] if state_stem.endswith(suffix) else state_stem
        return parent / f"{game_stem}_{PROGRAMMATIC_LOG_FILENAME}"
    return parent / PROGRAMMATIC_LOG_FILENAME


def _normalize_text(value: Any, *, max_chars: int | None = None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if max_chars is not None and max_chars > 0 and len(text) > max_chars:
        omitted = len(text) - max_chars
        return f"{text[:max_chars].rstrip()}... [{omitted} chars omitted]"
    return text


def _board_text(frame: Frame | None = None, *, ascii_grid: str | None = None) -> str:
    if ascii_grid is not None and str(ascii_grid).strip():
        return str(ascii_grid).rstrip()
    if frame is None:
        return "(empty grid)"
    rendered = frame.ascii if frame.ascii else format_grid_ascii(frame.grid)
    return rendered.rstrip() if rendered else "(empty grid)"


@dataclass
class ProgrammaticMemory:
    """Lossless append-only interaction log searched programmatically by the agent."""

    path: Path
    attempt: int = 1
    _initialized: bool = False
    _last_plan: str = ""
    _action_count_logged: int = -1
    _pending_analysis: list[str] = field(default_factory=list)

    @classmethod
    def for_state_path(cls, state_path: Path) -> ProgrammaticMemory:
        return cls(path=resolve_programmatic_log_path(state_path))

    def ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def clear(self) -> None:
        self.ensure_parent()
        if self.path.exists():
            self.path.unlink()
        self._initialized = False
        self.attempt = 1
        self._last_plan = ""
        self._action_count_logged = -1
        self._pending_analysis.clear()

    def _append(self, text: str) -> None:
        if not text:
            return
        self.ensure_parent()
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")

    def note_plan(self, plan: str) -> None:
        cleaned = _normalize_text(plan, max_chars=1200)
        if cleaned:
            self._last_plan = cleaned

    def note_analysis(
        self,
        *,
        content: str = "",
        reasoning: str = "",
        world_model: dict[str, str] | None = None,
    ) -> None:
        blocks: list[str] = []
        if world_model:
            lines = [
                f"{label}: {value}"
                for label, value in (
                    ("World model", world_model.get("world_model", "")),
                    ("Goal model", world_model.get("goal_model", "")),
                    ("Action model", world_model.get("action_model", "")),
                    ("Recent findings", world_model.get("recent_findings", "")),
                    ("Open questions", world_model.get("open_questions", "")),
                    ("Plan", world_model.get("current_plan", "")),
                    ("Cross-level notes", world_model.get("cross_level_notes", "")),
                )
                if _normalize_text(value)
            ]
            if lines:
                blocks.append("Working model:\n" + "\n".join(lines))
                plan = _normalize_text(world_model.get("current_plan", ""), max_chars=1200)
                if plan:
                    self._last_plan = plan
        content_text = _normalize_text(content, max_chars=2400)
        if content_text:
            blocks.append(content_text)
        reasoning_text = _normalize_text(reasoning, max_chars=1800)
        if reasoning_text:
            blocks.append(f"Reasoning digest:\n{reasoning_text}")
        if blocks:
            self._pending_analysis.append("\n\n".join(blocks))

    def mark_game_over_reset(self) -> None:
        self.attempt += 1

    def write_initial_state(
        self,
        *,
        frame: Frame | None,
        score: int = 0,
        level: int | None = None,
    ) -> None:
        if self._initialized and self.path.exists() and self.path.stat().st_size > 0:
            return
        level_num = int(level if level is not None else (frame.level if frame is not None else 1))
        board = _board_text(frame)
        self.clear()
        self._append(
            f"{'=' * 80}\n"
            f"Action 0 | Level {level_num} | Attempt {self.attempt} | INITIAL STATE | Score: {score}\n\n"
            f"[INITIAL BOARD STATE]\n{board}\n\n"
        )
        self._initialized = True
        self._action_count_logged = 0

    def append_action(
        self,
        *,
        action_num: int,
        action_display: str,
        frame: Frame | None,
        score: int,
        level: int,
        board_changed: bool | None = None,
        animated: bool | None = None,
        state: str | None = None,
        plan: str | None = None,
        plan_step: tuple[int, int] | None = None,
    ) -> None:
        if not self._initialized:
            self.write_initial_state(frame=frame, score=score, level=level)

        if int(action_num) <= self._action_count_logged:
            return

        plan_text = _normalize_text(plan or self._last_plan, max_chars=1200)
        plan_info = ""
        if plan_step is not None:
            current, total = plan_step
            if total > 0:
                plan_info = f" | Plan Step {current}/{total}"

        parts = [
            f"\n{'=' * 80}\n",
            f"Action {int(action_num)} | Level {int(level)} | Attempt {self.attempt}{plan_info} | Score: {int(score)}\n\n",
        ]
        if self._pending_analysis:
            parts.append("[ANALYSIS]\n")
            parts.append("\n\n---\n\n".join(self._pending_analysis))
            parts.append("\n\n")
            self._pending_analysis.clear()
        if plan_text:
            parts.append(f"[PLAN]\n{plan_text}\n\n")
        parts.append(f"Tool Call: {action_display}\n")
        meta: list[str] = []
        if board_changed is not None:
            meta.append(f"board_changed={bool(board_changed)}")
            if animated and not board_changed:
                meta.append(
                    "animated=True (transient effect in intermediate frames, "
                    "settled frame reverted -- do not treat board_changed=False "
                    "here as proof the action was inert)"
                )
        if state:
            meta.append(f"state={state}")
        if meta:
            parts.append("Outcome: " + ", ".join(meta) + "\n")
        parts.append(f"\n[POST-ACTION BOARD STATE]\n{_board_text(frame)}\n\n")
        self._append("".join(parts))
        self._action_count_logged = int(action_num)

    def append_reasoning_compaction(
        self,
        *,
        action_num: int,
        level: int,
        score: int,
        dropped_messages: Iterable[dict[str, Any]],
        digest: str = "",
    ) -> None:
        """Preserve reasoning that would otherwise be lost to context eviction."""
        excerpts: list[str] = []
        for message in dropped_messages:
            role = str(message.get("role", "")).strip().lower()
            if role == "assistant":
                content = _normalize_text(message.get("content", ""), max_chars=900)
                reasoning = _normalize_text(
                    message.get("reasoning") or message.get("reasoning_content") or "",
                    max_chars=900,
                )
                if content:
                    excerpts.append(f"assistant: {content}")
                if reasoning:
                    excerpts.append(f"thinking: {reasoning}")
            elif role == "tool":
                content = _normalize_text(message.get("content", ""), max_chars=500)
                if content:
                    excerpts.append(f"tool: {content}")
            elif role == "user":
                content = _normalize_text(message.get("content", ""), max_chars=400)
                if content:
                    excerpts.append(f"user: {content}")

        digest_text = _normalize_text(digest, max_chars=2000)
        if not excerpts and not digest_text:
            return

        body_parts: list[str] = []
        if digest_text:
            body_parts.append(digest_text)
        if excerpts:
            # Keep only the freshest excerpts from the dropped span.
            body_parts.append("Dropped-context excerpts:\n" + "\n".join(excerpts[-12:]))

        self._append(
            f"\n{'=' * 80}\n"
            f"Action {int(action_num)} | Level {int(level)} | Attempt {self.attempt} | "
            f"REASONING COMPACTION | Score: {int(score)}\n\n"
            f"[REASONING COMPACTION]\n"
            + "\n\n".join(body_parts)
            + "\n\n"
        )

    def stats(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"path": str(self.path), "exists": False, "bytes": 0, "lines": 0}
        text = self.path.read_text(encoding="utf-8")
        return {
            "path": str(self.path),
            "exists": True,
            "bytes": len(text.encode("utf-8")),
            "lines": text.count("\n") + (0 if text.endswith("\n") or not text else 1),
            "actions_logged": self._action_count_logged,
            "attempt": self.attempt,
        }


def grep_log_text(
    text: str,
    pattern: str,
    *,
    ignore_case: bool = False,
    context_before: int = 0,
    context_after: int = 0,
    max_matches: int = 50,
) -> str:
    flags = re.MULTILINE
    if ignore_case:
        flags |= re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        return f"Invalid regex: {exc}"

    lines = text.splitlines()
    hits: list[str] = []
    match_count = 0
    for index, line in enumerate(lines):
        if not regex.search(line):
            continue
        match_count += 1
        start = max(0, index - max(0, context_before))
        end = min(len(lines), index + 1 + max(0, context_after))
        if context_before or context_after:
            hits.append(f"-- match {match_count} @ line {index + 1} --")
        for line_no in range(start, end):
            hits.append(f"{line_no + 1}:{lines[line_no]}")
        if match_count >= max(1, max_matches):
            hits.append(f"... truncated after {max_matches} matches")
            break
    if not hits:
        return "(no matches)"
    return "\n".join(hits)


def tail_log_text(text: str, n: int = 80) -> str:
    lines = text.splitlines()
    if n <= 0:
        return ""
    clipped = lines[-n:]
    start = max(1, len(lines) - len(clipped) + 1)
    return "\n".join(f"{start + offset}:{line}" for offset, line in enumerate(clipped))


def list_action_headers(text: str) -> list[str]:
    headers: list[str] = []
    for line in text.splitlines():
        if _ACTION_HEADER_RE.match(line.strip()):
            headers.append(line.strip())
    return headers
