"""External history (compaction mode ``external_history``).

Instead of fighting the context window with summarisation, append the full
trajectory to a per-game JSONL file and give the agent search tools over it.
Recent turns stay in context; older detail is retrievable on demand.

Records are append-only, one JSON object per line::

    {"action_num": 12, "level": 1, "action": "UP", "board_changed": true,
     "board_ascii": "...", "note": "..."}

Everything here is generic: the helpers are plain text/JSONL search over the
agent's own past, with no conditioning on board content or game identity.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class ExternalHistory:
    """Append-only JSONL trajectory log with search/tail/at helpers."""

    def __init__(self, path: Path, max_matches: int = 20,
                 max_chars_per_record: int = 2000):
        self.path = Path(path)
        self.max_matches = max_matches
        self.max_chars_per_record = max_chars_per_record
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ---- writing ---------------------------------------------------------

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True, default=str))
            f.write("\n")

    # ---- reading ---------------------------------------------------------

    def _iter_records(self):
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def _clip(self, record: dict[str, Any]) -> dict[str, Any]:
        rendered = json.dumps(record, ensure_ascii=True, default=str)
        if len(rendered) <= self.max_chars_per_record:
            return record
        clipped = dict(record)
        for key in ("board_ascii", "note", "stdout"):
            value = clipped.get(key)
            if isinstance(value, str) and len(value) > 200:
                clipped[key] = value[:200] + f"... [{len(value) - 200} chars omitted]"
        return clipped

    def search(self, pattern: str, last_n: int | None = None) -> list[dict[str, Any]]:
        """Records whose serialized form matches ``pattern`` (regex, plain
        text falls back to a substring search). ``last_n`` limits the scan to
        the most recent N records."""
        records = list(self._iter_records())
        if last_n is not None:
            records = records[-int(last_n):]
        try:
            rx = re.compile(pattern, re.IGNORECASE)
            match = lambda text: bool(rx.search(text))  # noqa: E731
        except re.error:
            needle = pattern.lower()
            match = lambda text: needle in text.lower()  # noqa: E731
        hits = [r for r in records
                if match(json.dumps(r, ensure_ascii=True, default=str))]
        return [self._clip(r) for r in hits[-self.max_matches:]]

    def tail(self, n: int = 10) -> list[dict[str, Any]]:
        records = list(self._iter_records())[-max(1, int(n)):]
        return [self._clip(r) for r in records]

    def at(self, action_num: int) -> dict[str, Any] | None:
        for record in self._iter_records():
            if record.get("action_num") == action_num:
                return self._clip(record)
        return None

    def stats(self) -> dict[str, Any]:
        records = list(self._iter_records())
        return {
            "records": len(records),
            "first_action": records[0].get("action_num") if records else None,
            "last_action": records[-1].get("action_num") if records else None,
            "path": str(self.path),
        }
