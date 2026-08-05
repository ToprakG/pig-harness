"""LLM-based history compaction (C1): summarize dropped history blocks into a
rolling digest instead of silently truncating them.

Pure logic — no HTTP client of its own. The agent supplies ``llm_call`` (a
small chat call reusing its existing endpoint, headers, and client) and its
token estimator. Every path is guarded: if the compaction call fails, times
out, or returns garbage, the caller gets the digest unchanged and the drop
proceeds exactly as upstream (deterministic fallback) — compaction must
never stall the agent.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

DIGEST_LABEL = "PERSISTED HISTORY DIGEST (auto-compacted)"

DIGEST_SECTIONS = (
    "Confirmed rules",
    "Disproven hypotheses",
    "Level layout notes",
    "Current objective",
    "Recent failures to avoid",
)

_COMPACTION_INSTRUCTION = (
    "You maintain a compressed memory digest for a puzzle-game-playing agent. "
    "Old conversation turns are about to be deleted to free context. Merge any "
    "NEW durable knowledge from them into the existing digest. Keep exactly "
    "these sections, each as a short bullet list:\n"
    + "\n".join(f"- {section}" for section in DIGEST_SECTIONS)
    + "\nRules: keep it terse and factual; drop stale or superseded items; "
    "never invent facts; keep the whole digest under {max_tokens} tokens. "
    "Output ONLY the digest text."
)


def _default_estimate_tokens(value: Any) -> int:
    """Mirror of tool_agent._estimate_tokens (injected normally; this keeps
    the module import-clean without a circular dependency)."""
    import json

    try:
        rendered = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except TypeError:
        rendered = str(value)
    return max(1, (len(rendered) + 2) // 3)


@dataclass
class CompactionConfig:
    """Mirrors the ``compaction`` block in configs/inference.json."""

    enabled: bool = False
    digest_max_tokens: int = 700
    compaction_call_max_tokens: int = 900
    compaction_timeout_s: float = 20.0
    min_dropped_tokens_to_compact: int = 800
    wipe_knowledge_on_level_transition: bool = True
    wipe_knowledge_on_reset: bool = True

    @classmethod
    def from_dict(cls, data: dict | None) -> "CompactionConfig":
        data = dict(data or {})
        known = {k: data[k] for k in data if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class CompactionStats:
    """Per-run instrumentation (C4)."""

    compaction_events: int = 0
    compaction_fallbacks: int = 0
    compaction_wallclock_s: float = 0.0

    def summary_line(self, digest_tokens: int) -> str:
        return (
            f"compaction_summary: events={self.compaction_events} "
            f"fallbacks={self.compaction_fallbacks} "
            f"wallclock_s={self.compaction_wallclock_s:.2f} "
            f"digest_tokens={digest_tokens}"
        )


class HistoryDigest:
    """Rolling plain-text digest, clamped to a token budget."""

    def __init__(self, max_tokens: int = 700,
                 estimate_tokens: Callable[[Any], int] | None = None):
        self.max_tokens = max(50, int(max_tokens))
        self._estimate = estimate_tokens or _default_estimate_tokens
        self.text = ""

    def is_empty(self) -> bool:
        return not self.text.strip()

    def tokens(self) -> int:
        return 0 if self.is_empty() else self._estimate(self.text)

    def replace(self, new_text: str) -> None:
        """Set the digest, hard-clamping to the token budget."""
        text = str(new_text or "").strip()
        while text and self._estimate(text) > self.max_tokens:
            # trim whole lines from the tail first, then hard-cut
            lines = text.splitlines()
            text = "\n".join(lines[:-1]).strip() if len(lines) > 1 else text[: len(text) * 3 // 4].strip()
        self.text = text

    def clear(self) -> None:
        self.text = ""

    def render_lines(self) -> list[str]:
        """Delimited prompt block (empty when there is nothing to inject)."""
        if self.is_empty():
            return []
        return [
            f"===== {DIGEST_LABEL} =====",
            self.text,
            "===== END DIGEST =====",
        ]


def render_dropped_messages(messages: list[dict[str, Any]],
                            max_chars_per_message: int = 1500) -> str:
    """Plain-text rendering of the about-to-be-dropped block."""
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "?"))
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(
                str(p.get("text", "")) for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        content = str(content or "").strip()
        reasoning = str(message.get("reasoning") or "").strip()
        body = "\n".join(x for x in (reasoning, content) if x)
        if len(body) > max_chars_per_message:
            omitted = len(body) - max_chars_per_message
            body = f"{body[:max_chars_per_message]}\n... [{omitted} chars omitted]"
        if body:
            parts.append(f"[{role}]\n{body}")
    return "\n\n".join(parts)


def build_compaction_prompt(dropped_text: str, current_digest: str,
                            digest_max_tokens: int) -> str:
    instruction = _COMPACTION_INSTRUCTION.format(max_tokens=digest_max_tokens)
    digest_block = current_digest.strip() or "(empty)"
    return (
        f"{instruction}\n\n"
        f"CURRENT DIGEST:\n{digest_block}\n\n"
        f"TURNS BEING DELETED:\n{dropped_text}\n\n"
        f"UPDATED DIGEST:"
    )


def compact(
    dropped_messages: list[dict[str, Any]],
    digest: HistoryDigest,
    llm_call: Callable[[str, int, float], str],
    *,
    config: CompactionConfig,
    stats: CompactionStats,
    estimate_tokens: Callable[[Any], int] | None = None,
    context: str = "",
) -> HistoryDigest:
    """Merge an about-to-be-dropped message block into the digest.

    ``llm_call(prompt, max_tokens, timeout_s) -> str`` is supplied by the
    agent and reuses its existing HTTP client and headers. On any failure the
    digest is returned unchanged (upstream silent-drop behavior) and the
    fallback is logged/counted. ``context`` labels the [COMPACT] stdout
    lines (game id + pass, when known).
    """
    from inference.framework import observability

    estimate = estimate_tokens or _default_estimate_tokens
    dropped_text = render_dropped_messages(dropped_messages)
    if not dropped_text.strip():
        return digest
    dropped_tokens = estimate(dropped_text)
    if dropped_tokens < config.min_dropped_tokens_to_compact:
        return digest  # too small to be worth an LLM call

    prompt = build_compaction_prompt(dropped_text, digest.text,
                                     config.digest_max_tokens)
    started = time.monotonic()
    try:
        new_text = llm_call(prompt, config.compaction_call_max_tokens,
                            config.compaction_timeout_s)
    except Exception as exc:
        latency_ms = int((time.monotonic() - started) * 1000)
        stats.compaction_fallbacks += 1
        stats.compaction_wallclock_s += (time.monotonic() - started)
        logger.warning("compaction_fallback: %s: %s", type(exc).__name__, exc)
        observability.compact_fallback_line(
            context, f"{type(exc).__name__}", latency_ms)
        return digest
    latency_ms = int((time.monotonic() - started) * 1000)
    stats.compaction_wallclock_s += (time.monotonic() - started)
    new_text = str(new_text or "").strip()
    if not new_text:
        stats.compaction_fallbacks += 1
        logger.warning("compaction_fallback: empty compaction response")
        observability.compact_fallback_line(context, "empty_response", latency_ms)
        return digest
    digest.replace(new_text)
    stats.compaction_events += 1
    observability.compact_ok_line(context, dropped_tokens, digest.tokens(),
                                  latency_ms)
    return digest
