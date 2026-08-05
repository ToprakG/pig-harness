"""Run observability: startup banner, per-event stdout lines, end-of-run
accounting. Everything prints to stdout because Kaggle captures stdout and
the artifact bundle (~167 MB) is impractical to inspect.

Line grammar (grep-stable, one line per event):
    [META] restart game=<id> pass=<n> t=<n> level=<l> fails=<f> posterior=<p> reason=<r>
    [META] suppressed game=<id> pass=<n> t=<n> reason=<r>
    [COMPACT] ok game=<id> dropped_tokens=<n> digest_tokens=<n> latency_ms=<n>
    [COMPACT] fallback game=<id> reason=<r> latency_ms=<n>
    [BUDGET] game=<id> planned=<s> used=<s> util=<pct>
"""

from __future__ import annotations

import json
import time
from typing import Any


def _flush_print(line: str) -> None:
    print(line, flush=True)


def print_banner(
    *,
    meta_config: dict[str, Any] | None,
    compaction_config: dict[str, Any],
    model_id: str,
    max_runtime_s_per_game: float | None,
    max_actions_per_game: int | None,
    concurrency: int,
    budget_plan: dict[str, Any] | None = None,
) -> None:
    """Single self-describing block at run start. Git SHA/branch/dirty is
    already printed by the framework (git_info/git_status artifacts + deploy
    logs) — this block covers the resolved feature configs and budget."""
    meta_enabled = bool(meta_config and meta_config.get("enabled"))
    compaction_enabled = bool(compaction_config.get("enabled"))
    lines = [
        "==================== RUN CONFIG BANNER ====================",
        f"meta.enabled={meta_enabled} compaction.enabled={compaction_enabled}",
        f"meta.config={json.dumps(meta_config or {}, sort_keys=True)}",
        f"compaction.config={json.dumps(compaction_config, sort_keys=True)}",
        f"model_id={model_id}",
        f"max_runtime_s_per_game={max_runtime_s_per_game}",
        f"max_actions_per_game={max_actions_per_game}",
        f"concurrency={concurrency}",
    ]
    if budget_plan:
        lines.append(f"budget.plan={json.dumps(budget_plan, sort_keys=True)}")
    lines.append("===========================================================")
    _flush_print("\n".join(lines))


def meta_restart_line(game_id: str, pass_index: int, t: int, level: int,
                      fails: int, posterior: float, reason: str) -> None:
    _flush_print(
        f"[META] restart game={game_id} pass={pass_index} t={t} "
        f"level={level} fails={fails} posterior={posterior:.3f} reason={reason}"
    )


def meta_suppressed_line(game_id: str, pass_index: int, t: int,
                         reason: str) -> None:
    _flush_print(
        f"[META] suppressed game={game_id} pass={pass_index} t={t} reason={reason}"
    )


def compact_ok_line(context: str, dropped_tokens: int, digest_tokens: int,
                    latency_ms: int) -> None:
    _flush_print(
        f"[COMPACT] ok game={context} dropped_tokens={dropped_tokens} "
        f"digest_tokens={digest_tokens} latency_ms={latency_ms}"
    )


def compact_fallback_line(context: str, reason: str, latency_ms: int) -> None:
    _flush_print(
        f"[COMPACT] fallback game={context} reason={reason} latency_ms={latency_ms}"
    )


def budget_line(game_id: str, planned_s: float, used_s: float) -> None:
    util = (used_s / planned_s * 100.0) if planned_s > 0 else 0.0
    _flush_print(
        f"[BUDGET] game={game_id} planned={planned_s:.0f} used={used_s:.0f} "
        f"util={util:.0f}%"
    )


class RunAccounting:
    """Aggregates per-game summaries; prints the end-of-run block."""

    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self.games: list[dict[str, Any]] = []

    def record_game(self, **summary: Any) -> None:
        self.games.append(summary)

    def print_block(self, *, granted_seconds: float | None) -> None:
        used = time.monotonic() - self.started_at
        lines = ["==================== RUN ACCOUNTING ===================="]
        if granted_seconds:
            lines.append(
                f"wallclock used={used:.0f}s granted={granted_seconds:.0f}s "
                f"utilisation={used / granted_seconds * 100.0:.1f}%"
            )
        else:
            lines.append(f"wallclock used={used:.0f}s granted=unknown")
        total_actions = total_tokens = 0
        meta_fired = meta_suppressed = 0
        compact_ok = compact_fallback = 0
        for g in self.games:
            total_actions += g.get("actions", 0)
            total_tokens += g.get("tokens", 0)
            meta_fired += g.get("meta_restarts", 0)
            meta_suppressed += g.get("meta_suppressed", 0)
            compact_ok += g.get("compaction_ok", 0)
            compact_fallback += g.get("compaction_fallbacks", 0)
            lines.append(
                "game={game} pass={pass_index} actions={actions} "
                "tokens={tokens} levels={levels} score={score} "
                "meta_restarts={meta_restarts} "
                "compaction_ok={compaction_ok} "
                "compaction_fallbacks={compaction_fallbacks} "
                "elapsed_s={elapsed_s:.0f}".format(**{
                    "game": g.get("game", "?"),
                    "pass_index": g.get("pass_index", 0),
                    "actions": g.get("actions", 0),
                    "tokens": g.get("tokens", 0),
                    "levels": g.get("levels", 0),
                    "score": g.get("score", "n/a"),
                    "meta_restarts": g.get("meta_restarts", 0),
                    "meta_suppressed": g.get("meta_suppressed", 0),
                    "compaction_ok": g.get("compaction_ok", 0),
                    "compaction_fallbacks": g.get("compaction_fallbacks", 0),
                    "elapsed_s": g.get("elapsed_s", 0.0),
                })
            )
        lines.append(f"[META] fired={meta_fired} suppressed={meta_suppressed}")
        lines.append(f"[COMPACT] ok={compact_ok} fallback={compact_fallback}")
        if used > 0:
            lines.append(
                f"generated_tokens_per_sec={total_tokens / used:.2f} "
                f"mean_seconds_per_action="
                f"{(used / total_actions) if total_actions else 0.0:.2f}"
            )
        lines.append("========================================================")
        _flush_print("\n".join(lines))
