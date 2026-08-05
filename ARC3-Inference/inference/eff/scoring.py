"""The official ARC-AGI-3 score, plus efficiency counterfactuals.

Mirrors ``taaf.game.GameRun._compute_final_score`` (itself a mirror of
``arc_agi.scorecard.EnvironmentScoreCalculator`` v0.9.8) exactly:

    per cleared level: min(115, (baseline / actions)^2 * 100)
    weight            = level_index + 1
    score             = sum(level_score * weight) / sum(weight)
    capped at           max_weights / total_weights * 100

Score elasticity with respect to action count is **-2**: halving the actions
spent on a cleared level quadruples that level's contribution. Actions are
the only thing that costs score — LLM calls, Python inspection and re-reading
history are free.

This module is pure arithmetic over run summaries; it never sees board
content or game identity.
"""

from __future__ import annotations

from typing import Iterable, Sequence

LEVEL_SCORE_CAP = 115.0


def to_bool(v) -> bool:
    """Boolean-ish JSONL fields are ``bool`` in some files and ``"True"`` in
    others; normalizing everywhere has bitten us before."""
    return v is True or v == "True"


def level_score(baseline: float, actions: int) -> float:
    """Score contribution of one cleared level before weighting."""
    if actions <= 0 or baseline <= 0:
        return 0.0
    return min(LEVEL_SCORE_CAP, (baseline / actions) ** 2 * 100.0)


def final_score(
    *,
    actions_per_level: Sequence[int],
    base_actions_per_level: Sequence[float] | None,
    levels_completed: int,
    number_of_levels: int,
) -> float:
    """The official score. Returns 0.0 when baselines are hidden."""
    if base_actions_per_level is None or number_of_levels == 0:
        return 0.0
    total_score = 0.0
    total_weights = 0
    max_weights = 0
    for level_idx in range(number_of_levels):
        weight = level_idx + 1
        total_weights += weight
        completed = level_idx < levels_completed
        actions = (actions_per_level[level_idx]
                   if level_idx < len(actions_per_level) else 0)
        baseline = base_actions_per_level[level_idx]
        score_i = level_score(baseline, actions) if completed else 0.0
        if score_i > 0:
            max_weights += weight
        total_score += score_i * weight
    if total_weights == 0:
        return 0.0
    return min(total_score / total_weights, max_weights / total_weights * 100.0)


def score_with_alpha(
    *,
    actions_per_level: Sequence[int],
    base_actions_per_level: Sequence[float] | None,
    levels_completed: int,
    number_of_levels: int,
    alpha: float,
) -> float:
    """Counterfactual score if every CLEARED level had used ``alpha x`` its
    actual actions (alpha = 1.0 reproduces the real score). Uncleared levels
    are untouched: their counters never entered the score anyway."""
    if base_actions_per_level is None:
        return 0.0
    scaled = []
    for idx, actions in enumerate(actions_per_level):
        if idx < levels_completed:
            scaled.append(max(1, int(round(actions * alpha))))
        else:
            scaled.append(actions)
    return final_score(actions_per_level=scaled,
                       base_actions_per_level=base_actions_per_level,
                       levels_completed=levels_completed,
                       number_of_levels=number_of_levels)


def score_if_1x(
    *,
    base_actions_per_level: Sequence[float] | None,
    levels_completed: int,
    number_of_levels: int,
) -> float:
    """Score this run would have received had every cleared level used
    exactly the human baseline number of actions."""
    if base_actions_per_level is None:
        return 0.0
    ideal = [int(round(b)) for b in base_actions_per_level]
    return final_score(actions_per_level=ideal,
                       base_actions_per_level=base_actions_per_level,
                       levels_completed=levels_completed,
                       number_of_levels=number_of_levels)


def human_multipliers(actions_per_level: Sequence[int],
                      base_actions_per_level: Sequence[float] | None,
                      ) -> list[float] | None:
    """actions / baseline per level (how many times the human budget we
    spent). ``None`` when baselines are hidden."""
    if base_actions_per_level is None:
        return None
    out = []
    for idx, baseline in enumerate(base_actions_per_level):
        actions = (actions_per_level[idx]
                   if idx < len(actions_per_level) else 0)
        out.append((actions / baseline) if baseline else float("nan"))
    return out


def cleared_human_mult(actions_per_level: Sequence[int],
                       base_actions_per_level: Sequence[float] | None,
                       levels_completed: int) -> float | None:
    """Weight-averaged actions/baseline over CLEARED levels only — the number
    the agent can actually improve."""
    mults = human_multipliers(actions_per_level, base_actions_per_level)
    if mults is None or levels_completed <= 0:
        return None
    num = den = 0.0
    for idx in range(min(levels_completed, len(mults))):
        weight = idx + 1
        num += mults[idx] * weight
        den += weight
    return (num / den) if den else None


def fmt_list(values: Iterable[float] | None, digits: int = 2) -> str:
    if values is None:
        return "hidden"
    return ",".join(f"{v:.{digits}f}".rstrip("0").rstrip(".") if isinstance(v, float)
                    else str(v) for v in values)
