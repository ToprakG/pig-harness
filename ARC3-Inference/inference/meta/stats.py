"""Shared survival-statistics helpers for the meta-calibration layer.

Pure Python (no lifelines). Operates on run-level summaries:
each run is characterized by its first-clear action (or None) and its
total action count, which doubles as the right-censoring time.
"""

from __future__ import annotations

from itertools import groupby
from typing import Iterable, Sequence


def to_bool(v) -> bool:
    """Normalize boolean-ish JSONL fields.

    Some event files may encode booleans as the strings "True"/"False";
    statistics computed without this normalization are silently wrong.
    """
    return v is True or v == "True"


def km_curve(first_clears: Sequence[int | None], lengths: Sequence[int]) -> list[tuple[int, float]]:
    """Kaplan-Meier product-limit estimate of S(t) for first-clear time.

    Runs that never clear are right-censored at their total length.
    Returns a step function as [(t, S(t))] sorted by t, where S(t) is the
    survival value just after the events at time t.
    """
    pts = []
    for fc, ln in zip(first_clears, lengths):
        if fc is not None:
            pts.append((fc, 1))  # event
        else:
            pts.append((ln, 0))  # censored
    pts.sort()
    curve: list[tuple[int, float]] = []
    s = 1.0
    at_risk = len(pts)
    for t, grp in groupby(pts, key=lambda x: x[0]):
        grp = list(grp)
        d = sum(1 for _, kind in grp if kind == 1)
        if d > 0 and at_risk > 0:
            s *= 1.0 - d / at_risk
        curve.append((t, s))
        at_risk -= len(grp)
    return curve


def km_survival_at(curve: Sequence[tuple[int, float]], t: float) -> float:
    """Evaluate a km_curve step function at time t."""
    s = 1.0
    for tt, ss in curve:
        if tt <= t:
            s = ss
        else:
            break
    return s


def windowed_hazard(
    first_clears: Sequence[int | None],
    lengths: Sequence[int],
    bin_width: int = 20,
    max_t: int | None = None,
) -> list[dict]:
    """Windowed conditional hazard: P(first clear in [a, a+w) | at risk at a).

    A run is at risk at a if it has neither cleared nor been censored
    before a (first_clear >= a, or never cleared with length >= a).
    """
    if max_t is None:
        max_t = max(lengths)
    out = []
    for a in range(0, max_t, bin_width):
        b = a + bin_width
        at_risk = 0
        events = 0
        for fc, ln in zip(first_clears, lengths):
            if fc is not None:
                if fc >= a:
                    at_risk += 1
                    if fc < b:
                        events += 1
            elif ln >= a:
                at_risk += 1
        if at_risk == 0:
            continue
        out.append({"t_lo": a, "t_hi": b, "at_risk": at_risk, "events": events,
                    "hazard": events / at_risk})
    return out


def median(xs: Iterable[float]) -> float:
    """Plain median (mean of middle two for even n)."""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        raise ValueError("median of empty sequence")
    mid = n // 2
    return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2.0
