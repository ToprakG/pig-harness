"""Deploy-side restart policy: MetaConfig + RestartController.

This module is imported by the solver at runtime — keep it stdlib-only.

The v1 policy is M3-only (mixture posterior over game type). The covariate
signals ``noop_rate_20`` / ``novelty_rate_30`` are accepted as inputs for
forward compatibility but unused: in the LOGO tournament the hazard-gated
policy beat M3-alone by +2.8pp median, below the 3pp threshold required to
ship the M2 gating layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Feature names the policy may consume. guard.py checks this against its
# whitelist and fails the build on any addition outside it.
POLICY_INPUT_FEATURES = (
    "action_num",
    "level",
    "noop_rate_w",
    "novelty_rate_w",
    "budget_frac",
    "attempt_index",
    "posterior_tractable",
)


def default_mixture_params() -> dict:
    """Deploy-time mixture parameters, estimated from all 25 example-run
    games at deadline T=60 (see mixture.py)."""
    return {"pi": 0.36, "p_E": 0.83, "p_H": 0.15}


@dataclass
class MetaConfig:
    """Configuration for the probabilistic-restart meta layer."""

    enabled: bool = False
    first_clear_deadline: int = 60
    phi: float = 0.25
    min_actions_before_restart: int = 30
    max_restarts_per_pass: int = 2
    budget_reserve_frac: float = 0.25
    post_clear_restart: str = "forbidden"
    mixture: dict = field(default_factory=default_mixture_params)

    @classmethod
    def from_dict(cls, d: dict | None) -> "MetaConfig":
        d = dict(d or {})
        mixture = {**default_mixture_params(), **d.pop("mixture", {})}
        known = {k: d[k] for k in d if k in cls.__dataclass_fields__}
        return cls(mixture=mixture, **known)


def posterior_tractable(fails: int, pi: float, p_e: float, p_h: float) -> float:
    """P(game is tractable | ``fails`` attempts with no clear by T).

    A failed attempt multiplies the tractable likelihood by (1-p_E) vs
    (1-p_H) for resistant — with the default parameters the posterior
    collapses below every sensible phi after 1-2 failures.
    """
    a = pi * (1.0 - p_e) ** fails
    b = (1.0 - pi) * (1.0 - p_h) ** fails
    return a / (a + b) if a + b > 0 else 0.0


class RestartController:
    """Pure restart decision + counters.

    ``should_restart`` has no side effects; the session calls
    ``record_restart()`` after actually issuing a reset.
    """

    def __init__(self, config: MetaConfig):
        self.config = config
        self.restarts_used = 0
        self.failed_attempts = 0

    def should_restart(self, t: int, level: int, fails: int,
                       noop_rate_20: float, novelty_rate_30: float,
                       budget_used_frac: float) -> bool:
        """Decide whether to restart the current attempt.

        ``t`` is actions taken in the current attempt, ``level`` the current
        game level (level > 1 means a level was cleared — never restart),
        ``fails`` the number of prior attempts with no clear by the deadline,
        ``budget_used_frac`` the fraction of the pass budget consumed.
        The covariate rates are currently unused (M2 gating not shipped).
        """
        cfg = self.config
        if not cfg.enabled:
            return False
        if level > 1:
            return False  # post_clear_restart is forbidden
        if t < cfg.first_clear_deadline:
            return False
        if t < cfg.min_actions_before_restart:
            return False
        if self.restarts_used >= cfg.max_restarts_per_pass:
            return False
        if budget_used_frac > 1.0 - cfg.budget_reserve_frac:
            return False  # no restarts in the final reserve of the budget
        m = cfg.mixture
        if posterior_tractable(fails, m["pi"], m["p_E"], m["p_H"]) < cfg.phi:
            return False  # commit the remaining budget to one long attempt
        return True

    def record_restart(self) -> None:
        self.restarts_used += 1
        self.failed_attempts += 1

    @property
    def counters(self) -> dict:
        return {"restarts_used": self.restarts_used,
                "failed_attempts": self.failed_attempts}
