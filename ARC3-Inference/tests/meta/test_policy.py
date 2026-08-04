"""Boundary-condition unit tests for MetaConfig + RestartController."""

import pytest

from inference.meta.policy import MetaConfig, RestartController, posterior_tractable


def make_controller(**overrides) -> RestartController:
    cfg = MetaConfig(enabled=True, **overrides)
    return RestartController(cfg)


BASE_KW = dict(level=1, fails=0, noop_rate_20=0.0, novelty_rate_30=1.0,
               budget_used_frac=0.3)


def test_disabled_never_restarts():
    ctl = RestartController(MetaConfig(enabled=False))
    assert ctl.should_restart(t=60, **BASE_KW) is False


def test_restart_exactly_at_deadline():
    ctl = make_controller()
    assert ctl.should_restart(t=59, **BASE_KW) is False
    assert ctl.should_restart(t=60, **BASE_KW) is True


def test_any_level_above_one_never_restarts():
    ctl = make_controller()
    kw = {**BASE_KW, "level": 2}
    assert ctl.should_restart(t=60, **kw) is False
    kw["level"] = 5
    assert ctl.should_restart(t=200, **kw) is False


def test_reserve_zone_never_restarts():
    ctl = make_controller()
    kw = {**BASE_KW, "budget_used_frac": 0.76}
    assert ctl.should_restart(t=60, **kw) is False
    # exactly at the reserve boundary is still allowed
    kw["budget_used_frac"] = 0.75
    assert ctl.should_restart(t=60, **kw) is True


def test_restart_cap_never_exceeded():
    ctl = make_controller()
    assert ctl.should_restart(t=60, **BASE_KW) is True
    ctl.record_restart()
    ctl.record_restart()
    assert ctl.counters["restarts_used"] == 2
    assert ctl.should_restart(t=60, **BASE_KW) is False


def test_posterior_below_phi_never_restarts():
    ctl = make_controller()
    # with default params P(tractable | 1 fail) = 0.10 < phi = 0.25
    kw = {**BASE_KW, "fails": 1}
    assert ctl.should_restart(t=60, **kw) is False
    assert ctl.should_restart(t=60, **BASE_KW) is True  # 0 fails: 0.36 >= phi


def test_min_actions_floor():
    ctl = make_controller(first_clear_deadline=20, min_actions_before_restart=30)
    assert ctl.should_restart(t=20, **BASE_KW) is False
    assert ctl.should_restart(t=30, **BASE_KW) is True


def test_posterior_arithmetic():
    p0 = posterior_tractable(0, 0.36, 0.83, 0.15)
    p1 = posterior_tractable(1, 0.36, 0.83, 0.15)
    p2 = posterior_tractable(2, 0.36, 0.83, 0.15)
    assert p0 == pytest.approx(0.36)
    assert p1 == pytest.approx(0.1011, abs=1e-3)
    assert p2 == pytest.approx(0.0220, abs=1e-3)
    assert p0 > p1 > p2  # collapses after 1-2 failures


def test_from_dict_defaults_and_overrides():
    cfg = MetaConfig.from_dict(None)
    assert cfg.enabled is False and cfg.first_clear_deadline == 60
    cfg = MetaConfig.from_dict({"enabled": True, "phi": 0.15,
                                "mixture": {"pi": 0.30}, "unknown_key": 1})
    assert cfg.enabled is True and cfg.phi == 0.15
    assert cfg.mixture == {"pi": 0.30, "p_E": 0.83, "p_H": 0.15}
    assert cfg.post_clear_restart == "forbidden"
