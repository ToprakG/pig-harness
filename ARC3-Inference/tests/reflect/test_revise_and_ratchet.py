"""Phases 2 and 3: revise call (5 stub cases) and the rollback ratchet."""

from __future__ import annotations

from collections import deque

import pytest

from inference.reflect.detector import StallConfig, stall_state
from inference.reflect.revise import (
    BLOCK_HEADER,
    MAX_STACK_DEPTH,
    RevisionRatchet,
    parse_block,
    revise,
    validate_evidence,
)

VALID = (
    "WHY STUCK: SPACE has not changed the board in the last 12 actions.\n"
    "ABANDONED: that SPACE toggles the target.\n"
    "NEW HYPOTHESIS: the target only reacts after MOUSE selects it.\n"
    "NEXT PROBES: MOUSE, then SPACE; if unchanged try UP.\n"
)
PACKET = {
    "stall_window": [{"action": "SPACE", "changed": False},
                     {"action": "SPACE", "changed": False}],
    "untried_actions": ["MOUSE", "UP"],
    "repeated_states": [{"hash": "abc123", "count": 4}],
    "last_progress": {"action": "LEFT", "actions_ago": 14},
}
VALID_ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE"]


def cfg(**kw):
    return StallConfig(enabled=True, **kw)


# ---- the five stub cases --------------------------------------------------

def test_valid_output():
    r = revise(PACKET, "", VALID_ACTIONS, lambda *a: VALID, config=cfg())
    assert r.ok and r.reason == "ok" and r.block.startswith(BLOCK_HEADER)
    assert r.dropped_lines == []


def test_timeout():
    def boom(*a):
        raise TimeoutError("simulated")
    r = revise(PACKET, "prev", VALID_ACTIONS, boom, config=cfg())
    assert not r.ok and r.reason == "TimeoutError"


@pytest.mark.parametrize("text", [
    "", "I think we should try something else.",
    "WHY STUCK: stuck.\nABANDONED: x.",                       # missing sections
    ("WHY STUCK: <one line>\nABANDONED: <hypothesis>\n"
     "NEW HYPOTHESIS: <what>\nNEXT PROBES: <probes>\n"),      # template echo
])
def test_malformed_output(text):
    r = revise(PACKET, "", VALID_ACTIONS, lambda *a: text, config=cfg())
    assert not r.ok and r.reason == "malformed_output"


def test_hallucinated_action_id_is_dropped():
    text = VALID.replace("NEXT PROBES: MOUSE", "NEXT PROBES: TELEPORT")
    r = revise(PACKET, "", VALID_ACTIONS, lambda *a: text, config=cfg())
    assert r.ok, "one bad section must not kill the whole revision"
    assert "TELEPORT" not in r.block and "dropped" in r.block
    assert r.dropped_lines


def test_oversize_output():
    text = VALID.replace("SPACE has not changed", "x " * 2000)
    r = revise(PACKET, "", VALID_ACTIONS, lambda *a: text, config=cfg(max_tokens=40))
    assert not r.ok and r.reason in {"oversize_output", "malformed_output"}


def test_trace_dump_is_rejected():
    text = VALID + " stall_window: [...] untried_actions: [...]"
    r = revise(PACKET, "", VALID_ACTIONS, lambda *a: text, config=cfg())
    assert not r.ok and r.reason == "malformed_output"


def test_validator_accepts_only_real_action_ids():
    cleaned, dropped = validate_evidence(parse_block(VALID), PACKET, VALID_ACTIONS)
    assert dropped == []
    assert "MOUSE" in cleaned["NEXT PROBES"]


# ---- ratchet --------------------------------------------------------------

def test_gating_respects_budget_cooldown_and_warmup():
    r = RevisionRatchet(cfg(min_actions_before_first_revise=20,
                            cooldown_actions=25, max_revisions_per_level=2))
    assert not r.may_fire(action_num=10, stalled=True)      # too early
    assert not r.may_fire(action_num=30, stalled=False)     # not stalled
    assert r.may_fire(action_num=30, stalled=True)
    r.accept("B1", action_num=30, hashes={"h1"})
    assert not r.may_fire(action_num=40, stalled=True)      # evaluation open
    r.evaluate(action_num=50, stalled=True, hashes={"h1"}, level_cleared=False)
    assert not r.may_fire(action_num=50, stalled=True)      # cooldown
    assert r.may_fire(action_num=56, stalled=True)


def test_rollback_when_no_improvement():
    r = RevisionRatchet(cfg())
    r.accept("B1", action_num=30, hashes={"h1"})
    assert r.block == "B1"
    improved, index = r.evaluate(action_num=50, stalled=True, hashes={"h1"},
                                 level_cleared=False)
    assert improved is False and index == 1
    assert r.block == "", "a failed revision must be reverted"
    assert r.rollbacks == 1
    assert r.revisions_used == 1, "a rollback still consumes budget"


@pytest.mark.parametrize("kwargs,expected", [
    ({"stalled": False, "hashes": {"h1"}, "level_cleared": False}, True),
    ({"stalled": True, "hashes": {"h1", "h2"}, "level_cleared": False}, True),
    ({"stalled": True, "hashes": {"h1"}, "level_cleared": True}, True),
    ({"stalled": True, "hashes": {"h1"}, "level_cleared": False}, False),
])
def test_improvement_definition(kwargs, expected):
    r = RevisionRatchet(cfg())
    r.accept("B1", action_num=30, hashes={"h1"})
    improved, _ = r.evaluate(action_num=50, **kwargs)
    assert improved is expected
    assert (r.block == "B1") is expected


def test_two_revisions_both_roll_back_to_the_pre_revision_state():
    r = RevisionRatchet(cfg(cooldown_actions=5))
    base = r.block
    r.accept("B1", action_num=30, hashes={"h1"})
    r.evaluate(action_num=50, stalled=True, hashes={"h1"}, level_cleared=False)
    r.accept("B2", action_num=60, hashes={"h1"})
    r.evaluate(action_num=80, stalled=True, hashes={"h1"}, level_cleared=False)
    assert r.block == base == ""
    assert r.rollbacks == 2
    assert not r.may_fire(action_num=100, stalled=True)   # budget exhausted


def test_stack_never_exceeds_the_bound():
    r = RevisionRatchet(cfg(cooldown_actions=1, max_revisions_per_level=99))
    for i in range(10):
        t = i * 30 + 30
        r.accept(f"B{i}", action_num=t, hashes={"h"})
        r.evaluate(action_num=t + 20, stalled=False, hashes={"h"},
                   level_cleared=False)          # improved: keep the block
        assert len(r.stack) <= MAX_STACK_DEPTH, r.stack


def test_level_transition_clears_the_block():
    r = RevisionRatchet(cfg())
    r.accept("B1", action_num=30, hashes={"h"})
    r.evaluate(action_num=50, stalled=False, hashes={"h"}, level_cleared=False)
    assert r.block == "B1"
    r.reset_level()                       # asserts internally
    assert r.block == "" and r.revisions_used == 0


def test_detector_and_ratchet_agree_on_a_replayed_trajectory():
    """A stalling trajectory fires once, then respects cooldown."""
    config = cfg(min_actions_before_first_revise=20, cooldown_actions=25)
    ratchet = RevisionRatchet(config)
    changed, hashes = deque(maxlen=20), deque(maxlen=30)
    fires = []
    for t in range(1, 121):
        moved = t % 5 == 0                      # 80% no-op: a hard stall
        changed.append(moved)
        hashes.append(f"h{t // 7}")
        if len(changed) < 20:
            continue
        state = stall_state(changed, hashes, config)
        if ratchet.due(t):
            ratchet.evaluate(action_num=t, stalled=state.stalled,
                             hashes=set(hashes), level_cleared=False)
        if ratchet.may_fire(action_num=t, stalled=state.stalled):
            ratchet.accept(f"B{t}", action_num=t, hashes=set(hashes))
            fires.append(t)
    assert fires, "a heavily stalling trajectory must fire at least once"
    assert len(fires) <= config.max_revisions_per_level
    assert all(b - a >= config.cooldown_actions for a, b in zip(fires, fires[1:]))
    assert len(ratchet.stack) <= MAX_STACK_DEPTH
