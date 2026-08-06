"""Phase 1: the packet builder is a pure function over an event log."""

from __future__ import annotations

import glob
import json

import pytest

from inference.debrief.packet import (
    build_packet,
    describe_delta,
    estimate_tokens,
    split_levels,
    to_bool,
)

BG = 0


def grid(cells: dict[tuple[int, int], int], size: int = 3):
    return [[cells.get((r, c), BG) for c in range(size)] for r in range(size)]


def event(action="UP", changed=True, board=None, completed=False, num=1):
    return {"type": "action", "action_num": num, "action_display": action,
            "action_name": "ACTION1", "board_changed": changed,
            "level_completed": completed, "board": board or grid({})}


def test_to_bool_handles_the_string_form():
    assert to_bool(True) and to_bool("True")
    assert not to_bool(False) and not to_bool("False") and not to_bool(None)


@pytest.mark.parametrize("before,after,expected", [
    (grid({(0, 0): 1}), grid({(0, 0): 1}), "no_change"),
    (grid({(0, 0): 1}), grid({(0, 0): 1, (1, 1): 1}), "appeared"),
    (grid({(0, 0): 1, (1, 1): 1}), grid({(0, 0): 1}), "disappeared"),
    (grid({(0, 0): 1}), grid({(0, 1): 1}), "moved"),
    (grid({(0, 0): 1}), grid({(0, 0): 2}), "recoloured"),
])
def test_describe_delta_categories(before, after, expected):
    assert describe_delta(before, after) == expected


def test_packet_counts_effects_and_dead_actions():
    events = [
        event("UP", changed=True, board=grid({(0, 0): 1}), num=1),
        event("UP", changed=True, board=grid({(0, 1): 1}), num=2),
        event("SPACE", changed=False, board=grid({(0, 1): 1}), num=3),
        event("SPACE", changed=False, board=grid({(0, 1): 1}), num=4),
        event("LEFT", changed=True, board=grid({(0, 0): 1}), completed=True, num=5),
    ]
    packet = build_packet(events, level=1)
    effects = {e["action"]: e for e in packet.action_effects}
    assert effects["UP"]["changed"] == 2 and effects["UP"]["unchanged"] == 0
    assert effects["SPACE"]["unchanged"] == 2
    assert packet.dead_actions == ["SPACE"]
    assert packet.actions_used == 5
    assert packet.level == 1


def test_clear_trigger_is_the_last_three_actions():
    events = [event(a, num=i) for i, a in enumerate("ABCDE", start=1)]
    events[-1]["level_completed"] = True
    packet = build_packet(events, level=2)
    assert [t["action"] for t in packet.clear_trigger] == ["C", "D", "E"]


def test_stall_segments_record_context():
    events = [
        event("UP", changed=True, board=grid({(0, 0): 1}), num=1),
        event("DOWN", changed=False, board=grid({(0, 0): 1}), num=2),
        event("DOWN", changed=False, board=grid({(0, 0): 1}), num=3),
        event("LEFT", changed=True, board=grid({(0, 1): 1}), completed=True, num=4),
    ]
    stall = build_packet(events, level=1).stall_segments[0]
    assert stall["length"] == 2
    assert stall["preceded_by"] == "UP" and stall["broken_by"] == "LEFT"


def test_baseline_is_omitted_unless_explicitly_included():
    events = [event(completed=True)]
    assert build_packet(events, level=1, baseline_actions=10).human_multiple is None
    packet = build_packet(events, level=1, baseline_actions=10,
                          include_baseline=True)
    assert packet.human_multiple == pytest.approx(0.1)
    assert "human_multiple" not in build_packet(events, level=1).as_dict()


def test_packet_carries_no_game_identity_or_coordinates():
    events = [event("MOUSE(row=3, col=4)", num=1),
              event("UP", completed=True, num=2)]
    rendered = build_packet(events, level=1).render()
    assert "row=" not in rendered and "col=" not in rendered
    assert "MOUSE" in rendered            # the action id itself is fine
    for forbidden in ("game_id", "-0c556536", "colour"):
        assert forbidden not in rendered


def test_split_levels_drops_an_uncompleted_tail():
    events = [event(num=1), event(num=2, completed=True), event(num=3)]
    segments = split_levels(events)
    assert len(segments) == 1 and len(segments[0]) == 2


def test_action_ids_covers_everything_the_validator_must_check():
    events = [event("UP", num=1), event("SPACE", changed=False, num=2),
              event("LEFT", completed=True, num=3)]
    ids = build_packet(events, level=1).action_ids()
    assert {"UP", "SPACE", "LEFT"} <= ids


# ---- the acceptance criterion, against real data -------------------------

def test_builds_for_every_real_level_transition():
    files = sorted(glob.glob("../example-run/artifacts/*_events.jsonl"))
    if not files:
        pytest.skip("example-run artifacts not present")
    transitions = 0
    sizes = []
    for path in files:
        actions = [json.loads(line) for line in open(path)]
        actions = [a for a in actions if a.get("type") == "action"]
        for level, segment in enumerate(split_levels(actions), start=1):
            packet = build_packet(segment, level=level)   # must not raise
            sizes.append(estimate_tokens(packet.render()))
            transitions += 1
    assert transitions == 309, transitions
    assert max(sizes) <= 400, f"packet exceeded the token cap: {max(sizes)}"
