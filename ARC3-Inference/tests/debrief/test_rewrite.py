"""Phase 2: the rewrite call, its parser, its evidence validator, and all
four failure paths (valid / timeout / malformed / hallucinated action id)."""

from __future__ import annotations

import pytest

from inference.debrief.packet import build_packet
from inference.debrief.rewrite import (
    BLOCK_HEADER,
    DebriefConfig,
    build_prompt,
    parse_block,
    rewrite,
    validate_evidence,
)

VALID = (
    "CONFIRMED MECHANICS: UP and LEFT move the object; SPACE recolours it.\n"
    "DEAD ACTIONS: MOUSE\n"
    "WHAT CLEARED THE LAST LEVEL: LEFT made the target disappear.\n"
    "FIRST THINGS TO TEST ON THIS LEVEL: try UP once; then LEFT; watch SPACE.\n"
)


def packet():
    events = [
        {"type": "action", "action_display": "UP", "board_changed": True,
         "board": [[0, 1], [0, 0]], "level_completed": False},
        {"type": "action", "action_display": "MOUSE(row=1, col=2)",
         "board_changed": False, "board": [[0, 1], [0, 0]],
         "level_completed": False},
        {"type": "action", "action_display": "SPACE", "board_changed": True,
         "board": [[0, 2], [0, 0]], "level_completed": False},
        {"type": "action", "action_display": "LEFT", "board_changed": True,
         "board": [[0, 0], [0, 0]], "level_completed": True},
    ]
    return build_packet(events, level=1)


def config(**kw):
    return DebriefConfig(enabled=True, **kw)


# ---- (a) valid output -----------------------------------------------------

def test_valid_rewrite_produces_a_block():
    result = rewrite(packet(), "", lambda *a: VALID, config=config())
    assert result.ok and result.reason == "ok"
    assert result.block.startswith(BLOCK_HEADER)
    assert "CONFIRMED MECHANICS:" in result.block
    assert result.dropped_lines == []


# ---- (b) timeout ----------------------------------------------------------

def test_timeout_falls_back_without_raising():
    def boom(*a):
        raise TimeoutError("simulated read timeout")

    result = rewrite(packet(), "prev block", boom, config=config())
    assert not result.ok and result.reason == "TimeoutError"
    assert result.block is None            # caller keeps its existing block


# ---- (c) malformed --------------------------------------------------------

@pytest.mark.parametrize("text", [
    "", "   ",
    "Sure! Here are some thoughts about the level.",
    "CONFIRMED MECHANICS: UP moves.\nDEAD ACTIONS: none.",   # sections missing
])
def test_malformed_output_falls_back(text):
    result = rewrite(packet(), "", lambda *a: text, config=config())
    assert not result.ok and result.reason == "malformed_output"


# ---- (d) hallucinated action id ------------------------------------------

def test_hallucinated_action_id_is_dropped_not_kept():
    hallucinated = VALID.replace(
        "UP and LEFT move the object", "ACTION7 and TELEPORT move the object")
    result = rewrite(packet(), "", lambda *a: hallucinated, config=config())
    assert result.ok, "a hallucination must not kill the whole block"
    assert "ACTION7" not in result.block and "TELEPORT" not in result.block
    assert "dropped" in result.block
    assert result.dropped_lines and "CONFIRMED MECHANICS" in result.dropped_lines[0]


def test_validator_keeps_sections_that_only_use_evidenced_ids():
    sections = parse_block(VALID)
    cleaned, dropped = validate_evidence(sections, packet())
    assert dropped == []
    assert cleaned["CONFIRMED MECHANICS"] == sections["CONFIRMED MECHANICS"]


def test_validator_ignores_plan_sections():
    """Plans may name anything; only factual sections are evidence-checked."""
    text = VALID.replace("try UP once", "try WARP once")
    cleaned, dropped = validate_evidence(parse_block(text), packet())
    assert dropped == []
    assert "WARP" in cleaned["FIRST THINGS TO TEST ON THIS LEVEL"]


# ---- misc -----------------------------------------------------------------

def test_oversize_output_falls_back():
    huge = VALID.replace("UP and LEFT move the object",
                         "UP moves " + "x " * 3000)
    result = rewrite(packet(), "", lambda *a: huge, config=config(max_tokens=50))
    assert not result.ok and result.reason == "oversize_output"


def test_parser_tolerates_multiline_sections():
    text = VALID.replace("DEAD ACTIONS: MOUSE",
                         "DEAD ACTIONS: MOUSE\n  (never changed the board)")
    sections = parse_block(text)
    assert "never changed" in sections["DEAD ACTIONS"]


def test_temperature_is_clamped_to_compression_range():
    assert DebriefConfig.from_dict({"temperature": 0.9}).temperature == 0.3
    assert DebriefConfig.from_dict({"temperature": 0.1}).temperature == 0.1


def test_prompt_contains_the_trace_and_the_standing_block():
    prompt = build_prompt(packet(), "STANDING: keep left")
    assert "TRACE OF THE LEVEL JUST CLEARED" in prompt
    assert "STANDING: keep left" in prompt
    assert "MOUSE" in prompt and "row=" not in prompt


def test_llm_call_receives_the_configured_limits():
    seen = {}

    def spy(prompt, max_tokens, timeout_s, temperature):
        seen.update(max_tokens=max_tokens, timeout_s=timeout_s,
                    temperature=temperature)
        return VALID

    rewrite(packet(), "", spy, config=config(max_tokens=400, timeout_s=90,
                                             temperature=0.3))
    assert seen == {"max_tokens": 400, "timeout_s": 90, "temperature": 0.3}
