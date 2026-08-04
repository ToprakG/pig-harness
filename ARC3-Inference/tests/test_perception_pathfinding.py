from inference.utils.grid_utils import ARC_COLOR_CHARS
from inference.perception import find_path

WHITE = ARC_COLOR_CHARS[0]
BLUE = ARC_COLOR_CHARS[9]
RED = ARC_COLOR_CHARS[8]

_VALUES = {".": 0, "#": 9, "R": 8}


def _grid(rows):
    """Build a grid from strings: '.' white (open), '#' blue (wall), 'R' red."""
    return [[_VALUES[char] for char in row] for row in rows]


def _path(rows, start, goal, **kwargs):
    kwargs.setdefault("blocked_colors", [BLUE])
    return find_path(_grid(rows), ARC_COLOR_CHARS, start, goal, **kwargs)


def test_straight_corridor_gives_one_run_of_moves():
    result = _path(["....."], [0, 0], [0, 4])

    assert result["found"] is True
    assert result["steps"] == 4
    assert result["moves"] == ["RIGHT"] * 4
    assert result["runs"] == [["RIGHT", 4]]
    assert result["path"][0] == [0, 0]
    assert result["path"][-1] == [0, 4]


def test_up_means_decreasing_row():
    result = _path([".", ".", "."], [2, 0], [0, 0])

    assert result["moves"] == ["UP", "UP"]


def test_left_means_decreasing_column():
    result = _path(["..."], [0, 2], [0, 0])

    assert result["runs"] == [["LEFT", 2]]


def test_path_routes_around_a_wall():
    result = _path(
        [
            ".....",
            ".###.",
            ".....",
        ],
        [0, 2],
        [2, 2],
    )

    assert result["found"] is True
    assert result["steps"] == 6
    # Every cell on the path is open, and no step crosses the wall row directly.
    assert [1, 2] not in result["path"]


def test_walls_can_make_the_goal_unreachable():
    result = _path(
        [
            ".....",
            "#####",
            ".....",
        ],
        [0, 0],
        [2, 0],
    )

    assert result["found"] is False
    assert result["reason"] == "unreachable"
    assert result["steps"] is None
    assert result["moves"] == []
    # Only the open half of the board was reachable.
    assert result["explored"] == 5


def test_start_equals_goal_is_a_zero_step_path():
    result = _path(["..."], [0, 1], [0, 1])

    assert result["found"] is True
    assert result["steps"] == 0
    assert result["moves"] == []
    assert result["path"] == [[0, 1]]


def test_diagonal_moves_shorten_the_path_when_enabled():
    rows = ["...", "...", "..."]

    orthogonal = _path(rows, [0, 0], [2, 2])
    with_diagonals = _path(rows, [0, 0], [2, 2], diagonal=True)

    assert orthogonal["steps"] == 4
    assert with_diagonals["steps"] == 2
    assert with_diagonals["runs"] == [["DOWN_RIGHT", 2]]


def test_passable_colors_take_precedence_over_blocked():
    rows = ["...", "RRR", "..."]

    # Only white is walkable, so the red row is a wall even though nothing is blocked.
    result = find_path(
        _grid(rows), ARC_COLOR_CHARS, [0, 0], [2, 0], passable_colors=[WHITE], blocked_colors=[]
    )

    assert result["found"] is False


def test_start_and_goal_are_always_allowed():
    rows = ["#..", "...", "..#"]

    # Both endpoints sit on blocked cells; the route between them still has to be open.
    result = _path(rows, [0, 0], [2, 2])

    assert result["found"] is True
    assert result["steps"] == 4


def test_nearest_of_several_goals_wins():
    result = _path(["........."], [0, 4], [[0, 0], [0, 6], [0, 8]])

    assert result["goal_reached"] == [0, 6]
    assert result["steps"] == 2


def test_unreachable_goal_falls_back_to_a_reachable_one():
    rows = [
        ".#.",
        ".#.",
        ".#.",
    ]

    result = _path(rows, [0, 0], [[0, 2], [2, 0]])

    assert result["goal_reached"] == [2, 0]


def test_out_of_bounds_start_is_reported():
    result = _path(["..."], [5, 5], [0, 0])

    assert result["found"] is False
    assert result["reason"] == "start_out_of_bounds"


def test_out_of_bounds_goal_is_reported():
    result = _path(["..."], [0, 0], [9, 9])

    assert result["found"] is False
    assert result["reason"] == "goal_out_of_bounds"


def test_long_path_is_truncated_but_runs_stay_complete():
    rows = ["." * 300]

    result = _path(rows, [0, 0], [0, 299], max_path_steps=10)

    assert result["steps"] == 299
    assert result["truncated"] is True
    assert len(result["moves"]) == 10
    assert len(result["path"]) == 11
    # The compact form still describes the whole route.
    assert result["runs"] == [["RIGHT", 299]]


def test_short_path_is_not_marked_truncated():
    result = _path(["....."], [0, 0], [0, 2])

    assert result["truncated"] is False
    assert result["reason"] is None


def test_runs_collapse_each_direction_change():
    result = _path(
        [
            "..#",
            ".##",
            "...",
        ],
        [0, 0],
        [2, 2],
    )

    assert result["runs"] == [["DOWN", 2], ["RIGHT", 2]]


def test_explored_counts_visited_cells():
    result = _path(["...", "...", "..."], [0, 0], [0, 0])

    # The goal is the start, so the search stops immediately.
    assert result["explored"] == 1


def test_empty_grid_is_handled():
    result = find_path([], ARC_COLOR_CHARS, [0, 0], [1, 1])

    assert result["found"] is False
    assert result["reason"] == "start_out_of_bounds"
