"""Shortest-path search over grid cells, reported as directions to act on."""

from inference.perception.grid import _MISSING_CHAR, _cell_char, _dims


_ORTH_MOVES = (("UP", -1, 0), ("DOWN", 1, 0), ("LEFT", 0, -1), ("RIGHT", 0, 1))
_DIAG_MOVES = (
    ("UP_LEFT", -1, -1),
    ("UP_RIGHT", -1, 1),
    ("DOWN_LEFT", 1, -1),
    ("DOWN_RIGHT", 1, 1),
)


def _coords(value):
    """Normalise a coordinate or list of coordinates to a list of ``(row, col)``."""
    if not value:
        return []
    first = value[0]
    if isinstance(first, (list, tuple)):
        return [(int(item[0]), int(item[1])) for item in value]
    return [(int(value[0]), int(value[1]))]


def _compress_moves(moves):
    """``["RIGHT", "RIGHT", "UP"]`` becomes ``[["RIGHT", 2], ["UP", 1]]``."""
    runs = []
    for move in moves:
        if runs and runs[-1][0] == move:
            runs[-1][1] += 1
        else:
            runs.append([move, 1])
    return runs


def find_path(
    grid,
    color_chars,
    start,
    goal,
    blocked_colors=(),
    passable_colors=None,
    diagonal=False,
    max_path_steps=200,
):
    """Shortest path between two cells, returned as directions you can act on.

    Breadth-first over the grid, so the path is the shortest available. ``goal`` may be one
    ``[row, col]`` or a list of them, in which case the nearest reachable one wins.

    Passability comes from colors: give ``blocked_colors`` to treat those as walls and
    everything else as open, or ``passable_colors`` to allow only those (it takes
    precedence). The start and goal cells are always allowed, so you can path out of and
    into a colored cell. Cells outside the grid are walls.

    ``UP`` means decreasing row and ``LEFT`` decreasing column. With ``diagonal`` the four
    diagonal moves are added, each still counting as one step.

    Returns ``found``, ``steps`` (number of moves), ``moves`` (direction names),
    ``runs`` (``moves`` collapsed to ``[direction, count]`` pairs, the compact form to act
    on), ``path`` (cells from start to goal inclusive), ``goal_reached``, ``explored``
    (cells visited, showing how much of the board is reachable), plus ``truncated`` when
    ``moves``/``path`` were cut at ``max_path_steps``, and ``reason`` when no path exists.
    """
    rows, cols = _dims(grid)
    empty = {
        "found": False,
        "steps": None,
        "moves": [],
        "runs": [],
        "path": [],
        "goal_reached": None,
        "explored": 0,
        "truncated": False,
    }

    starts = _coords(start)
    goals = [cell for cell in _coords(goal) if 0 <= cell[0] < rows and 0 <= cell[1] < cols]
    if not starts or not (0 <= starts[0][0] < rows and 0 <= starts[0][1] < cols):
        return dict(empty, reason="start_out_of_bounds")
    if not goals:
        return dict(empty, reason="goal_out_of_bounds")

    origin = starts[0]
    goal_set = set(goals)
    allowed = set(passable_colors) if passable_colors is not None else None
    blocked = set(blocked_colors)
    exempt = {origin} | goal_set

    def passable(cell):
        if cell in exempt:
            return True
        char = _cell_char(grid, cell[0], cell[1], color_chars)
        if char == _MISSING_CHAR:
            return False
        if allowed is not None:
            return char in allowed
        return char not in blocked

    moves = _ORTH_MOVES + _DIAG_MOVES if diagonal else _ORTH_MOVES
    came_from = {origin: (None, None)}
    queue = [origin]
    head = 0
    reached = origin if origin in goal_set else None

    while head < len(queue) and reached is None:
        row, col = queue[head]
        head += 1
        for name, row_step, col_step in moves:
            neighbour = (row + row_step, col + col_step)
            if not (0 <= neighbour[0] < rows and 0 <= neighbour[1] < cols):
                continue
            if neighbour in came_from or not passable(neighbour):
                continue
            came_from[neighbour] = ((row, col), name)
            if neighbour in goal_set:
                reached = neighbour
                break
            queue.append(neighbour)

    if reached is None:
        return dict(empty, explored=len(came_from), reason="unreachable")

    path = []
    directions = []
    cursor = reached
    while cursor is not None:
        path.append([cursor[0], cursor[1]])
        previous, name = came_from[cursor]
        if name is not None:
            directions.append(name)
        cursor = previous
    path.reverse()
    directions.reverse()

    steps = len(directions)
    truncated = steps > max_path_steps
    return {
        "found": True,
        "steps": steps,
        "moves": directions[:max_path_steps],
        "runs": _compress_moves(directions),
        "path": path[: max_path_steps + 1],
        "goal_reached": [reached[0], reached[1]],
        "explored": len(came_from),
        "truncated": truncated,
        "reason": None,
    }
