"""Which actions have actually changed the board, and which are proven no-ops.

Wasted actions are scored against you directly, and the expensive mistake is repeating an
action that has never once done anything. The evidence for that is already in the action
history; this turns it into a count.
"""

from inference.perception.grid import _cell_char, _dims

_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))

_MOUSE_PREFIX = "MOUSE"


def _parse_mouse(action):
    """``"MOUSE(row=12, col=30)"`` -> ``(12, 30)``, or ``None`` for other actions."""
    if not action.startswith(_MOUSE_PREFIX):
        return None
    try:
        row = int(action.split("row=", 1)[1].split(",", 1)[0])
        col = int(action.split("col=", 1)[1].split(")", 1)[0])
    except (IndexError, ValueError):
        return None
    return row, col


def _component_at(grid, row, col, max_cells):
    """4-connected same-value cells containing ``(row, col)``, capped at ``max_cells``.

    Returns ``(cells, truncated)``. Truncation keeps a click on a huge background region
    from walking the whole board; such targets are reported as a region rather than a shape.
    """
    rows, cols = _dims(grid)
    if not (0 <= row < rows and 0 <= col < cols and col < len(grid[row])):
        return None, False
    target = grid[row][col]
    seen = {(row, col)}
    stack = [(row, col)]
    cells = []
    while stack:
        cell = stack.pop()
        cells.append(cell)
        if len(cells) >= max_cells:
            return cells, True
        for row_step, col_step in _ORTH:
            neighbour = (cell[0] + row_step, cell[1] + col_step)
            if neighbour in seen:
                continue
            if not (0 <= neighbour[0] < rows and 0 <= neighbour[1] < cols):
                continue
            if neighbour[1] >= len(grid[neighbour[0]]):
                continue
            if grid[neighbour[0]][neighbour[1]] != target:
                continue
            seen.add(neighbour)
            stack.append(neighbour)
    return cells, False


def _click_target(grid, row, col, color_chars, max_region_cells):
    """A stable name for the kind of thing at ``(row, col)``.

    Keyed by color, size, and bounding-box shape so that clicking a second object of the
    same kind counts against the same tally -- what matters is that this *type* of target
    does nothing, not that one particular instance did nothing.
    """
    color = _cell_char(grid, row, col, color_chars)
    cells, truncated = _component_at(grid, row, col, max_region_cells)
    if not cells:
        return {"target": f"{color}:unknown", "color": color, "pixels": 0, "shape": None}
    if truncated:
        return {"target": f"{color}:region", "color": color, "pixels": None, "shape": None}
    rows = [cell[0] for cell in cells]
    cols = [cell[1] for cell in cells]
    height = max(rows) - min(rows) + 1
    width = max(cols) - min(cols) + 1
    return {
        "target": f"{color}:{len(cells)}px:{height}x{width}",
        "color": color,
        "pixels": len(cells),
        "shape": [height, width],
    }


def summarize_action_effects(
    records,
    color_chars,
    min_attempts=2,
    max_actions=10,
    max_targets=8,
    max_sample_cells=4,
    max_region_cells=200,
):
    """Tally which actions changed the board and which never have.

    ``records`` is a sequence of ``(action_display, before_grid, after_grid)`` triples in
    chronological order; records without a ``before_grid`` are skipped because nothing can
    be concluded from them. Restrict ``records`` to the current level before calling, since
    mechanics change between levels.

    Clicks are grouped by what sat under the cursor rather than by coordinate, so twenty
    clicks on twenty instances of the same kind of block aggregate into one verdict.

    An entry is ``dead`` once it has been tried at least ``min_attempts`` times and has
    never changed the board. Treat dead entries as unavailable for the rest of the level
    rather than retrying them.

    Returns ``considered``, ``actions`` and ``click_targets`` (each with ``attempts``,
    ``changed``, ``dead``, and for targets a few sample ``cells``), the ``dead_actions`` and
    ``dead_click_targets`` shortlists, and ``wasted_actions`` -- how many actions have
    already gone into entries now known to be dead.
    """
    actions = {}
    targets = {}
    considered = 0

    for action, before, after in records:
        if before is None:
            continue
        considered += 1
        changed = before != after
        name = _MOUSE_PREFIX if action.startswith(_MOUSE_PREFIX) else action

        entry = actions.setdefault(name, {"action": name, "attempts": 0, "changed": 0})
        entry["attempts"] += 1
        entry["changed"] += 1 if changed else 0

        position = _parse_mouse(action)
        if position is None:
            continue
        described = _click_target(before, position[0], position[1], color_chars, max_region_cells)
        target = targets.setdefault(
            described["target"],
            dict(described, attempts=0, changed=0, cells=[]),
        )
        target["attempts"] += 1
        target["changed"] += 1 if changed else 0
        if len(target["cells"]) < max_sample_cells:
            target["cells"].append([position[0], position[1]])

    def _finish(entry):
        entry["dead"] = entry["attempts"] >= min_attempts and entry["changed"] == 0
        return entry

    action_list = sorted(
        (_finish(entry) for entry in actions.values()),
        key=lambda item: (-item["attempts"], item["action"]),
    )
    target_list = sorted(
        (_finish(entry) for entry in targets.values()),
        key=lambda item: (-item["attempts"], item["target"]),
    )

    wasted = sum(entry["attempts"] for entry in action_list if entry["dead"])
    wasted += sum(
        entry["attempts"]
        for entry in target_list
        if entry["dead"] and not actions.get(_MOUSE_PREFIX, {}).get("dead")
    )

    return {
        "considered": considered,
        "actions": action_list[:max_actions],
        "actions_total": len(action_list),
        "click_targets": target_list[:max_targets],
        "click_targets_total": len(target_list),
        "dead_actions": [entry["action"] for entry in action_list if entry["dead"]],
        "dead_click_targets": [entry["target"] for entry in target_list if entry["dead"]],
        "wasted_actions": wasted,
    }
