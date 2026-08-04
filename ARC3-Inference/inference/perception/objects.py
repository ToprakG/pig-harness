"""Segmentation-node geometry, indexing, and cross-frame object matching."""


def _node_bbox(node):
    """``(min_row, min_col, max_row, max_col)`` for a segmentation node, or ``None``.

    Prefers the node's own ``bbox``. Falls back to the boundary corners, which carry the
    full extent because a 4-connected component's extreme cells always lie on its outer
    perimeter at direction-change points.
    """
    bbox = node.get("bbox")
    if bbox:
        return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    points = node.get("boundary") or []
    if not points:
        return None
    rows = [int(point[0]) for point in points]
    cols = [int(point[1]) for point in points]
    return (min(rows), min(cols), max(rows), max(cols))


def _object_index(segmentation, background_pixels):
    """Group segmentation nodes by shape hash, keeping each node's anchor and size.

    The hash covers color plus position-normalized shape, so equal hashes mean the same
    object regardless of where it sits -- which is what makes cross-frame matching work.

    Nodes of at least ``background_pixels`` cells are split out as background. A
    background region is the complement of everything drawn on it, so its shape -- and
    therefore its hash -- changes whenever any object moves. Left in the object lists it
    would report as a huge appeared/disappeared pair on almost every frame and crowd out
    the objects that actually matter.
    """
    grouped = {}
    background = []
    for node in (segmentation or {}).get("nodes", []):
        bbox = _node_bbox(node)
        if bbox is None:
            continue
        centroid = node.get("centroid") or [
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2,
        ]
        entry = {
            "hash": node.get("hash"),
            "shape_hash": node.get("shape_hash"),
            "id": node.get("id"),
            "color": node.get("color"),
            "pixels": int(node.get("pixels", 0)),
            "anchor": (bbox[0], bbox[1]),
            "bbox": bbox,
            "centroid": (float(centroid[0]), float(centroid[1])),
        }
        if background_pixels > 0 and entry["pixels"] >= background_pixels:
            background.append(entry)
            continue
        grouped.setdefault(entry["hash"], []).append(entry)
    return grouped, background


def _pair_by_proximity(before_nodes, after_nodes):
    """Greedily pair same-hash nodes by closest anchor, nearest pairs first.

    Identical hash means identical shape and color, so an anchor delta is exactly the
    translation the object underwent.
    """
    candidates = []
    for before_index, before in enumerate(before_nodes):
        for after_index, after in enumerate(after_nodes):
            distance = abs(before["anchor"][0] - after["anchor"][0]) + abs(
                before["anchor"][1] - after["anchor"][1]
            )
            candidates.append((distance, before_index, after_index))
    candidates.sort()

    used_before = set()
    used_after = set()
    pairs = []
    for _distance, before_index, after_index in candidates:
        if before_index in used_before or after_index in used_after:
            continue
        used_before.add(before_index)
        used_after.add(after_index)
        pairs.append((before_nodes[before_index], after_nodes[after_index]))
    unmatched_before = [node for index, node in enumerate(before_nodes) if index not in used_before]
    unmatched_after = [node for index, node in enumerate(after_nodes) if index not in used_after]
    return pairs, unmatched_before, unmatched_after


def _bboxes_overlap(first, second):
    return not (
        first[2] < second[0] or second[2] < first[0] or first[3] < second[1] or second[3] < first[1]
    )


def _size_ratio(first, second):
    largest = max(first, second)
    return min(first, second) / largest if largest else 1.0


def _describe_change(before, after):
    """Name what differs between two matched objects.

    Geometry is compared with the color-free ``shape_hash`` so that a recolored object is
    not also reported as reshaped. A size change already implies a shape change, so only
    the more specific ``size`` is reported in that case.
    """
    changed = []
    if before["anchor"] != after["anchor"]:
        changed.append("moved")
    if before["color"] != after["color"]:
        changed.append("color")
    if before["pixels"] != after["pixels"]:
        changed.append("size")
    elif before["shape_hash"] != after["shape_hash"]:
        changed.append("shape")
    return changed


def _match_entry(before, after, by):
    return {
        "before_id": before["id"],
        "after_id": after["id"],
        "by": by,
        "color": before["color"] if before["color"] == after["color"] else None,
        "colors": [before["color"], after["color"]],
        "pixels": [before["pixels"], after["pixels"]],
        "from": [before["anchor"][0], before["anchor"][1]],
        "to": [after["anchor"][0], after["anchor"][1]],
        "delta": [
            after["anchor"][0] - before["anchor"][0],
            after["anchor"][1] - before["anchor"][1],
        ],
        "changed": _describe_change(before, after),
    }


def _match_loosely(before_nodes, after_nodes, max_shift, min_size_ratio, allow_color_change):
    """Pair leftover nodes that stayed put but changed shape, size, or color.

    A pair is only considered when the two are plausibly the same thing: similar size, and
    either overlapping bounding boxes or centroids within ``max_shift`` cells. Candidates
    are then taken closest-first, so the obvious pairings win before the marginal ones.
    """
    candidates = []
    for before_index, before in enumerate(before_nodes):
        for after_index, after in enumerate(after_nodes):
            if not allow_color_change and before["color"] != after["color"]:
                continue
            if _size_ratio(before["pixels"], after["pixels"]) < min_size_ratio:
                continue
            distance = abs(before["centroid"][0] - after["centroid"][0]) + abs(
                before["centroid"][1] - after["centroid"][1]
            )
            if distance > max_shift and not _bboxes_overlap(before["bbox"], after["bbox"]):
                continue
            same_color = 0 if before["color"] == after["color"] else 1
            candidates.append((same_color, distance, before_index, after_index))
    candidates.sort()

    used_before = set()
    used_after = set()
    pairs = []
    for _same_color, _distance, before_index, after_index in candidates:
        if before_index in used_before or after_index in used_after:
            continue
        used_before.add(before_index)
        used_after.add(after_index)
        pairs.append((before_nodes[before_index], after_nodes[after_index]))
    return (
        pairs,
        [node for index, node in enumerate(before_nodes) if index not in used_before],
        [node for index, node in enumerate(after_nodes) if index not in used_after],
    )


def match_objects(
    before_segmentation,
    after_segmentation,
    background_pixels=0,
    max_shift=3,
    min_size_ratio=0.5,
    allow_color_change=True,
    max_matches=8,
    max_unmatched=8,
):
    """Track objects across two frames, following them through shape and color changes.

    Matching runs in two stages. First, objects with identical shape hashes are paired by
    closest position, which is exact: same shape and color means the position delta is
    precisely the translation. Whatever is left over is then paired loosely, so an object
    that grew, shrank, or changed color is still recognised as the same object instead of
    reading as an unrelated disappearance plus appearance.

    A loose pair requires a size ratio of at least ``min_size_ratio`` and either
    overlapping bounding boxes or centroids within ``max_shift`` cells. Set
    ``allow_color_change`` to ``False`` to require the color to hold. Nodes of at least
    ``background_pixels`` cells are set aside as background, since a background region's
    shape changes whenever anything on top of it moves.

    This is the tracking primitive behind ``diff_frames``, which reports only exact
    matches. Reach for this when you need to follow something that is changing as it moves.

    Returns ``matched`` (each with ``before_id``, ``after_id``, ``by`` -- ``"shape"`` for an
    exact match or ``"position"`` for a loose one -- ``colors``, ``pixels``, ``from``,
    ``to``, ``delta``, and ``changed`` naming which of moved/color/size/shape differ),
    ``unmatched_before`` and ``unmatched_after`` (genuinely gone or genuinely new), each
    with a ``*_total`` count, plus ``stable`` for objects that matched exactly without
    moving and ``background`` colors.
    """
    before_groups, before_background = _object_index(before_segmentation, background_pixels)
    after_groups, after_background = _object_index(after_segmentation, background_pixels)

    matched = []
    stable = 0
    leftover_before = []
    leftover_after = []

    for hash_key in sorted(set(before_groups) | set(after_groups), key=lambda key: str(key)):
        pairs, unmatched_before, unmatched_after = _pair_by_proximity(
            before_groups.get(hash_key, []), after_groups.get(hash_key, [])
        )
        for before, after in pairs:
            if before["anchor"] == after["anchor"]:
                stable += 1
                continue
            matched.append(_match_entry(before, after, "shape"))
        leftover_before.extend(unmatched_before)
        leftover_after.extend(unmatched_after)

    loose_pairs, gone, new = _match_loosely(
        leftover_before, leftover_after, max_shift, min_size_ratio, allow_color_change
    )
    for before, after in loose_pairs:
        matched.append(_match_entry(before, after, "position"))

    matched.sort(key=lambda item: (-max(item["pixels"]), item["from"], item["to"]))
    gone_entries = sorted(
        (
            {
                "id": node["id"],
                "hash": node["hash"],
                "color": node["color"],
                "pixels": node["pixels"],
                "at": [node["anchor"][0], node["anchor"][1]],
            }
            for node in gone
        ),
        key=lambda item: (-item["pixels"], item["at"]),
    )
    new_entries = sorted(
        (
            {
                "id": node["id"],
                "hash": node["hash"],
                "color": node["color"],
                "pixels": node["pixels"],
                "at": [node["anchor"][0], node["anchor"][1]],
            }
            for node in new
        ),
        key=lambda item: (-item["pixels"], item["at"]),
    )

    return {
        "matched": matched[:max_matches],
        "matched_total": len(matched),
        "unmatched_before": gone_entries[:max_unmatched],
        "unmatched_before_total": len(gone_entries),
        "unmatched_after": new_entries[:max_unmatched],
        "unmatched_after_total": len(new_entries),
        "stable": stable,
        "background_colors": sorted({node["color"] for node in after_background}),
    }
