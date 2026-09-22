"""Source-only architecture folds with construction and contrast closure."""

from collections import defaultdict
from typing import Any

from core.learning.semantic_program_campaign import _sha


def construction_folds(examples: Any, *, count: int = 3, seed: int = 0) -> dict[str, Any]:
    """Keep connected construction/contrast groups on one side of every fit."""
    examples = tuple(examples)
    if type(count) is not int or count < 2 or type(seed) is not int:
        raise ValueError("invalid construction fold configuration")
    if not examples or any(item.split != "train" for item in examples):
        raise ValueError("architecture folds require source training examples only")
    rows = sorted((item.ir.source_text_sha256, item.construction_id, item.contrast_id)
                  for item in examples)
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError("repeated source identity in architecture folds")
    if any(not source or not construction for source, construction, _ in rows):
        raise ValueError("architecture fold provenance is incomplete")
    parents = list(range(len(rows)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners = {}
    for index, (source, construction, contrast) in enumerate(rows):
        keys = [("construction", construction), ("lineage", source)]
        if contrast:
            keys.append(("lineage", contrast))
        for key in keys:
            if key in owners:
                parents[root(index)] = root(owners[key])
            else:
                owners[key] = index
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[root(index)].append(row)
    if len(groups) < count:
        raise ValueError("too few independent construction/contrast groups")
    ordered = sorted(groups.values(), key=lambda group: (
        -len(group), _sha({"seed": seed, "sources": [row[0] for row in group]})))
    sizes = [0] * count
    assignments = {}
    for group in ordered:
        fold = min(range(count), key=lambda index: (sizes[index], index))
        sizes[fold] += len(group)
        assignments.update((row[0], fold) for row in group)
    body = {"schema": "aura.semantic_construction_folds.v1", "count": count,
            "seed": seed, "population": rows, "assignments": assignments,
            "fold_sizes": sizes, "independent_groups": len(groups),
            "validation_used": False, "test_used": False}
    return {**body, "receipt_sha256": _sha(body)}
