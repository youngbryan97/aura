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


def utterance_construction_folds(examples: Any, *, count: int = 3,
                                 seed: int = 0) -> dict[str, Any]:
    """Hold wording constructions while allowing shared program semantics.

    This tests paraphrase transfer, not transfer to unseen concepts or
    computation graphs. Contrasts may cross the split by design.
    """
    examples = tuple(examples)
    if type(count) is not int or count < 2 or type(seed) is not int:
        raise ValueError("invalid utterance fold configuration")
    if not examples or any(item.split != "train" for item in examples):
        raise ValueError("utterance folds require source training examples only")
    rows = sorted((item.ir.source_text_sha256, item.construction_id, item.contrast_id)
                  for item in examples)
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError("repeated source identity in utterance folds")
    groups = defaultdict(list)
    families = defaultdict(list)
    for row in rows:
        source, construction, _contrast = row
        family, separator, suffix = construction.partition(":")
        if not source or not separator or not family or not suffix:
            raise ValueError("utterance fold family provenance is incomplete")
        groups[construction].append(row)
    for construction in groups:
        families[construction.partition(":")[0]].append(construction)
    if any(len(constructions) < count for constructions in families.values()):
        raise ValueError("each utterance family needs one construction per fold")
    assignments = {}
    sizes = [0] * count
    for family, constructions in sorted(families.items()):
        family_sizes = [0] * count
        ordered = sorted(constructions, key=lambda construction: (
            -len(groups[construction]), _sha({"seed": seed, "construction": construction})))
        for construction in ordered:
            fold = min(range(count), key=lambda index: (family_sizes[index], sizes[index], index))
            assignments.update((row[0], fold) for row in groups[construction])
            family_sizes[fold] += 1
            sizes[fold] += len(groups[construction])
    contrast_folds = defaultdict(set)
    for source, _construction, contrast in rows:
        if contrast:
            contrast_folds[contrast].add(assignments[source])
    body = {"schema": "aura.semantic_utterance_construction_folds.v1",
            "count": count, "seed": seed, "population": rows,
            "assignments": assignments, "fold_sizes": sizes,
            "family_construction_counts": {name: len(values)
                                            for name, values in sorted(families.items())},
            "contrast_lineages_crossing_folds": sum(len(values) > 1
                                                   for values in contrast_folds.values()),
            "contrast_lineage_closure_enforced": False,
            "lexical_construction_disjoint": True,
            "independent_semantic_transfer_claim": False,
            "validation_used": False, "test_used": False}
    return {**body, "receipt_sha256": _sha(body)}
