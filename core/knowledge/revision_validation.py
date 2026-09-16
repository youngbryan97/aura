"""Executable checks for the AtomSpace revision contract."""

from core.knowledge.atomspace import AtomSpace, InferenceRule, TruthValue, concept, _fold


def revision_violations(space: AtomSpace) -> list[str]:
    failures = []
    with space._lock:
        expected = {}
        for atom, record in space._records.items():
            if record.derivations and record.tv != _fold(record):
                failures.append(f"truth does not match support: {atom}")
            for key, derivation in record.derivations.items():
                if not derivation.dependencies or not derivation.sources:
                    failures.append(f"unattributed derivation: {atom}")
                if set(derivation.premise_revisions) != derivation.dependencies or any(
                    revision != (space._records[p].revision if p in space._records else None)
                    for p, revision in derivation.premise_revisions.items()
                ):
                    failures.append(f"stale premise revision: {atom}")
                for parent in derivation.dependencies:
                    expected.setdefault(parent, set()).add((atom, key))
                    if space._depends_on_locked(parent, atom):
                        failures.append(f"circular derivation: {atom}")
        if expected != space._dependents:
            failures.append("dependency index disagrees with active derivations")
        for source, (_revision, claims) in space._observations.items():
            for atom, tv in claims.items():
                record = space._records.get(atom)
                if record is not None and record.sources.get(source) != tv:
                    failures.append(f"observation disagrees with atom: {source}")
    return failures


def revision_canary() -> bool:
    """Exercise correction and stale-delivery handling on the production class."""
    space = AtomSpace()
    a, b = concept("canary-premise"), concept("canary-conclusion")
    rule = InferenceRule("canary-copy", (a,), b,
        lambda _s, _b, tvs: TruthValue(tvs[0].strength, .9 * tvs[0].count))
    space.revise_observation("canary", 1, {a: TruthValue(.9, 4)})
    space.apply_rules((rule,), focus_only=False)
    space.revise_observation("canary", 2, {a: TruthValue(.1, 4)})
    if space.get_tv(b).count != 0:
        return False
    space.apply_rules((rule,), focus_only=False)
    space.revise_observation("canary", 1, {a: TruthValue(.9, 4)})
    return space.get_tv(b) == TruthValue(.1, 3.6) and not revision_violations(space)
