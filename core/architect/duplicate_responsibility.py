"""Explainable duplicate-responsibility detection."""
from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher

from core.architect.models import ArchitecturalSmell, ArchitectureGraph, MutationTier, SmellSeverity


class DuplicateResponsibilityDetector:
    """Find likely duplicate implementations using AST and semantic signals."""

    def detect(self, graph: ArchitectureGraph, *, limit: int = 50) -> list[ArchitecturalSmell]:
        smells: list[ArchitecturalSmell] = []
        smells.extend(self._identical_fingerprints(graph, limit=limit))
        smells.extend(self._similar_named_symbols(graph, limit=limit))
        return smells[:limit]

    def _identical_fingerprints(self, graph: ArchitectureGraph, *, limit: int) -> list[ArchitecturalSmell]:
        by_fp: dict[str, list[str]] = defaultdict(list)
        for node in graph.nodes.values():
            if node.kind not in {"function", "async_function", "method", "class"}:
                continue
            fp = str(node.metadata.get("fingerprint", ""))
            if fp:
                by_fp[fp].append(node.id)
        smells: list[ArchitecturalSmell] = []
        for fp, refs in by_fp.items():
            unique_paths = {graph.nodes[ref].path for ref in refs if ref in graph.nodes}
            if len(refs) < 2 or len(unique_paths) < 2:
                continue
            first = graph.nodes[refs[0]]
            smells.append(
                ArchitecturalSmell(
                    id=f"duplicate-ast-{fp[:12]}",
                    kind="duplicate_implementation",
                    severity=SmellSeverity.MEDIUM,
                    path=first.path,
                    symbol=first.name,
                    evidence=(f"identical normalized AST fingerprint appears in {len(refs)} symbols",),
                    graph_refs=tuple(refs[:10]),
                    suggested_tier=MutationTier.T2_REFACTOR,
                    proof_obligations=("behavior_fingerprint_equivalent", "relevant_tests_pass"),
                    auto_fixable=False,
                )
            )
            if len(smells) >= limit:
                return smells
        return smells

    def _similar_named_symbols(self, graph: ArchitectureGraph, *, limit: int) -> list[ArchitecturalSmell]:
        symbols = [
            node for node in graph.nodes.values()
            if node.kind in {"function", "async_function", "method", "class"} and len(node.name) >= 6
        ]
        buckets: dict[str, list[int]] = defaultdict(list)
        token_cache: list[set[str]] = []
        for idx, symbol in enumerate(symbols):
            tokens = _tokenize(symbol.name + " " + str(symbol.metadata.get("docstring", "")))
            token_cache.append(tokens)
            keys = sorted(tokens)[:4] or [symbol.name[:6].lower()]
            for key in keys:
                buckets[key].append(idx)
        smells: list[ArchitecturalSmell] = []
        seen_pairs: set[tuple[int, int]] = set()
        comparisons = 0
        for members in buckets.values():
            if len(members) > 250:
                members = members[:250]
            for left_pos, left_idx in enumerate(members):
                left = symbols[left_idx]
                left_tokens = token_cache[left_idx]
                for right_idx in members[left_pos + 1:]:
                    pair = (min(left_idx, right_idx), max(left_idx, right_idx))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    comparisons += 1
                    if comparisons > 50000:
                        return smells
                    right = symbols[right_idx]
                    if left.path == right.path:
                        continue
                    name_score = SequenceMatcher(a=left.name.lower(), b=right.name.lower()).ratio()
                    token_overlap = _jaccard(left_tokens, token_cache[right_idx])
                    effect_overlap = _jaccard(set(left.metadata.get("effects", ())), set(right.metadata.get("effects", ())))
                    combined = (name_score * 0.45) + (token_overlap * 0.35) + (effect_overlap * 0.20)
                    if combined < 0.64:
                        continue
                    smells.append(
                        ArchitecturalSmell(
                            id=f"duplicate-responsibility-{len(smells)+1}",
                            kind="duplicate_responsibility",
                            severity=SmellSeverity.LOW,
                            path=left.path,
                            symbol=left.name,
                            evidence=(
                                f"similarity {combined:.2f} between {left.qualified_name} and {right.qualified_name}",
                                f"name={name_score:.2f} docs/names={token_overlap:.2f} effects={effect_overlap:.2f}",
                            ),
                            graph_refs=(left.id, right.id),
                            suggested_tier=MutationTier.T2_REFACTOR,
                            proof_obligations=("caller_migration_plan", "behavior_fingerprint_equivalent"),
                            auto_fixable=False,
                        )
                    )
                    if len(smells) >= limit:
                        return smells
        return smells


def _tokenize(text: str) -> set[str]:
    raw = text.replace("_", " ").replace("-", " ").lower()
    return {part for part in raw.split() if len(part) >= 4}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))
