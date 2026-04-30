"""Architecture-smell detection for the ASA graph."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from core.architect.config import ASAConfig
from core.architect.duplicate_responsibility import DuplicateResponsibilityDetector
from core.architect.models import ArchitecturalSmell, ArchitectureGraph, MutationTier, SemanticSurface, SmellSeverity


class SmellDetector:
    """Detect structural, effect, governance, and coverage smells."""

    def __init__(self, config: ASAConfig | None = None):
        self.config = config or ASAConfig.from_env()
        self.duplicates = DuplicateResponsibilityDetector()

    def detect(self, graph: ArchitectureGraph) -> list[ArchitecturalSmell]:
        smells: list[ArchitecturalSmell] = []
        smells.extend(self._god_files(graph))
        smells.extend(self._god_classes(graph))
        smells.extend(self._fan_smells(graph))
        smells.extend(self._import_cycles(graph))
        smells.extend(self._effect_smells(graph))
        smells.extend(self._dead_symbols(graph))
        smells.extend(self._coverage_smells(graph))
        smells.extend(self._source_string_tests(graph))
        smells.extend(self._semantic_owner_smells(graph))
        smells.extend(self._stale_critical_markers(graph))
        smells.extend(self.duplicates.detect(graph, limit=40))
        return sorted(smells, key=lambda smell: (-int(smell.severity), smell.kind, smell.path, smell.symbol))

    def _god_files(self, graph: ArchitectureGraph) -> list[ArchitecturalSmell]:
        smells: list[ArchitecturalSmell] = []
        for node in graph.nodes.values():
            if node.kind != "file":
                continue
            line_count = int(node.metadata.get("line_count", 0))
            if line_count < self.config.god_file_lines:
                continue
            smells.append(
                ArchitecturalSmell(
                    id=f"god-file-{_slug(node.path)}",
                    kind="god_file",
                    severity=SmellSeverity.MEDIUM,
                    path=node.path,
                    evidence=(f"{line_count} lines exceeds {self.config.god_file_lines}",),
                    graph_refs=(node.id,),
                    suggested_tier=MutationTier.T2_REFACTOR,
                    proof_obligations=("staged_split_plan", "behavior_fingerprint_equivalent"),
                    auto_fixable=False,
                )
            )
        return smells

    def _god_classes(self, graph: ArchitectureGraph) -> list[ArchitecturalSmell]:
        smells: list[ArchitecturalSmell] = []
        method_counts: Counter[str] = Counter()
        for edge in graph.edges:
            if edge.kind == "defines" and edge.source.startswith("class:") and (edge.target.startswith("method:") or edge.target.startswith("function:")):
                method_counts[edge.source] += 1
        for node in graph.nodes.values():
            if node.kind != "class":
                continue
            line_count = int(node.metadata.get("line_count", 0))
            methods = method_counts.get(node.id, 0)
            if line_count < self.config.god_class_lines and methods < 20:
                continue
            smells.append(
                ArchitecturalSmell(
                    id=f"god-class-{_slug(node.qualified_name)}",
                    kind="god_class",
                    severity=SmellSeverity.MEDIUM,
                    path=node.path,
                    symbol=node.name,
                    evidence=(f"{line_count} lines, {methods} methods",),
                    graph_refs=(node.id,),
                    suggested_tier=MutationTier.T2_REFACTOR,
                    proof_obligations=("caller_migration_plan", "public_api_compatibility"),
                    auto_fixable=False,
                )
            )
        return smells

    def _fan_smells(self, graph: ArchitectureGraph) -> list[ArchitecturalSmell]:
        outgoing: Counter[str] = Counter()
        incoming: Counter[str] = Counter()
        for edge in graph.edges:
            if edge.kind != "imports":
                continue
            outgoing[edge.source] += 1
            incoming[edge.target] += 1
        smells: list[ArchitecturalSmell] = []
        for node_id, count in outgoing.items():
            path = node_id.removeprefix("file:")
            if count >= self.config.high_fan_out:
                smells.append(
                    ArchitecturalSmell(
                        id=f"high-fan-out-{_slug(path)}",
                        kind="high_fan_out",
                        severity=SmellSeverity.LOW,
                        path=path,
                        evidence=(f"{count} imports",),
                        graph_refs=(node_id,),
                        suggested_tier=MutationTier.T2_REFACTOR,
                        proof_obligations=("graph_proof",),
                        auto_fixable=False,
                    )
                )
        for target, count in incoming.items():
            if count >= self.config.high_fan_in:
                smells.append(
                    ArchitecturalSmell(
                        id=f"high-fan-in-{_slug(target)}",
                        kind="high_fan_in",
                        severity=SmellSeverity.INFO,
                        path=str(target),
                        evidence=(f"{count} importers",),
                        graph_refs=(target,),
                        suggested_tier=MutationTier.T2_REFACTOR,
                        proof_obligations=("public_api_compatibility",),
                        auto_fixable=False,
                    )
                )
        return smells

    def _import_cycles(self, graph: ArchitectureGraph) -> list[ArchitecturalSmell]:
        adjacency: dict[str, set[str]] = defaultdict(set)
        module_paths = {
            node.qualified_name: node.path
            for node in graph.nodes.values()
            if node.kind == "file"
        }
        path_by_file_id = {f"file:{node.path}": node.path for node in graph.nodes.values() if node.kind == "file"}
        for edge in graph.edges:
            if edge.kind != "imports":
                continue
            src_path = path_by_file_id.get(edge.source)
            if not src_path:
                continue
            target_path = _resolve_import_path(edge.target, module_paths)
            if target_path:
                adjacency[src_path].add(target_path)
        cycles = _strongly_connected(adjacency)
        smells: list[ArchitecturalSmell] = []
        for component in cycles:
            smells.append(
                ArchitecturalSmell(
                    id=f"import-cycle-{_slug('-'.join(sorted(component))[:80])}",
                    kind="import_cycle",
                    severity=SmellSeverity.HIGH,
                    path=sorted(component)[0],
                    evidence=(f"cycle across {len(component)} modules",),
                    graph_refs=tuple(sorted(component)),
                    suggested_tier=MutationTier.T2_REFACTOR,
                    proof_obligations=("shadow_imports_pass", "graph_rebuild_succeeds"),
                    auto_fixable=False,
                )
            )
        return smells

    def _effect_smells(self, graph: ArchitectureGraph) -> list[ArchitecturalSmell]:
        smells: list[ArchitecturalSmell] = []
        for node in graph.nodes.values():
            if node.kind != "file":
                continue
            effects = set(node.metadata.get("effects", ()))
            surfaces = set(graph.semantic_surfaces.get(node.path, ()))
            if "broad_exception" in effects:
                smells.append(self._effect_smell("broad_exception_cluster", node.path, node.id, "broad exception handler detected", SmellSeverity.HIGH, MutationTier.T1_CLEANUP, False))
            if "service_get_default_none" in effects and surfaces & {SemanticSurface.BOOT_RUNTIME_KERNEL, SemanticSurface.AUTHORITY_GOVERNANCE, SemanticSurface.MEMORY_WRITE_READ, SemanticSurface.STATE_MUTATION}:
                smells.append(self._effect_smell("critical_service_default_none", node.path, node.id, "ServiceContainer.get(... default=None) in a critical surface", SmellSeverity.HIGH, MutationTier.T4_GOVERNANCE_SENSITIVE, False))
            if "state_write" in effects and "gateway" not in node.path.lower():
                smells.append(self._effect_smell("state_write_bypass", node.path, node.id, "state write outside canonical gateway path", SmellSeverity.CRITICAL, MutationTier.T4_GOVERNANCE_SENSITIVE, False))
            if "memory_write" in effects and "gateway" not in node.path.lower() and not node.path.startswith("core/memory/"):
                smells.append(self._effect_smell("memory_write_bypass", node.path, node.id, "memory write outside memory owner surface", SmellSeverity.HIGH, MutationTier.T4_GOVERNANCE_SENSITIVE, False))
            if "tool_execution" in effects and "authority_call" not in effects and "capability_token" not in effects:
                smells.append(self._effect_smell("tool_authority_bypass", node.path, node.id, "tool execution without local authority/capability evidence", SmellSeverity.CRITICAL, MutationTier.T4_GOVERNANCE_SENSITIVE, False))
            if "subprocess" in effects and not _approved_subprocess_path(node.path):
                smells.append(self._effect_smell("subprocess_outside_approved_module", node.path, node.id, "direct subprocess usage outside approved execution modules", SmellSeverity.MEDIUM, MutationTier.T2_REFACTOR, False))
            if "dynamic_import" in effects or "string_dispatch" in effects:
                smells.append(self._effect_smell("dynamic_import_hazard", node.path, node.id, "dynamic import or string dispatch detected", SmellSeverity.MEDIUM, MutationTier.T2_REFACTOR, False))
            if ("compat" in node.path.lower() or "legacy" in node.path.lower() or "shim" in node.path.lower()) and not _has_receipt(graph, node.path):
                smells.append(self._effect_smell("compatibility_shim_without_receipt", node.path, node.id, "compatibility surface has no runtime receipt evidence", SmellSeverity.LOW, MutationTier.T1_CLEANUP, False))
            if "raw_asyncio_create_task" in effects:
                smells.append(self._effect_smell("untracked_background_task", node.path, node.id, "background task lacks TaskTracker lifecycle evidence", SmellSeverity.MEDIUM, MutationTier.T1_CLEANUP, False))
            if "direct_state_mutation" in effects and (surfaces & {SemanticSurface.STATE_MUTATION, SemanticSurface.CONSCIOUSNESS_SUBSTRATE}):
                smells.append(self._effect_smell("direct_protected_state_mutation", node.path, node.id, "direct mutation of protected state-like object", SmellSeverity.HIGH, MutationTier.T4_GOVERNANCE_SENSITIVE, False))
        return smells

    def _dead_symbols(self, graph: ArchitectureGraph) -> list[ArchitecturalSmell]:
        inbound: Counter[str] = Counter(edge.target for edge in graph.edges if edge.kind in {"calls", "inherits", "tests"})
        runtime_hits = _runtime_hit_index(graph)
        smells: list[ArchitecturalSmell] = []
        for node in graph.nodes.values():
            if node.kind not in {"function", "async_function", "class"}:
                continue
            if node.name.startswith("_") or node.name in {"main", "setup", "teardown"}:
                continue
            if inbound.get(node.name, 0) or inbound.get(node.qualified_name, 0) or inbound.get(node.id, 0):
                continue
            if self.config.is_protected(node.path) or node.path.startswith("tests/"):
                continue
            if _has_runtime_hit(runtime_hits, node):
                continue
            smells.append(
                ArchitecturalSmell(
                    id=f"dead-symbol-{_slug(node.qualified_name)}",
                    kind="dead_symbol_candidate",
                    severity=SmellSeverity.LOW,
                    path=node.path,
                    symbol=node.name,
                    evidence=("zero static inbound call/test/inheritance edges", "zero runtime receipt or coverage hits"),
                    graph_refs=(node.id,),
                    suggested_tier=MutationTier.T1_CLEANUP,
                    proof_obligations=("static_reachability_proof", "coverage_backed_deadness", "shadow_imports_pass", "quarantine_manifest"),
                    auto_fixable=False,
                )
            )
        return smells[:200]

    def _coverage_smells(self, graph: ArchitectureGraph) -> list[ArchitecturalSmell]:
        tested = {edge.target.removeprefix("file:") for edge in graph.edges if edge.kind == "tests"}
        receipt_paths = {receipt.path for receipt in graph.runtime_receipts if receipt.path}
        smells: list[ArchitecturalSmell] = []
        for node in graph.nodes.values():
            if node.kind != "file" or node.path.startswith(("tests/", "docs/", "scripts/")):
                continue
            if self.config.is_protected(node.path):
                continue
            if node.path in tested or node.path in receipt_paths:
                continue
            smells.append(
                ArchitecturalSmell(
                    id=f"missing-tests-receipts-{_slug(node.path)}",
                    kind="module_without_tests_or_receipts",
                    severity=SmellSeverity.INFO,
                    path=node.path,
                    evidence=("no mapped test edge and no runtime receipt",),
                    graph_refs=(node.id,),
                    suggested_tier=MutationTier.T1_CLEANUP,
                    proof_obligations=("add_behavior_test_before_refactor",),
                    auto_fixable=False,
                )
            )
        return smells[:200]

    def _source_string_tests(self, graph: ArchitectureGraph) -> list[ArchitecturalSmell]:
        smells: list[ArchitecturalSmell] = []
        for node in graph.nodes.values():
            if node.kind != "file" or not node.path.startswith("tests/"):
                continue
            effects = set(node.metadata.get("effects", ()))
            if "file_write" in effects:
                continue
            text_like = any("read_text" in edge.target or "getsource" in edge.target for edge in graph.edges if edge.source == node.id or edge.path == node.path)
            if text_like:
                smells.append(
                    ArchitecturalSmell(
                        id=f"source-string-test-{_slug(node.path)}",
                        kind="source_string_test",
                        severity=SmellSeverity.LOW,
                        path=node.path,
                        evidence=("test appears to inspect source text instead of behavior",),
                        graph_refs=(node.id,),
                        suggested_tier=MutationTier.T1_CLEANUP,
                        proof_obligations=("replace_with_behavior_assertion",),
                        auto_fixable=False,
                    )
                )
        return smells

    def _semantic_owner_smells(self, graph: ArchitectureGraph) -> list[ArchitecturalSmell]:
        smells: list[ArchitecturalSmell] = []
        owner_by_concern: dict[str, set[str]] = defaultdict(set)
        for domain in graph.ownership.values():
            owner_by_concern[domain.concern.lower()].add(domain.owner)
        for concern, owners in owner_by_concern.items():
            if len(owners) > 1:
                smells.append(
                    ArchitecturalSmell(
                        id=f"duplicate-ownership-{_slug(concern)}",
                        kind="duplicate_ownership",
                        severity=SmellSeverity.HIGH,
                        path="OWNERSHIP.md",
                        evidence=(f"{concern} has owners {sorted(owners)}",),
                        graph_refs=tuple(sorted(owners)),
                        suggested_tier=MutationTier.T4_GOVERNANCE_SENSITIVE,
                        proof_obligations=("ownership_migration_plan",),
                        auto_fixable=False,
                    )
                )
        surface_files: dict[SemanticSurface, list[str]] = defaultdict(list)
        for path, surfaces in graph.semantic_surfaces.items():
            for surface in surfaces:
                surface_files[surface].append(path)
        for surface, threshold, kind in (
            (SemanticSurface.LLM_MODEL_ROUTING, 8, "multiple_model_routing_owners"),
            (SemanticSurface.MEMORY_WRITE_READ, 30, "multiple_memory_owners"),
            (SemanticSurface.SELF_MODIFICATION, 12, "multiple_autonomy_repair_loops"),
        ):
            if len(surface_files.get(surface, ())) >= threshold:
                smells.append(
                    ArchitecturalSmell(
                        id=f"{kind}-{len(surface_files[surface])}",
                        kind=kind,
                        severity=SmellSeverity.MEDIUM,
                        path=surface_files[surface][0],
                        evidence=(f"{len(surface_files[surface])} files classified as {surface.value}",),
                        graph_refs=tuple(surface_files[surface][:20]),
                        suggested_tier=MutationTier.T4_GOVERNANCE_SENSITIVE,
                        proof_obligations=("canonical_owner_plan",),
                        auto_fixable=False,
                    )
                )
        return smells

    def _stale_critical_markers(self, graph: ArchitectureGraph) -> list[ArchitecturalSmell]:
        marker_a = "TO" + "DO"
        marker_b = "FIX" + "ME"
        smells: list[ArchitecturalSmell] = []
        for node in graph.nodes.values():
            if node.kind != "file":
                continue
            surfaces = set(graph.semantic_surfaces.get(node.path, ()))
            if not surfaces & {
                SemanticSurface.AUTHORITY_GOVERNANCE,
                SemanticSurface.BOOT_RUNTIME_KERNEL,
                SemanticSurface.MEMORY_WRITE_READ,
                SemanticSurface.STATE_MUTATION,
                SemanticSurface.SELF_MODIFICATION,
            }:
                continue
            path = self.config.repo_root / node.path
            try:
                source = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if marker_a in source or marker_b in source:
                smells.append(
                    ArchitecturalSmell(
                        id=f"stale-critical-marker-{_slug(node.path)}",
                        kind="stale_marker_in_critical_path",
                        severity=SmellSeverity.MEDIUM,
                        path=node.path,
                        evidence=("critical surface contains unresolved marker text",),
                        graph_refs=(node.id,),
                        suggested_tier=MutationTier.T4_GOVERNANCE_SENSITIVE,
                        proof_obligations=("proposal_only",),
                        auto_fixable=False,
                    )
                )
        return smells

    def _effect_smell(
        self,
        kind: str,
        path: str,
        ref: str,
        evidence: str,
        severity: SmellSeverity,
        tier: MutationTier,
        auto_fixable: bool,
    ) -> ArchitecturalSmell:
        return ArchitecturalSmell(
            id=f"{kind}-{_slug(path)}",
            kind=kind,
            severity=severity,
            path=path,
            evidence=(evidence,),
            graph_refs=(ref,),
            suggested_tier=tier,
            proof_obligations=("graph_rebuild_succeeds", "shadow_imports_pass"),
            auto_fixable=auto_fixable,
        )


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value.lower()).strip("-")[:96]


def _approved_subprocess_path(path: str) -> bool:
    return path.startswith(("core/runtime/", "core/discovery/", "core/self_modification/", "core/capability_engine.py", "scripts/", "tests/"))


def _has_receipt(graph: ArchitectureGraph, path: str) -> bool:
    return any(receipt.path == path or path in str(receipt.payload) for receipt in graph.runtime_receipts)


def _runtime_hit_index(graph: ArchitectureGraph) -> dict[str, set[str]]:
    hits: dict[str, set[str]] = {"paths": set(), "symbols": set(), "payload_text": set()}
    for receipt in graph.runtime_receipts:
        if receipt.path:
            hits["paths"].add(receipt.path)
        symbol = receipt.payload.get("symbol") or receipt.payload.get("qualified_name") or receipt.payload.get("function")
        if isinstance(symbol, str) and symbol:
            hits["symbols"].add(symbol)
        payload_text = str(receipt.payload)
        if len(payload_text) <= 12000:
            hits["payload_text"].add(payload_text)
    return hits


def _has_runtime_hit(hits: dict[str, set[str]], node) -> bool:
    if node.path in hits["paths"]:
        return True
    if node.qualified_name in hits["symbols"] or node.name in hits["symbols"]:
        return True
    needles = (node.qualified_name, node.name, node.path)
    return any(any(needle and needle in payload for needle in needles) for payload in hits["payload_text"])


def _resolve_import_path(target: str, module_paths: dict[str, str]) -> str:
    parts = target.split(".")
    for end in range(len(parts), 0, -1):
        module = ".".join(parts[:end])
        if module in module_paths:
            return module_paths[module]
    return ""


def _strongly_connected(adjacency: dict[str, set[str]]) -> list[set[str]]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    components: list[set[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in adjacency.get(node, set()):
            if target not in indices:
                visit(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[target])
        if lowlinks[node] == indices[node]:
            component: set[str] = set()
            while stack:
                item = stack.pop()
                on_stack.remove(item)
                component.add(item)
                if item == node:
                    break
            if len(component) > 1:
                components.append(component)

    for node in sorted(adjacency):
        if node not in indices:
            visit(node)
    return components
