"""Multi-layered live architecture graph builder."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.architect.config import ASAConfig
from core.architect.errors import GraphBuildError
from core.architect.models import (
    ArchitectureEdge,
    ArchitectureGraph,
    ArchitectureNode,
    OwnershipDomain,
    RuntimeReceipt,
    SemanticSurface,
)
from core.architect.semantic_classifier import SemanticClassifier
from core.runtime.atomic_writer import atomic_write_text


class LiveArchitectureGraphBuilder:
    """Build Aura's structural, effect, semantic, ownership, and receipt graph."""

    def __init__(self, config: ASAConfig | None = None):
        self.config = config or ASAConfig.from_env()
        self.classifier = SemanticClassifier()

    def build(self, *, persist: bool = True) -> ArchitectureGraph:
        root = self.config.repo_root
        if not root.exists():
            raise GraphBuildError(f"repo root does not exist: {root}")
        graph = ArchitectureGraph(root=str(root))
        parse_errors: list[str] = []
        py_files = self._python_files(root)
        for path in py_files:
            rel = path.relative_to(root).as_posix()
            try:
                source = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                parse_errors.append(f"{rel}:decode:{exc}")
                continue
            except OSError as exc:
                parse_errors.append(f"{rel}:read:{exc}")
                continue
            try:
                tree = ast.parse(source, filename=rel)
            except SyntaxError as exc:
                parse_errors.append(f"{rel}:syntax:{exc.lineno}:{exc.msg}")
                self._add_module_node(graph, rel, source, effects={"syntax_error"}, names=set(), parse_error=str(exc))
                continue
            self._index_file(graph, rel, source, tree)
        self._map_tests(graph)
        graph.ownership = self._load_ownership(root)
        graph.runtime_receipts = self._load_receipts(root)
        self._finalize_metrics(graph, parse_errors)
        if persist:
            self.persist(graph)
        return graph

    def persist(self, graph: ArchitectureGraph) -> None:
        artifact_root = self.config.artifacts
        reports = artifact_root / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        graph.persist_json(artifact_root / "architecture_graph.json")
        self._persist_jsonl(graph, artifact_root / "architecture_graph.jsonl")
        self._persist_sqlite(graph, artifact_root / "architecture_graph.sqlite")
        report_path = reports / f"architecture-graph-{int(graph.created_at)}.json"
        atomic_write_text(report_path, json.dumps({"metrics": graph.metrics}, indent=2, sort_keys=True, default=str))

    def _python_files(self, root: Path) -> list[Path]:
        files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            current = Path(dirpath)
            try:
                rel_dir = current.relative_to(root).as_posix()
            except ValueError:
                rel_dir = current.name
            dirnames[:] = [
                dirname for dirname in dirnames
                if not self.config.is_excluded(dirname)
                and not self.config.is_excluded(f"{rel_dir}/{dirname}" if rel_dir != "." else dirname)
            ]
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                path = current / filename
                rel = path.relative_to(root).as_posix()
                if self.config.is_excluded(rel):
                    continue
                files.append(path)
        return sorted(files)

    def _index_file(self, graph: ArchitectureGraph, rel: str, source: str, tree: ast.AST) -> None:
        visitor = _FileVisitor(rel, self.classifier.module_name_for_path(rel))
        visitor.visit(tree)
        self._add_module_node(graph, rel, source, effects=visitor.file_effects, names=visitor.names)
        for node in visitor.nodes:
            graph.add_node(node)
        for edge in visitor.edges:
            graph.add_edge(edge)
        surfaces = self.classifier.classify_path(rel, names=visitor.names, effects=visitor.file_effects)
        graph.semantic_surfaces[rel] = surfaces
        module_id = f"file:{rel}"
        for surface in surfaces:
            graph.add_edge(
                ArchitectureEdge(
                    source=module_id,
                    target=f"surface:{surface.value}",
                    kind="semantic_surface",
                    path=rel,
                )
            )

    def _add_module_node(
        self,
        graph: ArchitectureGraph,
        rel: str,
        source: str,
        *,
        effects: set[str],
        names: set[str],
        parse_error: str = "",
    ) -> None:
        lines = source.splitlines()
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        graph.add_node(
            ArchitectureNode(
                id=f"file:{rel}",
                kind="file",
                name=Path(rel).name,
                path=rel,
                line_start=1,
                line_end=max(1, len(lines)),
                qualified_name=self.classifier.module_name_for_path(rel),
                metadata={
                    "line_count": len(lines),
                    "hash": digest,
                    "effects": tuple(sorted(effects)),
                    "names": tuple(sorted(names)),
                    "protected": self.config.is_protected(rel),
                    "sealed": self.config.is_sealed(rel),
                    "parse_error": parse_error,
                },
            )
        )

    def _map_tests(self, graph: ArchitectureGraph) -> None:
        files = [node for node in graph.nodes.values() if node.kind == "file"]
        module_paths = {node.path for node in files if not node.path.startswith("tests/")}
        for node in files:
            if not node.path.startswith("tests/"):
                continue
            base = Path(node.path).stem
            candidates = []
            if base.startswith("test_"):
                needle = base.removeprefix("test_")
                candidates = [path for path in module_paths if needle in Path(path).stem or needle in path]
            for target in candidates[:10]:
                graph.add_edge(ArchitectureEdge(source=f"file:{node.path}", target=f"file:{target}", kind="tests", path=node.path))

    def _load_ownership(self, root: Path) -> dict[str, OwnershipDomain]:
        ownership_path = root / "OWNERSHIP.md"
        if not ownership_path.exists():
            return {}
        rows: dict[str, OwnershipDomain] = {}
        try:
            lines = ownership_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return rows
        for line in lines:
            if not line.startswith("|") or "`" not in line:
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) < 3 or cells[0].lower() in {"concern", "domain"}:
                continue
            file_cell = cells[-1]
            file_path = _strip_ticks(file_cell)
            if not file_path.endswith(".py"):
                continue
            concern = _strip_ticks(cells[0])
            owner = _strip_ticks(cells[1])
            role = _strip_ticks(cells[2]) if len(cells) > 2 else ""
            rows[file_path] = OwnershipDomain(concern=concern, owner=owner, role=role, file=file_path, source="OWNERSHIP.md")
        return rows

    def _load_receipts(self, root: Path) -> list[RuntimeReceipt]:
        local_receipts: list[RuntimeReceipt] = []
        external_receipts: list[RuntimeReceipt] = []
        local_candidates = [
            root / ".aura_architect" / "receipts",
            root / "logs" / "receipts",
            root / "data" / "receipts",
        ]
        external_candidates = [
            Path.home() / ".aura" / "receipts",
            Path.home() / ".aura" / "logs" / "receipts",
        ]
        for directory in local_candidates:
            if not directory.exists() or not directory.is_dir():
                continue
            local_receipts.extend(self._load_jsonl_receipts(directory))
            local_receipts.extend(self._load_json_receipts(directory))
        for directory in external_candidates:
            if not directory.exists() or not directory.is_dir():
                continue
            external_receipts.extend(self._load_jsonl_receipts(directory))
            external_receipts.extend(self._load_json_receipts(directory))
        local_receipts.extend(self._load_trace_receipts(root / "data" / "traces"))
        local_receipts.extend(self._load_life_trace_receipts(root / "data" / "life_trace.sqlite3"))
        local_receipts.extend(self._load_coverage_receipts(root))
        external_receipts.extend(self._load_life_trace_receipts(Path.home() / ".aura" / "life_trace.sqlite3"))
        local_receipts.sort(key=lambda receipt: receipt.timestamp, reverse=True)
        external_receipts.sort(key=lambda receipt: receipt.timestamp, reverse=True)
        return (local_receipts + external_receipts)[: self.config.runtime_receipt_limit]

    def _load_jsonl_receipts(self, directory: Path) -> list[RuntimeReceipt]:
        receipts: list[RuntimeReceipt] = []
        files = sorted(directory.rglob("*.jsonl"), key=_safe_mtime, reverse=True)[:200]
        for path in files:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines[-500:]:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                receipt = self._receipt_from_payload(payload, source_hint=path.stem)
                if receipt is not None:
                    receipts.append(receipt)
        return receipts

    def _load_json_receipts(self, directory: Path) -> list[RuntimeReceipt]:
        receipts: list[RuntimeReceipt] = []
        files = sorted(directory.rglob("*.json"), key=_safe_mtime, reverse=True)[: self.config.runtime_receipt_limit]
        for path in files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and isinstance(payload.get("payload"), dict):
                payload = dict(payload["payload"])
            receipt = self._receipt_from_payload(payload, source_hint=path.stem)
            if receipt is not None:
                receipts.append(receipt)
        return receipts

    def _load_trace_receipts(self, directory: Path) -> list[RuntimeReceipt]:
        if not directory.exists() or not directory.is_dir():
            return []
        receipts: list[RuntimeReceipt] = []
        files = sorted(directory.glob("*.jsonl"), key=_safe_mtime, reverse=True)[:100]
        for path in files:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines[-200:]:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                receipt = self._receipt_from_payload(payload, source_hint=f"trace:{path.stem}")
                if receipt is not None:
                    receipts.append(receipt)
        return receipts

    def _load_life_trace_receipts(self, db_path: Path) -> list[RuntimeReceipt]:
        if not db_path.exists():
            return []
        receipts: list[RuntimeReceipt] = []
        try:
            conn = sqlite3.connect(db_path)
        except sqlite3.Error:
            return receipts
        try:
            rows = conn.execute(
                "SELECT event_type, payload, timestamp FROM life_trace ORDER BY timestamp DESC LIMIT ?",
                (min(self.config.runtime_receipt_limit, 500),),
            ).fetchall()
        except sqlite3.Error:
            return receipts
        finally:
            conn.close()
        for event_type, payload_text, timestamp in rows:
            try:
                payload = json.loads(payload_text)
            except (TypeError, json.JSONDecodeError):
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            payload.setdefault("kind", str(event_type or "life_trace"))
            payload.setdefault("timestamp", float(timestamp or time.time()))
            receipt = self._receipt_from_payload(payload, source_hint="life_trace")
            if receipt is not None:
                receipts.append(receipt)
        return receipts

    def _load_coverage_receipts(self, root: Path) -> list[RuntimeReceipt]:
        receipts: list[RuntimeReceipt] = []
        receipts.extend(self._coverage_json_receipts(root / "coverage.json"))
        receipts.extend(self._coverage_sqlite_receipts(root / ".coverage"))
        receipts.extend(self._symbol_hit_receipts(root / ".aura_architect" / "telemetry" / "symbol_hits.jsonl"))
        return receipts[: self.config.coverage_hit_limit]

    def _coverage_json_receipts(self, path: Path) -> list[RuntimeReceipt]:
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        files = payload.get("files") if isinstance(payload, dict) else None
        if not isinstance(files, dict):
            return []
        receipts: list[RuntimeReceipt] = []
        for file_path, file_payload in list(files.items())[: self.config.coverage_hit_limit]:
            rel = self._normalize_receipt_path(file_path)
            receipts.append(RuntimeReceipt(source="coverage.json", path=rel, timestamp=_safe_mtime(path), kind="coverage", payload={"path": rel, "coverage": file_payload}))
        return receipts

    def _coverage_sqlite_receipts(self, path: Path) -> list[RuntimeReceipt]:
        if not path.exists():
            return []
        try:
            conn = sqlite3.connect(path)
        except sqlite3.Error:
            return []
        try:
            rows = conn.execute("SELECT path FROM file LIMIT ?", (self.config.coverage_hit_limit,)).fetchall()
        except sqlite3.Error:
            return []
        finally:
            conn.close()
        receipts: list[RuntimeReceipt] = []
        for (file_path,) in rows:
            rel = self._normalize_receipt_path(str(file_path))
            receipts.append(RuntimeReceipt(source=".coverage", path=rel, timestamp=_safe_mtime(path), kind="coverage", payload={"path": rel}))
        return receipts

    def _symbol_hit_receipts(self, path: Path) -> list[RuntimeReceipt]:
        if not path.exists():
            return []
        receipts: list[RuntimeReceipt] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return receipts
        for line in lines[-self.config.coverage_hit_limit:]:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            receipt = self._receipt_from_payload(payload, source_hint="symbol_hits")
            if receipt is not None:
                receipts.append(receipt)
        return receipts

    def _receipt_from_payload(self, payload: Any, *, source_hint: str) -> RuntimeReceipt | None:
        if not isinstance(payload, dict):
            return None
        timestamp = payload.get("created_at", payload.get("timestamp", payload.get("when", time.time())))
        try:
            timestamp_f = float(timestamp)
        except (TypeError, ValueError):
            timestamp_f = time.time()
        return RuntimeReceipt(
            source=str(payload.get("source", payload.get("event", source_hint))),
            path=self._normalize_receipt_path(_path_from_payload(payload)),
            timestamp=timestamp_f,
            kind=str(payload.get("kind", payload.get("type", payload.get("event_type", "runtime")))),
            payload=payload,
        )

    def _normalize_receipt_path(self, value: str) -> str:
        raw = str(value or "").replace("\\", "/")
        if not raw:
            return ""
        path = Path(raw)
        if path.is_absolute():
            try:
                return path.resolve().relative_to(self.config.repo_root).as_posix()
            except ValueError:
                return raw
        return raw[2:] if raw.startswith("./") else raw

    def _finalize_metrics(self, graph: ArchitectureGraph, parse_errors: list[str]) -> None:
        kind_counts = Counter(node.kind for node in graph.nodes.values())
        edge_counts = Counter(edge.kind for edge in graph.edges)
        surface_counts = Counter(surface.value for surfaces in graph.semantic_surfaces.values() for surface in surfaces)
        effect_counts = Counter(effect for node in graph.nodes.values() for effect in node.metadata.get("effects", ()))
        receipt_kinds = Counter(receipt.kind for receipt in graph.runtime_receipts)
        receipt_paths = {receipt.path for receipt in graph.runtime_receipts if receipt.path}
        graph.metrics.update(
            {
                "files": kind_counts.get("file", 0),
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
                "node_kinds": dict(kind_counts),
                "edge_kinds": dict(edge_counts),
                "semantic_surfaces": dict(surface_counts),
                "effects": dict(effect_counts),
                "parse_errors": parse_errors,
                "runtime_receipts": len(graph.runtime_receipts),
                "runtime_receipts_by_kind": dict(receipt_kinds),
                "runtime_receipt_paths": len(receipt_paths),
            }
        )

    def _persist_jsonl(self, graph: ArchitectureGraph, path: Path) -> None:
        lines: list[str] = []
        for node in graph.nodes.values():
            lines.append(json.dumps({"type": "node", "payload": asdict(node)}, sort_keys=True, default=str))
        for edge in graph.edges:
            lines.append(json.dumps({"type": "edge", "payload": asdict(edge)}, sort_keys=True, default=str))
        atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))

    def _persist_sqlite(self, graph: ArchitectureGraph, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY, kind TEXT, path TEXT, payload TEXT);
                CREATE TABLE IF NOT EXISTS edges (idx INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT, target TEXT, kind TEXT, path TEXT, payload TEXT);
                DELETE FROM nodes;
                DELETE FROM edges;
                """
            )
            conn.executemany(
                "INSERT INTO nodes (id, kind, path, payload) VALUES (?, ?, ?, ?)",
                [
                    (node.id, node.kind, node.path, json.dumps(asdict(node), sort_keys=True, default=str))
                    for node in graph.nodes.values()
                ],
            )
            conn.executemany(
                "INSERT INTO edges (source, target, kind, path, payload) VALUES (?, ?, ?, ?, ?)",
                [
                    (edge.source, edge.target, edge.kind, edge.path, json.dumps(asdict(edge), sort_keys=True, default=str))
                    for edge in graph.edges
                ],
            )
            conn.commit()
        finally:
            conn.close()


class _FileVisitor(ast.NodeVisitor):
    def __init__(self, rel: str, module_name: str):
        self.rel = rel
        self.module_name = module_name
        self.nodes: list[ArchitectureNode] = []
        self.edges: list[ArchitectureEdge] = []
        self.names: set[str] = set()
        self.file_effects: set[str] = set()
        self.scope_stack: list[str] = [f"file:{rel}"]
        self.class_stack: list[str] = []

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            imported = alias.name
            self.names.add(alias.asname or imported.split(".")[0])
            self.edges.append(ArchitectureEdge(source=f"file:{self.rel}", target=imported, kind="imports", path=self.rel, line=node.lineno))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        module = node.module or ""
        for alias in node.names:
            imported = f"{module}.{alias.name}" if module else alias.name
            self.names.add(alias.asname or alias.name)
            self.edges.append(ArchitectureEdge(source=f"file:{self.rel}", target=imported, kind="imports", path=self.rel, line=node.lineno))
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        qual = f"{self.module_name}.{node.name}"
        self.names.add(node.name)
        metadata = self._common_metadata(node)
        metadata["decorators"] = tuple(self._name(dec) for dec in node.decorator_list if self._name(dec))
        metadata["fingerprint"] = self._fingerprint(node)
        metadata["docstring"] = ast.get_docstring(node) or ""
        arch_node = ArchitectureNode(
            id=f"class:{qual}",
            kind="class",
            name=node.name,
            path=self.rel,
            line_start=node.lineno,
            line_end=getattr(node, "end_lineno", node.lineno) or node.lineno,
            qualified_name=qual,
            metadata=metadata,
        )
        self.nodes.append(arch_node)
        self.edges.append(ArchitectureEdge(source=f"file:{self.rel}", target=arch_node.id, kind="defines", path=self.rel, line=node.lineno))
        for base in node.bases:
            base_name = self._name(base)
            if base_name:
                self.edges.append(ArchitectureEdge(source=arch_node.id, target=base_name, kind="inherits", path=self.rel, line=node.lineno))
        self.scope_stack.append(arch_node.id)
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._visit_function(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._visit_function(node, "async_function")

    def visit_Assign(self, node: ast.Assign) -> Any:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                self.names.add(target.id)
                self.nodes.append(
                    ArchitectureNode(
                        id=f"constant:{self.module_name}.{target.id}",
                        kind="constant",
                        name=target.id,
                        path=self.rel,
                        line_start=node.lineno,
                        line_end=getattr(node, "end_lineno", node.lineno) or node.lineno,
                        qualified_name=f"{self.module_name}.{target.id}",
                        metadata={"value_type": type(node.value).__name__},
                    )
                )
            if isinstance(target, ast.Attribute):
                target_text = self._name(target) or ""
                if any(token in target_text.lower() for token in ("state", "aura_state", "cognition", "memory")):
                    self.file_effects.add("direct_state_mutation")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        if isinstance(node.target, ast.Name) and node.target.id.isupper():
            self.names.add(node.target.id)
        if isinstance(node.target, ast.Attribute):
            target_text = self._name(node.target) or ""
            if any(token in target_text.lower() for token in ("state", "aura_state", "cognition", "memory")):
                self.file_effects.add("direct_state_mutation")
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> Any:
        if node.type is None:
            self.file_effects.add("broad_exception")
        else:
            name = self._name(node.type) or ""
            if name in {"Exception", "BaseException"}:
                self.file_effects.add("broad_exception")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        call = self._name(node.func)
        if call:
            self.edges.append(
                ArchitectureEdge(
                    source=self.scope_stack[-1],
                    target=call,
                    kind="calls",
                    path=self.rel,
                    line=getattr(node, "lineno", 0),
                )
            )
            self._add_call_effects(call, node)
        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, kind: str) -> None:
        owner = ".".join(self.class_stack + [node.name]) if self.class_stack else node.name
        qual = f"{self.module_name}.{owner}"
        effects = set(self._effects_in(node))
        self.file_effects.update(effects)
        metadata = self._common_metadata(node)
        metadata["effects"] = tuple(sorted(effects))
        metadata["decorators"] = tuple(self._name(dec) for dec in node.decorator_list if self._name(dec))
        metadata["fingerprint"] = self._fingerprint(node)
        metadata["docstring"] = ast.get_docstring(node) or ""
        metadata["args"] = tuple(arg.arg for arg in node.args.args)
        node_kind = "method" if self.class_stack else kind
        arch_node = ArchitectureNode(
            id=f"{node_kind}:{qual}",
            kind=node_kind,
            name=node.name,
            path=self.rel,
            line_start=node.lineno,
            line_end=getattr(node, "end_lineno", node.lineno) or node.lineno,
            qualified_name=qual,
            metadata=metadata,
        )
        self.names.add(node.name)
        self.nodes.append(arch_node)
        self.edges.append(ArchitectureEdge(source=self.scope_stack[-1], target=arch_node.id, kind="defines", path=self.rel, line=node.lineno))
        self.scope_stack.append(arch_node.id)
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call = self._name(child.func)
                if call:
                    self.edges.append(ArchitectureEdge(source=arch_node.id, target=call, kind="calls", path=self.rel, line=getattr(child, "lineno", node.lineno)))
                    self._add_call_effects(call, child)
        self.generic_visit(node)
        self.scope_stack.pop()

    def _common_metadata(self, node: ast.AST) -> dict[str, Any]:
        return {
            "line_count": max(1, (getattr(node, "end_lineno", getattr(node, "lineno", 0)) or 0) - (getattr(node, "lineno", 0) or 0) + 1),
        }

    def _effects_in(self, node: ast.AST) -> set[str]:
        effects: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call = self._name(child.func) or ""
                effects.update(self._classify_call(call, child))
            elif isinstance(child, ast.ExceptHandler):
                name = self._name(child.type) if child.type is not None else ""
                if name in {"", "Exception", "BaseException"}:
                    effects.add("broad_exception")
        return effects

    def _add_call_effects(self, call: str, node: ast.Call) -> None:
        self.file_effects.update(self._classify_call(call, node))

    def _classify_call(self, call: str, node: ast.Call) -> set[str]:
        lowered = call.lower()
        effects: set[str] = set()
        if "import_module" in lowered or call == "__import__":
            effects.add("dynamic_import")
        if call.endswith("getattr") or call == "getattr":
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                effects.add("string_dispatch")
        if "servicecontainer.get" in lowered or lowered.endswith(".get"):
            effects.add("service_container_get")
            if any(kw.arg == "default" and isinstance(kw.value, ast.Constant) and kw.value.value is None for kw in node.keywords):
                effects.add("service_get_default_none")
        if "servicecontainer.register" in lowered or lowered.endswith(".register") or "register_instance" in lowered:
            effects.add("service_container_register")
        if lowered.endswith(".publish") or lowered.endswith(".publish_threadsafe"):
            effects.add("event_bus_emit")
        if lowered.endswith(".subscribe"):
            effects.add("event_bus_subscribe")
        if any(token in lowered for token in ("route", ".get", ".post", ".put", ".delete")) and self.rel.startswith(("server", "api", "interface", "core/server")):
            effects.add("api_route")
        if any(token in lowered for token in ("write_text", "write_bytes", "atomic_write", "os.replace", "shutil.move", "open")):
            effects.add("file_write")
        if any(token in lowered for token in ("sqlite", ".execute", ".executemany")):
            effects.add("database_write")
        if any(token in lowered for token in ("subprocess.", "os.system", "create_subprocess_exec", "create_subprocess_shell")):
            effects.add("subprocess")
        if any(token in lowered for token in ("requests.", "httpx.", "urllib.", "urlopen")):
            effects.add("network")
        if any(token in lowered for token in ("llm", "generate", "completion", "chat", "mlx", "router")):
            effects.add("llm_call")
        if any(token in lowered for token in ("execute_tool", "tool_execution", "capabilityengine.execute", "run_skill")):
            effects.add("tool_execution")
        if any(token in lowered for token in ("capability", "token", "issue", "verify")):
            effects.add("capability_token")
        if any(token in lowered for token in ("unifiedwill", "authoritygateway", "constitution", ".decide", ".approve")):
            effects.add("authority_call")
        if "create_task" in lowered:
            effects.add("background_task")
            if "get_task_tracker" not in lowered:
                effects.add("raw_asyncio_create_task")
        if any(token in lowered for token in ("memory", "store_episode", "write_memory", "remember")):
            effects.add("memory_write")
        if any(token in lowered for token in ("state_repo", "state_gateway", "state.write", "mutate_state")):
            effects.add("state_write")
        return effects

    def _fingerprint(self, node: ast.AST) -> str:
        clone = ast.fix_missing_locations(node)
        dumped = ast.dump(clone, annotate_fields=False, include_attributes=False)
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

    def _name(self, node: ast.AST | None) -> str:
        if node is None:
            return ""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = self._name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        if isinstance(node, ast.Call):
            return self._name(node.func)
        if isinstance(node, ast.Constant):
            return str(node.value)
        if isinstance(node, ast.Subscript):
            return self._name(node.value)
        return ""


def _strip_ticks(value: str) -> str:
    return value.strip().strip("`").strip()


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _path_from_payload(payload: dict[str, Any]) -> str:
    direct_keys = (
        "path",
        "file",
        "file_path",
        "source_path",
        "target_path",
        "target_module",
        "module",
    )
    for key in direct_keys:
        value = payload.get(key)
        if isinstance(value, str) and _looks_like_source_path(value):
            return value
    for container_key in ("metadata", "context", "action_taken", "result", "memory_update", "verification_evidence"):
        nested = payload.get(container_key)
        if isinstance(nested, dict):
            found = _path_from_payload(nested)
            if found:
                return found
    target = payload.get("target")
    if isinstance(target, str) and _looks_like_source_path(target):
        return target
    return ""


def _looks_like_source_path(value: str) -> bool:
    lowered = value.lower()
    return lowered.endswith(".py") or "/" in lowered or lowered.startswith("core.") or lowered.startswith("tests.")
