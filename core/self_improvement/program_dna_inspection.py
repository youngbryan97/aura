"""Reading an artifact before saying anything about it.

An app bundle, a source tree, a Python file, a manifest — each read for what it
actually contains rather than what its name suggests. Every walk is bounded,
because an unbounded one on a source tree is how a reconstruction spends its
whole budget listing files.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation only; that module imports this one
    from .program_dna import ProgramDNAEvidence

import ast
import importlib
import json
import os
import plistlib
import tomllib
from pathlib import Path
from typing import Any


class _LooksAtWhatIsThere:
    """Lifted whole from ProgramDNAReconstructionEngine; see program_dna.py."""

    def _inspect_path(self, path: Path) -> list[ProgramDNAEvidence]:
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.

        if not path.exists():
            raise FileNotFoundError(str(path))
        if path.is_dir():
            if path.suffix == ".app":
                return self._inspect_app_bundle(path)
            return self._inspect_source_tree(path)
        return self._inspect_file(path)

    def _inspect_app_bundle(self, path: Path) -> list[ProgramDNAEvidence]:
        from .program_dna import (
            ProgramDNAEvidence,
        )

        evidence = [
            ProgramDNAEvidence(
                kind="app_bundle",
                source=str(path),
                summary=f"macOS app bundle detected: {path.name}",
                confidence=0.86,
                details={"bundle_name": path.name},
            )
        ]
        plist_path = path / "Contents" / "Info.plist"
        if plist_path.exists():
            data = plistlib.loads(plist_path.read_bytes())
            keys = {
                key: data.get(key)
                for key in (
                    "CFBundleName",
                    "CFBundleIdentifier",
                    "CFBundleExecutable",
                    "CFBundleShortVersionString",
                    "NSMicrophoneUsageDescription",
                    "NSCameraUsageDescription",
                    "NSAppleEventsUsageDescription",
                )
                if key in data
            }
            evidence.append(
                ProgramDNAEvidence(
                    kind="app_metadata",
                    source=str(plist_path),
                    summary=f"Bundle metadata exposes {len(keys)} operational identifiers/permission hints.",
                    confidence=0.90,
                    details=keys,
                    sha256=self._sha256(plist_path),
                )
            )
        return evidence

    def _inspect_source_tree(self, root: Path) -> list[ProgramDNAEvidence]:
        from .program_dna import (
            MANIFEST_NAMES,
            ProgramDNAEvidence,
        )

        counts: dict[str, int] = {}
        manifests: list[str] = []
        public_symbols: list[str] = []
        sampled_files = 0
        for file_path in self._walk_limited(root, max_files=400):
            sampled_files += 1
            rel = str(file_path.relative_to(root))
            if file_path.name in MANIFEST_NAMES:
                manifests.append(rel)
            counts[file_path.suffix or "<none>"] = counts.get(file_path.suffix or "<none>", 0) + 1
            if file_path.suffix == ".py" and len(public_symbols) < 80:
                public_symbols.extend(self._python_public_symbols(file_path)[:20])

        details = {
            "root": str(root),
            "sampled_files": sampled_files,
            "extension_counts": counts,
            "manifests": manifests[:30],
            "public_symbols": public_symbols[:80],
        }
        return [
            ProgramDNAEvidence(
                kind="source_tree",
                source=str(root),
                summary=(
                    f"Readable source tree with {sampled_files} sampled files, "
                    f"{len(manifests)} manifest(s), and {len(public_symbols[:80])} public symbol hints."
                ),
                confidence=0.92,
                details=details,
            )
        ]

    def _inspect_file(self, path: Path) -> list[ProgramDNAEvidence]:
        from .program_dna import (
            SOURCE_EXTENSIONS,
            ProgramDNAEvidence,
        )

        suffix = path.suffix.lower()
        if suffix == ".py":
            return [self._inspect_python_file(path)]
        if path.name == "pyproject.toml" or suffix == ".toml":
            return [self._inspect_toml_manifest(path)]
        if path.name == "package.json" or suffix == ".json":
            return [self._inspect_json_manifest(path)]
        if suffix in SOURCE_EXTENSIONS:
            text = path.read_text(encoding="utf-8", errors="replace")
            return [
                ProgramDNAEvidence(
                    kind="source_file",
                    source=str(path),
                    summary=f"Readable source file: {path.name} ({len(text.splitlines())} lines).",
                    confidence=0.78,
                    details={"suffix": suffix, "lines": len(text.splitlines())},
                    sha256=self._sha256(path),
                )
            ]
        return [
            ProgramDNAEvidence(
                kind="file_signature",
                source=str(path),
                summary=f"File signature only: {path.name} ({path.stat().st_size} bytes).",
                confidence=0.35,
                details={"suffix": suffix, "bytes": path.stat().st_size},
                sha256=self._sha256(path),
            )
        ]

    def _inspect_python_file(self, path: Path) -> ProgramDNAEvidence:
        from .program_dna import (
            ProgramDNAEvidence,
        )

        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and not node.name.startswith("_")
        ]
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module.split(".")[0])
        return ProgramDNAEvidence(
            kind="python_api",
            source=str(path),
            summary=f"Python API hints: {len(classes)} class(es), {len(functions)} public function(s).",
            confidence=0.88,
            details={
                "classes": classes[:80],
                "functions": functions[:120],
                "imports": sorted(set(imports))[:80],
                "module_docstring": ast.get_docstring(tree) or "",
            },
            sha256=self._sha256(path),
        )

    def _inspect_toml_manifest(self, path: Path) -> ProgramDNAEvidence:
        from .program_dna import (
            ProgramDNAEvidence,
        )

        data = tomllib.loads(path.read_text(encoding="utf-8"))
        project = data.get("project") if isinstance(data, dict) else {}
        tool = data.get("tool") if isinstance(data, dict) else {}
        details = {
            "name": project.get("name") if isinstance(project, dict) else None,
            "dependencies": project.get("dependencies", [])[:80] if isinstance(project, dict) else [],
            "tool_sections": sorted(tool)[:40] if isinstance(tool, dict) else [],
        }
        return ProgramDNAEvidence(
            kind="manifest",
            source=str(path),
            summary=f"TOML manifest found for {details.get('name') or path.parent.name}.",
            confidence=0.82,
            details=details,
            sha256=self._sha256(path),
        )

    def _inspect_json_manifest(self, path: Path) -> ProgramDNAEvidence:
        from .program_dna import (
            ProgramDNAEvidence,
        )

        data = json.loads(path.read_text(encoding="utf-8"))
        details = {}
        if isinstance(data, dict):
            details = {
                "name": data.get("name"),
                "version": data.get("version"),
                "scripts": sorted((data.get("scripts") or {}).keys())[:40]
                if isinstance(data.get("scripts"), dict)
                else [],
                "dependencies": sorted((data.get("dependencies") or {}).keys())[:80]
                if isinstance(data.get("dependencies"), dict)
                else [],
            }
        return ProgramDNAEvidence(
            kind="manifest",
            source=str(path),
            summary=f"JSON manifest found for {details.get('name') or path.parent.name}.",
            confidence=0.80,
            details=details,
            sha256=self._sha256(path),
        )

    def _surface_entries(
        self,
        evidence: list[ProgramDNAEvidence],
        *,
        kinds: set[str],
        markers: tuple[str, ...],
    ) -> list[dict[str, Any]]:

        surfaces: list[dict[str, Any]] = []
        for item in evidence:
            text = f"{item.summary} {json.dumps(item.details, sort_keys=True)}".lower()
            if item.kind not in kinds and not any(marker in text for marker in markers):
                continue
            surfaces.append(
                {
                    "category": item.kind,
                    "source": item.source,
                    "summary": item.summary,
                    "confidence": item.confidence,
                    "observed": item.kind in kinds,
                    "markers": [marker for marker in markers if marker in text][:8],
                }
            )
        return surfaces[:40]

    def _collect_live_host_snapshot(self) -> list[ProgramDNAEvidence]:
        """Collect a bounded local host snapshot for explicit defensive study.

        This is intentionally shallow: it records process/network shape, not
        memory contents, credentials, packet payloads, or private app internals.
        """
        from core.runtime.resource_observation import get_resource_observer

        from .program_dna import (
            ProgramDNAEvidence,
        )

        evidence: list[ProgramDNAEvidence] = []
        observer = get_resource_observer()
        provenance = observer.provenance
        if not provenance.host_observed:
            self._record_degradation(
                "program_dna_reconstruction.host_snapshot",
                RuntimeError(
                    "live host snapshot refused non-host observation "
                    f"source={provenance.source.value}"
                ),
                severity="debug",
            )
            return evidence
        try:
            psutil = importlib.import_module("psutil")
        except ImportError as exc:
            self._record_degradation("program_dna_reconstruction.host_snapshot", exc, severity="debug")
            return evidence

        processes: list[dict[str, Any]] = []
        try:
            for process in observer.processes():
                cmdline = " ".join(str(part) for part in process.cmdline[:8])
                processes.append(
                    {
                        "pid": process.pid,
                        "name": process.name,
                        "username": process.username,
                        "cmdline_hint": cmdline[:240],
                    }
                )
                if len(processes) >= 40:
                    break
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._record_degradation("program_dna_reconstruction.process_snapshot", exc, severity="debug")
        if processes:
            evidence.append(
                ProgramDNAEvidence(
                    kind="process_observation",
                    source=f"{provenance.source.value}_host_snapshot:processes",
                    summary=f"Bounded process snapshot captured {len(processes)} visible process record(s).",
                    confidence=0.58,
                    details={
                        "processes": processes,
                        "observation": provenance.to_dict(),
                    },
                )
            )

        connections: list[dict[str, Any]] = []
        try:
            for conn in psutil.net_connections(kind="inet")[:80]:
                laddr = getattr(conn, "laddr", None)
                raddr = getattr(conn, "raddr", None)
                connections.append(
                    {
                        "fd": getattr(conn, "fd", None),
                        "family": str(getattr(conn, "family", "")),
                        "type": str(getattr(conn, "type", "")),
                        "local": f"{getattr(laddr, 'ip', '')}:{getattr(laddr, 'port', '')}" if laddr else "",
                        "remote_present": bool(raddr),
                        "status": getattr(conn, "status", ""),
                        "pid": getattr(conn, "pid", None),
                    }
                )
        except (psutil.Error, RuntimeError, TypeError, ValueError) as exc:
            self._record_degradation("program_dna_reconstruction.network_snapshot", exc, severity="debug")
        if connections:
            evidence.append(
                ProgramDNAEvidence(
                    kind="network_observation",
                    source=f"{provenance.source.value}_host_snapshot:inet_connections",
                    summary=f"Bounded network socket snapshot captured {len(connections)} visible connection record(s).",
                    confidence=0.54,
                    details={
                        "connections": connections,
                        "observation": provenance.to_dict(),
                    },
                )
            )
        return evidence

    def _walk_limited(self, root: Path, *, max_files: int) -> list[Path]:
        from .program_dna import (
            SKIP_DIRS,
        )

        files: list[Path] = []
        for current_root, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
            for name in names:
                if name.startswith("."):
                    continue
                path = Path(current_root) / name
                if path.is_file():
                    files.append(path)
                    if len(files) >= max_files:
                        return files
        return files

    def _python_public_symbols(self, path: Path) -> list[str]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError):
            return []
        symbols: list[str] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and not node.name.startswith("_"):
                symbols.append(node.name)
        return symbols

    def _inspect_raw_path(self, value: str | os.PathLike[str]) -> list[ProgramDNAEvidence]:

        return self._inspect_path(self._expanded_path(value))
