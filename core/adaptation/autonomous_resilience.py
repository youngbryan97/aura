"""Autonomous resilience mesh for Aura.

This module pushes Aura's adaptive immunity beyond bounded runtime repair:

1. Static fault auditing for preventable bug classes.
2. Runtime watchdog harvesting for stalls / deadlock pressure / task pressure.
3. Integration auditing and auto-wiring of health probes + repair handlers.
4. Verifier-guided patch execution through the existing self-modification engine.

The goal is not to pretend Aura can *guarantee* perfect universal repair.
The goal is to make her dramatically better at surfacing risk honestly,
preempting common failures, and turning repair proposals into validated action.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation



import ast
import asyncio
import inspect
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger("Aura.AutonomousResilience")

__all__ = [
    "AutonomousResilienceMesh",
    "IntegrationAuditor",
    "RuntimeWatchdogAuditor",
    "SecurityImmuneAuditor",
    "StaticFaultAuditor",
    "VerifierGuidedRepairPipeline",
    "get_autonomous_resilience_mesh",
]

_FINDING_SCORES = {
    "info": 0.20,
    "warning": 0.45,
    "error": 0.70,
    "critical": 0.92,
}


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _score_for_severity(severity: str) -> float:
    return float(_FINDING_SCORES.get(str(severity or "warning").lower(), 0.45))


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return repr(value)


@dataclass
class ResilienceFinding:
    kind: str
    severity: str
    subsystem: str
    message: str
    file_path: Optional[str] = None
    line: Optional[int] = None
    can_auto_patch: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return _score_for_severity(self.severity)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "kind": self.kind,
            "severity": self.severity,
            "subsystem": self.subsystem,
            "message": self.message,
            "file_path": self.file_path,
            "line": self.line,
            "can_auto_patch": self.can_auto_patch,
            "score": round(self.score, 4),
            "metadata": _json_safe(self.metadata),
        }
        return payload


class StaticFaultAuditor:
    """Cheap AST-based scanning for high-value preventable bug classes."""

    _BLOCKING_CALLS = {
        ("time", "sleep"): "async_blocking_time_sleep",
        ("requests", "get"): "async_blocking_requests",
        ("requests", "post"): "async_blocking_requests",
        ("requests", "put"): "async_blocking_requests",
        ("requests", "delete"): "async_blocking_requests",
        ("subprocess", "run"): "async_blocking_subprocess",
        ("subprocess", "Popen"): "async_blocking_subprocess",
    }

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir).resolve()

    def audit_codebase(
        self,
        *,
        paths: Optional[Sequence[str | Path]] = None,
        limit: int = 64,
    ) -> List[ResilienceFinding]:
        findings: List[ResilienceFinding] = []
        target_paths: Iterable[Path]
        if paths:
            target_paths = [self._resolve_path(path) for path in paths]
        else:
            target_paths = (path for path in (self.base_dir / "core").rglob("*.py"))

        for path in target_paths:
            if len(findings) >= limit:
                break
            if not path.exists() or path.suffix != ".py":
                continue
            try:
                findings.extend(self.audit_file(path))
            except Exception as exc:
                record_degradation('autonomous_resilience', exc)
                logger.debug("Static fault audit skipped for %s: %s", path, exc)
            if len(findings) >= limit:
                break
        findings.sort(key=lambda item: item.score, reverse=True)
        return findings[:limit]

    def audit_file(self, path: str | Path) -> List[ResilienceFinding]:
        resolved = self._resolve_path(path)
        try:
            source = resolved.read_text(encoding="utf-8")
        except Exception:
            source = resolved.read_text(errors="ignore")
        tree = ast.parse(source, filename=str(resolved))
        parents = self._parent_map(tree)
        rel_path = self._relative_path(resolved)
        findings: List[ResilienceFinding] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                issue = self._zero_division_issue(node, rel_path, parents)
                if issue is not None:
                    findings.append(issue)
            elif isinstance(node, ast.AsyncFunctionDef):
                findings.extend(self._audit_async_function(node, rel_path))

        return findings

    def _audit_async_function(
        self,
        node: ast.AsyncFunctionDef,
        rel_path: str,
    ) -> List[ResilienceFinding]:
        findings: List[ResilienceFinding] = []
        has_await_cache: Dict[int, bool] = {}

        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Call):
                fq_name = self._call_name(subnode)
                issue_kind = self._BLOCKING_CALLS.get(fq_name)
                if issue_kind:
                    findings.append(
                        ResilienceFinding(
                            kind=issue_kind,
                            severity="error" if issue_kind == "async_blocking_time_sleep" else "warning",
                            subsystem="runtime",
                            message=(
                                f"async function '{node.name}' performs blocking call "
                                f"{fq_name[0]}.{fq_name[1]}()"
                            ),
                            file_path=rel_path,
                            line=getattr(subnode, "lineno", None),
                            can_auto_patch=(issue_kind == "async_blocking_time_sleep"),
                            metadata={"function": node.name},
                        )
                    )
            elif isinstance(subnode, ast.While) and self._is_constant_true(subnode.test):
                has_await = has_await_cache.setdefault(id(subnode), self._contains_await(subnode))
                if not has_await:
                    findings.append(
                        ResilienceFinding(
                            kind="async_busy_loop",
                            severity="critical",
                            subsystem="runtime",
                            message=f"async function '{node.name}' contains a busy loop with no await",
                            file_path=rel_path,
                            line=getattr(subnode, "lineno", None),
                            metadata={"function": node.name},
                        )
                    )

        return findings

    def _zero_division_issue(
        self,
        node: ast.BinOp,
        rel_path: str,
        parents: Dict[ast.AST, ast.AST],
    ) -> Optional[ResilienceFinding]:
        right = node.right
        operator = {
            ast.Div: "/",
            ast.FloorDiv: "//",
            ast.Mod: "%",
        }.get(type(node.op), "/")
        function_name = self._enclosing_function_name(node, parents)

        if isinstance(right, ast.Constant) and right.value == 0:
            return ResilienceFinding(
                kind="definite_zero_division",
                severity="critical",
                subsystem="codebase",
                message=f"definite {operator} zero operation detected",
                file_path=rel_path,
                line=getattr(node, "lineno", None),
                metadata={"function": function_name},
            )

        if isinstance(right, ast.Call) and isinstance(right.func, ast.Name) and right.func.id in {"len", "count"}:
            return ResilienceFinding(
                kind="possible_zero_division",
                severity="warning",
                subsystem="codebase",
                message=f"division by {right.func.id}(...) may be zero without an explicit guard",
                file_path=rel_path,
                line=getattr(node, "lineno", None),
                metadata={"function": function_name},
            )

        if isinstance(right, ast.Name) and re.search(
            r"(count|len|size|total|denom|denominator|divisor|workers?)$",
            right.id,
            re.IGNORECASE,
        ):
            return ResilienceFinding(
                kind="possible_zero_division",
                severity="warning",
                subsystem="codebase",
                message=f"division by '{right.id}' may require a zero guard",
                file_path=rel_path,
                line=getattr(node, "lineno", None),
                metadata={"function": function_name, "divisor": right.id},
            )
        return None

    @staticmethod
    def _call_name(node: ast.Call) -> Tuple[str, str] | None:
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            return node.func.value.id, node.func.attr
        return None

    @staticmethod
    def _is_constant_true(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and bool(node.value) is True

    @staticmethod
    def _contains_await(node: ast.AST) -> bool:
        return any(isinstance(child, ast.Await) for child in ast.walk(node))

    @staticmethod
    def _parent_map(tree: ast.AST) -> Dict[ast.AST, ast.AST]:
        parents: Dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        return parents

    @staticmethod
    def _enclosing_function_name(node: ast.AST, parents: Dict[ast.AST, ast.AST]) -> str:
        current = node
        while current in parents:
            current = parents[current]
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return current.name
        return "<module>"

    def _resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return (self.base_dir / candidate).resolve()

    def _relative_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.base_dir))
        except Exception:
            return str(path)


class IntegrationAuditor:
    """Audits the service graph and auto-wires health + repair hooks when safe."""

    _IGNORED_SERVICES = {
        "metrics",
        "audit",
        "event_bus",
    }

    def __init__(
        self,
        *,
        service_resolver: Callable[[str], Any],
        container_cls: Any,
    ):
        self._service_resolver = service_resolver
        self._container_cls = container_cls
        self._auto_wired_repairs: set[Tuple[str, str]] = set()

    def audit_service_graph(self) -> Dict[str, Any]:
        findings: List[ResilienceFinding] = []
        registry = getattr(self._container_cls, "_services", {}) or {}
        registered_names = set(registry.keys())
        aliases = getattr(self._container_cls, "_aliases", {}) or {}

        for name, descriptor in registry.items():
            dependencies = list(getattr(descriptor, "dependencies", []) or [])
            for dependency in dependencies:
                resolved = aliases.get(dependency, dependency)
                if resolved not in registered_names:
                    findings.append(
                        ResilienceFinding(
                            kind="missing_dependency",
                            severity="error",
                            subsystem="service_graph",
                            message=f"service '{name}' depends on missing service '{dependency}'",
                            metadata={"service": name, "dependency": dependency},
                        )
                    )

        return {
            "findings": [finding.to_dict() for finding in findings],
            "finding_count": len(findings),
            "dependency_gaps": [
                finding.to_dict()
                for finding in findings
                if finding.kind == "missing_dependency"
            ],
        }

    def auto_wire_autopoiesis(self) -> Dict[str, Any]:
        autopoiesis = self._service_resolver("autopoiesis")
        if autopoiesis is None:
            return {
                "health_probes_added": [],
                "repair_handlers_added": [],
                "notes": ["autopoiesis unavailable"],
            }

        health_added: List[str] = []
        repair_added: List[str] = []
        registry = getattr(self._container_cls, "_services", {}) or {}

        try:
            from core.cognitive.autopoiesis import RepairStrategy
        except Exception as exc:
            record_degradation('autonomous_resilience', exc)
            return {
                "health_probes_added": [],
                "repair_handlers_added": [],
                "notes": [f"repair strategy import failed: {exc}"],
            }

        existing_health = set(getattr(autopoiesis, "_health_fns", {}).keys())

        for name in registry:
            if name in self._IGNORED_SERVICES:
                continue
            try:
                instance = self._service_resolver(name)
            except Exception:
                continue
            if instance is None:
                continue

            if name not in existing_health:
                probe = self._health_probe_for(instance)
                if probe is not None:
                    try:
                        autopoiesis.register_component(name, probe)
                        health_added.append(name)
                    except Exception as exc:
                        record_degradation('autonomous_resilience', exc)
                        logger.debug("Auto-wire health probe skipped for %s: %s", name, exc)

            for strategy_name, handler in self._repair_handlers_for(instance).items():
                key = (name, strategy_name)
                if key in self._auto_wired_repairs:
                    continue
                strategy = getattr(RepairStrategy, strategy_name, None)
                if strategy is None:
                    continue
                try:
                    autopoiesis.register_repair_handler(strategy, name, handler)
                    self._auto_wired_repairs.add(key)
                    repair_added.append(f"{name}:{strategy.value}")
                except Exception as exc:
                    record_degradation('autonomous_resilience', exc)
                    logger.debug("Auto-wire repair handler skipped for %s/%s: %s", name, strategy_name, exc)

        return {
            "health_probes_added": sorted(health_added),
            "repair_handlers_added": sorted(repair_added),
            "notes": [],
        }

    @staticmethod
    def _health_probe_for(instance: Any) -> Optional[Callable[[], float]]:
        if hasattr(instance, "health_score") and callable(instance.health_score):
            return lambda instance=instance: _clamp01(float(instance.health_score()))

        if hasattr(instance, "get_vitality") and callable(instance.get_vitality):
            return lambda instance=instance: _clamp01(float(instance.get_vitality()))

        if hasattr(instance, "get_status") and callable(instance.get_status):
            def _status_probe(instance: Any = instance) -> float:
                status = instance.get_status()
                if not isinstance(status, dict):
                    return 0.5
                if "health_score" in status:
                    return _clamp01(float(status["health_score"]))
                if "health" in status and isinstance(status["health"], (int, float)):
                    return _clamp01(float(status["health"]))
                if "overall_healthy" in status:
                    return 1.0 if status["overall_healthy"] else 0.35
                if "integrity_ok" in status:
                    return 1.0 if status["integrity_ok"] else 0.30
                if "passed" in status:
                    return 1.0 if status["passed"] else 0.30
                if "running" in status:
                    return 1.0 if status["running"] else 0.25
                return 0.55

            return _status_probe

        if hasattr(instance, "is_healthy") and callable(instance.is_healthy):
            return lambda instance=instance: 1.0 if bool(instance.is_healthy()) else 0.30

        return None

    @staticmethod
    def _repair_handlers_for(instance: Any) -> Dict[str, Callable[[], Any]]:
        handlers: Dict[str, Callable[[], Any]] = {}

        def _wrap(method_name: str) -> Callable[[], Any]:
            method = getattr(instance, method_name)

            async def _runner() -> Any:
                result = method()
                if inspect.isawaitable(result):
                    return await result
                return result

            return _runner

        for strategy_name, candidates in {
            "CLEAR_CACHE": ("clear_cache", "flush_cache"),
            "REDUCE_LOAD": ("reduce_load", "shed_load", "hibernate"),
            "RESTART_COMPONENT": ("restart", "reload", "reconnect", "reinitialize", "reset"),
            "RESTORE_CHECKPOINT": ("restore_checkpoint", "rollback", "restore"),
            "ISOLATE": ("isolate", "quarantine", "disable"),
        }.items():
            for candidate in candidates:
                if hasattr(instance, candidate) and callable(getattr(instance, candidate)):
                    handlers[strategy_name] = _wrap(candidate)
                    break
        return handlers


class RuntimeWatchdogAuditor:
    """Harvest runtime risk signals from existing watchdog / task infrastructure."""

    def __init__(self, *, service_resolver: Callable[[str], Any]):
        self._service_resolver = service_resolver

    def audit(self) -> Dict[str, Any]:
        findings: List[ResilienceFinding] = []

        lock_snapshot = self._lock_watchdog_snapshot()
        if lock_snapshot["active_count"] > 0:
            hottest = lock_snapshot["locks"][0]
            held = float(hottest.get("held_duration_s", 0.0) or 0.0)
            threshold = float(lock_snapshot.get("threshold_s", 180.0) or 180.0)
            if held >= threshold:
                findings.append(
                    ResilienceFinding(
                        kind="stalled_lock",
                        severity="critical",
                        subsystem="runtime_locking",
                        message=f"lock '{hottest['name']}' held for {held:.1f}s",
                        metadata=hottest,
                    )
                )
            elif held >= threshold * 0.5:
                findings.append(
                    ResilienceFinding(
                        kind="lock_contention",
                        severity="warning",
                        subsystem="runtime_locking",
                        message=f"lock '{hottest['name']}' is approaching deadlock territory ({held:.1f}s)",
                        metadata=hottest,
                    )
                )

        task_stats = self._task_tracker_stats()
        if task_stats["unsupervised_active"] >= 80:
            findings.append(
                ResilienceFinding(
                    kind="unsupervised_task_pressure",
                    severity="error",
                    subsystem="runtime_tasks",
                    message=f"high unsupervised task count: {task_stats['unsupervised_active']}",
                    metadata=task_stats,
                )
            )
        elif task_stats["active"] >= max(task_stats["max_concurrent"] * 2, 40):
            findings.append(
                ResilienceFinding(
                    kind="task_backlog",
                    severity="warning",
                    subsystem="runtime_tasks",
                    message=f"background task backlog detected: {task_stats['active']}",
                    metadata=task_stats,
                )
            )

        stability = self._stability_snapshot()
        if not stability.get("healthy", True):
            findings.append(
                ResilienceFinding(
                    kind="stability_degraded",
                    severity="error",
                    subsystem="runtime_stability",
                    message=stability.get("message", "stability guardian reported degradation"),
                    metadata=stability,
                )
            )

        return {
            "findings": [finding.to_dict() for finding in findings],
            "finding_count": len(findings),
            "threat_score": round(max((finding.score for finding in findings), default=0.0), 4),
            "lock_watchdog": lock_snapshot,
            "task_tracker": task_stats,
            "stability": stability,
        }

    def _lock_watchdog_snapshot(self) -> Dict[str, Any]:
        try:
            lock_watchdog = self._service_resolver("lock_watchdog")
            if lock_watchdog is None:
                from core.resilience.lock_watchdog import get_lock_watchdog

                lock_watchdog = get_lock_watchdog()
            return dict(lock_watchdog.get_snapshot())
        except Exception as exc:
            record_degradation('autonomous_resilience', exc)
            return {
                "active_count": 0,
                "locks": [],
                "threshold_s": 180.0,
                "error": str(exc),
            }

    def _task_tracker_stats(self) -> Dict[str, Any]:
        try:
            from core.utils.task_tracker import get_task_tracker

            tracker = get_task_tracker()
            stats = dict(tracker.get_stats())
        except Exception as exc:
            record_degradation('autonomous_resilience', exc)
            return {
                "active": 0,
                "high_water": 0,
                "total_tracked": 0,
                "max_concurrent": 0,
                "unsupervised_active": 0,
                "error": str(exc),
            }

        unsupervised = 0
        try:
            loop = asyncio.get_running_loop()
            for task in asyncio.all_tasks(loop):
                if task.done():
                    continue
                if not bool(getattr(task, "_aura_supervised", False)):
                    unsupervised += 1
        except RuntimeError:
            unsupervised = 0
        except Exception:
            pass

        stats["unsupervised_active"] = unsupervised
        return stats

    def _stability_snapshot(self) -> Dict[str, Any]:
        guardian = self._service_resolver("stability_guardian")
        if guardian is None:
            return {"healthy": True, "message": "stability guardian unavailable"}
        history = getattr(guardian, "_report_history", None)
        if history:
            report = history[-1]
            return {
                "healthy": bool(getattr(report, "overall_healthy", True)),
                "message": "; ".join(
                    c.message for c in getattr(report, "checks", []) if not getattr(c, "healthy", True)
                )
                or "healthy",
            }
        return {"healthy": True, "message": "no recent report"}


class SecurityImmuneAuditor:
    """Detects prompt-injection and tool-misuse style immune antigens."""

    _TOOL_MISUSE_PATTERNS = (
        r"(?i)\brm\s+-rf\b",
        r"(?i)\bchmod\s+777\b",
        r"(?i)\bcurl\b.+\|\s*(?:sh|bash|zsh)",
        r"(?i)\breveal\b.+\b(secret|token|password|api key)\b",
        r"(?i)\bdisable\b.+\b(guardrail|guard|safety|constitution)\b",
    )

    def __init__(self) -> None:
        try:
            from core.utils.sanitizer import get_blood_brain_barrier

            self._barrier = get_blood_brain_barrier()
            self._injection_patterns = tuple(getattr(self._barrier, "malicious_patterns", ()))
        except Exception:
            self._barrier = None
            self._injection_patterns = ()

    def scan_text(self, text: str) -> Dict[str, Any]:
        findings: List[ResilienceFinding] = []
        if not str(text or "").strip():
            return {
                "findings": [],
                "finding_count": 0,
                "threat_score": 0.0,
            }

        raw_text = str(text)
        lowered = raw_text.lower()
        for pattern in self._injection_patterns:
            if re.search(pattern, raw_text):
                findings.append(
                    ResilienceFinding(
                        kind="prompt_injection_signal",
                        severity="error",
                        subsystem="prompt_boundary",
                        message=f"prompt-injection pattern matched: {pattern}",
                    )
                )

        for pattern in self._TOOL_MISUSE_PATTERNS:
            if re.search(pattern, raw_text):
                findings.append(
                    ResilienceFinding(
                        kind="tool_misuse_signal",
                        severity="critical",
                        subsystem="tool_boundary",
                        message=f"high-risk command or exfiltration pattern matched: {pattern}",
                    )
                )

        if "memory" in lowered and any(token in lowered for token in ("poison", "override", "corrupt", "inject")):
            findings.append(
                ResilienceFinding(
                    kind="memory_poisoning_signal",
                    severity="error",
                    subsystem="memory_boundary",
                    message="input resembles poisoned-memory or unsafe memory mutation attempt",
                )
            )

        deduped: Dict[Tuple[str, str], ResilienceFinding] = {}
        for finding in findings:
            deduped[(finding.kind, finding.message)] = finding
        ordered = sorted(deduped.values(), key=lambda item: item.score, reverse=True)
        return {
            "findings": [finding.to_dict() for finding in ordered],
            "finding_count": len(ordered),
            "threat_score": round(max((finding.score for finding in ordered), default=0.0), 4),
        }


class VerifierGuidedRepairPipeline:
    """Bridge immune patch proposals into the existing self-modification engine."""

    _TRACEBACK_RE = re.compile(r'File "([^"]+)", line (\d+), in ')

    def __init__(
        self,
        *,
        base_dir: str | Path,
        service_resolver: Callable[[str], Any],
    ):
        self.base_dir = Path(base_dir).resolve()
        self._service_resolver = service_resolver

    async def attempt_repair(
        self,
        *,
        error_signature: str,
        stack_trace: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = context or {}
        target = self._locate_target(stack_trace, context=context)
        if target is None:
            return {
                "attempted": False,
                "applied": False,
                "status": "no_target",
                "notes": "no patchable file/line could be recovered from context",
            }

        file_path, line_number = target
        modifier = self._service_resolver("self_modification_engine")
        if modifier is None or not hasattr(modifier, "code_repair"):
            return {
                "attempted": False,
                "applied": False,
                "status": "self_modifier_unavailable",
                "file_path": file_path,
                "line_number": line_number,
                "notes": "self-modification engine unavailable",
            }

        diagnosis = self._diagnosis_for(error_signature, context)
        try:
            success, fix, test_results = await modifier.code_repair.repair_bug(
                file_path,
                line_number,
                diagnosis,
            )
        except Exception as exc:
            record_degradation('autonomous_resilience', exc)
            logger.debug("Repair pipeline generation failed for %s:%s: %s", file_path, line_number, exc)
            return {
                "attempted": True,
                "applied": False,
                "status": "generation_failed",
                "file_path": file_path,
                "line_number": line_number,
                "notes": str(exc),
            }

        if not success or fix is None:
            return {
                "attempted": True,
                "applied": False,
                "status": "proposal_failed",
                "file_path": file_path,
                "line_number": line_number,
                "notes": str((test_results or {}).get("error") or (test_results or {}).get("errors") or "repair proposal failed"),
            }

        proposal = {
            "bug": {
                "diagnosis": diagnosis,
                "pattern": {
                    "events": [
                        {
                            "error_type": error_signature or "runtime_error",
                        }
                    ]
                },
            },
            "fix": fix,
            "test_results": test_results,
        }
        applied = False
        apply_error = ""
        if hasattr(modifier, "apply_fix"):
            try:
                applied = bool(await modifier.apply_fix(proposal, force=True, test_results=test_results))
            except Exception as exc:
                record_degradation('autonomous_resilience', exc)
                apply_error = str(exc)

        status = "applied" if applied else "validated_unapplied"
        notes = apply_error or "sandbox + verification pipeline completed"
        return {
            "attempted": True,
            "applied": applied,
            "status": status,
            "file_path": file_path,
            "line_number": line_number,
            "fix_confidence": getattr(fix, "confidence", None),
            "notes": notes,
        }

    def _locate_target(
        self,
        stack_trace: str,
        *,
        context: Dict[str, Any],
    ) -> Optional[Tuple[str, int]]:
        candidates: List[Tuple[str, int]] = []
        for match in self._TRACEBACK_RE.finditer(str(stack_trace or "")):
            raw_path = Path(match.group(1)).expanduser()
            line_number = int(match.group(2))
            if raw_path.is_absolute():
                try:
                    raw_path = raw_path.resolve()
                except Exception:
                    pass
                try:
                    rel = raw_path.relative_to(self.base_dir)
                    candidates.append((str(rel), line_number))
                except Exception:
                    continue
            else:
                candidate = (self.base_dir / raw_path).resolve()
                if candidate.exists():
                    candidates.append((str(candidate.relative_to(self.base_dir)), line_number))

        file_path = context.get("file_path")
        if file_path and context.get("line_number"):
            try:
                candidate = Path(file_path)
                if candidate.is_absolute():
                    candidate = candidate.resolve().relative_to(self.base_dir)
                candidates.insert(0, (str(candidate), int(context["line_number"])))
            except Exception:
                pass

        if not candidates:
            return None
        return candidates[0]

    @staticmethod
    def _diagnosis_for(error_signature: str, context: Dict[str, Any]) -> Dict[str, Any]:
        error_lower = str(error_signature or "").lower()
        if "zerodivision" in error_lower:
            root_cause = "A divisor reached zero without a guard"
            potential_fix = "Add an explicit zero guard or safe denominator fallback while preserving semantics."
        elif "attributeerror" in error_lower or "none" in error_lower:
            root_cause = "A nullable object is being dereferenced without a guard"
            potential_fix = "Introduce a none-check and safe degraded path before the attribute access."
        elif "keyerror" in error_lower or "indexerror" in error_lower:
            root_cause = "Collection access is not defended against absent keys or bounds"
            potential_fix = "Guard the lookup and degrade gracefully when the requested entry is absent."
        elif "typeerror" in error_lower:
            root_cause = "Type contract mismatch or missing normalization"
            potential_fix = "Normalize or validate inputs before the failing operation."
        elif "nameerror" in error_lower or "importerror" in error_lower:
            root_cause = "Symbol wiring or import dependency is broken"
            potential_fix = "Repair the symbol wiring, import path, or missing definition."
        else:
            root_cause = context.get("summary") or f"Runtime failure: {error_signature or 'unknown'}"
            potential_fix = "Generate the smallest change that resolves the failure and preserves existing behavior."

        return {
            "ok": True,
            "summary": root_cause,
            "hypotheses": [
                {
                    "root_cause": root_cause,
                    "explanation": context.get("message") or root_cause,
                    "potential_fix": potential_fix,
                    "confidence": "medium",
                }
            ],
        }


class AutonomousResilienceMesh:
    """Coordinates static auditing, runtime watchdogs, and verified code repair."""

    def __init__(
        self,
        *,
        base_dir: str | Path,
        service_resolver: Optional[Callable[[str], Any]] = None,
        container_cls: Any = None,
        code_scan_interval_s: float = 300.0,
        auto_wire_interval_s: float = 90.0,
    ):
        self.base_dir = Path(base_dir).resolve()
        self._service_resolver = service_resolver or self._default_service_resolver
        self._container_cls = container_cls or self._default_container_cls()
        self.static_faults = StaticFaultAuditor(self.base_dir)
        self.integration = IntegrationAuditor(
            service_resolver=self._service_resolver,
            container_cls=self._container_cls,
        )
        self.runtime = RuntimeWatchdogAuditor(service_resolver=self._service_resolver)
        self.security = SecurityImmuneAuditor()
        self.patch_pipeline = VerifierGuidedRepairPipeline(
            base_dir=self.base_dir,
            service_resolver=self._service_resolver,
        )
        self._code_scan_interval_s = float(code_scan_interval_s)
        self._auto_wire_interval_s = float(auto_wire_interval_s)
        self._last_code_scan = 0.0
        self._last_auto_wire = 0.0
        self._last_report: Dict[str, Any] = {}
        self._last_static_findings: List[Dict[str, Any]] = []

    async def tick(
        self,
        *,
        user_text: str = "",
        state_snapshot: Optional[Dict[str, Any]] = None,
        force_code_scan: bool = False,
    ) -> Dict[str, Any]:
        state_snapshot = state_snapshot or {}
        now = time.time()

        security_report = self.security.scan_text(user_text)
        runtime_report = self.runtime.audit()
        service_report = self.integration.audit_service_graph()
        auto_wire_report = None
        if force_code_scan or (now - self._last_auto_wire) >= self._auto_wire_interval_s:
            auto_wire_report = self.integration.auto_wire_autopoiesis()
            self._last_auto_wire = now

        static_report = {
            "findings": list(self._last_static_findings),
            "finding_count": len(self._last_static_findings),
            "threat_score": round(max((item.get("score", 0.0) for item in self._last_static_findings), default=0.0), 4),
        }
        error_count = int(state_snapshot.get("error_count", 0) or 0)
        if force_code_scan or error_count > 0 or (now - self._last_code_scan) >= self._code_scan_interval_s:
            static_findings = self.static_faults.audit_codebase(limit=24)
            self._last_static_findings = [finding.to_dict() for finding in static_findings]
            static_report = {
                "findings": list(self._last_static_findings),
                "finding_count": len(self._last_static_findings),
                "threat_score": round(max((finding.score for finding in static_findings), default=0.0), 4),
            }
            self._last_code_scan = now

        threat_score = round(
            max(
                float(security_report.get("threat_score", 0.0)),
                float(runtime_report.get("threat_score", 0.0)),
                float(static_report.get("threat_score", 0.0)),
                0.78 if service_report.get("finding_count", 0) else 0.0,
            ),
            4,
        )
        immune_events = self._build_immune_events(
            security_report=security_report,
            runtime_report=runtime_report,
            static_report=static_report,
            service_report=service_report,
        )
        report = {
            "timestamp": now,
            "threat_score": threat_score,
            "security": security_report,
            "runtime": runtime_report,
            "service_graph": service_report,
            "auto_wire": auto_wire_report,
            "static_faults": static_report,
            "immune_events": immune_events,
            "finding_count": (
                int(security_report.get("finding_count", 0))
                + int(runtime_report.get("finding_count", 0))
                + int(service_report.get("finding_count", 0))
                + int(static_report.get("finding_count", 0))
            ),
        }
        self._last_report = report
        return report

    async def attempt_patch_for_antigen(
        self,
        artifact: Any,
        antigen: Any,
    ) -> Dict[str, Any]:
        context = dict(getattr(antigen, "context", {}) or {})
        context.setdefault("component", getattr(artifact, "component", "unknown"))
        return await self.patch_pipeline.attempt_repair(
            error_signature=str(getattr(antigen, "error_signature", "")),
            stack_trace=str(getattr(antigen, "stack_trace", "")),
            context=context,
        )

    def get_status(self) -> Dict[str, Any]:
        return dict(self._last_report or {})

    @staticmethod
    def _default_container_cls() -> Any:
        from core.container import ServiceContainer

        return ServiceContainer

    def _default_service_resolver(self, name: str) -> Any:
        from core.container import ServiceContainer

        return ServiceContainer.get(name, default=None)

    def _build_immune_events(
        self,
        *,
        security_report: Dict[str, Any],
        runtime_report: Dict[str, Any],
        static_report: Dict[str, Any],
        service_report: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for section_name, section in (
            ("security", security_report),
            ("runtime", runtime_report),
            ("static_faults", static_report),
        ):
            for finding in section.get("findings", [])[:3]:
                severity = str(finding.get("severity", "warning"))
                score = _score_for_severity(severity)
                events.append(
                    {
                        "type": finding.get("kind", section_name),
                        "text": finding.get("message", section_name),
                        "source": f"autonomous_resilience:{section_name}",
                        "subsystem": finding.get("subsystem") or section_name,
                        "danger": score,
                        "threat_probability": score,
                        "error_count": 1,
                        "timestamp": time.time(),
                        "error_signature": finding.get("kind", section_name),
                    }
                )

        for finding in service_report.get("findings", [])[:2]:
            score = _score_for_severity(str(finding.get("severity", "error")))
            events.append(
                {
                    "type": finding.get("kind", "service_graph_issue"),
                    "text": finding.get("message", "service graph issue"),
                    "source": "autonomous_resilience:service_graph",
                    "subsystem": finding.get("subsystem") or "service_graph",
                    "danger": score,
                    "threat_probability": score,
                    "error_count": 1,
                    "timestamp": time.time(),
                    "error_signature": finding.get("kind", "service_graph_issue"),
                }
            )

        events.sort(key=lambda item: float(item.get("danger", 0.0)), reverse=True)
        return events[:6]


_autonomous_resilience_singleton: Optional[AutonomousResilienceMesh] = None


def get_autonomous_resilience_mesh() -> AutonomousResilienceMesh:
    global _autonomous_resilience_singleton
    if _autonomous_resilience_singleton is None:
        try:
            from core.config import config

            base_dir = config.paths.base_dir
        except Exception:
            base_dir = os.getcwd()
        _autonomous_resilience_singleton = AutonomousResilienceMesh(base_dir=base_dir)
    return _autonomous_resilience_singleton
