"""Closed-loop fault-to-patch autonomy pipeline.

The loop is deterministic first and semantic second:

runtime error -> causal trace -> bug packet -> localization -> reproduction ->
targeted test -> risk tier -> deterministic patch candidate -> shadow evidence
-> promotion decision.

LLM patch synthesis can be attached after a precise bug packet exists, but this
module deliberately handles the common mechanical repairs without tokens.
"""
from __future__ import annotations

import ast
import hashlib
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from core.runtime.causal_trace import current_trace_id
from core.self_modification.mutation_tiers import classify_mutation_path
from core.self_modification.patch_genealogy import PatchNode, get_patch_genealogy
from core.self_modification.repair_approval import get_repair_approval_policy
from core.self_modification.repair_calibration import get_repair_calibration


@dataclass(frozen=True)
class TraceFrame:
    file: str
    line: int
    function: str
    code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"file": self.file, "line": self.line, "function": self.function, "code": self.code}


@dataclass(frozen=True)
class BugFingerprint:
    error_type: str
    normalized_message: str
    file: str
    line: int
    function: str
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type,
            "normalized_message": self.normalized_message,
            "file": self.file,
            "line": self.line,
            "function": self.function,
            "digest": self.digest,
        }


@dataclass(frozen=True)
class BugPacket:
    error_type: str
    message: str
    file: str
    line: int
    function: str
    minimal_repro: str
    failing_test: str
    risk_tier: str
    trace_id: str
    fingerprint: BugFingerprint
    frames: tuple[TraceFrame, ...] = field(default_factory=tuple)
    deterministic_repairable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "file": self.file,
            "line": self.line,
            "function": self.function,
            "minimal_repro": self.minimal_repro,
            "failing_test": self.failing_test,
            "risk_tier": self.risk_tier,
            "trace_id": self.trace_id,
            "fingerprint": self.fingerprint.to_dict(),
            "frames": [frame.to_dict() for frame in self.frames],
            "deterministic_repairable": self.deterministic_repairable,
        }


@dataclass(frozen=True)
class PatchCandidate:
    target_file: str
    before_source: str
    after_source: str
    explanation: str
    confidence: float
    deterministic: bool
    risk_tier: str
    blocked_reason: str = ""

    @property
    def changed(self) -> bool:
        return self.before_source != self.after_source

    def diff_digest(self) -> str:
        return hashlib.sha256((self.before_source + "\n---\n" + self.after_source).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_file": self.target_file,
            "explanation": self.explanation,
            "confidence": self.confidence,
            "deterministic": self.deterministic,
            "risk_tier": self.risk_tier,
            "changed": self.changed,
            "diff_digest": self.diff_digest(),
            "blocked_reason": self.blocked_reason,
        }


@dataclass(frozen=True)
class FaultPipelineResult:
    packet: BugPacket
    candidate: Optional[PatchCandidate]
    promotion_allowed: bool
    lineage_patch_id: str
    calibration_probability: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet": self.packet.to_dict(),
            "candidate": self.candidate.to_dict() if self.candidate else None,
            "promotion_allowed": self.promotion_allowed,
            "lineage_patch_id": self.lineage_patch_id,
            "calibration_probability": round(self.calibration_probability, 4),
            "reasons": list(self.reasons),
        }


class TracebackHarvester:
    def __init__(self, codebase_root: str | Path = ".") -> None:
        self.codebase_root = Path(codebase_root).resolve()

    def from_exception(self, exc: BaseException) -> tuple[TraceFrame, ...]:
        extracted = traceback.extract_tb(exc.__traceback__)
        return tuple(
            TraceFrame(
                file=self._rel(frame.filename),
                line=int(frame.lineno),
                function=str(frame.name),
                code=str(frame.line or ""),
            )
            for frame in extracted
        )

    def _rel(self, filename: str) -> str:
        path = Path(filename).resolve()
        try:
            return path.relative_to(self.codebase_root).as_posix()
        except ValueError:
            return str(path)


class BugLocalizer:
    def __init__(self, codebase_root: str | Path = ".") -> None:
        self.codebase_root = Path(codebase_root).resolve()

    def localize(self, frames: Iterable[TraceFrame]) -> TraceFrame:
        frame_list = list(frames)
        if not frame_list:
            return TraceFrame(file="", line=0, function="")
        for frame in reversed(frame_list):
            if frame.file.startswith("core/") or frame.file.startswith("skills/") or frame.file.startswith("interface/"):
                return frame
        return frame_list[-1]


class ReproductionSynthesizer:
    def synthesize(self, frame: TraceFrame, exc: BaseException) -> str:
        return (
            "import ast\n"
            "import py_compile\n"
            f"target_file = {frame.file!r}\n"
            "py_compile.compile(target_file, doraise=True)\n"
            f"tree = ast.parse(open(target_file).read())\n"
            f"funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]\n"
            f"assert {frame.function!r} in funcs, f'Function {frame.function!r} not found in {{funcs}}'\n"
            f"# Expected failure before repair: {type(exc).__name__}: {str(exc)[:160]}\n"
        )

    @staticmethod
    def _module_name(file_path: str) -> str:
        return file_path.replace("/", ".").removesuffix(".py")


class TestSynthesizer:
    def synthesize(self, frame: TraceFrame, exc: BaseException) -> str:
        module_name = frame.file.replace("/", ".").removesuffix(".py")
        test_name = re.sub(r"[^a-zA-Z0-9_]+", "_", f"test_repair_{module_name}_{frame.function}")[:120]
        return (
            "import ast\n"
            "import py_compile\n\n"
            f"def {test_name}():\n"
            f"    py_compile.compile({frame.file!r}, doraise=True)\n"
            f"    tree = ast.parse(open({frame.file!r}).read())\n"
            f"    funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]\n"
            f"    assert {frame.function!r} in funcs\n"
            f"    # Regression packet: {type(exc).__name__}: {str(exc)[:120]!r}\n"
        )


class StaticRiskClassifier:
    def classify(self, file_path: str) -> str:
        return classify_mutation_path(file_path).tier.label


class DeterministicPatchGenerator:
    """Mechanical repairs that do not need a semantic model."""

    SAFE_NAME_IMPORTS: dict[str, str] = {
        "asyncio": "import asyncio",
        "json": "import json",
        "logging": "import logging",
        "time": "import time",
        "math": "import math",
        "re": "import re",
        "Path": "from pathlib import Path",
        "Any": "from typing import Any",
        "Dict": "from typing import Dict",
        "List": "from typing import List",
        "Optional": "from typing import Optional",
        "get_mlx_client": "from core.brain.llm.mlx_client import get_mlx_client",
    }

    def __init__(self, codebase_root: str | Path = ".") -> None:
        self.codebase_root = Path(codebase_root).resolve()

    def generate(self, packet: BugPacket) -> Optional[PatchCandidate]:
        if packet.error_type == "NameError":
            missing = self._missing_name(packet.message)
            if missing:
                return self._inject_import(packet, missing)
        return None

    @staticmethod
    def _missing_name(message: str) -> str:
        match = re.search(r"name ['\"]([^'\"]+)['\"] is not defined", message)
        return match.group(1) if match else ""

    def _inject_import(self, packet: BugPacket, missing: str) -> Optional[PatchCandidate]:
        import_line = self.SAFE_NAME_IMPORTS.get(missing)
        if not import_line:
            return None
        target = self.codebase_root / packet.file
        if not target.exists():
            return None
        before = target.read_text(encoding="utf-8")
        after = self._insert_import(before, import_line)
        ast.parse(after)
        return PatchCandidate(
            target_file=packet.file,
            before_source=before,
            after_source=after,
            explanation=f"Deterministically inject safe import for undefined name {missing!r}.",
            confidence=0.92,
            deterministic=True,
            risk_tier=packet.risk_tier,
        )

    @staticmethod
    def _insert_import(source: str, import_line: str) -> str:
        if import_line in source:
            return source
        lines = source.splitlines()
        insert_at = 0
        if lines and lines[0].startswith("#!"):
            insert_at = 1
        if insert_at < len(lines) and "coding" in lines[insert_at]:
            insert_at += 1
        if insert_at < len(lines) and (lines[insert_at].startswith('"""') or lines[insert_at].startswith("'''")):
            quote = lines[insert_at][:3]
            insert_at += 1
            while insert_at < len(lines) and quote not in lines[insert_at]:
                insert_at += 1
            insert_at = min(len(lines), insert_at + 1)
        while insert_at < len(lines) and (
            lines[insert_at].startswith("from __future__")
            or lines[insert_at].strip() == ""
        ):
            insert_at += 1
        lines.insert(insert_at, import_line)
        return "\n".join(lines) + ("\n" if source.endswith("\n") else "")


class FaultToPatchPipeline:
    """Coordinates deterministic fault diagnosis and patch packet creation."""

    def __init__(self, codebase_root: str | Path = ".") -> None:
        self.codebase_root = Path(codebase_root).resolve()
        self.harvester = TracebackHarvester(self.codebase_root)
        self.localizer = BugLocalizer(self.codebase_root)
        self.repro = ReproductionSynthesizer()
        self.tests = TestSynthesizer()
        self.risk = StaticRiskClassifier()
        self.patch_generator = DeterministicPatchGenerator(self.codebase_root)
        self.genealogy = get_patch_genealogy()
        self.calibration = get_repair_calibration()
        self.approval_policy = get_repair_approval_policy()

    def build_packet(self, exc: BaseException) -> BugPacket:
        frames = self.harvester.from_exception(exc)
        localized = self.localizer.localize(frames)
        risk_tier = self.risk.classify(localized.file)
        message = str(exc)
        normalized = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", message)
        digest = hashlib.sha256(
            f"{type(exc).__name__}|{normalized}|{localized.file}|{localized.function}".encode("utf-8")
        ).hexdigest()[:16]
        fingerprint = BugFingerprint(
            error_type=type(exc).__name__,
            normalized_message=normalized,
            file=localized.file,
            line=localized.line,
            function=localized.function,
            digest=digest,
        )
        deterministic = type(exc).__name__ == "NameError"
        return BugPacket(
            error_type=type(exc).__name__,
            message=message,
            file=localized.file,
            line=localized.line,
            function=localized.function,
            minimal_repro=self.repro.synthesize(localized, exc),
            failing_test=self.tests.synthesize(localized, exc),
            risk_tier=risk_tier,
            trace_id=current_trace_id("trace_unavailable"),
            fingerprint=fingerprint,
            frames=frames,
            deterministic_repairable=deterministic,
        )

    def diagnose(self, exc: BaseException) -> FaultPipelineResult:
        packet = self.build_packet(exc)
        candidate = self.patch_generator.generate(packet)
        reasons: list[str] = []
        if candidate is None:
            reasons.append("no_deterministic_patch")

        module_family = packet.file.split("/", 2)[0] if packet.file else "unknown"
        model_conf = candidate.confidence if candidate else 0.0
        bucket = self.calibration.bucket(
            bug_class=packet.error_type,
            risk_tier=packet.risk_tier,
            module_family=module_family,
        )
        calibration_probability = self.calibration.calibrated_probability(
            bug_class=packet.error_type,
            risk_tier=packet.risk_tier,
            module_family=module_family,
            model_confidence=model_conf,
        )
        approval = self.approval_policy.decide(
            target_file=packet.file,
            candidate_changed=bool(candidate and candidate.changed),
            deterministic=bool(candidate and candidate.deterministic),
            candidate_confidence=model_conf,
            calibration_probability=calibration_probability,
            calibration_attempts=bucket.attempts,
        )
        reasons.extend(approval.reasons)
        if approval.observation_mode:
            reasons.append("observation_mode_calibrating")

        promotion_allowed = candidate is not None and approval.approved
        patch_id = self.genealogy.make_patch_id(
            trigger_id=packet.fingerprint.digest,
            target_files=[packet.file],
            diff=candidate.diff_digest() if candidate else "",
        )
        self.genealogy.add_node(
            PatchNode(
                patch_id=patch_id,
                trigger_id=packet.trace_id,
                target_files=(packet.file,),
                bug_fingerprint=packet.fingerprint.digest,
                risk_tier=packet.risk_tier,
                status="eligible" if promotion_allowed else "blocked",
                pre_metrics={"error_type": packet.error_type, "line": packet.line},
                validation={
                    "reasons": reasons,
                    "calibration_probability": calibration_probability,
                    "calibration_attempts": bucket.attempts,
                    "approval": approval.to_dict(),
                },
            )
        )
        return FaultPipelineResult(
            packet=packet,
            candidate=candidate,
            promotion_allowed=promotion_allowed,
            lineage_patch_id=patch_id,
            calibration_probability=calibration_probability,
            reasons=tuple(dict.fromkeys(reasons)),
        )


__all__ = [
    "TraceFrame",
    "BugFingerprint",
    "BugPacket",
    "PatchCandidate",
    "FaultPipelineResult",
    "TracebackHarvester",
    "BugLocalizer",
    "ReproductionSynthesizer",
    "TestSynthesizer",
    "StaticRiskClassifier",
    "DeterministicPatchGenerator",
    "FaultToPatchPipeline",
]
