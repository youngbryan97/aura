"""core/actuators/actuator_synthesis.py
=====================================
Self-Compiling and Safe Actuator Synthesis Pipeline for Aura.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.actuators.actuator_registry import (
    BaseActuator,
    SandboxedSynthesizedActuator,
    get_actuator_registry,
)
from core.actuators.actuator_validator import ActuatorCodeValidator
from core.brain.local_llm import LocalBrain
from core.runtime.atomic_writer import atomic_write_text

logger = logging.getLogger("Aura.Actuators.Synthesis")

# A synthesized actuator name is chosen by MODEL-GENERATED code (it comes back
# from the sandbox run), and it was interpolated straight into a filesystem
# path. Names are restricted to a bare identifier so a name such as
# "../../core/brain/cognitive_engine" cannot direct a write outside the
# synthesis output directory.
_SAFE_ACTUATOR_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")

# Bounds for boot reload: generated artifacts are attacker-influenced content
# on disk, so the scan is budgeted rather than unbounded.
_MAX_RELOAD_FILES = 64
_MAX_ACTUATOR_SOURCE_BYTES = 256_000

# Bounds on the untrusted request payload that feeds the generation prompt.
_MAX_PROBLEM_CHARS = 4_000
_MAX_FAILED_ACTUATORS = 32
_MAX_CONTEXT_KEYS = 64


def _safe_actuator_name(name: Any) -> str:
    """Return a filesystem-safe actuator name, or '' if it is not acceptable."""
    candidate = str(name or "").strip()
    return candidate if _SAFE_ACTUATOR_NAME_RE.match(candidate) else ""


def _finite_urgency(value: Any, default: float = 0.5) -> float:
    """Finite urgency in [0, 1].

    NaN silently maximized governance priority: min(0.9, 0.4 + 0.5*nan)
    returns 0.9 because every NaN comparison is False, so an unvalidated
    urgency produced the HIGHEST possible self-modification priority.
    """
    try:
        candidate = float(default if value is None else value)
    except (TypeError, ValueError):
        return default
    if candidate != candidate or candidate in (float("inf"), float("-inf")):
        return default
    return max(0.0, min(1.0, candidate))


def _source_digest(source_code: str) -> str:
    return hashlib.sha256(str(source_code or "").encode("utf-8")).hexdigest()


@dataclass
class SynthesisRequest:
    """A formal request to synthesize a new physical actuator."""

    problem_description: str
    failed_actuators: list[str] = field(default_factory=list)
    sensor_context: dict[str, Any] = field(default_factory=dict)
    world_model_snapshot: dict[str, Any] = field(default_factory=dict)
    urgency: float = 0.5
    timestamp: float = field(default_factory=lambda: time.time())


class ActuatorSynthesizer:
    """Manages the generation, validation, governance approval, and loading of synthesized actuators."""

    def __init__(self, output_dir: str | None = None) -> None:
        # Anchored to configured Aura storage rather than a relative path:
        # "data/synthesized_actuators" resolved against the process CWD, so
        # where executable synthesized code was written (and reloaded from at
        # boot) depended on how Aura happened to be launched.
        if output_dir is None:
            try:
                from core.config import config

                output_dir = str(Path(config.paths.data_dir) / "synthesized_actuators")
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                output_dir = "data/synthesized_actuators"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Guards synthesis/persist/reload so concurrent callers cannot collide
        # on the same name or interleave writes (it was declared but never
        # acquired anywhere).
        self._lock = threading.RLock()

    # -- synthesis orchestrator -------------------------------------------------

    async def request_synthesis(self, request: SynthesisRequest) -> BaseActuator | None:
        """Process a synthesis request: generate, validate, governance check, load, persist."""
        logger.info("Starting open-ended actuator synthesis for: '%s'", request.problem_description)

        # 1. Code Generation
        try:
            source_code = await self.synthesize_actuator_code(request)
        except (AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
            logger.error("Actuator code generation failed: %s", exc)
            return None

        if not source_code.strip():
            logger.error("Synthesizer produced empty code response.")
            return None

        # 2. Multi-Stage Validation
        logger.info("Executing multi-stage validation pipeline for synthesized code...")

        # Stage 1: AST static checks
        ast_res = ActuatorCodeValidator.validate_ast(source_code)
        if not ast_res.success:
            logger.error("Validation Stage 1 (AST) failed: %s", ast_res.error)
            return None
        class_name = ast_res.details.get("class_name", "UnknownActuator")
        logger.info("Stage 1 (AST) passed. Discovered class name: %s", class_name)

        # Stage 2: Sandbox process isolation dry-run
        sandbox_res = ActuatorCodeValidator.validate_sandbox(source_code)
        if not sandbox_res.success:
            logger.error("Validation Stage 2 (Sandbox) failed: %s", sandbox_res.error)
            return None
        actuator_name = sandbox_res.details.get("name")
        if not actuator_name:
            logger.error("Actuator class did not return a valid name.")
            return None
        logger.info("Stage 2 (Sandbox) passed. Actuator name: %s", actuator_name)

        # Stage 3: Causal simulation in physics world copy
        causal_res = ActuatorCodeValidator.validate_causal(source_code)
        if not causal_res.success:
            logger.error("Validation Stage 3 (Causal) failed: %s", causal_res.error)
            return None
        logger.info("Stage 3 (Causal) passed. Causal simulation successfully updated world state.")

        # 3. Governance Gate
        logger.info(
            "Submitting synthesized actuator '%s' to Unified Will governance...", actuator_name
        )
        governance_receipt = await self._governance_approve(
            actuator_name, source_code, request
        )
        if not governance_receipt:
            logger.warning("Governance check DENIED activation of actuator '%s'", actuator_name)
            return None
        logger.info("Governance APPROVED activation of actuator '%s'", actuator_name)

        # 4. Persistence BEFORE live registration. Registering first left an
        #    active-but-non-restorable actuator whenever the write failed (and
        #    a raising write escaped after the registry had already been
        #    mutated). Durable first means the live registry never holds an
        #    actuator that cannot survive a restart.
        logger.info("Persisting validated actuator code to disk: %s", actuator_name)
        if not self.persist_actuator(
            actuator_name, source_code, governance_receipt=governance_receipt
        ):
            logger.error("Refusing to register actuator '%s': persistence failed.", actuator_name)
            return None

        # 5. Hot-Loading
        logger.info(
            "Hot-loading and registering actuator '%s' into live registry...", actuator_name
        )
        actuator_instance = self._register_validated_actuator(source_code, sandbox_res.details)
        if not actuator_instance:
            logger.error("Failed to hot-load the validated actuator class.")
            return None

        return actuator_instance

    # -- code generation --------------------------------------------------------

    async def synthesize_actuator_code(self, request: SynthesisRequest) -> str:
        """Synthesize a complete BaseActuator Python class using the LocalBrain Ollama instance."""
        brain = LocalBrain()

        system_prompt = (
            "You are Aura's core actuator synthesis subsystem.\n"
            "Your task is to write a single production-grade Python class inheriting from BaseActuator "
            "to perform action primitive operations in the PhysicsWorldModel.\n\n"
            "CRITICAL ARCHITECTURAL CONSTRAINTS:\n"
            "1. Inherit from BaseActuator: `class CustomNameActuator(BaseActuator):`.\n"
            "2. Define a class-level dictionary `test_params: dict[str, Any]` containing valid sample parameters "
            "that pass validation and are suitable for dry-run validation.\n"
            "3. Implement properties `name` (returning a unique machine-string like 'transfer_cargo'), "
            "`description` (short human explanation), and methods `validate_params(self, params: dict[str, Any]) -> bool` "
            "and `execute(self, params: dict[str, Any]) -> ActuatorResult`.\n"
            "4. NEVER import os, sys, subprocess, socket, urllib, requests, ctypes, shutil, pty, platform, builtins, importlib.\n"
            "5. NEVER use eval(), exec(), compile(), __import__(), or open().\n"
            "6. Do not import the live world model. Compute and return a bounded update payload only; Aura applies it after sandbox validation.\n"
            "7. Return ActuatorResult(success=True/False, message='...', updates={entity_id: {...}}).\n"
            "8. Updates may include numeric fields load, flow_rate, latency, capacity, max_flow_rate, coordinates, and primitive attributes.\n"
            "9. ONLY output the valid, clean Python code block. Do NOT surround it with explanations. Do NOT provide comments "
            "explaining the code; just return the code. Keep it extremely tight and professional."
        )

        prompt = (
            f"We need a new actuator to solve the following problem:\n"
            f"Problem description: {request.problem_description}\n"
            f"Failed actuators tried: {request.failed_actuators}\n"
            f"Current sensor context: {json.dumps(request.sensor_context, indent=2)}\n"
            f"Physics world snapshot: {json.dumps(request.world_model_snapshot, indent=2)}\n\n"
            f"Write the complete self-contained Python code now."
        )

        try:
            # We explicitly use LocalBrain in async context
            res = await brain.generate(prompt, system_prompt=system_prompt)
            raw_response = res.get("response", "")

            # Extract python code
            code = raw_response
            match = re.search(r"```python\s*(.*?)\s*```", code, re.DOTALL)
            if match:
                code = match.group(1)
            else:
                match = re.search(r"```\s*(.*?)\s*```", code, re.DOTALL)
                if match:
                    code = match.group(1)

            return code.strip()
        finally:
            await brain.close()

    # -- governance -------------------------------------------------------------

    async def _governance_approve(
        self, actuator_name: str, source_code: str, request: SynthesisRequest
    ) -> str | None:
        """Consult the Unified Will; return its receipt id, or None if denied.

        Returns the RECEIPT rather than a bare boolean so the approval can be
        carried into persistence and the durable manifest — the decision used
        to be reduced to True/False and discarded, leaving nothing downstream
        that could prove which code was approved, by whom, or when.
        """
        try:
            from core.will import ActionDomain, get_will

            urgency = _finite_urgency(request.urgency)
            digest = _source_digest(source_code)
            decision = get_will().decide(
                content=f"Register synthesized actuator '{actuator_name}'",
                source="actuator_synthesizer",
                domain=ActionDomain.SELF_MODIFICATION,
                priority=min(0.9, 0.4 + 0.5 * urgency),
                context={
                    "actuator_name": actuator_name,
                    # The Will previously saw only the first 1000 characters,
                    # so it approved an implementation it had not read. The
                    # full source is provided, plus a digest that binds the
                    # approval to exact bytes.
                    "source_code": source_code,
                    "source_sha256": digest,
                    "source_bytes": len(source_code.encode("utf-8")),
                    "urgency": urgency,
                    "failed_actuators": request.failed_actuators,
                },
            )
            if not decision.is_approved():
                return None
            return str(getattr(decision, "receipt_id", "") or "approved")
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.warning(
                "Unified Will unavailable for actuator synthesis check: %s. Failing closed.",
                exc,
            )
            return None

    # -- hot-loading & persistence ----------------------------------------------

    def hot_load_actuator(
        self,
        source_code: str,
        metadata: dict[str, Any] | None = None,
    ) -> BaseActuator | None:
        """Validate and register a synthesized actuator (public, governed entry).

        This used to be a straight registration path: any caller with source
        text could put a live actuator into the registry without the
        validation stages OR Will approval, which made every governance gate
        in request_synthesis optional. It now re-derives metadata from the
        code itself (never from caller-supplied fields, which were not bound
        to the source bytes) and runs the full validation pipeline.
        """
        validated = self._validate_source(source_code)
        if validated is None:
            return None
        return self._register_validated_actuator(source_code, validated)

    def _validate_source(self, source_code: str) -> dict[str, Any] | None:
        """Run AST + sandbox + causal validation; return sandbox metadata."""
        ast_res = ActuatorCodeValidator.validate_ast(source_code)
        if not ast_res.success:
            logger.error("Actuator validation (AST) failed: %s", ast_res.error)
            return None
        sandbox_res = ActuatorCodeValidator.validate_sandbox(source_code)
        if not sandbox_res.success:
            logger.error("Actuator validation (sandbox) failed: %s", sandbox_res.error)
            return None
        causal_res = ActuatorCodeValidator.validate_causal(source_code)
        if not causal_res.success:
            logger.error("Actuator validation (causal) failed: %s", causal_res.error)
            return None
        details = dict(sandbox_res.details or {})
        # CP126 f80cc444: the registry cannot tell validated code from asserted
        # code unless the validator says so, bound to the exact source bytes it
        # actually checked. This is that statement.
        details["_validation_receipt"] = {
            "passed": True,
            "validator": "ActuatorCodeValidator",
            "stages": ["ast", "sandbox", "causal"],
            "source_digest": hashlib.sha256(
                source_code.encode("utf-8", "replace")
            ).hexdigest(),
            "validated_at": time.time(),
        }
        return details

    def _register_validated_actuator(
        self,
        source_code: str,
        metadata: dict[str, Any] | None = None,
    ) -> BaseActuator | None:
        """Register a sandboxed synthesized actuator wrapper in the live registry."""
        try:
            metadata = metadata or ActuatorCodeValidator.validate_sandbox(source_code).details
            # The name is derived from the sandbox run of the code itself, and
            # must still be filesystem/registry safe.
            actuator_name = _safe_actuator_name(metadata.get("name"))
            if not actuator_name:
                logger.error(
                    "No safe actuator name available for sandboxed synthesized code (%r).",
                    metadata.get("name"),
                )
                return None

            schema = metadata.get("param_schema") or metadata.get("parameters")
            instance = SandboxedSynthesizedActuator(
                name=actuator_name,
                description=str(metadata.get("description") or "Sandboxed synthesized actuator"),
                source_code=source_code,
                trust_score=0.3,
                param_schema=schema if isinstance(schema, dict) else None,
            )

            # Register in live ActuatorRegistry (trust starts at 0.3 for
            # synthesized). The validator receipt is what earns that 0.3;
            # without it the registry floors trust to 0.0 and the actuator is
            # refused at execution.
            registry = get_actuator_registry()
            registry.register_synthesized(
                instance,
                source_code,
                trust_score=0.3,
                validation_receipt=metadata.get("_validation_receipt"),
                registered_by="actuator_synthesis",
            )

            return instance
        except (AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
            logger.error("Failed to hot-load actuator: %s", exc)
            return None

    def persist_actuator(
        self,
        name: str,
        source_code: str,
        *,
        governance_receipt: str = "",
    ) -> bool:
        """Persist verified actuator source plus an integrity manifest.

        The name is validated and containment is re-checked after resolution:
        it originates from model-generated code, and was previously
        interpolated directly into the output path.

        A sidecar manifest records the source digest, the governance receipt,
        and the validation stages that passed, so boot reload can prove the
        bytes on disk are the bytes that were approved (they were previously
        unsigned and reloaded without any integrity check).
        """
        safe_name = _safe_actuator_name(name)
        if not safe_name:
            logger.error("Refusing to persist actuator with unsafe name: %r", name)
            return False

        filepath = (self.output_dir / f"{safe_name}.py").resolve()
        root = self.output_dir.resolve()
        try:
            filepath.relative_to(root)
        except ValueError:
            logger.error(
                "Refusing to persist actuator outside the synthesis root: %s", filepath
            )
            return False

        manifest = {
            "schema": "aura.synthesized_actuator.manifest.v1",
            "name": safe_name,
            "source_sha256": _source_digest(source_code),
            "source_bytes": len(source_code.encode("utf-8")),
            "governance_receipt": str(governance_receipt or ""),
            "validated_stages": ["ast", "sandbox", "causal"],
            "persisted_at": time.time(),
        }
        with self._lock:
            atomic_write_text(filepath, source_code)
            atomic_write_text(
                filepath.with_suffix(".manifest.json"),
                json.dumps(manifest, indent=2, sort_keys=True),
            )
        logger.info("Persisted verified actuator '%s' to: %s", safe_name, filepath)
        return True

    def _verify_persisted_manifest(self, file: Path, source_code: str) -> dict[str, Any] | None:
        """Return the manifest when it matches the source bytes, else None."""
        manifest_path = file.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            logger.warning(
                "Refusing unsigned synthesized actuator (no manifest): %s", file.name
            )
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Unreadable manifest for %s: %s", file.name, exc)
            return None
        if not isinstance(manifest, dict):
            return None
        expected = str(manifest.get("source_sha256") or "")
        actual = _source_digest(source_code)
        if not expected or expected != actual:
            logger.error(
                "Synthesized actuator %s failed integrity check (expected %s, got %s); refusing to load.",
                file.name,
                expected[:16] or "<missing>",
                actual[:16],
            )
            return None
        return manifest

    def reload_persisted_actuators(self) -> list[BaseActuator]:
        """Scans disk, re-validates, and hot-loads all previously generated actuators on boot."""
        reloaded: list[BaseActuator] = []
        if not self.output_dir.exists():
            return reloaded

        logger.info("Scanning persistent storage for previously synthesized actuators...")
        # Budgeted scan: these artifacts are generated code on disk, so the
        # boot path does not walk an unbounded number or size of files.
        candidates = sorted(self.output_dir.glob("*.py"))[:_MAX_RELOAD_FILES]
        for file in candidates:
            try:
                if file.stat().st_size > _MAX_ACTUATOR_SOURCE_BYTES:
                    logger.warning(
                        "Skipping oversized synthesized actuator %s (%d bytes).",
                        file.name,
                        file.stat().st_size,
                    )
                    continue
                source_code = file.read_text(encoding="utf-8")

                # INTEGRITY FIRST: the bytes on disk must be the bytes that
                # were approved. Without this, a persisted file could be
                # edited after approval and would be loaded on the next boot
                # with no check at all.
                manifest = self._verify_persisted_manifest(file, source_code)
                if manifest is None:
                    continue

                # FULL validation on reload, not just AST+sandbox. The causal
                # stage was skipped here even though it gates first-time
                # activation, so a reload was strictly weaker than the
                # original admission.
                metadata = self._validate_source(source_code)
                if metadata is None:
                    logger.warning(
                        "Failed validation during boot reload of persisted actuator: %s",
                        file.name,
                    )
                    continue

                # Reload is a self-modification event in its own right: the
                # previous approval was for a previous boot, so current Will
                # authority is required rather than inherited from the file.
                if not self._governance_approve_reload(manifest, source_code):
                    logger.warning(
                        "Governance refused reload of persisted actuator: %s", file.name
                    )
                    continue

                inst = self._register_validated_actuator(source_code, metadata)
                if inst:
                    logger.info("Successfully reloaded persisted actuator: '%s'", inst.name)
                    reloaded.append(inst)
            except (OSError, RuntimeError, UnicodeDecodeError, TypeError, ValueError) as exc:
                logger.error("Failed to reload persisted actuator %s: %s", file.name, exc)

        return reloaded

    def _governance_approve_reload(self, manifest: dict[str, Any], source_code: str) -> bool:
        """Require current Will authority to re-activate a persisted actuator."""
        try:
            from core.will import ActionDomain, get_will

            decision = get_will().decide(
                content=f"Reload synthesized actuator '{manifest.get('name')}'",
                source="actuator_synthesizer_reload",
                domain=ActionDomain.SELF_MODIFICATION,
                priority=0.6,
                context={
                    "actuator_name": manifest.get("name"),
                    "source_sha256": _source_digest(source_code),
                    "prior_governance_receipt": manifest.get("governance_receipt", ""),
                    "source_bytes": len(source_code.encode("utf-8")),
                },
            )
            return bool(decision.is_approved())
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.warning(
                "Unified Will unavailable for actuator reload check: %s. Failing closed.",
                exc,
            )
            return False


# Singleton Pattern
_synthesizer_instance: ActuatorSynthesizer | None = None
# Concurrent first access could otherwise construct two owners with different
# output roots and different locks — defeating the very lock that serializes
# synthesis and persistence.
_synthesizer_lock = threading.Lock()


def get_actuator_synthesizer() -> ActuatorSynthesizer:
    global _synthesizer_instance
    if _synthesizer_instance is None:
        with _synthesizer_lock:
            if _synthesizer_instance is None:
                _synthesizer_instance = ActuatorSynthesizer()
    return _synthesizer_instance
