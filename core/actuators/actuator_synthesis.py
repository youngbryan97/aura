"""core/actuators/actuator_synthesis.py
=====================================
Self-Compiling and Safe Actuator Synthesis Pipeline for Aura.
"""

from __future__ import annotations

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


@dataclass
class SynthesisRequest:
    """A formal request to synthesize a new physical actuator."""

    problem_description: str
    failed_actuators: list[str] = field(default_factory=list)
    sensor_context: dict[str, Any] = field(default_factory=dict)
    world_model_snapshot: dict[str, Any] = field(default_factory=dict)
    urgency: float = 0.5
    timestamp: float = field(default_factory=time.time)


class ActuatorSynthesizer:
    """Manages the generation, validation, governance approval, and loading of synthesized actuators."""

    def __init__(self, output_dir: str = "data/synthesized_actuators") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

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
        approved = await self._governance_approve(actuator_name, source_code, request)
        if not approved:
            logger.warning("Governance check DENIED activation of actuator '%s'", actuator_name)
            return None
        logger.info("Governance APPROVED activation of actuator '%s'", actuator_name)

        # 4. Hot-Loading
        logger.info(
            "Hot-loading and registering actuator '%s' into live registry...", actuator_name
        )
        actuator_instance = self.hot_load_actuator(source_code, sandbox_res.details)
        if not actuator_instance:
            logger.error("Failed to hot-load the validated actuator class.")
            return None

        # 5. Persistence
        logger.info("Persisting validated actuator code to disk: %s", actuator_name)
        self.persist_actuator(actuator_name, source_code)

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
    ) -> bool:
        """Consult the Unified Will to get explicit registration approval."""
        try:
            from core.will import ActionDomain, get_will

            decision = get_will().decide(
                content=f"Register synthesized actuator '{actuator_name}'",
                source="actuator_synthesizer",
                domain=ActionDomain.SELF_MODIFICATION,
                priority=min(0.9, 0.4 + 0.5 * request.urgency),
                context={
                    "actuator_name": actuator_name,
                    "source_code": source_code[:1000],
                    "urgency": request.urgency,
                    "failed_actuators": request.failed_actuators,
                },
            )
            return bool(decision.is_approved())
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.warning(
                "Unified Will unavailable for actuator synthesis check: %s. Failing closed.",
                exc,
            )
            return False

    # -- hot-loading & persistence ----------------------------------------------

    def hot_load_actuator(
        self,
        source_code: str,
        metadata: dict[str, Any] | None = None,
    ) -> BaseActuator | None:
        """Register a sandboxed synthesized actuator wrapper in the live registry."""
        try:
            metadata = metadata or ActuatorCodeValidator.validate_sandbox(source_code).details
            actuator_name = str(metadata.get("name") or "").strip()
            if not actuator_name:
                logger.error("No actuator name available for sandboxed synthesized code.")
                return None

            instance = SandboxedSynthesizedActuator(
                name=actuator_name,
                description=str(metadata.get("description") or "Sandboxed synthesized actuator"),
                source_code=source_code,
                trust_score=0.3,
            )

            # Register in live ActuatorRegistry (trust starts at 0.3 for synthesized)
            registry = get_actuator_registry()
            registry.register_synthesized(instance, source_code, trust_score=0.3)

            return instance
        except (AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
            logger.error("Failed to hot-load actuator: %s", exc)
            return None

    def persist_actuator(self, name: str, source_code: str) -> None:
        """Persist verified actuator class to disk for boot reloading."""
        filepath = self.output_dir / f"{name}.py"
        atomic_write_text(filepath, source_code)
        logger.info("Persisted verified actuator '%s' to: %s", name, filepath)

    def reload_persisted_actuators(self) -> list[BaseActuator]:
        """Scans disk, re-validates, and hot-loads all previously generated actuators on boot."""
        reloaded: list[BaseActuator] = []
        if not self.output_dir.exists():
            return reloaded

        logger.info("Scanning persistent storage for previously synthesized actuators...")
        for file in self.output_dir.glob("*.py"):
            try:
                source_code = file.read_text(encoding="utf-8")
                # Fast AST and Sandbox checks on boot to ensure code integrity
                ast_res = ActuatorCodeValidator.validate_ast(source_code)
                sandbox_res = ActuatorCodeValidator.validate_sandbox(source_code)
                if ast_res.success and sandbox_res.success:
                    inst = self.hot_load_actuator(source_code, sandbox_res.details)
                    if inst:
                        logger.info("Successfully reloaded persisted actuator: '%s'", inst.name)
                        reloaded.append(inst)
                else:
                    logger.warning(
                        "Failed validation during boot reload of persisted actuator: %s", file.name
                    )
            except (OSError, RuntimeError, UnicodeDecodeError, TypeError, ValueError) as exc:
                logger.error("Failed to reload persisted actuator %s: %s", file.name, exc)

        return reloaded


# Singleton Pattern
_synthesizer_instance: ActuatorSynthesizer | None = None


def get_actuator_synthesizer() -> ActuatorSynthesizer:
    global _synthesizer_instance
    if _synthesizer_instance is None:
        _synthesizer_instance = ActuatorSynthesizer()
    return _synthesizer_instance
