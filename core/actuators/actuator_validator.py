"""core/actuators/actuator_validator.py
======================================
Multi-stage validation pipeline for runtime-synthesized actuator code.
"""

from __future__ import annotations

import ast
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Set

logger = logging.getLogger("Aura.Actuators.Validator")


@dataclass
class ValidationResult:
    """The result of running a validation stage."""

    success: bool
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class ActuatorCodeValidator:
    """Multi-stage validation pipeline for synthesized actuator code."""

    FORBIDDEN_IMPORTS: Set[str] = {
        "os", "sys", "subprocess", "socket", "urllib", "requests", "ctypes",
        "shutil", "pty", "platform", "builtins", "importlib", "gc", "multiprocessing",
        "threading", "asyncio", "http", "tempfile", "pdb"
    }

    FORBIDDEN_BUILTINS: Set[str] = {
        "eval", "exec", "compile", "__import__", "open", "input"
    }

    REQUIRED_METHODS: Set[str] = {"name", "description", "validate_params", "execute"}

    @classmethod
    def validate_ast(cls, source: str) -> ValidationResult:
        """Stage 1: Static AST validation to check for illegal imports, builtins, and structure."""
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return ValidationResult(False, f"Syntax error in source code: {exc}")

        # Check for forbidden imports and builtins
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    base_module = name.name.split('.')[0]
                    if base_module in cls.FORBIDDEN_IMPORTS:
                        return ValidationResult(False, f"Forbidden import: {name.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    base_module = node.module.split('.')[0]
                    if base_module in cls.FORBIDDEN_IMPORTS:
                        return ValidationResult(False, f"Forbidden import: {node.module}")

            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in cls.FORBIDDEN_BUILTINS:
                        return ValidationResult(False, f"Forbidden function call: {node.func.id}")

        # Verify class definition
        class_nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        if not class_nodes:
            return ValidationResult(False, "No class defined in the source code")
        if len(class_nodes) > 1:
            return ValidationResult(False, "Multiple classes defined; only one class inheriting from BaseActuator is allowed")

        cls_node = class_nodes[0]
        # Check base classes (one of them should be BaseActuator)
        has_base_actuator = False
        for base in cls_node.bases:
            if isinstance(base, ast.Name) and base.id == "BaseActuator":
                has_base_actuator = True
                break
            elif isinstance(base, ast.Attribute) and base.attr == "BaseActuator":
                has_base_actuator = True
                break

        if not has_base_actuator:
            return ValidationResult(False, f"Class '{cls_node.name}' must inherit from BaseActuator")

        # Check methods defined
        defined_methods = {node.name for node in cls_node.body if isinstance(node, ast.FunctionDef)}
        missing = cls.REQUIRED_METHODS - defined_methods
        if missing:
            return ValidationResult(False, f"Class '{cls_node.name}' is missing required methods/properties: {missing}")

        return ValidationResult(True, details={"class_name": cls_node.name})

    @classmethod
    def validate_sandbox(cls, source: str) -> ValidationResult:
        """Stage 2: Runs the source in a secure isolated process via run_untrusted to test instantiation."""
        from core.sandbox.runner import run_untrusted

        # Build a script that wraps the untrusted code and verifies it instantiates and validates params
        test_script = f"""
class BaseActuator:
    pass

class ActuatorResult:
    def __init__(self, success, message, updates=None):
        self.success = success
        self.message = message
        self.updates = updates or {{}}

{source}

import json
# Find the actuator class
classes = [obj for name, obj in list(globals().items()) if isinstance(obj, type) and obj.__name__ != 'BaseActuator' and issubclass(obj, BaseActuator)]
if not classes:
    print(json.dumps({{"success": False, "error": "No BaseActuator subclass found"}}))
else:
    try:
        inst = classes[0]()
        # Call validate_params with empty dict (should not raise exception, should return a bool)
        val = inst.validate_params({{}})
        # Check test_params attribute
        test_params = getattr(inst, "test_params", {{}})
        print(json.dumps({{
            "success": True,
            "name": inst.name,
            "description": inst.description,
            "validate_empty": val,
            "has_test_params": isinstance(test_params, dict)
        }}))
    except Exception as e:
        print(json.dumps({{"success": False, "error": "Instantiation error: " + str(e)}}))
"""
        try:
            res = run_untrusted(test_script, timeout=5, mem_bytes=100 * 1024 * 1024)
            status = res.get("status")
            if status != "completed" and status != "ok":
                stderr = res.get("stderr", "")
                stdout = res.get("stdout", "")
                return ValidationResult(False, f"Sandbox failed with status: {status}. stderr: {stderr}. stdout: {stdout}")

            stdout = res.get("stdout", "")
            try:
                payload = json.loads(stdout.strip())
            except json.JSONDecodeError:
                return ValidationResult(False, f"Sandbox returned invalid JSON: {stdout}")

            if not payload.get("success"):
                return ValidationResult(False, f"Sandbox validation failed: {payload.get('error')}")

            return ValidationResult(True, details=payload)
        except Exception as exc:
            return ValidationResult(False, f"Sandbox execution failed with exception: {exc}")

    @classmethod
    def validate_causal(cls, source: str, world_snapshot: dict | None = None) -> ValidationResult:
        """Stage 3: Load the actuator inside this process, run against a temporary copy of PhysicsWorldModel, and verify safety/boundedness."""
        try:
            # 1. Compile in standard namespace
            from core.actuators.actuator_registry import BaseActuator, ActuatorResult

            namespace = {
                "BaseActuator": BaseActuator,
                "ActuatorResult": ActuatorResult,
                "__builtins__": __builtins__
            }

            code_obj = compile(source, "<synthesized_actuator>", "exec")
            exec(code_obj, namespace)

            classes = [obj for obj in namespace.values() if isinstance(obj, type) and obj != BaseActuator and issubclass(obj, BaseActuator)]
            if not classes:
                return ValidationResult(False, "No BaseActuator subclass found after compilation")

            actuator_cls = classes[0]
            instance = actuator_cls()

            # 2. Setup a sandboxed/copied PhysicsWorldModel
            from core.world.world_model import get_physics_world_model, PhysicsWorldModel, WorldEntity
            
            original_model = get_physics_world_model()
            copied_model = PhysicsWorldModel(seed=original_model.seed)
            copied_model.sim_time = original_model.sim_time
            copied_model.entities = {}
            for eid, ent in original_model.entities.items():
                copied_model.entities[eid] = WorldEntity(
                    entity_id=ent.entity_id,
                    kind=ent.kind,
                    capacity=ent.capacity,
                    load=ent.load,
                    flow_rate=ent.flow_rate,
                    max_flow_rate=ent.max_flow_rate,
                    latency=ent.latency,
                    coordinates=ent.coordinates,
                    attributes=ent.attributes.copy()
                )

            # 3. Patch global singleton temporarily during test execution
            import core.world.world_model
            old_inst = core.world.world_model._instance
            
            test_params = getattr(instance, "test_params", {})
            try:
                core.world.world_model._instance = copied_model
                
                # Check parameter validation
                if not instance.validate_params(test_params):
                    return ValidationResult(False, f"Actuator rejected its own test_params: {test_params}")
                
                # Execute actuator
                res = instance.execute(test_params)
                if not res.success:
                    return ValidationResult(False, f"Actuator execution returned success=False: {res.message}")
                
                # Verify safety & boundedness of state
                for ent in copied_model.entities.values():
                    if not math.isfinite(ent.load) or ent.load < 0.0 or ent.load > ent.capacity:
                        return ValidationResult(False, f"Actuator caused invalid entity load state: {ent.entity_id} load={ent.load}")
                    if not math.isfinite(ent.flow_rate) or ent.flow_rate < 0.0:
                        return ValidationResult(False, f"Actuator caused invalid flow rate: {ent.entity_id} flow={ent.flow_rate}")
                
                return ValidationResult(True, details={"message": res.message, "updates": res.updates})
            finally:
                core.world.world_model._instance = old_inst
                
        except Exception as exc:
            return ValidationResult(False, f"Causal simulation validation failed: {exc}")
