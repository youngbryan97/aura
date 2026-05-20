"""core/adaptation/immune_executor.py
==================================
Immune Heuristic Executor.

A sandboxed, symbolic interpreter that safely parses and executes evolved
behavioral rules (instruction graphs) from active immune cells, translating
them into physical actions via the ActuatorRegistry.
"""

from __future__ import annotations

import ast
import logging
import math
import operator
import re
from typing import Any

from core.actuators.actuator_registry import ActuatorResult, get_actuator_registry
from core.sensors.sensor_registry import get_sensor_registry

logger = logging.getLogger("Aura.ImmuneHeuristicExecutor")

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class ImmuneHeuristicExecutor:
    """Symbolic, sandboxed instruction interpreter for immune cell behavioral rules."""

    def __init__(self) -> None:
        pass

    def evaluate_condition(self, condition: dict[str, Any], sensors_data: dict[str, float]) -> bool:
        """Evaluates a single condition against current sensor values safely."""
        sensor_id = condition.get("sensor")
        operator = condition.get("operator")
        target_value = condition.get("value")

        if not sensor_id or not operator or target_value is None:
            logger.warning("Malformed condition in behavioral rule: %s", condition)
            return False

        if sensor_id not in sensors_data:
            logger.warning("Sensor ID '%s' in condition not found in telemetry", sensor_id)
            return False

        sensor_val = sensors_data[sensor_id]

        try:
            target_val = float(target_value)
            sensor_val = float(sensor_val)
            if not math.isfinite(target_val) or not math.isfinite(sensor_val):
                return False

            if operator == ">":
                return sensor_val > target_val
            elif operator == "<":
                return sensor_val < target_val
            elif operator == ">=":
                return sensor_val >= target_val
            elif operator == "<=":
                return sensor_val <= target_val
            elif operator == "==":
                return abs(sensor_val - target_val) < 1e-6
            elif operator == "!=":
                return abs(sensor_val - target_val) >= 1e-6
            else:
                logger.warning("Unknown operator '%s' in condition", operator)
                return False
        except (ValueError, TypeError) as exc:
            logger.error("Failed evaluating condition %s: %s", condition, exc)
            return False

    def resolve_params(
        self, params: dict[str, Any], sensors_data: dict[str, float]
    ) -> dict[str, Any]:
        """Resolves dynamic sensor reference variables in parameters (e.g. '$port_east_load * 0.5')."""
        resolved = {}
        for key, val in params.items():
            if isinstance(val, str) and val.startswith("$"):
                # Clean and parse variable
                expr = val[1:].strip()
                # Basic safety validation: only allow alphanumeric, underscores, space, operators, decimals
                if not re.match(r"^[\w\s\.\+\-\*\/]+$", expr):
                    logger.warning("Safety violation in param expression '%s'", val)
                    resolved[key] = val
                    continue

                # Find any sensor names in the expression and replace them
                substituted = expr
                for s_id, s_val in sensors_data.items():
                    if s_id in substituted:
                        substituted = re.sub(
                            r"\b" + re.escape(s_id) + r"\b", str(s_val), substituted
                        )

                try:
                    res_val = self._safe_eval_numeric(substituted)
                    resolved[key] = float(res_val)
                except Exception as exc:
                    logger.warning(
                        "Failed resolving expression '%s' (substituted as '%s'): %s",
                        val,
                        substituted,
                        exc,
                    )
                    resolved[key] = val
            else:
                resolved[key] = val
        return resolved

    def _safe_eval_numeric(self, expression: str) -> float:
        """Evaluate a sanitized arithmetic expression without Python eval."""
        parsed = ast.parse(expression, mode="eval")
        result = self._eval_ast_node(parsed.body)
        if not math.isfinite(result):
            raise ValueError("non-finite expression result")
        return result

    def _eval_ast_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            value = float(node.value)
            if not math.isfinite(value):
                raise ValueError("non-finite literal")
            return value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            left = self._eval_ast_node(node.left)
            right = self._eval_ast_node(node.right)
            if isinstance(node.op, ast.Div) and abs(right) < 1e-12:
                raise ZeroDivisionError("division by zero")
            return float(_ALLOWED_BINOPS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
            return float(_ALLOWED_UNARYOPS[type(node.op)](self._eval_ast_node(node.operand)))
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    def execute_rule(self, rule: dict[str, Any]) -> dict[str, Any]:
        """Parses and executes a behavioral rule graph.

        Example Rule Format:
        {
          "conditions": [
            {"sensor": "port_east_load", "operator": ">", "value": 750.0}
          ],
          "actions": [
            {
              "actuator": "reallocate_flow",
              "params": {
                "source_id": "Port_East",
                "target_id": "Port_West",
                "amount": "$port_east_load * 0.25"
              }
            }
          ]
        }
        """
        registry = get_sensor_registry()
        # Make sure we pull the latest values from the physics simulator
        registry.sync_from_world_model()
        sensors_data = registry.read_all()

        conditions = rule.get("conditions", [])
        actions = rule.get("actions", [])

        if not actions:
            return {
                "conditions_met": False,
                "actions_executed": [],
                "success": True,
                "message": "No actions to execute.",
            }

        # 1. Evaluate conditions
        conditions_met = True
        for cond in conditions:
            if not self.evaluate_condition(cond, sensors_data):
                conditions_met = False
                break

        if not conditions_met:
            return {
                "conditions_met": False,
                "actions_executed": [],
                "success": True,
                "message": "Conditions not satisfied, skipped execution.",
            }

        # 2. Execute actions
        actuator_registry = get_actuator_registry()
        executed_actions = []
        overall_success = True
        messages = []

        for action in actions:
            actuator_name = action.get("actuator")
            raw_params = action.get("params", {})

            if not actuator_name:
                logger.warning("Action missing 'actuator' identifier")
                overall_success = False
                continue

            resolved_params = self.resolve_params(raw_params, sensors_data)
            logger.info(
                "Executing immune action '%s' with params: %s", actuator_name, resolved_params
            )

            res: ActuatorResult = actuator_registry.execute_action(actuator_name, resolved_params)
            executed_actions.append(
                {
                    "actuator": actuator_name,
                    "params": resolved_params,
                    "success": res.success,
                    "message": res.message,
                }
            )

            if not res.success:
                overall_success = False
            messages.append(res.message)

        # Sync telemetry again post-execution
        registry.sync_from_world_model()

        return {
            "conditions_met": True,
            "actions_executed": executed_actions,
            "success": overall_success,
            "message": "; ".join(messages),
        }


# Singleton pattern
_instance: ImmuneHeuristicExecutor | None = None


def get_immune_executor() -> ImmuneHeuristicExecutor:
    global _instance
    if _instance is None:
        _instance = ImmuneHeuristicExecutor()
    return _instance
