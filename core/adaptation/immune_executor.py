"""core/adaptation/immune_executor.py
==================================
Immune Heuristic Executor.

A sandboxed, symbolic interpreter that safely parses and executes evolved
behavioral rules (instruction graphs) from active immune cells, translating
them into physical actions via the ActuatorRegistry.
"""

from __future__ import annotations

import ast
import asyncio
import logging
import math
import operator
import re
from typing import Any

from core.actuators.actuator_registry import ActuatorResult, get_actuator_registry
from core.runtime.errors import record_degradation
from core.runtime.service_access import resolve_orchestrator
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

    _SIMULATION_SOURCES = frozenset(
        {
            "causal_fitness_lab",
            "offline_coevolution_lab",
            "immune_simulation",
            "test_simulation",
        }
    )
    _MAINTENANCE_SOURCES = frozenset(
        {
            "adaptive_immune_system",
            "adaptive_immunity",
            "immune_maintenance",
        }
    )

    # A remedy that keeps failing is a finding, not background noise. The live
    # 2026-07-25 idle window fired reallocate_flow 247 times with identical
    # parameters; 192 of those failed and produced no log line at all, because
    # the failure only ever landed in a returned dict nobody read. An immune
    # system that cannot tell "I treated it" from "I tried and it did nothing"
    # is not an immune system.
    _REPEAT_FAILURE_ESCALATION = 5
    _FAILURE_LEDGER_CAP = 64
    _failure_streaks: dict[str, int] = {}

    @classmethod
    def _note_action_outcome(
        cls, actuator_name: str, params: dict[str, Any], result: ActuatorResult
    ) -> None:
        """Make a failing immune action visible, and escalate a stuck one."""

        key = f"{actuator_name}:{sorted(params.items(), key=lambda kv: kv[0])}"
        if result.success:
            cls._failure_streaks.pop(key, None)
            return

        streak = cls._failure_streaks.get(key, 0) + 1
        if len(cls._failure_streaks) >= cls._FAILURE_LEDGER_CAP and key not in cls._failure_streaks:
            cls._failure_streaks.clear()
        cls._failure_streaks[key] = streak

        if streak == 1:
            logger.warning(
                "Immune action '%s' did not take effect: %s",
                actuator_name, result.message,
            )
        elif streak == cls._REPEAT_FAILURE_ESCALATION:
            record_degradation(
                "immune_executor",
                RuntimeError(
                    f"immune action {actuator_name} failed {streak} times with "
                    f"identical parameters: {result.message}"
                ),
                severity="warning",
                action=(
                    "kept re-issuing an immune remedy that never took effect; "
                    "the underlying condition is not being relieved"
                ),
                extra={"actuator": actuator_name, "streak": streak},
            )

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
            if condition.get("relative_to_baseline"):
                baseline = float(condition.get("baseline"))
                if not math.isfinite(baseline):
                    return False
                target_val = baseline * target_val
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
                except (ArithmeticError, SyntaxError, TypeError, ValueError) as exc:
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

    @staticmethod
    def _rule_action_contract(
        actions: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> tuple[bool, str]:
        """Reject evolved actions outside the explicit immune contract."""
        source = str(context.get("source") or "").strip().lower()
        governed_immune_source = source in (
            ImmuneHeuristicExecutor._SIMULATION_SOURCES
            | ImmuneHeuristicExecutor._MAINTENANCE_SOURCES
        )
        if not governed_immune_source:
            return True, ""
        registry = get_actuator_registry()
        for action in actions:
            name = str(action.get("actuator") or "").strip()
            actuator = registry.get_actuator(name) if name else None
            if actuator is None:
                return False, f"immune rule actuator '{name or '<missing>'}' is not registered"
            if not bool(getattr(actuator, "immune_rule_compatible", False)):
                return False, f"actuator '{name}' has no immune-rule simulation contract"
            if bool(getattr(actuator, "requires_authority", True)):
                return (
                    False,
                    f"actuator '{name}' is consequential and cannot be evolved as an immune rule",
                )
        return True, ""

    def _authorization_preflight(
        self,
        context: dict[str, Any],
    ) -> tuple[bool | None, str, str]:
        """Classify simulation, denial, deferral, or required authority."""
        source = str(context.get("source") or "").strip().lower()
        simulation_only = bool(
            context.get("isolated_simulation")
            and context.get("world_model_isolated")
            and source in self._SIMULATION_SOURCES
        )
        if simulation_only:
            return True, "simulation_authorized", "isolated simulation context"
        if source not in self._MAINTENANCE_SOURCES:
            return (
                False,
                "governance_denied",
                "immune behavioral rules require an isolated simulation or adaptive-maintenance context",
            )

        try:
            from core.runtime.background_policy import (
                MAINTENANCE_BACKGROUND_POLICY,
                background_activity_reason,
            )

            orchestrator = resolve_orchestrator()
            defer_reason = str(
                background_activity_reason(
                    orchestrator,
                    profile=MAINTENANCE_BACKGROUND_POLICY,
                    allow_no_user_anchor=True,
                )
                or ""
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Immune behavioral rule background-policy probe failed: %s", exc)
            return (
                False,
                "governance_denied",
                "background policy unavailable for immune behavioral rule",
            )
        if defer_reason:
            return (
                False,
                "deferred",
                f"immune behavioral action deferred by maintenance policy: {defer_reason}",
            )
        return None, "authority_required", "adaptive maintenance requires authority"

    def _authorize_execution(self, context: dict[str, Any]) -> tuple[bool, str, str]:
        """Authorize a synchronous immune behavioral rule before actuation."""
        allowed, status, message = self._authorization_preflight(context)
        if allowed is not None:
            return allowed, status, message
        try:
            from core.executive.authority_gateway import get_authority_gateway

            decision = get_authority_gateway().authorize_state_mutation_sync(
                "adaptive_immune_system",
                "adaptive_immune_behavioral_rule",
                priority=float(context.get("priority", 0.68) or 0.68),
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Immune behavioral rule AuthorityGateway probe failed: %s", exc)
            return (
                False,
                "governance_denied",
                "AuthorityGateway unavailable for immune behavioral rule",
            )
        if not getattr(decision, "approved", False):
            reason = str(getattr(decision, "reason", "") or "").strip()
            return (
                False,
                "governance_denied",
                f"immune behavioral action denied by AuthorityGateway: {reason}",
            )
        return True, "authorized", "adaptive maintenance authorized"

    async def _authorize_execution_async(
        self,
        context: dict[str, Any],
    ) -> tuple[bool, str, str]:
        """Keep blocking policy observation off-loop and authority on-loop."""
        allowed, status, message = await asyncio.to_thread(
            self._authorization_preflight,
            context,
        )
        if allowed is not None:
            return allowed, status, message
        try:
            from core.executive.authority_gateway import get_authority_gateway

            decision = await get_authority_gateway().authorize_state_mutation(
                "adaptive_immune_system",
                "adaptive_immune_behavioral_rule",
                priority=float(context.get("priority", 0.68) or 0.68),
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Immune behavioral rule AuthorityGateway probe failed: %s", exc)
            return (
                False,
                "governance_denied",
                "AuthorityGateway unavailable for immune behavioral rule",
            )
        if not getattr(decision, "approved", False):
            reason = str(getattr(decision, "reason", "") or "").strip()
            return (
                False,
                "governance_denied",
                f"immune behavioral action denied by AuthorityGateway: {reason}",
            )
        return True, "authorized", "adaptive maintenance authorized"

    @staticmethod
    def _schedule_missing_actuator_synthesis(
        actuator_name: str,
        sensors_data: dict[str, float],
    ) -> None:
        logger.warning(
            "Actuator '%s' not found. Launching open-ended actuator synthesis...",
            actuator_name,
        )
        try:
            from core.actuators.actuator_synthesis import (
                SynthesisRequest,
                get_actuator_synthesizer,
            )
            from core.utils.task_tracker import get_task_tracker
            from core.world.world_model import get_physics_world_model

            world_snapshot = {
                entity_id: {
                    "kind": entity.kind,
                    "load": entity.load,
                    "capacity": entity.capacity,
                    "flow_rate": entity.flow_rate,
                    "latency": entity.latency,
                }
                for entity_id, entity in get_physics_world_model().entities.items()
            }
            request = SynthesisRequest(
                problem_description=(
                    f"Action requested but actuator '{actuator_name}' is not found in the registry."
                ),
                failed_actuators=[actuator_name],
                sensor_context=sensors_data,
                world_model_snapshot=world_snapshot,
                urgency=0.8,
            )

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                get_task_tracker().create_task(
                    get_actuator_synthesizer().request_synthesis(request),
                    name=f"actuator_synthesis_{actuator_name}",
                )
                return

            import threading

            def run_synthesis() -> None:
                asyncio.run(get_actuator_synthesizer().request_synthesis(request))

            threading.Thread(
                target=run_synthesis,
                daemon=True,
                name=f"actuator-synthesis-{actuator_name}",
            ).start()
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            logger.error("Failed to trigger background actuator synthesis: %s", exc)

    def execute_rule(
        self,
        rule: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
        context = dict(context or {})
        actions = list(rule.get("actions", []) or [])
        contract_ok, contract_message = self._rule_action_contract(actions, context)
        if not contract_ok:
            return {
                "conditions_met": False,
                "actions_executed": [],
                "success": False,
                "status": "unsupported_rule",
                "message": contract_message,
            }
        authorized, status, message = self._authorize_execution(context)
        if not authorized:
            return {
                "conditions_met": False,
                "actions_executed": [],
                "success": False,
                "status": status,
                "deferred": status == "deferred",
                "message": message,
            }

        registry = get_sensor_registry()
        # Make sure we pull the latest values from the physics simulator
        registry.sync_from_world_model()
        sensors_data = registry.read_all()

        conditions = rule.get("conditions", [])
        if not actions:
            return {
                "conditions_met": False,
                "actions_executed": [],
                "success": True,
                "status": "no_actions",
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
                "status": "conditions_not_met",
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

            res: ActuatorResult = actuator_registry.execute_action(
                actuator_name,
                resolved_params,
                context={
                    "source": context.get("source") or "adaptive_immune_system",
                    "priority": float(context.get("priority", 0.68) or 0.68),
                    "is_critical": bool(context.get("is_critical", False)),
                    "isolated_simulation": bool(context.get("isolated_simulation", False)),
                },
            )

            # Trigger dynamic synthesis if actuator is missing
            self._note_action_outcome(actuator_name, resolved_params, res)
            if not res.success and ("not found" in res.message or "not registered" in res.message):
                self._schedule_missing_actuator_synthesis(actuator_name, sensors_data)

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
            "status": "executed" if executed_actions else "no_actions_executed",
            "message": "; ".join(messages),
        }

    async def execute_rule_async(
        self,
        rule: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Async runtime path for immune actions and their authority lifecycle."""
        context = dict(context or {})
        actions = list(rule.get("actions", []) or [])
        contract_ok, contract_message = self._rule_action_contract(actions, context)
        if not contract_ok:
            return {
                "conditions_met": False,
                "actions_executed": [],
                "success": False,
                "status": "unsupported_rule",
                "message": contract_message,
            }
        authorized, status, message = await self._authorize_execution_async(context)
        if not authorized:
            return {
                "conditions_met": False,
                "actions_executed": [],
                "success": False,
                "status": status,
                "deferred": status == "deferred",
                "message": message,
            }

        registry = get_sensor_registry()
        registry.sync_from_world_model()
        sensors_data = registry.read_all()
        conditions = list(rule.get("conditions", []) or [])
        if not actions:
            return {
                "conditions_met": False,
                "actions_executed": [],
                "success": True,
                "status": "no_actions",
                "message": "No actions to execute.",
            }
        if not all(self.evaluate_condition(condition, sensors_data) for condition in conditions):
            return {
                "conditions_met": False,
                "actions_executed": [],
                "success": True,
                "status": "conditions_not_met",
                "message": "Conditions not satisfied, skipped execution.",
            }

        actuator_registry = get_actuator_registry()
        executed_actions: list[dict[str, Any]] = []
        messages: list[str] = []
        overall_success = True
        for action in actions:
            actuator_name = str(action.get("actuator") or "").strip()
            if not actuator_name:
                logger.warning("Action missing 'actuator' identifier")
                overall_success = False
                continue
            resolved_params = self.resolve_params(
                dict(action.get("params", {}) or {}),
                sensors_data,
            )
            logger.info(
                "Executing immune action '%s' with params: %s",
                actuator_name,
                resolved_params,
            )
            result = await actuator_registry.execute_action_async(
                actuator_name,
                resolved_params,
                context={
                    "source": context.get("source") or "adaptive_immune_system",
                    "priority": float(context.get("priority", 0.68) or 0.68),
                    "is_critical": bool(context.get("is_critical", False)),
                    "isolated_simulation": bool(
                        context.get("isolated_simulation", False)
                    ),
                },
            )
            self._note_action_outcome(actuator_name, resolved_params, result)
            if not result.success and (
                "not found" in result.message or "not registered" in result.message
            ):
                self._schedule_missing_actuator_synthesis(
                    actuator_name,
                    sensors_data,
                )
            executed_actions.append(
                {
                    "actuator": actuator_name,
                    "params": resolved_params,
                    "success": result.success,
                    "message": result.message,
                }
            )
            overall_success = overall_success and result.success
            messages.append(result.message)

        registry.sync_from_world_model()
        return {
            "conditions_met": True,
            "actions_executed": executed_actions,
            "success": overall_success,
            "status": "executed" if executed_actions else "no_actions_executed",
            "message": "; ".join(messages),
        }


# Singleton pattern
_instance: ImmuneHeuristicExecutor | None = None


def get_immune_executor() -> ImmuneHeuristicExecutor:
    global _instance
    if _instance is None:
        _instance = ImmuneHeuristicExecutor()
    return _instance
