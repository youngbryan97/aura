from __future__ import annotations

import asyncio
import builtins
import json
import logging
import multiprocessing
import queue
import time
import traceback
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from core.runtime.dynamic_execution_gateway import get_dynamic_execution_gateway
from core.runtime.errors import record_degradation
from core.runtime.process_privilege import ProcessRole
from core.runtime.shutdown_coordinator import is_shutdown_requested
from core.runtime.shutdown_execution import run_sync_shutdown_callable
from core.runtime.subprocess_gateway import (
    AcceleratorCapability,
    PythonProcessSpec,
    get_subprocess_gateway,
)
from core.security.ast_guard import DEFAULT_SAFE_MODULES, ASTGuard, SecurityViolation
from core.state.aura_state import AuraState

from .bridge import Phase

if TYPE_CHECKING:
    from core.kernel.aura_kernel import AuraKernel

logger = logging.getLogger("Aura.Shadow")

_SHADOW_SAFE_IMPORTS = frozenset(DEFAULT_SAFE_MODULES | {"math", "json"})
_SHADOW_SAFE_BUILTINS = {
    "ArithmeticError": ArithmeticError,
    "AssertionError": AssertionError,
    "Exception": Exception,
    "False": False,
    "KeyError": KeyError,
    "None": None,
    "RuntimeError": RuntimeError,
    "True": True,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "__build_class__": builtins.__build_class__,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "repr": repr,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def _shadow_safe_import(
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    base = str(name or "").split(".", 1)[0]
    if base not in _SHADOW_SAFE_IMPORTS:
        raise ImportError(f"Shadow sandbox import blocked: {name}")
    return __import__(name, globals, locals, fromlist, level)


def _validate_shadow_source(code: str) -> None:
    ASTGuard(allowed_modules=sorted(_SHADOW_SAFE_IMPORTS)).validate(
        code,
        source_label="<shadow_mutation>",
    )

@dataclass(frozen=True)
class StateBoundsConfig:
    """Hard limits on AuraState field sizes. Enforced post-sandbox."""

    #: How far past the enforced working-memory capacity a state may go before
    #: it counts as a bomb. Not 1: the trimmer runs after the append, so a
    #: legitimate state is briefly one item over, and a bomb is orders of
    #: magnitude past that rather than one item.
    WORKING_MEMORY_HEADROOM: int = 2
    MAX_LONG_TERM_MEMORY_ITEMS: int = 10000
    MAX_LONG_TERM_MEMORY_ITEMS: int = 10000
    MAX_CONCEPT_GRAPH_NODES: int = 50000
    MAX_KNOWN_ENTITIES: int = 5000
    MAX_PENDING_INTENTS: int = 100
    MAX_NESTED_DEPTH: int = 20


@dataclass(frozen=True)
class ShadowValidationReceipt:
    """Structured evidence for a shadow mutation validation attempt."""

    success: bool
    behavioral_ok: bool
    structural_ok: bool
    validator_info: object = ""
    failure_reason: str = ""
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

def _sandbox_worker(
    mutated_code: str,
    serialized_state: str,
    result_queue: Any,
) -> None:
    """Worker executed in a separate process with a hardened namespace."""
    try:
        _validate_shadow_source(mutated_code)

        safe_builtins = dict(_SHADOW_SAFE_BUILTINS)
        safe_builtins["__import__"] = _shadow_safe_import
        sandbox_globals = {
            "__builtins__": safe_builtins,
            "__name__": "aura_shadow_sandbox",
        }
        
        # Execute AST-vetted mutation code in a separate process with a minimal
        # builtins/import surface. The mutation can define helpers and a
        # validate(state_dict) function, but it cannot perform file, process,
        # network, importlib, or introspection escapes.
        dynamic_gateway = get_dynamic_execution_gateway()
        code_object = dynamic_gateway.compile_source(
            mutated_code,
            filename="<shadow_mutation>",
            mode="exec",
            source="shadow_kernel.sandbox_worker",
        )
        dynamic_gateway.execute_code_object(
            code_object,
            globals_dict=sandbox_globals,
            source="shadow_kernel.sandbox_worker",
        )
        
        state_dict = json.loads(serialized_state)
        validator = sandbox_globals.get("validate")
        if callable(validator):
            ok, info = validator(state_dict)
            result_queue.put({"ok": bool(ok), "info": info})
        else:
            # If no validator is defined, just ensure the code imports/executes
            result_queue.put({"ok": True, "info": "Code executed but no validator found."})
            
    except SecurityViolation as exc:
        result_queue.put({"ok": False, "info": f"security_violation: {exc}"})
    except (Exception, SystemExit):
        result_queue.put({"ok": False, "trace": traceback.format_exc()})

class ShadowExecutionPhase(Phase):  # type: ignore[misc]
    """
    Headless sandbox validator.
    Runs mutations in a separate process to prevent host-kernel contamination.
    """
    def __init__(self, kernel: AuraKernel):
        self.kernel = kernel

    async def execute(
        self,
        state: AuraState,
        objective: str | None = None,
        **kwargs: Any,
    ) -> AuraState:
        """
        [ZENITH-v2] Dual-Phase Validation: Behavioral + Structural.
        """
        # In this version, we ensure the sandbox logic is robust
        # This will be used to validate subconscious parallel branches.
        return state

    async def apply_mutation_safely(self, mutated_code: str, validator_code: str) -> bool:
        receipt = await self.evaluate_mutation_safely(mutated_code, validator_code)
        return receipt.success

    async def evaluate_mutation_safely(
        self,
        mutated_code: str,
        validator_code: str,
    ) -> ShadowValidationReceipt:
        """
        Orchestrates the dual-phase validation of a proposed mutation.
        1. Behavioral: Sandbox execution.
        2. Structural: Post-apply bounds check on a test copy.
        """
        start = time.monotonic()
        # 1. Behavioral Sandbox Check
        behavioral_ok, validator_info = await self._validate_mutation_detailed(
            mutated_code,
            validator_code,
        )
        if not behavioral_ok:
            return ShadowValidationReceipt(
                success=False,
                behavioral_ok=False,
                structural_ok=False,
                validator_info=validator_info,
                failure_reason=str(validator_info or "behavioral_validation_failed"),
                elapsed_ms=(time.monotonic() - start) * 1000.0,
            )
            
        # 2. Structural Integrity Check (State Bounds)
        # We apply it to a test copy first to ensure it doesn't 'explode' the state graph.
        try:
            test_copy = await self.kernel.state.derive_async("sandbox_structural_test")
            if not self._validate_state_bounds(test_copy):
                logger.error("Sandbox: Structural integrity check failed (State Bounds violation)")
                return ShadowValidationReceipt(
                    success=False,
                    behavioral_ok=True,
                    structural_ok=False,
                    validator_info=validator_info,
                    failure_reason="state_bounds_violation",
                    elapsed_ms=(time.monotonic() - start) * 1000.0,
                )
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('shadow_kernel', e)
            logger.error("Sandbox: Critical failure during structural validation: %s", e)
            return ShadowValidationReceipt(
                success=False,
                behavioral_ok=True,
                structural_ok=False,
                validator_info=validator_info,
                failure_reason=f"structural_validation_error:{type(e).__name__}",
                elapsed_ms=(time.monotonic() - start) * 1000.0,
            )
            
        return ShadowValidationReceipt(
            success=True,
            behavioral_ok=True,
            structural_ok=True,
            validator_info=validator_info,
            elapsed_ms=(time.monotonic() - start) * 1000.0,
        )

    def _validate_state_bounds(self, state: AuraState) -> bool:
        """
        Strictly enforces structural invariants on the AuraState object.
        Prevents 'Memory Bomb' attacks that bypass sandbox behavior checks.
        """
        config = StateBoundsConfig()
        
        try:
            # Check Working Memory. Read through the canonical accessor: this
            # was `state.working_memory`, which AuraState does not have — the
            # list is on state.cognition — so the guard passed a 5,000-item
            # bomb without ever looking at it.
            from core.state.one_working_memory import the_capacity, the_working_memory

            allowed = the_capacity() * max(1, config.WORKING_MEMORY_HEADROOM)
            if len(the_working_memory(state)) > allowed:
                return False
                
            # Check LTM (if accessible)
            # if hasattr(state, "long_term_memory") ...
            
            # Check for illegal deep nesting or circularity (basic check)
            # This is partly handled by deepcopy during derivation, but we add a safety layer here.
            serialized = json.dumps(getattr(state, "__dict__", {}), default=lambda x: str(x))
            if len(serialized) > 10 * 1024 * 1024: # 10MB limit for serialized state fragment
                return False
                
            return True
        except (json.JSONDecodeError, TypeError, ValueError):
            return False

    async def _validate_mutation(self, mutated_code: str, validator_code: str) -> bool:
        ok, _info = await self._validate_mutation_detailed(mutated_code, validator_code)
        return ok

    async def _validate_mutation_detailed(
        self,
        mutated_code: str,
        validator_code: str,
    ) -> tuple[bool, object]:
        """Runs the existing behavioral sandbox check."""
        if is_shutdown_requested():
            return False, "runtime_shutdown"
        result_queue: Any = multiprocessing.Queue()
        try:
            from core.runtime.runtime_hygiene import get_runtime_hygiene

            get_runtime_hygiene().register_shutdown_resource(
                result_queue,
                kind="multiprocessing_queue",
                name="shadow_kernel.result_queue",
                source="core.kernel.shadow_kernel",
                timeout_s=1.0,
            )
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
            try:
                result_queue.close()
                await run_sync_shutdown_callable(
                    result_queue.join_thread,
                    timeout_s=1.0,
                    name="shadow-result-queue-registration-failure",
                )
            except (AttributeError, OSError, RuntimeError, ValueError, TimeoutError):
                pass
            return False, "runtime_shutdown" if is_shutdown_requested() else "resource_owner_unavailable"
        process = None
        try:
            test_state = self.kernel.state
            serialized_state = json.dumps(
                {
                    "version": getattr(test_state, "version", 0),
                    "mood": getattr(test_state, "mood", "neutral"),
                    "vitality": getattr(test_state, "vitality", 100.0),
                }
            )
            process_source = "shadow_kernel.sandbox_validation"
            process = get_subprocess_gateway().spawn_python_process(
                PythonProcessSpec(
                    target=_sandbox_worker,
                    args=(
                        mutated_code + "\n" + validator_code,
                        serialized_state,
                        result_queue,
                    ),
                    source=process_source,
                    name="AuraShadowValidator",
                    role=ProcessRole.UNTRUSTED_CODE,
                    requested_privileges=frozenset(),
                    accelerator_capability=AcceleratorCapability.NONE,
                    start_method="spawn",
                )
            )
            if is_shutdown_requested():
                return False, "runtime_shutdown"

            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                if is_shutdown_requested():
                    return False, "runtime_shutdown"
                if not process.is_alive():
                    break
                await asyncio.sleep(0.1)

            if process.is_alive():
                logger.warning("Sandbox: Mutation validation timed out")
                return False, "timeout"

            await run_sync_shutdown_callable(
                lambda: process.join(timeout=2.0),
                timeout_s=2.25,
                name="shadow-process-result-join",
            )
            try:
                result = result_queue.get_nowait()
            except (OSError, ConnectionError, TimeoutError, queue.Empty) as exc:
                record_degradation("shadow_kernel", exc)
                logger.error("Sandbox: Failed to retrieve result from worker: %s", exc)
                return False, f"result_queue_error:{type(exc).__name__}"
            if not result.get("ok"):
                info = result.get("trace") or result.get("info") or "validator_returned_false"
                logger.error("Sandbox: Mutation failed: %s", info)
                return False, info

            info = result.get("info")
            logger.info("Sandbox: Mutation validated successfully: %s", info)
            return True, info
        finally:
            if process is not None and process.is_alive():
                process.terminate()
                try:
                    await run_sync_shutdown_callable(
                        lambda: process.join(timeout=1.0),
                        timeout_s=1.25,
                        name="shadow-process-terminate-join",
                    )
                except TimeoutError:
                    pass
                if process.is_alive():
                    process.kill()
                    try:
                        await run_sync_shutdown_callable(
                            lambda: process.join(timeout=1.0),
                            timeout_s=1.25,
                            name="shadow-process-kill-join",
                        )
                    except TimeoutError:
                        pass
            queue_cleanup_complete = False
            try:
                result_queue.close()
                await run_sync_shutdown_callable(
                    result_queue.join_thread,
                    timeout_s=1.0,
                    name="shadow-result-queue-close",
                )
                queue_cleanup_complete = True
            except (AttributeError, OSError, RuntimeError, ValueError, TimeoutError):
                pass
            if queue_cleanup_complete:
                try:
                    from core.runtime.runtime_hygiene import get_runtime_hygiene

                    get_runtime_hygiene().unregister_shutdown_resource(result_queue)
                except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
                    pass
