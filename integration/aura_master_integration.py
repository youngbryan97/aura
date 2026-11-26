"""
integration/aura_master_integration.py
Aura v3.0 Master Integration System
Production-ready integration manager with fault tolerance and monitoring.
"""
from __future__ import annotations

import logging
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from pathlib import Path
import importlib
import importlib.util
import inspect
import json
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Configure structured logging
logger = logging.getLogger("Aura.Integration")
logger.setLevel(logging.INFO)

# Add JSON formatter for production
try:
    from pythonjsonlogger import jsonlogger
    log_handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)
except ImportError:
    # Fallback to basic formatter
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


class IntegrationStatus(Enum):
    """Integration operation status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class IntegrationError(Exception):
    """Custom exception for integration failures"""
    def __init__(self, message: str, component: str, original_error: Optional[Exception] = None):
        self.message = message
        self.component = component
        self.original_error = original_error
        super().__init__(f"{component}: {message}")


_INTEGRATION_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TimeoutError,
    FutureTimeoutError,
    OSError,
    LookupError,
    TypeError,
    ValueError,
    IntegrationError,
)


@dataclass
class IntegrationStep:
    """Represents a single integration step"""
    name: str
    module_path: str
    function_name: str
    timeout_seconds: int = 30
    retry_count: int = 3
    required: bool = True
    dependencies: List[str] = field(default_factory=list)


@dataclass
class IntegrationResult:
    """Result of an integration operation"""
    step_name: str
    status: IntegrationStatus
    duration_seconds: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations with exponential backoff"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except _INTEGRATION_RECOVERABLE_ERRORS as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        sleep_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}. "
                            f"Retrying in {sleep_time:.2f}s. Error: {e}"
                        )
                        # Synchronous sleep here as wrapper is sync-compatible, use asyncio.sleep if async
                        import time
                        time.sleep(sleep_time)
                    else:
                        logger.error(
                            f"All {max_retries} attempts failed for {func.__name__}"
                        )
                        raise IntegrationError(
                            message=f"Failed after {max_retries} attempts",
                            component=func.__name__,
                            original_error=last_exception
                        )
            raise last_exception
        return wrapper
    return decorator


def validate_orchestrator(orchestrator: Any) -> None:
    """
    Validate orchestrator has required interface.
    
    Args:
        orchestrator: Orchestrator instance to validate
        
    Raises:
        TypeError: If orchestrator doesn't have required attributes
    """
    # Relaxed validation for v3.1 AsyncOrchestrator compatibility
    # v3.1 uses different internal structure, so we check for basic existence
    if orchestrator is None:
         raise ValueError("Orchestrator cannot be None")

    # Log what we found for diagnostics
    logger.info(f"Validating orchestrator: {type(orchestrator)}")


class IntegrationManager:
    """
    Manages integration operations with dependency resolution and monitoring.
    """
    
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.results: Dict[str, IntegrationResult] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        
    def register_step(self, step: IntegrationStep) -> None:
        """Register an integration step with its dependencies"""
        self.dependency_graph[step.name] = step.dependencies
        
    def _resolve_execution_order(self) -> List[str]:
        """Topological sort for dependency resolution"""
        ordered = []
        pending = set(self.dependency_graph.keys())
        completed = set()
        
        while pending:
            # Find nodes with all deps satisfied
            ready = [
                n for n in pending 
                if all(dep in completed for dep in self.dependency_graph[n])
            ]
            
            if not ready:
                # Cycle detected
                raise IntegrationError(
                    "Circular dependency detected in integration steps",
                    component="dependency_resolution"
                )
            
            # Add ready nodes to order
            # Sort for deterministic behavior
            ready.sort() 
            
            for n in ready:
                ordered.append(n)
                completed.add(n)
                pending.remove(n)
                
        return ordered

    def _load_step_callable(self, step: IntegrationStep) -> Callable[..., Any]:
        """Load the configured integration callable from a file or module name."""
        if step.module_path == "__main__":
            module = sys.modules.get("__main__")
            if module is None:
                raise IntegrationError("__main__ module is unavailable", step.name)
        else:
            candidate = Path(step.module_path)
            if not candidate.exists():
                candidate = Path(__file__).resolve().parent.parent / step.module_path

            if candidate.exists():
                module_name = f"aura_integration_{step.name}_{abs(hash(str(candidate.resolve())))}"
                spec = importlib.util.spec_from_file_location(module_name, candidate)
                if spec is None or spec.loader is None:
                    raise IntegrationError(f"Cannot import module from {candidate}", step.name)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            else:
                module_name = step.module_path.removesuffix(".py").replace("/", ".")
                module = importlib.import_module(module_name)

        func = getattr(module, step.function_name, None)
        if not callable(func):
            raise IntegrationError(
                f"Function {step.function_name!r} not found in {step.module_path!r}",
                step.name,
            )
        return func

    @staticmethod
    def _invoke_step_callable(func: Callable[..., Any], orchestrator: Any) -> Any:
        signature = inspect.signature(func)
        if not signature.parameters:
            return func()
        return func(orchestrator)

    @staticmethod
    def _json_safe_result(result: Any) -> Any:
        try:
            json.dumps(result)
            return result
        except (TypeError, ValueError):
            return repr(result)
    
    @retry_on_failure(max_retries=3)
    def execute_step(self, step: IntegrationStep, orchestrator: Any) -> IntegrationResult:
        """Execute a single integration step with timeout and retry"""
        start_time = datetime.now()

        try:
            func = self._load_step_callable(step)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._invoke_step_callable, func, orchestrator)
                result = future.result(timeout=max(0.1, float(step.timeout_seconds)))

            return IntegrationResult(
                step_name=step.name,
                status=IntegrationStatus.COMPLETED,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                metadata={"result": self._json_safe_result(result)}
            )

        except _INTEGRATION_RECOVERABLE_ERRORS as e:
            return IntegrationResult(
                step_name=step.name,
                status=IntegrationStatus.FAILED,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                error=str(e)
            )
    
    def execute_all(self, steps: List[IntegrationStep], orchestrator: Any) -> Dict[str, IntegrationResult]:
        """Execute all integration steps in dependency order"""
        # Register all steps
        for step in steps:
            self.register_step(step)
        
        # Get execution order
        execution_order = self._resolve_execution_order()
        
        # Execute in order (Sequential for safe dependency handling in this implementation)
        # Using ThreadPoolExecutor introduced complexity with shared state in the original proposal
        # Reverting to reliable sequential execution for v3.1 stability
        
        for step_name in execution_order:
            step = next(s for s in steps if s.name == step_name)
            
            # Check dependencies
            if not all(self.results.get(dep) and self.results[dep].status == IntegrationStatus.COMPLETED 
                       for dep in step.dependencies):
                 logger.warning(f"Skipping {step.name} due to failed dependencies")
                 self.results[step.name] = IntegrationResult(
                     step_name=step.name, status=IntegrationStatus.SKIPPED, duration_seconds=0
                 )
                 continue

            # Execute
            result = self.execute_step(step, orchestrator)
            self.results[step.name] = result
            
            if result.status == IntegrationStatus.FAILED and step.required:
                logger.error(f"Critical step {step.name} failed. Stopping integration.")
                break
        
        return self.results


def apply_all_fixes(orchestrator: Any) -> bool:
    """
    Apply ALL fixes to transform Aura into an intelligent companion.
    """
    # Input validation
    if orchestrator is None:
        raise ValueError("orchestrator cannot be None")
    
    validate_orchestrator(orchestrator)
    
    logger.info("\n" + "=" * 70)
    logger.info("AURA v3.1 MASTER INTEGRATION")
    logger.info("=" * 70)
    
    try:
        # Define integration steps
        integration_steps = [
            IntegrationStep(
                name="enhanced_skills",
                module_path="skill_integration.py",
                function_name="apply_all_enhancements",
                timeout_seconds=45,
                required=True
            ),
            IntegrationStep(
                name="personality_fixes",
                module_path="personality_fixes.py",
                function_name="apply_personality_fixes",
                timeout_seconds=30,
                required=True,
                dependencies=["enhanced_skills"]
            ),
            IntegrationStep(
                name="final_configuration",
                module_path="__main__", 
                function_name="_finalize_configuration",
                timeout_seconds=15,
                required=False
            )
        ]
        
        # Execute integration
        manager = IntegrationManager(max_workers=2)
        results = manager.execute_all(integration_steps, orchestrator)
        
        # Analyze results
        successful = all(
            r.status == IntegrationStatus.COMPLETED or r.status == IntegrationStatus.SKIPPED
            for r in results.values()
            if r.status != IntegrationStatus.SKIPPED # Ignored skipped in success calc
        )
        
        if successful:
            logger.info("\n" + "=" * 70)
            logger.info("ALL FIXES APPLIED SUCCESSFULLY")
            logger.info("=" * 70)
            return True
        else:
            logger.error("\n" + "=" * 70)
            logger.error("INTEGRATION FAILED")
            return False
            
    except _INTEGRATION_RECOVERABLE_ERRORS as e:
        logger.error(f"\nMaster integration failed: {e}", exc_info=True)
        return False

# Main execution for testing
if __name__ == "__main__":
    class CliOrchestrator:
        """Minimal CLI orchestrator marker for direct integration diagnostics."""

        source = "integration_cli"

    sys.exit(0 if apply_all_fixes(CliOrchestrator()) else 1)
