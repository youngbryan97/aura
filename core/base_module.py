from core.runtime.errors import record_degradation
import logging
import time
import asyncio
import traceback
from typing import Any, Dict, Optional, Callable, TypeVar, Union
from functools import wraps

T = TypeVar("T")

class AuraBaseModule:
    """Enterprise-grade base class for all Aura core modules.
    
    This base class provides standardized logging, performance metrics,
    and error boundaries to ensure system stability and observability.
    
    Attributes:
        module_name (str): The name of the module for logging and metrics.
        logger (logging.Logger): Standardized logger instance for the module.
        metrics (Dict[str, Any]): Dictionary containing performance and error metrics.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initializes the base module.
        
        Args:
            name: Optional explicit name for the module. Defaults to class name.
        """
        self.module_name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"Aura.{self.module_name}")
        self.metrics: Dict[str, Any] = {
            "calls": 0,
            "errors": 0,
            "avg_latency": 0.0,
            "last_error": None
        }
        self.logger.debug("Initialized %s", self.module_name)

    def error_boundary(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator to wrap methods in a standardized error boundary.
        
        Automatically logs errors, updates failure metrics, and tracks latency.
        Works for both sync and async methods.
        
        Args:
            func: The method to wrap.
            
        Returns:
            The wrapped method with error protection.
        """
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                self.metrics["calls"] += 1
                try:
                    result = await func(*args, **kwargs)
                    latency = time.time() - start_time
                    self._update_latency(latency)
                    return result
                except Exception as e:
                    record_degradation('base_module', e)
                    self.metrics["errors"] += 1
                    self.metrics["last_error"] = str(e)
                    self.logger.error("Error in %s: %s", func.__name__, e)
                    self.logger.debug(traceback.format_exc())
                    return self.handle_error(e, func.__name__)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                self.metrics["calls"] += 1
                try:
                    result = func(*args, **kwargs)
                    latency = time.time() - start_time
                    self._update_latency(latency)
                    return result
                except Exception as e:
                    record_degradation('base_module', e)
                    self.metrics["errors"] += 1
                    self.metrics["last_error"] = str(e)
                    self.logger.error("Error in %s: %s", func.__name__, e)
                    self.logger.debug(traceback.format_exc())
                    return self.handle_error(e, func.__name__)
            return sync_wrapper

    def _update_latency(self, current_latency: float) -> None:
        """Update moving average of latency.
        
        Args:
            current_latency: The latency of the current call in seconds.
        """
        n = self.metrics["calls"]
        old_avg = self.metrics["avg_latency"]
        self.metrics["avg_latency"] = old_avg + (current_latency - old_avg) / n

    def handle_error(self, error: Union[Exception, str], context: str) -> Dict[str, Any]:
        """Overrideable error handler for module-specific recovery.
        
        Args:
            error: The exception that was caught.
            context: The name of the method where the error occurred.
            
        Returns:
            A dictionary containing the error details.
        """
        return {"ok": False, "error": str(error), "context": context}

    def get_health(self) -> Dict[str, Any]:
        """Retrieves module health statistics.
        
        Returns:
            Dict[str, Any]: Health metrics including call count, error count, 
                            average latency, and current status.
        """
        return {
            "module": self.module_name,
            "calls": self.metrics["calls"],
            "errors": self.metrics["errors"],
            "avg_latency": round(self.metrics["avg_latency"], 4),
            "status": "healthy" if self.metrics["errors"] == 0 else "degraded"
        }