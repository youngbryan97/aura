import asyncio
import logging
import signal
import sys
from typing import Any, Callable, List, Union, Awaitable, ClassVar, Optional

logger = logging.getLogger("Aura.Shutdown")

class GracefulShutdown:
    """Manages system-wide shutdown hooks with LIFO execution order.
    
    Handles SIGINT and SIGTERM to ensure all services (e.g., Vault flushing, 
    Monitor stopping) complete their tasks before the process exits.
    """
    
    _hooks: ClassVar[List[Union[Callable[[], Any], Callable[[], Awaitable[Any]]]]] = []
    _is_shutting_down: ClassVar[bool] = False
    _shutdown_event: ClassVar[Optional[asyncio.Event]] = None

    @classmethod
    def register(cls, hook: Union[Callable[[], Any], Callable[[], Awaitable[Any]]]):
        """Register a cleanup hook. Hooks are executed in LIFO order."""
        if hook not in cls._hooks:
            cls._hooks.append(hook)
            logger.debug(f"Registered shutdown hook: {hook.__name__ if hasattr(hook, '__name__') else str(hook)}")

    @classmethod
    def setup_signals(cls):
        """Bind OS signals to the shutdown handler."""
        loop = asyncio.get_running_loop()
        cls._shutdown_event = asyncio.Event()
        
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(cls.trigger_shutdown(s)))
            except NotImplementedError as _e:
                # Fallback for Windows or certain environments
                logger.debug('Ignored NotImplementedError in graceful_shutdown.py: %s', _e)

    @classmethod
    async def trigger_shutdown(cls, sig=None):
        """Executes all registered hooks in reverse order (LIFO)."""
        if cls._is_shutting_down:
            return
        cls._is_shutting_down = True
        
        prefix = f"Received signal {sig}: " if sig else ""
        logger.warning(f"🛑 {prefix}Initiating graceful shutdown of Aura components...")
        
        # Shutdown Services in reverse order
        while cls._hooks:
            hook = cls._hooks.pop() # LIFO
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook()
                else:
                    res = hook()
                    if asyncio.iscoroutine(res) or hasattr(res, "__await__"):
                        await res
                logger.info(f"   [✓] Shutdown hook completed.")
            except Exception as e:
                logger.error(f"   [!] Shutdown hook failed: {e}")

        # Also trigger ServiceContainer shutdown if it exists
        try:
            from .container import get_container
            container = get_container()
            await container.shutdown()
        except Exception as e:
            logger.error(f"Error during container shutdown: {e}")

        logger.info("✅ All Aura core services gracefully terminated. Goodbye.")
        
        if cls._shutdown_event:
            cls._shutdown_event.set()
        
        # Give logs a moment to flush then exit if we were the one who started the loop
        # Note: In some hosting environments, we shouldn't call sys.exit()
        # but let the main loop finish.
        if sig:
            # Re-raise or exit
            sys.exit(0)

    @classmethod
    async def wait_for_shutdown(cls):
        """Block until shutdown is complete."""
        if cls._shutdown_event is None:
            cls.setup_signals()
        await cls._shutdown_event.wait()

# Global singleton helper
def register_shutdown_hook(hook: Union[Callable[[], Any], Callable[[], Awaitable[Any]]]):
    GracefulShutdown.register(hook)
