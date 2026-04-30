from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
from typing import Optional, Any
from prometheus_client import start_http_server, Gauge, Counter, REGISTRY
import psutil
import time

logger = logging.getLogger("Aura.Metrics")

# Core System Metrics
MEM_USAGE = Gauge('aura_memory_usage_bytes', 'Current RSS memory usage in bytes')
CPU_USAGE = Gauge('aura_cpu_usage_percent', 'Current CPU usage percentage')
UPTIME = Gauge('aura_uptime_seconds', 'System uptime in seconds')

# LLM Metrics (to be populated by providers)
TOKEN_COUNT = Counter('aura_llm_tokens_total', 'Total tokens processed', ['model', 'type'])
LATENCY = Gauge('aura_llm_latency_seconds', 'Last request latency', ['model'])

class MetricsExporter:
    """
    Background service that exports Prometheus metrics.
    """
    def __init__(self, port: int = 9090):
        self.port = port
        self.actual_port: Optional[int] = None
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self._start_time = time.time()

    def _find_free_port(self, start_port: int, max_attempts: int = 10) -> int:
        """Find an available port starting from start_port."""
        import socket
        for p in range(start_port, start_port + max_attempts):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('', p))
                    return p
                except OSError:
                    continue
        raise OSError(f"Could not find a free port in range {start_port}-{start_port + max_attempts}")

    async def start(self):
        if self.running:
            return
        
        self.running = True
        try:
            # v44: Handle port collisions
            try:
                self.actual_port = self._find_free_port(self.port)
            except OSError as e:
                logger.warning(f"Default port {self.port} busy, searching for alternative: {e}")
                self.actual_port = self._find_free_port(self.port + 1, max_attempts=50)

            # Phase 33: start_http_server is synchronous and can block on DNS (socket.getfqdn)
            # We wrap it in to_thread to prevent event loop stalls during boot.
            await asyncio.to_thread(start_http_server, self.actual_port)
            logger.info(f"📊 Metrics Exporter ONLINE (port {self.actual_port})")
            self._task = get_task_tracker().create_task(self._monitor_loop())
        except Exception as e:
            logger.error(f"Failed to start Metrics Exporter: {e}")
            self.running = False

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _e:
                logger.debug('Ignored asyncio.CancelledError in metrics_exporter.py: %s', _e)
        logger.info("📊 Metrics Exporter OFFLINE")

    async def _monitor_loop(self):
        process = psutil.Process()
        while self.running:
            try:
                # Update system metrics
                MEM_USAGE.set(process.memory_info().rss)
                CPU_USAGE.set(psutil.cpu_percent())
                UPTIME.set(time.time() - self._start_time)
                
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Metrics monitor tick failed: {e}")
                await asyncio.sleep(10)

# Global helper for counting tokens
def report_tokens(model: str, count: int, token_type: str = "output"):
    TOKEN_COUNT.labels(model=model, type=token_type).inc(count)

def report_latency(model: str, seconds: float):
    LATENCY.labels(model=model).set(seconds)
