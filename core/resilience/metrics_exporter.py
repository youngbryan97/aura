import asyncio
import logging
import socket
import time
from typing import Any

from prometheus_client import Counter, Gauge, start_http_server

from core.runtime import resource_psutil as psutil
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.shutdown_execution import run_sync_shutdown_callable
from core.utils.concurrency import cancel_and_join
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.Metrics")

_METRICS_EXPORTER_ERRORS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _record_metrics_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict | None = None,
) -> None:
    record_degradation(
        "metrics_exporter",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )

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
        self.actual_port: int | None = None
        self.running = False
        self._task: asyncio.Task | None = None
        self._start_time = time.time()
        self._monitor_failures = 0
        self._http_server: Any | None = None
        self._http_thread: Any | None = None

    def _find_free_port(self, start_port: int, max_attempts: int = 10) -> int:
        """Find an available port starting from start_port."""
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
                logger.warning("Default port %s busy, searching for alternative: %s", self.port, e)
                self.actual_port = self._find_free_port(self.port + 1, max_attempts=50)

            # Phase 33: start_http_server is synchronous and can block on DNS (socket.getfqdn)
            # We wrap it in to_thread to prevent event loop stalls during boot.
            self._http_server, self._http_thread = await asyncio.to_thread(
                start_http_server,
                self.actual_port,
            )
            try:
                from core.runtime.runtime_hygiene import get_runtime_hygiene

                get_runtime_hygiene().register_shutdown_resource(
                    self._http_server,
                    kind="tcp_listener",
                    name=f"metrics_exporter:{self.actual_port}",
                    source="core.resilience.metrics_exporter",
                    closer=self._close_http_server_blocking,
                    timeout_s=3.0,
                    blocking=True,
                )
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
                await run_sync_shutdown_callable(
                    self._close_http_server_blocking,
                    timeout_s=3.0,
                    name="metrics-exporter-registration-rollback",
                )
                raise
            logger.info("📊 Metrics Exporter ONLINE (port %s)", self.actual_port)
            self._task = get_task_tracker().create_task(
                self._monitor_loop(),
                name="metrics_exporter.monitor_loop",
            )
        except _METRICS_EXPORTER_ERRORS as e:
            _record_metrics_degradation(
                e,
                action="left metrics exporter offline after Prometheus startup failed",
                severity="warning",
                extra={"requested_port": self.port, "actual_port": self.actual_port},
            )
            logger.error("Failed to start Metrics Exporter: %s", e)
            self.running = False
            self.actual_port = None

    async def stop(self):
        self.running = False
        if self._task:
            await cancel_and_join(self._task, owner="core.resilience.metrics_exporter")
            self._task = None
        server = self._http_server
        if server is not None:
            await run_sync_shutdown_callable(
                self._close_http_server_blocking,
                timeout_s=3.0,
                name="metrics-exporter-stop",
            )
            try:
                from core.runtime.runtime_hygiene import get_runtime_hygiene

                get_runtime_hygiene().unregister_shutdown_resource(server)
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                logger.debug(
                    "%s unavailable (%s: %s); the exporter's server stayed registered with runtime hygiene",
                    "get_runtime_hygiene",
                    type(exc).__name__,
                    exc,
                )
        self.actual_port = None
        logger.info("📊 Metrics Exporter OFFLINE")

    def _close_http_server_blocking(self) -> None:
        server, self._http_server = self._http_server, None
        thread, self._http_thread = self._http_thread, None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
            if thread.is_alive():
                raise TimeoutError("metrics HTTP server thread did not stop")

    async def _monitor_loop(self):
        process = psutil.Process()
        while self.running:
            try:
                # Update system metrics
                MEM_USAGE.set(process.memory_info().rss)
                CPU_USAGE.set(psutil.cpu_percent())
                UPTIME.set(time.time() - self._start_time)
                self._monitor_failures = 0
                
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except _METRICS_EXPORTER_ERRORS as e:
                self._monitor_failures += 1
                backoff_s = min(60.0, 5.0 * (2 ** min(self._monitor_failures - 1, 4)))
                _record_metrics_degradation(
                    e,
                    action="kept metrics exporter loop alive after sample collection failed",
                    severity="warning",
                    extra={
                        "stage": "monitor_loop",
                        "consecutive_errors": self._monitor_failures,
                        "backoff_s": backoff_s,
                    },
                )
                logger.debug("Metrics monitor tick failed: %s", e)
                await asyncio.sleep(backoff_s)

# Global helper for counting tokens
def report_tokens(model: str, count: int, token_type: str = "output"):
    TOKEN_COUNT.labels(model=model, type=token_type).inc(count)

def report_latency(model: str, seconds: float):
    LATENCY.labels(model=model).set(seconds)
