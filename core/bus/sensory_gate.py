from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from core.bus.local_pipe_bus import LocalPipeBus
from core.phantom_browser import PhantomBrowser
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.SensoryGate")

SENSORY_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
    urllib.error.URLError,
)


def _record_sensory_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, object] | None = None,
) -> None:
    record_degradation(
        "sensory_gate",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=False,
        extra=extra,
    )


class SensoryGateActor:
    """Isolated sensory actor for browser/search observation work."""

    def __init__(self, connection: Any):
        self.bus = LocalPipeBus(is_child=True, connection=connection)
        self.browser: PhantomBrowser | None = None
        self._is_active = True
        self._heartbeat_interval = 3.0
        self._heartbeat_failures = 0
        self._background_tasks: set[asyncio.Task] = set()
        self._shutdown_event: asyncio.Event | None = None

    def _track_task(self, coro: Any, *, name: str | None = None) -> asyncio.Task:
        task = get_task_tracker().create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def _request_shutdown(self) -> None:
        self._is_active = False
        if self._shutdown_event is not None:
            self._shutdown_event.set()

    async def _cancel_background_tasks(self) -> None:
        tasks = [task for task in self._background_tasks if not task.done()]
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._background_tasks.clear()

    async def run(self) -> None:
        """Main actor loop."""
        logger.info("👁️ SensoryGate Actor starting...")
        self._shutdown_event = asyncio.Event()
        try:
            try:
                self.browser = PhantomBrowser(visible=False)
            except SENSORY_RECOVERABLE_ERRORS as exc:
                _record_sensory_degradation(
                    exc,
                    action="failed sensory gate startup before accepting browser/search work",
                    severity="critical",
                )
                raise

            self.bus.register_handler("browse", self._handle_browse)
            self.bus.register_handler("search", self._handle_search)
            self.bus.register_handler("ping", lambda payload, tid: "pong")
            self.bus.register_handler("shutdown", self._handle_shutdown)

            self.bus.start()
            self._track_task(self._heartbeat_loop(), name="sensory_gate.heartbeat")

            logger.info("👁️ SensoryGate Actor ready.")
            await self._shutdown_event.wait()
        finally:
            self._request_shutdown()
            await self._cancel_background_tasks()
            try:
                await self.bus.stop()
            except SENSORY_RECOVERABLE_ERRORS as exc:
                _record_sensory_degradation(
                    exc,
                    action="continued sensory gate shutdown after bus stop failed",
                    severity="degraded",
                )
                logger.error("❌ SensoryGate bus shutdown failed: %s", exc)
            if self.browser is not None:
                try:
                    await self.browser.close()
                except SENSORY_RECOVERABLE_ERRORS as exc:
                    _record_sensory_degradation(
                        exc,
                        action="continued sensory gate shutdown after browser close failed",
                        severity="degraded",
                    )
                    logger.error("❌ SensoryGate browser shutdown failed: %s", exc)
            logger.info("👁️ SensoryGate Actor stopped.")

    async def _heartbeat_loop(self) -> None:
        """Send heartbeats to the supervisor."""
        while self._is_active:
            try:
                await self.bus.send(
                    "heartbeat",
                    {
                        "pid": os.getpid(),
                        "ts": time.time(),
                        "status": "healthy" if self._heartbeat_failures == 0 else "degraded",
                    },
                )
                self._heartbeat_failures = 0
            except SENSORY_RECOVERABLE_ERRORS as exc:
                self._heartbeat_failures += 1
                _record_sensory_degradation(
                    exc,
                    action="kept sensory actor alive and retried supervisor heartbeat on next interval",
                    severity="warning",
                    extra={"consecutive_heartbeat_failures": self._heartbeat_failures},
                )
                logger.error("❌ Heartbeat failed: %s", exc)

            shutdown_event = self._shutdown_event
            if shutdown_event is None:
                await asyncio.sleep(self._heartbeat_interval)
                continue
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=self._heartbeat_interval)
            except TimeoutError:
                continue

    async def _handle_browse(self, payload: dict[str, Any], trace_id: str) -> dict[str, Any]:
        """Handle browse request."""
        url = str(payload.get("url") or "").strip()
        if not url:
            return {"error": "No URL provided"}
        if self.browser is None:
            error = RuntimeError("browser_unavailable")
            _record_sensory_degradation(
                error,
                action="blocked browse request because browser was not initialized",
                severity="degraded",
                extra={"trace_id": trace_id},
            )
            return {"error": "browser_unavailable"}

        logger.info("🌐 [%s] Browsing: %s", trace_id[:8], url)
        try:
            result = await self.browser.browse(url)
            return {
                "url": url,
                "content": result,
                "observation_only": True,
                "requires_governance_for_effects": True,
            }
        except SENSORY_RECOVERABLE_ERRORS as exc:
            _record_sensory_degradation(
                exc,
                action="returned browse error result without crashing sensory actor",
                severity="warning",
                extra={"trace_id": trace_id, "url": url},
            )
            logger.error("❌ [%s] Browse failed: %s", trace_id[:8], exc)
            return {"error": str(exc)}

    async def _handle_search(self, payload: dict[str, Any], trace_id: str) -> dict[str, Any]:
        """Handle search request via Wikipedia OpenSearch API."""
        query = str(payload.get("query") or "").strip()
        if not query:
            return {"error": "No query provided"}

        logger.info("🔍 [%s] Wikipedia search: %s", trace_id[:8], query)
        try:
            url = (
                "https://en.wikipedia.org/w/api.php?action=opensearch"
                f"&search={urllib.parse.quote(query)}&limit=3&namespace=0&format=json"
            )

            def fetch() -> Any:
                req = urllib.request.Request(url, headers={"User-Agent": "Aura/1.0"})
                with urllib.request.urlopen(req, timeout=5) as response:
                    return json.loads(response.read().decode("utf-8", errors="replace"))

            data = await asyncio.to_thread(fetch)
            return {
                "query": query,
                "source": "wikipedia",
                "results": self._format_search_results(data),
                "observation_only": True,
                "requires_governance_for_effects": True,
            }
        except SENSORY_RECOVERABLE_ERRORS as exc:
            _record_sensory_degradation(
                exc,
                action="returned search error result without crashing sensory actor",
                severity="warning",
                extra={"trace_id": trace_id, "query": query},
            )
            logger.error("❌ [%s] Wikipedia search failed: %s", trace_id[:8], exc)
            return {"error": str(exc)}

    @staticmethod
    def _format_search_results(data: Any) -> list[str]:
        if not isinstance(data, list) or len(data) < 4:
            return []
        titles = data[1] if isinstance(data[1], list) else []
        snippets = data[2] if isinstance(data[2], list) else []
        urls = data[3] if isinstance(data[3], list) else []
        return [
            f"{title}: {snippet} ({url})"
            for title, snippet, url in zip(titles, snippets, urls, strict=False)
        ]

    async def _handle_shutdown(self, payload: Any, trace_id: str) -> str:
        self._request_shutdown()
        return "Acknowledged"


def start_sensory_gate(connection: Any, *args: Any, **kwargs: Any) -> None:
    """Process entry point."""
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except SENSORY_RECOVERABLE_ERRORS as exc:
        _record_sensory_degradation(
            exc,
            action="continued sensory gate startup after signal handler setup failed",
            severity="debug",
        )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    actor = SensoryGateActor(connection)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(actor.run())
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
    finally:
        asyncio.set_event_loop(None)
        loop.close()
