
from core.runtime.errors import record_degradation
import asyncio
import logging
import multiprocessing
import os
import time
from typing import Any, Dict, Optional

from core.bus.local_pipe_bus import LocalPipeBus
from core.phantom_browser import PhantomBrowser
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.SensoryGate")

class SensoryGateActor:
    """
    Isolated Sensory Gate Actor.
    Runs in a separate process to prevent I/O blocking and browser overhead 
    from affecting the main cognitive loop.
    """
    def __init__(self, connection):
        self.bus = LocalPipeBus(is_child=True, connection=connection)
        self.browser = None
        self._is_active = True
        self._heartbeat_interval = 3.0
        self._background_tasks: set[asyncio.Task] = set()

    def _track_task(self, coro: Any, *, name: Optional[str] = None) -> asyncio.Task:
        task = get_task_tracker().create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _cancel_background_tasks(self):
        tasks = [task for task in self._background_tasks if not task.done()]
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._background_tasks.clear()

    async def run(self):
        """Main actor loop."""
        logger.info("👁️ SensoryGate Actor starting...")
        try:
            # Initialize browser in the child process
            self.browser = PhantomBrowser(visible=False)

            # Register handlers
            self.bus.register_handler("browse", self._handle_browse)
            self.bus.register_handler("search", self._handle_search)
            self.bus.register_handler("ping", lambda payload, tid: "pong")
            self.bus.register_handler("shutdown", self._handle_shutdown)

            self.bus.start()

            # Start heartbeat loop after bus is active
            self._track_task(
                self._heartbeat_loop(),
                name="sensory_gate.heartbeat",
            )

            logger.info("👁️ SensoryGate Actor ready.")

            while self._is_active:
                await asyncio.sleep(1.0)
        finally:
            self._is_active = False
            await self._cancel_background_tasks()
            try:
                await self.bus.stop()
            except Exception as exc:
                record_degradation('sensory_gate', exc)
                logger.error("❌ SensoryGate bus shutdown failed: %s", exc)
            if self.browser is not None:
                try:
                    await self.browser.close()
                except Exception as exc:
                    record_degradation('sensory_gate', exc)
                    logger.error("❌ SensoryGate browser shutdown failed: %s", exc)
            logger.info("👁️ SensoryGate Actor stopped.")

    async def _heartbeat_loop(self):
        """Send heartbeats to the supervisor."""
        while self._is_active:
            try:
                await self.bus.send("heartbeat", {
                    "pid": os.getpid(),
                    "ts": time.time(),
                    "status": "healthy"
                })
            except Exception as e:
                record_degradation('sensory_gate', e)
                logger.error("❌ Heartbeat failed: %s", e)
            await asyncio.sleep(self._heartbeat_interval)

    async def _handle_browse(self, payload: Dict, trace_id: str):
        """Handle browse request."""
        url = payload.get("url")
        if not url:
            return {"error": "No URL provided"}
        
        logger.info("🌐 [%s] Browsing: %s", trace_id[:8], url)
        try:
            result = await self.browser.browse(url)
            return {
                "url": url,
                "content": result,
                "observation_only": True,
                "requires_governance_for_effects": True,
            }
        except Exception as e:
            record_degradation('sensory_gate', e)
            logger.error("❌ [%s] Browse failed: %s", trace_id[:8], e)
            return {"error": str(e)}

    async def _handle_search(self, payload: Dict, trace_id: str):
        """Handle search request via Wikipedia OpenSearch API."""
        import urllib.parse
        import urllib.request
        import json

        query = payload.get("query")
        if not query:
            return {"error": "No query provided"}

        logger.info("🔍 [%s] Wikipedia search: %s", trace_id[:8], query)
        try:
            url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote(query)}&limit=3&namespace=0&format=json"

            def fetch():
                req = urllib.request.Request(url, headers={'User-Agent': 'Aura/1.0'})
                with urllib.request.urlopen(req, timeout=5) as response:
                    return json.loads(response.read().decode())

            data = await asyncio.to_thread(fetch)

            if len(data) >= 4 and len(data[1]) > 0:
                titles = data[1]
                snippets = data[2]
                urls = data[3]

                results = []
                for i in range(len(titles)):
                    results.append(f"{titles[i]}: {snippets[i]} ({urls[i]})")

                return {
                    "query": query,
                    "source": "wikipedia",
                    "results": results,
                    "observation_only": True,
                    "requires_governance_for_effects": True,
                }
            else:
                return {
                    "query": query,
                    "source": "wikipedia",
                    "results": [],
                    "observation_only": True,
                    "requires_governance_for_effects": True,
                }

        except Exception as e:
            record_degradation('sensory_gate', e)
            logger.error("❌ [%s] Wikipedia search failed: %s", trace_id[:8], e)
            return {"error": str(e)}

    async def _handle_shutdown(self, payload: Any, trace_id: str):
        self._is_active = False
        return "Acknowledged"

def start_sensory_gate(connection, *args, **kwargs):
    """Process entry point."""
    # Set up logging for the child process
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    actor = SensoryGateActor(connection)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(actor.run())
        loop.run_until_complete(loop.shutdown_asyncgens())
    finally:
        asyncio.set_event_loop(None)
        loop.close()


if __name__ == "__main__":
    # This is normally started via multiprocessing.Process
    pass  # no-op: intentional
