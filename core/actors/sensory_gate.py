
import asyncio
import logging
import multiprocessing
import os
import sys
import time
from typing import Any, Dict, Optional

# Ensure we can import from the root
sys.path.append(os.getcwd())

from core.bus.local_pipe_bus import LocalPipeBus
from core.phantom_browser import PhantomBrowser

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

    async def run(self):
        """Main actor loop."""
        logger.info("👁️ SensoryGate Actor starting...")
        
        # Initialize browser in the child process
        self.browser = PhantomBrowser(visible=False)
        
        # Register handlers
        self.bus.register_handler("browse", self._handle_browse)
        self.bus.register_handler("search", self._handle_search)
        self.bus.register_handler("ping", lambda payload, tid: "pong")
        self.bus.register_handler("shutdown", self._handle_shutdown)
        
        self.bus.start()
        
        # Start heartbeat loop after bus is active
        asyncio.create_task(self._heartbeat_loop())
        
        logger.info("👁️ SensoryGate Actor ready.")
        
        while self._is_active:
            await asyncio.sleep(1.0)
            
        await self.bus.stop()
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
            return {"url": url, "content": result}
        except Exception as e:
            logger.error("❌ [%s] Browse failed: %s", trace_id[:8], e)
            return {"error": str(e)}

    async def _handle_search(self, payload: Dict, trace_id: str):
        """Handle search request via Live Wikipedia OpenSearch API."""
        import urllib.parse
        import urllib.request
        import json
        
        query = payload.get("query")
        if not query:
            return {"error": "No query provided"}
        
        logger.info("🔍 [%s] Searching Knowledge Base (Wikipedia): %s", trace_id[:8], query)
        try:
            url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote(query)}&limit=3&namespace=0&format=json"
            
            def fetch():
                req = urllib.request.Request(url, headers={'User-Agent': 'AuraContextBuilder/1.0'})
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
                    
                return {"query": query, "results": results}
            else:
                return {"query": query, "results": ["No reliable external knowledge found for query."]}
                
        except Exception as e:
            logger.error("❌ [%s] Search failed: %s", trace_id[:8], e)
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
    asyncio.run(actor.run())


if __name__ == "__main__":
    # This is normally started via multiprocessing.Process
    pass
