from core.utils.task_tracker import get_task_tracker
import os
import sys
import time
import json
import logging
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("Aura.Tunnel")

class TunnelManager:
    """
    Manages the lifecycle of a cloudflared tunnel for the Aura API.
    """
    def __init__(self, port: int = 8000):
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.public_url: Optional[str] = None
        self.stop_event = threading.Event()
        self.log_file = Path("logs/tunnel.log")
        get_task_tracker().create_task(get_storage_gateway().create_dir(self.log_file.parent, cause='TunnelManager.__init__'))

    async def find_cloudflared(self) -> Optional[str]:
        """Check if cloudflared is installed (Async)."""
        try:
            process = await asyncio.create_subprocess_exec(
                "which", "cloudflared",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                return stdout.decode().strip()
        except Exception:
            pass
        return None

    async def start_tunnel(self):
        """Start a quick-tunnel or a permanent tunnel (Async)."""
        binary = await self.find_cloudflared()
        if not binary:
            logger.error("cloudflared binary not found. Please install it to enable remote access.")
            return False

        logger.info(f"Initiating tunnel for port {self.port}...")
        
        # Build command for a quick-tunnel (persists while process lives)
        cmd = [binary, "tunnel", "--url", f"http://localhost:{self.port}"]
        
        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Start background task to monitor output
            get_task_tracker().create_task(self._monitor_logs())
            
            # Wait for URL to be detected
            timeout = 30
            start_time = time.time()
            while not self.public_url and time.time() - start_time < timeout:
                if self.process.returncode is not None:
                    logger.error("Tunnel process exited unexpectedly.")
                    return False
                await asyncio.sleep(1)

            if self.public_url:
                logger.info(f"Tunnel establish successful")
                logger.info(f"PUBLIC ACCESS URL: {self.public_url}")
                await asyncio.to_thread(self._save_url, self.public_url)
                return True
            else:
                logger.error("Timeout reached waiting for tunnel URL.")
                # Kill zombie process on timeout
                self.process.terminate()
                await self.process.wait()
                self.process = None
                return False

        except Exception as e:
            logger.error(f"Failed to start tunnel: {e}")
            return False

    async def _monitor_logs(self):
        """Parse stderr for that .trycloudflare.com URL (Async)."""
        if not self.process or not self.process.stderr:
            return

        # Ensure directory exists for logging
        get_task_tracker().create_task(get_storage_gateway().create_dir(self.log_file.parent, cause='TunnelManager._monitor_logs'))
        
        async with asyncio.Lock(): # Simple guard for log file access if needed
            with open(self.log_file, "a") as log:
                async for line in self.process.stderr:
                    line_str = line.decode()
                    log.write(line_str)
                    if ".trycloudflare.com" in line_str:
                        # Extract URL: https://some-slug.trycloudflare.com
                        parts = line_str.split()
                        for p in parts:
                            if "https://" in p and ".trycloudflare.com" in p:
                                self.public_url = p.strip()
                                break
                    if self.stop_event.is_set():
                        break

    def _save_url(self, url: str):
        """Save the URL to a JSON file for the UI or Rebooter to find."""
        try:
            data = {
                "url": url,
                "timestamp": time.time(),
                "port": self.port
            }
            with open("data/active_tunnel.json", "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save tunnel metadata: {e}")

    async def stop_tunnel(self):
        """Safe shutdown (Async)."""
        self.stop_event.set()
        if self.process:
            logger.info("Stopping tunnel process...")
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
            self.process = None
        
        # Use robust path for temporary metadata
        try:
            from core.config import config
            data_path = config.paths.data_dir / "active_tunnel.json"
        except ImportError:
            data_path = Path("data/active_tunnel.json")
            
        if data_path.exists():
            try:
                get_task_tracker().create_task(get_storage_gateway().delete(data_path, cause='TunnelManager.stop_tunnel'))
            except Exception:
                pass

async def main_async():
    manager = TunnelManager()
    if await manager.start_tunnel():
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            await manager.stop_tunnel()
    else:
        raise SystemExit(1)

if __name__ == "__main__":
    if sys.argv[1] == "--check":
        tm = TunnelManager()
        # Non-ideal: running async in a check script, but keeps consistency
        binary = asyncio.run(tm.find_cloudflared())
        if binary:
            print("cloudflared: FOUND")
            raise SystemExit(0)
        else:
            print("cloudflared: NOT FOUND")
            raise SystemExit(1)
            
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Tunnel shutdown requested.")
