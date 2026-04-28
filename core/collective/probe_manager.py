"""core/collective/probe_manager.py
Phase 16.4: Ghost Deployment - Resource Monitoring Probes.
Spawns and manages lightweight monitoring scripts.
"""
from core.runtime.errors import record_degradation
from core.runtime.atomic_writer import atomic_write_text
from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from core.container import ServiceContainer

logger = logging.getLogger("Aura.Collective.ProbeManager")

class ProbeManager:
    """Manages external 'Ghost Probes' for long-term monitoring."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.probes: Dict[str, asyncio.subprocess.Process] = {}
        self.probe_metadata: Dict[str, Dict[str, Any]] = {}

    async def deploy_probe(self, probe_id: str, target: str, type: str = "file", duration: int = 3600) -> bool:
        """Spawn a ghost probe process."""
        if probe_id in self.probes:
            logger.warning("Probe %s already active.", probe_id)
            return False

        # Create a simple probe script if it doesn't exist
        # In a real impl, we'd have a template. For now, we'll use a python one-liner or simple script.
        # This probe will write to a local log or socket that we watch, or just report via stdout.
        
        probe_script = f"""
import time, os, sys
target = "{target}"
probe_type = "{type}"
print(f"ghost_probe_start:{{probe_type}}:{{target}}")
try:
    start_time = time.time()
    while time.time() - start_time < {duration}:
        if probe_type == "file":
            if os.path.exists(target):
                mtime = os.path.getmtime(target)
                print(f"ghost_update:file_exists:{{mtime}}")
        elif probe_type == "ping":
             # Simple ping simulation
             print(f"ghost_update:ping_ok")
        
        sys.stdout.flush()
        await asyncio.sleep(60) # Scan every minute
except Exception as e:
    print(f"ghost_error:{{e}}")
"""
        probe_path = Path(tempfile.gettempdir()) / f"aura_probe_{probe_id}.py"
        atomic_write_text(probe_path, probe_script)
        
        try:
            # Spawn in background with asyncio
            process = await asyncio.create_subprocess_exec(
                "python3", str(probe_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True
            )
            
            self.probes[probe_id] = process
            self.probe_metadata[probe_id] = {
                "target": target,
                "type": type,
                "start_time": time.time(),
                "expiry": time.time() + duration,
                "path": str(probe_path)
            }
            
            # Start a background listener for this probe
            get_task_tracker().create_task(self._listen_to_probe(probe_id))
            
            logger.info("👻 Ghost Probe '%s' deployed to watch %s.", probe_id, target)
            return True
        except Exception as e:
            record_degradation('probe_manager', e)
            logger.error("Failed to deploy probe %s: %s", probe_id, e)
            return False

    async def _listen_to_probe(self, probe_id: str):
        """Listen for telemetry from a specific probe."""
        process = self.probes.get(probe_id)
        if not process: return

        while process.returncode is None:
            line_bytes = await process.stdout.readline()
            if not line_bytes: break
            
            line = line_bytes.decode().strip()
            if line.startswith("ghost_update:"):
                update = line.split(":", 2)[1:]
                self.orchestrator.enqueue_message(f"Impulse [GHOST:{probe_id}]: {update}")
            elif line.startswith("ghost_error:"):
                err = line.split(":", 1)[1]
                logger.error("Ghost Probe %s error: %s", probe_id, err)

        # Cleanup
        await self.cleanup_probe(probe_id)

    async def cleanup_probe(self, probe_id: str):
        """Terminate and cleanup a probe's resources."""
        if probe_id in self.probes:
            proc = self.probes.pop(probe_id)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception as e:
                record_degradation('probe_manager', e)
                logger.debug("Failed to kill probe process group %d: %s", proc.pid, e)
            
            meta = self.probe_metadata.pop(probe_id, {})
            path = meta.get("path")
            if path and os.path.exists(path):
                os.remove(path)
                
            logger.info("👻 Ghost Probe '%s' cleaned up.", probe_id)

    async def auto_cleanup_loop(self):
        """Periodically remove expired probes."""
        while True:
            await asyncio.sleep(60)
            now = time.time()
            to_remove = [pid for pid, meta in self.probe_metadata.items() if now > meta["expiry"]]
            for pid in to_remove:
                await self.cleanup_probe(pid)