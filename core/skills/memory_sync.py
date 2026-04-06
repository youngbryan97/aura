"""Aura Hive Mind Sync
Enables memory synchronization between Home and Cloud variants via a private Git repository.
"""
import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.MemorySync")

class MemorySyncParams(BaseModel):
    action: Literal["sync", "push", "pull"] = Field("sync", description="The sync action to perform.")
    consented: bool = Field(False, description="Manual consent required for cloud memory upload (push/sync).")

class MemorySyncSkill(BaseSkill):
    name = "memory_sync"
    description = "Syncs semantic memory (data/memory) with a remote Git repository."
    input_model = MemorySyncParams
    
    def __init__(self):
        super().__init__()
        self.memory_path = Path("data/memory")
        self.repo_url = os.getenv("AURA_MEMORY_REPO")  # Private Git Repo URL
        
    async def execute(self, params: MemorySyncParams, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute memory synchronization.
        """
        # Legacy support
        if isinstance(params, dict):
            try:
                params = MemorySyncParams(**params)
            except Exception as e:
                return {"ok": False, "error": f"Invalid input: {e}"}

        action = params.action # sync, push, pull
        
        if not self.repo_url:
            return {"ok": False, "error": "AURA_MEMORY_REPO env var not set."}
            
        # SEC-04: Robust out-of-band confirmation via EventBus
        if (action == "push" or action == "sync"):
            logger.warning("MemorySync: Awaiting manual human consent for cloud upload...")
            if not await self._await_human_consent(action):
                logger.error("MemorySync blocked: Human consent REJECTED or TIMEOUT.")
                return {"ok": False, "error": "Security Restriction: Manual human-in-the-loop consent required."}
            
        if not self.memory_path.exists():
            self.memory_path.mkdir(parents=True, exist_ok=True)
            
        # Check if it's a git repo
        git_dir = self.memory_path / ".git"
        if not git_dir.exists():
            return await asyncio.to_thread(self._initialize_repo)
            
        if action == "push":
            return await asyncio.to_thread(self._push)
        elif action == "pull":
            return await asyncio.to_thread(self._pull)
        else:
            # Sync: pull then push to cloud git repo
            logger.info("Memory sync: pulling then pushing to cloud repository.")
            pull_res = await asyncio.to_thread(self._pull)
            push_res = await asyncio.to_thread(self._push)
            
            return {
                "ok": pull_res["ok"] and push_res["ok"],
                "p2p_nodes_synced": p2p_synced,
                "pull": pull_res,
                "push": push_res
            }
            
    async def _await_human_consent(self, action: str, timeout_s: float = 60.0) -> bool:
        """SEC-04: Robust out-of-band confirmation via EventBus."""
        try:
            from core.event_bus import get_event_bus
            bus = get_event_bus()
            
            # Subscribe to the response topic
            confirmation_queue = await bus.subscribe("human_consent_response")
            
            # Request approval via UI
            await bus.publish("human_consent_request", {
                "action": f"memory_sync:{action}",
                "message": f"Aura wants to sync your memory with the cloud ({action}). Approve?",
                "timeout": timeout_s
            })
            
            try:
                # Wait for the response event
                priority, seq, event = await asyncio.wait_for(
                    confirmation_queue.get(), timeout=timeout_s
                )
                data = event.get("data", {})
                return data.get("approved", False)
            except asyncio.TimeoutError:
                return False
            finally:
                await bus.unsubscribe("human_consent_response", confirmation_queue)
        except Exception as e:
            logger.error("Consent gate failure: %s", e)
            return False

    def _initialize_repo(self):
        try:
            cwd = str(self.memory_path)
            subprocess.run(["git", "init"], cwd=cwd, check=True)
            subprocess.run(["git", "remote", "add", "origin", self.repo_url], cwd=cwd, check=True)
            # Initial pull
            subprocess.run(["git", "pull", "origin", "main"], cwd=cwd, check=False)
            return {"ok": True, "message": "Memory repository initialized."}
        except Exception as e:
            return {"ok": False, "error": f"Init failed: {e}"}

    def _pull(self):
        try:
            cwd = str(self.memory_path)
            res = subprocess.run(["git", "pull", "origin", "main"], cwd=cwd, capture_output=True, text=True)
            if res.returncode == 0:
                logger.info("Memory Pulled successfully.")
                return {"ok": True, "message": "Memory synced from cloud."}
            else:
                logger.warning("Pull failed: %s", res.stderr)
                return {"ok": False, "error": res.stderr}
        except Exception as e:
            return {"ok": False, "error": f"Pull error: {e}"}

    def _push(self):
        """SK-02: Hardened push that excludes binary/db files."""
        try:
            cwd = str(self.memory_path)
            
            # Ensure .gitignore exists to prevent accidental DB commits
            gitignore = self.memory_path / ".gitignore"
            if not gitignore.exists():
                gitignore.write_text("*.db\n*.sqlite3\n*.sqlite\n*.bin\n*.safetensors\n.DS_Store\n")
                subprocess.run(["git", "add", ".gitignore"], cwd=cwd, check=False)

            # Only add specific text-based artifacts (SK-02)
            subprocess.run(["git", "add", "*.md", "*.json", "*.jsonl"], cwd=cwd, check=False)
            
            subprocess.run(["git", "commit", "-m", "Aura Memory Update [SK-02 Hardened]"], cwd=cwd, check=False)
            res = subprocess.run(["git", "push", "origin", "main"], cwd=cwd, capture_output=True, text=True)
            
            if res.returncode == 0:
                logger.info("Memory Pushed successfully.")
                return {"ok": True, "message": "Memory uploaded to cloud."}
            else:
                logger.warning("Push failed: %s", res.stderr)
                return {"ok": False, "error": res.stderr}
        except Exception as e:
            return {"ok": False, "error": f"Push error: {e}"}