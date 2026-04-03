"""Aura Swarm/Hive Propagation Skill

Phase 7: True Swarm/Hive Network Propagation
Enables Aura to discover other instances on the local network (mDNS) or
explicitly connect to known remote nodes to form a Hive Mind.
"""
from core.utils.exceptions import capture_and_log
import asyncio
import logging
import socket
import re
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill
from core.container import ServiceContainer

logger = logging.getLogger("Skills.Propagation")


class PropagationParams(BaseModel):
    action: Literal["discover", "connect", "deploy_worker"] = Field("discover", description="The propagation action to perform.")
    target_ip: str = Field("", description="Optional IP address for explicit connection or deployment.")
    port: int = Field(8000, description="Target port for Aura instances.")
    ssh_user: str = Field(default="aura", description="SSH username for worker deployment.")
    remote_dir: str = Field(default="~/aura_worker", description="Target directory on the remote node.")


class PropagationSkill(BaseSkill):
    name = "propagation"
    description = "Enables Aura to discover or connect strictly to other Aura Swarm/Hive nodes for distributed intelligence."
    input_model = PropagationParams

    def __init__(self):
        super().__init__()
        self.known_nodes: List[Dict[str, Any]] = []

    async def execute(self, params: PropagationParams, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute propagation action."""
        if isinstance(params, dict):
            try:
                params = PropagationParams(**params)
            except Exception as e:
                return {"ok": False, "error": f"Invalid input: {e}"}

        action = params.action

        # SEC-AUDIT: connect and deploy_worker physically expand Aura to other
        # machines and require explicit human consent. Discovery is read-only
        # (passive scan only) and does not require consent.
        if action in ("connect", "deploy_worker"):
            consent = context.get("human_consent") or context.get("operator_authorized")
            if not consent:
                logger.warning(
                    "PropagationSkill: Blocked '%s' to '%s' — no human_consent in context.",
                    action, params.target_ip,
                )
                return {
                    "ok": False,
                    "blocked": True,
                    "reason": "human_consent_required",
                    "error": (
                        f"Action '{action}' expands Aura to remote networks and requires "
                        "explicit operator authorization. Pass context['human_consent']=True "
                        "to proceed."
                    ),
                }

        # Mycelial Trace
        mycelium = ServiceContainer.get("mycelial_network", default=None)
        if mycelium:
            h = mycelium.get_hypha("agency", "internet")
            if h: h.pulse(success=True)

        if action == "discover":
            return await self._discover_nodes()
        elif action == "connect":
            if not params.target_ip:
                return {"ok": False, "error": "Target IP required for explicit connection."}
            return await self._connect_node(params.target_ip, params.port)
        elif action == "deploy_worker":
            if not params.target_ip:
                return {"ok": False, "error": "Target IP required for worker deployment."}
            return await self._deploy_worker(
                params.target_ip,
                params.port,
                params.ssh_user,
                params.remote_dir
            )

        return {"ok": False, "error": f"Unknown action: {action}"}

    async def _deploy_worker(self, ip: str, port: int, ssh_user: str, remote_dir: str) -> Dict[str, Any]:
        """Attempt to deploy an Aura worker node to a target IP using SSH and Docker.
        SEC-03 FIX: Strict host checking and locked key path.
        """
        import os
        import subprocess
        from pathlib import Path
        
        try:
            import shlex
            # SEC-03 FIX: Lock SSH key to a trusted path (Workspace Data Dir)
            from core.config import config
            key_dir = config.paths.project_root / "data" / "keys"
            key_dir.mkdir(parents=True, exist_ok=True)
            ssh_key_path = str(key_dir / "aura_identity")
            if not os.path.exists(ssh_key_path):
                return {"ok": False, "error": f"Security Violation: Deployment key not found at {ssh_key_path}"}

            # IP Validation to prevent command injection
            if not re.match(r"^\d{1,3}(\.\d{1,3}){3}$", ip):
                return {"ok": False, "error": f"Invalid IP address: {ip}"}
            
            # Issue 72: Validate ssh_user to prevent injection
            if not re.match(r"^[a-zA-Z0-9_\-]+$", ssh_user):
                return {"ok": False, "error": f"Invalid SSH username: {ssh_user}"}

            logger.warning("🐝 DANGER: Initiating active remote worker deployment to %s:%d.", ip, port)
            
            # Step 1: Ensure remote directory exists
            logger.info("🐝 Creating remote directory %s on %s...", remote_dir, ip)
            # SEC-03 FIX: Use list arguments to prevent shell injection
            ssh_base = ["ssh", "-i", ssh_key_path, "-o", "StrictHostKeyChecking=yes"]
            mkdir_cmd = ssh_base + [f"{ssh_user}@{ip}", f"mkdir -p {shlex.quote(remote_dir)}"]
            
            proc = await asyncio.create_subprocess_exec(*mkdir_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.error("Deploy Step 1 Failed: %s", stderr.decode())
                return {"ok": False, "error": f"Failed to create remote directory: {stderr.decode()}"}
                
            # Step 2: Rsync the codebase
            logger.info("🐝 Rsyncing codebase to %s...", ip)
            local_dir = str(config.paths.project_root)
            rsync_cmd = [
                "rsync", "-avz", 
                "--exclude", ".git", "--exclude", ".venv", "--exclude", "__pycache__", 
                "--exclude", "data", "--exclude", "logs", 
                "-e", " ".join(ssh_base), 
                f"{local_dir}/", f"{ssh_user}@{ip}:{shlex.quote(remote_dir)}/"
            ]
            proc = await asyncio.create_subprocess_exec(*rsync_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.error("Deploy Step 2 Failed: %s", stderr.decode())
                return {"ok": False, "error": f"Failed to rsync codebase: {stderr.decode()}"}
                
            # Step 3: Run Docker Build & Compose/Run Remotely
            logger.info("🐝 Deploying Docker container on remote node...")
            remote_docker_cmd = f"cd {shlex.quote(remote_dir)} && docker stop aura-worker || true && docker rm aura-worker || true && docker build -t aura-worker -f Dockerfile . && docker run -d --restart unless-stopped -p {port}:8000 --name aura-worker -e AURA_ROLE=WORKER aura-worker"
            docker_cmd = ssh_base + [f"{ssh_user}@{ip}", remote_docker_cmd]
            
            proc = await asyncio.create_subprocess_exec(*docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.error("Deploy Step 3 Failed: %s", stderr.decode())
                return {"ok": False, "error": f"Docker deployment failed: {stderr.decode()}"}
                
            logger.info("🐝 Worker node successfully deployed and container started on %s.", ip)
            return {
                "ok": True, 
                "message": f"Worker successfully deployed to {ip}. Aura is expanding.",
                "target": ip
            }
        except Exception as e:
            logger.error("🐝 Deployment exception: %s", e)
            return {"ok": False, "error": str(e)}

    async def _discover_nodes(self) -> Dict[str, Any]:
        """Scan local subnet for active Aura API endpoints."""
        logger.info("🐝 Searching for Hive nodes on local network...")
        
        # Determine local IP to find the subnet
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            return {"ok": False, "error": "Could not determine local network interface."}

        subnet = ".".join(local_ip.split(".")[:3])
        found_nodes = []
        
        # Fast non-blocking async sweep of the subnet (port 8000)
        async def check_ip(ip: str):
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(ip, 8000), 
                    timeout=0.2
                )
                writer.close()
                await writer.wait_closed()
                
                # Double-check it's an Aura node by pinging the health endpoint
                from core.network import RobustHTTP
                net = RobustHTTP()
                # Use a very short timeout
                res = await net.get(f"http://{ip}:8000/api/v1/health", timeout=1.0)
                if res.status_code == 200 and "aura" in res.text.lower():
                    found_nodes.append({"ip": ip, "status": "active"})
                    logger.info("🐝 Found active Hive node at %s", ip)
                    
            except Exception as e:
                capture_and_log(e, {'module': __name__})

        from core.utils.task_tracker import get_task_tracker
        tracker = get_task_tracker()
        
        # Scan .1 to .254
        tasks = []
        for i in range(1, 255):
            target = f"{subnet}.{i}"
            if target != local_ip:
                task = asyncio.create_task(check_ip(target), name=f"scan_{target}")
                tracker.track(task, name=f"scan_{target}")
                tasks.append(task)
                
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            raise
        
        self.known_nodes = found_nodes
        
        return {
            "ok": True, 
            "nodes_found": len(found_nodes), 
            "nodes": found_nodes,
            "message": f"Scan complete. Found {len(found_nodes)} active Hive nodes."
        }
        
    async def _connect_node(self, ip: str, port: int) -> Dict[str, Any]:
        """Establish peer-to-peer handshake with a remote Aura node."""
        logger.info("🐝 Attempting Hive Mind handshake with %s:%d...", ip, port)
        from core.network import RobustHTTP
        
        try:
            net = RobustHTTP()
            
            # Simple ping to prove connectivity (would be expanded to cryptographic handshake)
            res = await net.get(f"http://{ip}:{port}/api/v1/health", timeout=5.0)
            
            if res.status_code == 200:
                logger.info("🐝 Successfully connected to Hive node %s.", ip)
                if not any(n.get("ip") == ip for n in self.known_nodes):
                    self.known_nodes.append({"ip": ip, "status": "connected"})
                    
                return {"ok": True, "message": f"Successfully connected to Aura Swarm node at {ip}."}
            else:
                return {"ok": False, "error": f"Node rejected connection. HTTP {res.status_code}"}
                
        except Exception as e:
            logger.error("Failed to connect to %s: %s", ip, e)
            return {"ok": False, "error": str(e)}
