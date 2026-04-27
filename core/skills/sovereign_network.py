import asyncio
import ipaddress
import itertools
import logging
import platform
import socket
import subprocess
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.SovereignNetwork")

TCP_DISCOVERY_BATCH_SIZE = 16
TCP_DISCOVERY_TIMEOUT_S = 0.35

class NetworkInput(BaseModel):
    mode: str = Field("status", description="Mode: 'status', 'recon', 'scan', 'audit', 'discovery'")
    target: Optional[str] = Field(None, description="Target IP, subnet, or host (e.g., '192.168.1.0/24' or '8.8.8.8').")
    stealth: bool = Field(True, description="Whether to use stealthy (ARP) discovery in 'recon' mode.")
    ports: Optional[str] = Field("8000", description="Comma-separated ports for 'audit' or 'discovery' mode.")

class SovereignNetworkSkill(BaseSkill):
    """The unified network capability for Aura.
    Handles connectivity checks, stealthy recon, and advanced scanning.
    """
    
    name = "sovereign_network"
    description = "Monitor connectivity, discover local devices, and audit network services."
    input_model = NetworkInput
    
    def __init__(self):
        super().__init__()
    
    async def execute(self, params: NetworkInput, context: Dict[str, Any]) -> Dict[str, Any]:
        """Unified entry point for all network activities."""
        if isinstance(params, dict):
            try:
                params = NetworkInput(**params)
            except Exception as e:
                return {"ok": False, "error": f"Invalid input: {e}"}

        mode = params.mode
        target = params.target
        
        try:
            if mode == "status":
                return await self._get_status()
            elif mode == "recon":
                return await self._perform_recon(params.stealth)
            elif mode == "scan":
                return await self._perform_scan(target or "localhost")
            elif mode == "audit":
                return await self._perform_audit(target or "localhost", params.ports or "80,443,8000")
            elif mode == "discovery":
                return await self._perform_discovery(target or "192.168.1.0/24", params.ports or "80")
            else:
                return {"ok": False, "error": f"Unsupported network mode: {mode}"}
        except Exception as e:
            logger.error("Network skill failed: %s", e)
            return {"ok": False, "error": str(e)}

    async def _get_status(self) -> Dict[str, Any]:
        """Connectivity and interface status."""
        local_ip = self._get_primary_ip()
        internet = await self._check_internet()
        
        system = platform.system()
        interfaces = "Unknown"
        if system == "Darwin":
            res = await asyncio.to_thread(subprocess.run, ["networksetup", "-listallhardwareports"], capture_output=True, text=True)
            interfaces = res.stdout[:500]
        elif system == "Linux":
            res = await asyncio.to_thread(subprocess.run, ["ip", "link", "show"], capture_output=True, text=True)
            interfaces = res.stdout[:500]
            
        return {
            "ok": True,
            "local_ip": local_ip,
            "internet_accessible": internet,
            "interfaces": interfaces,
            "os": system
        }

    async def _perform_recon(self, stealth: bool) -> Dict[str, Any]:
        """ARP-based (stealth) or Ping-based discovery."""
        local_ip = self._get_primary_ip()
        if local_ip == "127.0.0.1":
            return {"ok": False, "error": "No primary network interface found."}
            
        devices = []
        # Attempt ARP cache check (Max Stealth)
        try:
            cmd_args = ["arp", "-a"] if platform.system() != "Windows" else ["arp", "-g"]
            output = await asyncio.to_thread(subprocess.check_output, cmd_args)
            output = output.decode()
            for line in output.split('\n'):
                match = re.search(r"\(([\d\.]+)\) at ([\w:]+)", line) # macOS/Linux format
                if match:
                    devices.append({"ip": match.group(1), "mac": match.group(2), "source": "arp"})
        except Exception as e:
            logger.debug("ARP discovery failed: %s", e)
        
        return {
            "ok": True,
            "local_ip": local_ip,
            "devices": list({d['ip']: d for d in devices}.values()),
            "count": len(devices),
            "mode": "stealth_recon"
        }

    async def _perform_scan(self, target: str) -> Dict[str, Any]:
        """Fast Nmap host discovery."""
        target = target or self._guess_subnet()
        logger.info("📡 Nmap Scanning: %s", target)
        try:
            # -sn: Ping scan (no port scan)
            process = await asyncio.create_subprocess_exec(
                "nmap", "-sn", target,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=60)
            return {"ok": True, "output": stdout.decode(), "target": target}
        except FileNotFoundError:
            return {"ok": False, "error": "The 'nmap' utility is not installed on this system. Aura cannot perform deep network scans without it. Please install nmap or use 'recon' mode for basic ARP-based discovery."}


    async def _perform_audit(self, target: str, ports: str) -> Dict[str, Any]:
        """Nmap port/service audit."""
        if not target:
            return {"ok": False, "error": "Audit mode requires a 'target' IP."}
        logger.info("🔍 Auditing %s on ports %s", target, ports)
        try:
            # -F: Fast, -sV: Version detection
            process = await asyncio.create_subprocess_exec(
                "nmap", "-p", ports, "-sV", target,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=90)
            return {"ok": True, "output": stdout.decode()}
        except FileNotFoundError:
            return {"ok": False, "error": "The 'nmap' utility is required for network auditing. Please install it to enable this capability."}

    async def _perform_discovery(self, target: str, ports: str) -> Dict[str, Any]:
        """Discover other Aura instances on the network."""
        target = target or self._guess_subnet()
        logger.info("📡 Aura Peer Discovery starting on %s:%s", target, ports)
        
        peers = []
        try:
            # -p: port, --open: only show open, -oG -: grepable output
            process = await asyncio.create_subprocess_exec(
                "nmap", "-p", ports, "--open", "-oG", "-", target,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=120)
            output = stdout.decode()
            
            # Simple parser for Nmap grepable output: Host: 192.168.1.5 (hostname)	Ports: 8000/open/tcp//...
            for line in output.split('\n'):
                if "Host:" in line and "/open/" in line:
                    match = re.search(r"Host: ([\d\.]+)", line)
                    if match:
                        ip = match.group(1)
                        peers.append({"address": ip, "rpc_port": int(ports.split(',')[0])})
        except FileNotFoundError:
            logger.info("nmap unavailable; falling back to bounded TCP peer discovery.")
            return await self._perform_tcp_peer_discovery(target, ports)
        except Exception as e:
            return {"ok": False, "error": str(e)}

        return {"ok": True, "peers": peers, "count": len(peers)}

    async def _perform_tcp_peer_discovery(self, target: str, ports: str) -> Dict[str, Any]:
        """Best-effort peer discovery that avoids hard dependency on Homebrew nmap."""
        first_port = self._first_port(ports)
        hosts = self._candidate_hosts(target)
        if not hosts:
            return {
                "ok": False,
                "error": f"No candidate hosts could be derived from target '{target}'.",
                "fallback": "tcp_connect",
            }

        semaphore = asyncio.Semaphore(TCP_DISCOVERY_BATCH_SIZE)

        async def probe(host: str) -> Optional[Dict[str, Any]]:
            async with semaphore:
                try:
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_connection(host, first_port),
                        timeout=TCP_DISCOVERY_TIMEOUT_S,
                    )
                    writer.close()
                    try:
                        await writer.wait_closed()
                    except Exception as close_exc:
                        logger.debug("TCP peer writer close failed for %s:%s: %s", host, first_port, close_exc)
                    return {"address": host, "rpc_port": first_port, "source": "tcp_connect"}
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.debug("TCP peer probe failed for %s:%s: %s", host, first_port, e)
                    return None

        tracker = None
        try:
            from core.utils.task_tracker import get_task_tracker

            tracker = get_task_tracker()
        except Exception:
            tracker = None

        peers: List[Dict[str, Any]] = []
        for start in range(0, len(hosts), TCP_DISCOVERY_BATCH_SIZE):
            batch = hosts[start:start + TCP_DISCOVERY_BATCH_SIZE]
            tasks: List[asyncio.Task] = []
            for host in batch:
                task_name = f"sovereign_network.tcp_probe.{host}"
                if tracker is not None:
                    tasks.append(tracker.create_task(probe(host), name=task_name))
                else:
                    tasks.append(get_task_tracker().create_task(probe(host), name=task_name))
            try:
                batch_results = await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            peers.extend(peer for peer in batch_results if peer)
        return {
            "ok": True,
            "peers": peers,
            "count": len(peers),
            "target": target,
            "fallback": "tcp_connect",
            "note": "nmap unavailable; used bounded TCP connect discovery instead.",
        }

    def _first_port(self, ports: str) -> int:
        try:
            return int(str(ports).split(",")[0].strip())
        except (TypeError, ValueError):
            return 8000

    def _candidate_hosts(self, target: str) -> List[str]:
        try:
            network = ipaddress.ip_network(target, strict=False)
        except ValueError:
            return [target]

        hosts = list(itertools.islice(network.hosts(), 256))
        if not hosts and network.num_addresses == 1:
            hosts = [network.network_address]
        return [str(host) for host in hosts[:256]]

    def _get_primary_ip(self) -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception as e:
            logger.debug("Primary IP discovery failed, defaulting to localhost: %s", e)
            return "127.0.0.1"

    async def _check_internet(self) -> bool:
        try:
            # Run blocking call in thread
            await asyncio.to_thread(socket.create_connection, ("8.8.8.8", 53), 3)
            return True
        except Exception as e:
            logger.debug("Internet connectivity check failed: %s", e)
            return False

    def _guess_subnet(self) -> str:
        ip = self._get_primary_ip()
        return ".".join(ip.split('.')[:3]) + ".0/24"


# Compatibility alias for older class-name derivation logic.
Sovereign_networkSkill = SovereignNetworkSkill
