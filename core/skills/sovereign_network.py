import asyncio
import ipaddress
import itertools
import logging
import platform
import re
import shutil
import socket
import subprocess
from typing import Any

from pydantic import BaseModel, Field

from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT
from core.runtime.errors import NetworkEffectDenied, record_degradation
from core.runtime.network_gateway import build_stream_endpoint, get_network_gateway
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.runtime.task_ownership import create_tracked_task
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.SovereignNetwork")

TCP_DISCOVERY_BATCH_SIZE = 16
TCP_DISCOVERY_TIMEOUT_S = 0.35
BACKGROUND_NETWORK_MIN_IDLE_S = 1800.0

class NetworkInput(BaseModel):
    mode: str = Field("status", description="Mode: 'status', 'recon', 'scan', 'audit', 'discovery'")
    target: str | None = Field(None, description="Target IP, subnet, or host (e.g., '192.168.1.0/24' or '8.8.8.8').")
    stealth: bool = Field(True, description="Whether to use stealthy (ARP) discovery in 'recon' mode.")
    ports: str | None = Field("8000", description="Comma-separated ports for 'audit' or 'discovery' mode.")

class SovereignNetworkSkill(BaseSkill):
    """The unified network capability for Aura.
    Handles connectivity checks, stealthy recon, and advanced scanning.
    """
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT

    
    name = "sovereign_network"
    description = "Monitor connectivity, discover local devices, and audit network services."
    input_model = NetworkInput
    
    def __init__(self):
        super().__init__()
    
    async def execute(self, params: NetworkInput, context: dict[str, Any]) -> dict[str, Any]:
        """Unified entry point for all network activities."""
        context = context or {}
        if isinstance(params, dict):
            try:
                params = NetworkInput(**params)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('sovereign_network', e)
                return {"ok": False, "error": f"Invalid input: {e}"}

        mode = params.mode
        target = params.target
        source = str(
            context.get("origin")
            or context.get("source")
            or context.get("intent_source")
            or context.get("request_origin")
            or ""
        ).strip().lower()
        user_initiated = source in {"user", "api", "chat", "desktop", "voice", "web"}
        background_declared = bool(context.get("is_background") or context.get("background") or (source and not user_initiated))
        if background_declared and mode in {"recon", "scan", "audit", "discovery"}:
            try:
                from core.runtime.background_policy import background_activity_reason

                reason = background_activity_reason(
                    context.get("orchestrator"),
                    min_idle_seconds=BACKGROUND_NETWORK_MIN_IDLE_S,
                    max_memory_percent=72.0,
                    max_failure_pressure=0.20,
                    require_conversation_ready=False,
                )
            except (ImportError, AttributeError, RuntimeError) as policy_exc:
                record_degradation(
                    "sovereign_network",
                    policy_exc,
                    severity="warning",
                    action="denied network action because background policy was unavailable",
                    extra={"mode": mode},
                )
                reason = "background_policy_unavailable"
            if reason:
                return {
                    "ok": False,
                    "status": "deferred",
                    "reason": reason,
                    "message": f"Network {mode} deferred while foreground conversation is protected ({reason}).",
                }
        
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
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('sovereign_network', e)
            logger.error("Network skill failed: %s", e)
            return {"ok": False, "error": str(e)}

    async def _get_status(self) -> dict[str, Any]:
        """Connectivity and interface status."""
        local_ip = self._get_primary_ip()
        internet = await self._check_internet()
        
        system = platform.system()
        interfaces = "Unknown"
        if system == "Darwin":
            res = await get_subprocess_gateway().run_async(
                ["networksetup", "-listallhardwareports"],
                timeout=10.0,
                read_only=True,
                source="skills.sovereign_network.status.interfaces",
                accelerator_capability="none",
            )
            interfaces = res.stdout[:500]
        elif system == "Linux":
            res = await get_subprocess_gateway().run_async(
                ["ip", "link", "show"],
                timeout=10.0,
                read_only=True,
                source="skills.sovereign_network.status.interfaces",
                accelerator_capability="none",
            )
            interfaces = res.stdout[:500]
            
        return {
            "ok": True,
            "local_ip": local_ip,
            "internet_accessible": internet,
            "interfaces": interfaces,
            "os": system
        }

    async def _perform_recon(self, stealth: bool) -> dict[str, Any]:
        """ARP-based (stealth) or Ping-based discovery."""
        local_ip = self._get_primary_ip()
        if local_ip == "127.0.0.1":
            return {"ok": False, "error": "No primary network interface found."}
            
        devices = []
        # Attempt ARP cache check (Max Stealth)
        try:
            cmd_args = ["arp", "-a"] if platform.system() != "Windows" else ["arp", "-g"]
            result = await get_subprocess_gateway().run_async(
                cmd_args,
                timeout=10.0,
                read_only=True,
                check=True,
                source="skills.sovereign_network.recon.arp_cache",
                accelerator_capability="auto",
            )
            output = result.stdout or ""
            for line in output.split('\n'):
                match = re.search(r"\(([\d\.]+)\) at ([\w:]+)", line) # macOS/Linux format
                if match:
                    devices.append({"ip": match.group(1), "mac": match.group(2), "source": "arp"})
        except (subprocess.SubprocessError, OSError) as e:
            record_degradation('sovereign_network', e)
            logger.debug("ARP discovery failed: %s", e)
        
        return {
            "ok": True,
            "local_ip": local_ip,
            "devices": list({d['ip']: d for d in devices}.values()),
            "count": len(devices),
            "mode": "stealth_recon"
        }

    async def _perform_scan(self, target: str) -> dict[str, Any]:
        """Fast Nmap host discovery."""
        target = target or self._guess_subnet()
        if not shutil.which("nmap"):
            return {"ok": False, "error": "The 'nmap' utility is not installed on this system. Aura cannot perform deep network scans without it. Please install nmap or use 'recon' mode for basic ARP-based discovery."}
        logger.info("📡 Nmap Scanning: %s", target)
        try:
            # -sn: Ping scan (no port scan)
            process = await get_subprocess_gateway().spawn_async(
                ["nmap", "-sn", target],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                source="skills.sovereign_network.scan.nmap",
                accelerator_capability="none",
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=60)
            return {"ok": True, "output": stdout.decode(), "target": target}
        except FileNotFoundError:
            return {"ok": False, "error": "The 'nmap' utility is not installed on this system. Aura cannot perform deep network scans without it. Please install nmap or use 'recon' mode for basic ARP-based discovery."}


    async def _perform_audit(self, target: str, ports: str) -> dict[str, Any]:
        """Nmap port/service audit."""
        if not target:
            return {"ok": False, "error": "Audit mode requires a 'target' IP."}
        if not shutil.which("nmap"):
            return {"ok": False, "error": "The 'nmap' utility is required for network auditing. Please install it to enable this capability."}
        logger.info("🔍 Auditing %s on ports %s", target, ports)
        try:
            # -F: Fast, -sV: Version detection
            process = await get_subprocess_gateway().spawn_async(
                ["nmap", "-p", ports, "-sV", target],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                source="skills.sovereign_network.audit.nmap",
                accelerator_capability="none",
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=90)
            return {"ok": True, "output": stdout.decode()}
        except FileNotFoundError:
            return {"ok": False, "error": "The 'nmap' utility is required for network auditing. Please install it to enable this capability."}

    async def _perform_discovery(self, target: str, ports: str) -> dict[str, Any]:
        """Discover other Aura instances on the network."""
        target = target or self._guess_subnet()
        logger.info("📡 Aura Peer Discovery starting on %s:%s", target, ports)
        if not shutil.which("nmap"):
            logger.info("nmap unavailable; falling back to bounded TCP peer discovery.")
            return await self._perform_tcp_peer_discovery(target, ports)

        peers = []
        try:
            # -p: port, --open: only show open, -oG -: grepable output
            process = await get_subprocess_gateway().spawn_async(
                ["nmap", "-p", ports, "--open", "-oG", "-", target],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                source="skills.sovereign_network.discovery.nmap",
                accelerator_capability="none",
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
        except (subprocess.SubprocessError, OSError) as e:
            record_degradation('sovereign_network', e)
            return {"ok": False, "error": str(e)}

        return {"ok": True, "peers": peers, "count": len(peers)}

    async def _perform_tcp_peer_discovery(self, target: str, ports: str) -> dict[str, Any]:
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

        async def probe(host: str) -> dict[str, Any] | None:
            async with semaphore:
                try:
                    admission = await get_network_gateway().connect_stream(
                        build_stream_endpoint(host, first_port),
                        open_timeout=TCP_DISCOVERY_TIMEOUT_S,
                        source="skills:sovereign_network.peer_discovery",
                        read_only=True,
                        allow_private_target=True,
                    )
                    writer = admission.writer
                    writer.close()
                    try:
                        await asyncio.wait_for(
                            writer.wait_closed(),
                            timeout=TCP_DISCOVERY_TIMEOUT_S,
                        )
                    except asyncio.CancelledError:
                        raise
                    except (RuntimeError, OSError, TimeoutError, AttributeError) as close_exc:
                        logger.debug("TCP peer writer close failed for %s:%s: %s", host, first_port, close_exc)
                    return {"address": host, "rpc_port": first_port, "source": "tcp_connect"}
                except asyncio.CancelledError:
                    raise
                except (
                    NetworkEffectDenied,
                    RuntimeError,
                    OSError,
                    TimeoutError,
                    AttributeError,
                ) as e:
                    logger.debug("TCP peer probe failed for %s:%s: %s", host, first_port, e)
                    return None

        peers: list[dict[str, Any]] = []
        for start in range(0, len(hosts), TCP_DISCOVERY_BATCH_SIZE):
            batch = hosts[start:start + TCP_DISCOVERY_BATCH_SIZE]
            tasks: list[asyncio.Task] = []
            for host in batch:
                task_name = f"sovereign_network.tcp_probe.{host}"
                tasks.append(create_tracked_task(probe(host), name=task_name))
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

    def _candidate_hosts(self, target: str) -> list[str]:
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
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            logger.debug("Primary IP discovery failed, defaulting to localhost: %s", e)
            return "127.0.0.1"

    async def _check_internet(self) -> bool:
        try:
            # Run blocking call in thread
            await asyncio.to_thread(socket.create_connection, ("8.8.8.8", 53), 3)
            return True
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            logger.debug("Internet connectivity check failed: %s", e)
            return False

    def _guess_subnet(self) -> str:
        ip = self._get_primary_ip()
        return ".".join(ip.split('.')[:3]) + ".0/24"


# Compatibility alias for older class-name derivation logic.
Sovereign_networkSkill = SovereignNetworkSkill
