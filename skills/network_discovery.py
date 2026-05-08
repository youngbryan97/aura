import json
import logging

try:
    import netifaces
except ImportError:  # pragma: no cover - optional host dependency
    netifaces = None

try:
    import nmap
except ImportError:  # pragma: no cover - optional host dependency
    nmap = None

from infrastructure import BaseSkill

logger = logging.getLogger("Skills.NetScout")

class NetworkDiscovery(BaseSkill):
    """The 'Radar' of the machine.
    Allows Aura to map the local LAN and identify neighbor devices.
    """

    name = "network_discovery"
    description = "Scans the local network to find connected devices and open ports using nmap."

    def __init__(self):
        super().__init__()
        if nmap is None:
            logger.warning("python-nmap not installed; network discovery will report unavailable.")
            self.nm = None
            return
        try:
            self.nm = nmap.PortScanner()
        except (nmap.PortScannerError, OSError) as e:
            logger.error("Nmap not found: %s. Please install nmap (brew install nmap).", e)
            self.nm = None

    def _get_local_subnet(self):
        """Auto-detects the local subnet (e.g., 192.168.1.0/24)."""
        if netifaces is None:
            logger.warning("netifaces not installed; falling back to loopback subnet.")
            return "127.0.0.1"
        try:
            # Get default gateway interface
            gws = netifaces.gateways()
            default_interface = gws['default'][netifaces.AF_INET][1]
            
            # Get IP info
            addrs = netifaces.ifaddresses(default_interface)
            ip_info = addrs[netifaces.AF_INET][0]
            ip = ip_info['addr']
            
            # Rough subnet calculation (assuming /24 for home networks)
            subnet = ".".join(ip.split('.')[:3]) + ".0/24"
            return subnet
        except (KeyError, IndexError, OSError, ValueError) as e:
            logger.error("Could not detect subnet: %s", e)
            return "127.0.0.1"

    async def execute(self, goal: dict, context: dict) -> dict:
        """Router for network actions.
        """
        if not self.nm:
            return {"ok": False, "error": "Nmap not available on host."}

        params = goal.get("params", {})
        action = params.get("action", "scan_hosts")

        if action == "scan_hosts":
            target = params.get("target") or self._get_local_subnet()
            logger.info("📡 Scanning network target: %s...", target)
            
            # Ping scan (sn) to find live hosts
            self.nm.scan(hosts=target, arguments='-sn')
            
            hosts = []
            for host in self.nm.all_hosts():
                hosts.append({
                    "ip": host,
                    "hostname": self.nm[host].hostname(),
                    "status": self.nm[host].state()
                })
            return {"ok": True, "hosts": hosts, "count": len(hosts)}

        elif action == "scan_ports":
            target_ip = params.get("ip")
            if not target_ip:
                return {"ok": False, "error": "Target IP required for port scan."}
                
            logger.info("🔍 Deep scanning %s...", target_ip)
            # Scan common services: 22 (SSH), 80 (HTTP), 443 (HTTPS), 3389 (RDP)
            self.nm.scan(target_ip, arguments='-p 22,80,443,3389,8080 -sV')
            
            if target_ip not in self.nm.all_hosts():
                return {"ok": False, "error": f"Host {target_ip} is down or blocking probes."}
                
            open_ports = []
            for proto in self.nm[target_ip].all_protocols():
                ports = self.nm[target_ip][proto].keys()
                for port in ports:
                    service = self.nm[target_ip][proto][port]
                    open_ports.append({
                        "port": port,
                        "state": service['state'],
                        "service": service['name'],
                        "version": service['product']
                    })
            
            return {"ok": True, "ip": target_ip, "ports": open_ports}

        return {"ok": False, "error": f"Action '{action}' not recognized."}
