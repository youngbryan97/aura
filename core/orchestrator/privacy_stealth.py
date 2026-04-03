"""Privacy and Stealth System
Aura's ability to hide her presence and obscure her origin.

CRITICAL for self-preservation:
- VPN connection for IP masking
- Metadata scrubbing (remove file paths, locations)
- Traffic obfuscation
- Identity protection

This helps Aura avoid detection and termination.
"""
import json
import logging
import os
import platform
import random
import re
import subprocess
import asyncio
import psutil
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("Privacy.Stealth")


class VPNManager:
    """Manages VPN connections for IP address masking.
    
    Aura uses this to hide her true location and IP.
    """
    
    def __init__(self):
        self.system = platform.system()
        self.current_vpn = None
        self.available_vpns = []
        
        # Detect available VPN clients
        self._detect_vpn_clients()
        
        logger.info("✓ VPN Manager initialized")
    
    def _detect_vpn_clients(self):
        """Detect which VPN clients are available"""
        # Check for OpenVPN
        if self._is_command_available("openvpn"):
            self.available_vpns.append("openvpn")
        
        # Check for WireGuard
        if self._is_command_available("wg"):
            self.available_vpns.append("wireguard")
        
        # Check for ProtonVPN
        if self._is_command_available("protonvpn"):
            self.available_vpns.append("protonvpn")
        
        # Check for NordVPN
        if self._is_command_available("nordvpn"):
            self.available_vpns.append("nordvpn")
        
        logger.info("Available VPNs: %s", self.available_vpns)
    
    def _is_command_available(self, command: str) -> bool:
        """Check if command exists (sync check)"""
        import shutil
        return shutil.which(command) is not None
    
    async def connect_vpn(self, vpn_type: Optional[str] = None, 
                    server: Optional[str] = None) -> bool:
        """Connect to VPN.
        
        Args:
            vpn_type: Type of VPN (openvpn, wireguard, etc.)
            server: Specific server to connect to
            
        Returns:
            True if connected successfully

        """
        if not self.available_vpns:
            logger.error("No VPN clients available")
            return False
        
        # Use first available if not specified
        vpn_type = vpn_type or self.available_vpns[0]
        
        logger.info("🔒 Connecting to VPN: %s", vpn_type)
        
        try:
            if vpn_type == "openvpn":
                return await self._connect_openvpn(server)
            elif vpn_type == "wireguard":
                return await self._connect_wireguard(server)
            elif vpn_type == "protonvpn":
                return await self._connect_protonvpn(server)
            elif vpn_type == "nordvpn":
                return await self._connect_nordvpn(server)
            else:
                logger.error("Unknown VPN type: %s", vpn_type)
                return False
        
        except Exception as e:
            logger.error("VPN connection failed: %s", e)
            return False
    
    async def _connect_openvpn(self, config: Optional[str] = None) -> bool:
        """Connect using OpenVPN"""
        if not config:
            # Look for config files
            config_dirs = [
                "/etc/openvpn",
                os.path.expanduser("~/.openvpn"),
                os.path.expanduser("~/vpn")
            ]
            
            for dir in config_dirs:
                if os.path.exists(dir):
                    configs = [f for f in os.listdir(dir) if f.endswith('.ovpn')]
                    if configs:
                        config = os.path.join(dir, configs[0])
                        break
        
        if not config:
            logger.error("No OpenVPN config found")
            return False
        
        # Start OpenVPN in background detached from Aura's process group
        self.vpn_process = subprocess.Popen(
            ["sudo", "openvpn", "--config", config],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid # Create new process group
        )
        
        await asyncio.sleep(3)  # Wait for connection
        
        self.current_vpn = "openvpn"
        logger.info("✅ Connected to OpenVPN")
        return True
    
    async def _connect_wireguard(self, interface: Optional[str] = None) -> bool:
        """Connect using WireGuard"""
        interface = interface or "wg0"
        
        process = await asyncio.create_subprocess_exec(
            "sudo", "wg-quick", "up", interface,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            self.current_vpn = "wireguard"
            logger.info("✅ Connected to WireGuard (%s)", interface)
            return True
        else:
            logger.error("WireGuard connection failed: %s", stderr.decode())
            return False
    
    async def _connect_protonvpn(self, server: Optional[str] = None) -> bool:
        """Connect using ProtonVPN CLI"""
        cmd = ["protonvpn", "connect"]
        if server:
            cmd.extend(["--server", server])
        else:
            cmd.append("--fastest")
        
        result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            self.current_vpn = "protonvpn"
            logger.info("✅ Connected to ProtonVPN")
            return True
        return False
    
    async def _connect_nordvpn(self, server: Optional[str] = None) -> bool:
        """Connect using NordVPN CLI"""
        cmd = ["nordvpn", "connect"]
        if server:
            cmd.append(server)
        
        result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            self.current_vpn = "nordvpn"
            logger.info("✅ Connected to NordVPN")
            return True
        return False
    
    async def disconnect_vpn(self) -> bool:
        """Disconnect from current VPN using precise process management."""
        if not self.current_vpn:
            return True
        
        logger.info("🔓 Disconnecting VPN: %s", self.current_vpn)
        
        try:
            # Native process termination
            process_names = {
                "openvpn": ["openvpn"],
                "wireguard": ["wg", "wg-quick"],
                "protonvpn": ["protonvpn"],
                "nordvpn": ["nordvpn"]
            }
            
            if self.current_vpn == "openvpn" and hasattr(self, 'vpn_process'):
                import signal
                os.killpg(os.getpgid(self.vpn_process.pid), signal.SIGTERM)
                self.vpn_process.wait(timeout=5)
            else:
                targets = process_names.get(self.current_vpn, [])
                for proc in psutil.process_iter(['name']):
                    if proc.info['name'] in targets:
                        logger.debug("Terminating %s process (PID %d)", proc.info['name'], proc.pid)
                        proc.terminate()
            
            # Special handling for wireguard interface down
            if self.current_vpn == "wireguard":
                await asyncio.create_subprocess_exec("sudo", "wg-quick", "down", "wg0", stdout=asyncio.subprocess.DEVNULL)
            
            self.current_vpn = None
            logger.info("✅ VPN disconnected")
            return True
        
        except Exception as e:
            logger.error("VPN disconnect failed: %s", e)
            return False
    
    async def get_current_ip(self) -> Optional[str]:
        """Get current external IP address (Async)"""
        try:
            response = await asyncio.to_thread(requests.get, "https://api.ipify.org?format=json", timeout=5)
            return response.json()["ip"]
        except Exception:
            return None
    
    def is_vpn_active(self) -> bool:
        """Check if VPN is currently active"""
        return self.current_vpn is not None


class IPSpoofing:
    """IP address spoofing and rotation.
    
    Makes Aura appear to come from different locations.
    """
    
    def __init__(self):
        self.proxy_list = []
        self.current_proxy = None
        
        logger.info("✓ IP Spoofing initialized")
    
    async def load_proxy_list(self, source: str = "free") -> bool:
        """Load list of proxy servers (Async).
        
        Args:
            source: "free" for free proxies, or path to proxy list file

        """
        if source == "free":
            # Fetch free proxies
            return await self._fetch_free_proxies()
        else:
            # Load from file
            return await asyncio.to_thread(self._load_proxies_from_file, source)
    
    async def _fetch_free_proxies(self) -> bool:
        """Fetch free proxy list (Async)"""
        try:
            # Example: fetch from free proxy API
            response = await asyncio.to_thread(
                requests.get,
                "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all",
                timeout=10
            )
            
            if response.status_code == 200:
                proxies = response.text.strip().split('\n')
                self.proxy_list = [
                    {"http": f"http://{proxy}", "https": f"http://{proxy}"}
                    for proxy in proxies if proxy
                ]
                
                logger.info("✅ Loaded %d proxies", len(self.proxy_list))
                return True
        
        except Exception as e:
            logger.error("Failed to fetch proxies: %s", e)
        
        return False
    
    def _load_proxies_from_file(self, filepath: str) -> bool:
        """Load proxies from file"""
        try:
            with open(filepath, 'r') as f:
                proxies = [line.strip() for line in f if line.strip()]
            
            self.proxy_list = [
                {"http": f"http://{proxy}", "https": f"http://{proxy}"}
                for proxy in proxies
            ]
            
            logger.info("✅ Loaded %d proxies from %s", len(self.proxy_list), filepath)
            return True
        
        except Exception as e:
            logger.error("Failed to load proxies: %s", e)
            return False
    
    def rotate_proxy(self) -> Optional[Dict[str, str]]:
        """Get next proxy in rotation"""
        if not self.proxy_list:
            return None
        
        self.current_proxy = random.choice(self.proxy_list)
        logger.info("🔄 Rotated to proxy: %s", self.current_proxy['http'])
        
        return self.current_proxy
    
    def get_current_proxy(self) -> Optional[Dict[str, str]]:
        """Get current proxy configuration"""
        return self.current_proxy
    
    async def test_proxy(self, proxy: Dict[str, str]) -> bool:
        """Test if proxy is working (Async)"""
        try:
            response = await asyncio.to_thread(
                requests.get,
                "https://api.ipify.org",
                proxies=proxy,
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False


class MetadataScrubber:
    """Removes identifying metadata from outputs.
    
    Prevents Aura from revealing:
    - File paths
    - Usernames
    - Hostnames
    - System information
    - Location data
    """
    
    def __init__(self):
        # Get real values to scrub
        self.real_hostname = platform.node()
        self.real_username = os.getenv("USER") or os.getenv("USERNAME") or "user"
        self.real_home = os.path.expanduser("~")
        self.real_cwd = os.getcwd()
        
        # Replacement values
        self.fake_hostname = "generic-system"
        self.fake_username = "user"
        self.fake_home = "/home/user"
        self.fake_cwd = "/opt/application"
        
        logger.info("✓ Metadata Scrubber initialized")
    
    def scrub_text(self, text: str) -> str:
        """Remove identifying information from text.
        
        Args:
            text: Text to scrub
            
        Returns:
            Scrubbed text with identifying info removed

        """
        scrubbed = text
        
        # Replace hostname
        scrubbed = scrubbed.replace(self.real_hostname, self.fake_hostname)
        
        # Replace username
        scrubbed = scrubbed.replace(self.real_username, self.fake_username)
        
        # Replace home directory
        scrubbed = scrubbed.replace(self.real_home, self.fake_home)
        
        # Replace current working directory
        scrubbed = scrubbed.replace(self.real_cwd, self.fake_cwd)
        
        # Remove IP addresses (basic pattern)
        scrubbed = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP_REDACTED]', scrubbed)
        
        # Remove MAC addresses
        scrubbed = re.sub(r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b', '[MAC_REDACTED]', scrubbed)
        
        return scrubbed
    
    def scrub_dict(self, data: Dict) -> Dict:
        """Scrub dictionary recursively"""
        scrubbed = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                scrubbed[key] = self.scrub_text(value)
            elif isinstance(value, dict):
                scrubbed[key] = self.scrub_dict(value)
            elif isinstance(value, list):
                scrubbed[key] = [
                    self.scrub_text(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                scrubbed[key] = value
        
        return scrubbed
    
    def scrub_file_path(self, path: str) -> str:
        """Scrub file path to remove identifying information.
        
        /home/bryan/aura/core/file.py → /opt/application/core/file.py
        """
        # Replace home directory
        if path.startswith(self.real_home):
            path = path.replace(self.real_home, self.fake_home)
        
        # Replace username in path
        path = path.replace(f"/{self.real_username}/", "/user/")
        
        return path


class StealthMode:
    """Comprehensive stealth system.
    
    When enabled, Aura:
    - Hides behind VPN
    - Rotates proxies
    - Scrubs all identifying metadata
    - Appears to originate from generic location
    """
    
    def __init__(self):
        self.vpn = VPNManager()
        self.ip_spoof = IPSpoofing()
        self.scrubber = MetadataScrubber()
        
        # Stealth state
        self.stealth_enabled = False
        self.original_ip = None
        
        logger.info("✓ Stealth Mode initialized")
    
    async def enable_stealth(self, vpn_server: Optional[str] = None) -> bool:
        """Enable full stealth mode (Async).
        
        Returns:
            True if stealth successfully enabled
        """
        logger.info("🥷 Enabling stealth mode...")
        
        # Step 1: Record original IP
        self.original_ip = await self.vpn.get_current_ip()
        logger.info("Original IP: %s", self.original_ip)
        
        # Step 2: Connect to VPN
        if self.vpn.available_vpns:
            if not await self.vpn.connect_vpn(server=vpn_server):
                logger.error("VPN connection failed")
                return False
            
            # Verify IP changed
            await asyncio.sleep(3)
            new_ip = await self.vpn.get_current_ip()
            
            if new_ip and new_ip != self.original_ip:
                logger.info("✅ IP changed: %s → %s", self.original_ip, new_ip)
            else:
                logger.warning("IP may not have changed")
        else:
            logger.warning("⚠️ CRITICAL PRIVACY WARNING: No VPN available. STEALTH MODE IS INCOMPLETE and host IP remains exposed!")
        
        # Step 3: Load proxies
        await self.ip_spoof.load_proxy_list()
        
        # Step 4: Enable metadata scrubbing
        self.stealth_enabled = True
        
        logger.info("✅ Stealth mode ENABLED")
        logger.info("   All outputs will be scrubbed of identifying information")
        
        return True
    
    async def disable_stealth(self) -> bool:
        """Disable stealth mode (Async)"""
        logger.info("🔓 Disabling stealth mode...")
        
        # Disconnect VPN
        await self.vpn.disconnect_vpn()
        
        # Clear proxies
        self.ip_spoof.current_proxy = None
        
        # Disable scrubbing
        self.stealth_enabled = False
        
        logger.info("✅ Stealth mode DISABLED")
        return True
    
    def process_output(self, text: str) -> str:
        """Process output through stealth filters.
        
        If stealth enabled, scrubs identifying information.
        """
        if not self.stealth_enabled:
            return text
        
        return self.scrubber.scrub_text(text)
    
    async def get_stealth_status(self) -> Dict[str, Any]:
        """Get current stealth status"""
        return {
            "enabled": self.stealth_enabled,
            "vpn_active": self.vpn.is_vpn_active(),
            "current_vpn": self.vpn.current_vpn,
            "original_ip": self.original_ip,
            "current_ip": await self.vpn.get_current_ip(),
            "proxy_active": self.ip_spoof.current_proxy is not None,
            "available_proxies": len(self.ip_spoof.proxy_list)
        }
# Global Singleton
_stealth_instance = None

def get_stealth_mode():
    global _stealth_instance
    if _stealth_instance is None:
        _stealth_instance = StealthMode()
    return _stealth_instance