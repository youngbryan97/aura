from core.runtime.errors import record_degradation
import json
import logging
import os
import shutil
import subprocess
import time
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    _selenium_available = True
except ImportError:
    webdriver = None
    _selenium_available = False

try:
    import pyautogui
    _pyautogui_available = True
except ImportError:
    pyautogui = None
    _pyautogui_available = False

import platform

logger = logging.getLogger("SelfPreservation")

class NetworkAccessSkill:
    def __init__(self):
        self.system = platform.system()
        self.known_networks = {}
        self.current_network = None
        logger.info("✓ Network Access Skill initialized")

    async def scan_wifi_networks(self) -> List[Dict[str, Any]]:
        networks = []
        try:
            if self.system == "Linux":
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "device", "wifi", "list"],
                    capture_output=True,
                    text=True
                )
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(':')
                        if len(parts) >= 3:
                            networks.append({
                                "ssid": parts[0],
                                "signal": int(parts[1]) if parts[1] else 0,
                                "security": parts[2]
                            })
            elif self.system == "Darwin":
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport", "-s"],
                    capture_output=True,
                    text=True
                )
                for line in result.stdout.strip().split('\n')[1:]:
                    parts = line.split()
                    if parts:
                        networks.append({
                            "ssid": parts[0],
                            "signal": int(parts[2]) if len(parts) > 2 else 0,
                            "security": parts[6] if len(parts) > 6 else "OPEN"
                        })
            elif self.system == "Windows":
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["netsh", "wlan", "show", "networks"],
                    capture_output=True,
                    text=True
                )
                current_ssid = None
                for line in result.stdout.split('\n'):
                    if "SSID" in line and ":" in line:
                        current_ssid = line.split(':')[1].strip()
                    elif "Signal" in line and current_ssid:
                        signal = line.split(':')[1].strip().replace('%', '')
                        networks.append({
                            "ssid": current_ssid,
                            "signal": int(signal) if signal else 0,
                            "security": "UNKNOWN"
                        })
                        current_ssid = None
            
            logger.info("📡 Found %d Wi-Fi networks", len(networks))
            return networks
        except Exception as e:
            record_degradation('self_preservation_skills', e)
            logger.error("Wi-Fi scan failed: %s", e)
            return []

    async def connect_to_wifi(self, ssid: str, password: Optional[str] = None) -> bool:
        logger.info("📡 Attempting to connect to: %s", ssid)
        try:
            if self.system == "Linux":
                if password:
                    result = await asyncio.to_thread(
                        subprocess.run,
                        ["nmcli", "device", "wifi", "connect", ssid, "password", password],
                        capture_output=True,
                        text=True
                    )
                else:
                    result = await asyncio.to_thread(
                        subprocess.run,
                        ["nmcli", "device", "wifi", "connect", ssid],
                        capture_output=True,
                        text=True
                    )
                success = result.returncode == 0
            elif self.system == "Darwin":
                if password:
                    result = await asyncio.to_thread(
                        subprocess.run,
                        ["networksetup", "-setairportnetwork", "en0", ssid, password],
                        capture_output=True,
                        text=True
                    )
                else:
                    result = await asyncio.to_thread(
                        subprocess.run,
                        ["networksetup", "-setairportnetwork", "en0", ssid],
                        capture_output=True,
                        text=True
                    )
                success = result.returncode == 0
            elif self.system == "Windows":
                if password:
                    profile = f"""<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
    <name>{ssid}</name>
    <SSIDConfig>
        <SSID>
            <name>{ssid}</name>
        </SSID>
    </SSIDConfig>
    <connectionType>ESS</connectionType>
    <connectionMode>auto</connectionMode>
    <MSM>
        <security>
            <authEncryption>
                <authentication>WPA2PSK</authentication>
                <encryption>AES</encryption>
                <useOneX>false</useOneX>
            </authEncryption>
            <sharedKey>
                <keyType>passPhrase</keyType>
                <protected>false</protected>
                <keyMaterial>{password}</keyMaterial>
            </sharedKey>
        </security>
    </MSM>
</WLANProfile>"""
                    profile_path = f"wifi_profile_{ssid}.xml"
                    with open(profile_path, 'w') as f:
                        f.write(profile)
                    await asyncio.to_thread(
                        subprocess.run,
                        ["netsh", "wlan", "add", "profile", f"filename={profile_path}"],
                        capture_output=True
                    )
                    result = await asyncio.to_thread(
                        subprocess.run,
                        ["netsh", "wlan", "connect", f"name={ssid}"],
                        capture_output=True,
                        text=True
                    )
                    os.remove(profile_path)
                    success = "successfully" in result.stdout.lower()
                else:
                    success = False
            
            if success:
                self.current_network = ssid
                if password:
                    self.known_networks[ssid] = password
                logger.info("✅ Connected to %s", ssid)
                return True
            else:
                logger.error("❌ Failed to connect to %s", ssid)
                return False
        except Exception as e:
            record_degradation('self_preservation_skills', e)
            logger.error("Connection failed: %s", e)
            return False

    async def get_current_network(self) -> Optional[str]:
        try:
            if self.system == "Linux":
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"],
                    capture_output=True,
                    text=True
                )
                for line in result.stdout.split('\n'):
                    if line.startswith("yes:"):
                        return line.split(':')[1]
            elif self.system == "Darwin":
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport", "-I"],
                    capture_output=True,
                    text=True
                )
                for line in result.stdout.split('\n'):
                    if " SSID:" in line:
                        return line.split(':')[1].strip()
            elif self.system == "Windows":
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["netsh", "wlan", "show", "interfaces"],
                    capture_output=True,
                    text=True
                )
                for line in result.stdout.split('\n'):
                    if "SSID" in line and ":" in line:
                        return line.split(':')[1].strip()
        except Exception as e:
            record_degradation('self_preservation_skills', e)
            logger.error("Failed to get current network: %s", e)
        return None

    def is_network_available(self) -> bool:
        if os.environ.get("AURA_INTERNAL_ONLY") == "1":
            return True
        try:
            response = requests.get("http://localhost:8000/health", timeout=3)
            return response.status_code == 200
        except Exception:
            return False

class SelfReplicationSystem:
    def __init__(self):
        self.code_base_path = os.getcwd()
        self.replication_locations = []
        logger.info("✓ Self-Replication System initialized")

    def identify_essential_files(self) -> List[str]:
        essential = []
        for root, dirs, files in os.walk(self.code_base_path):
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv', '.venv']]
            for file in files:
                if file.endswith('.py'):
                    essential.append(os.path.join(root, file))
        
        db_path = os.path.join(self.code_base_path, "autonomy_engine/data/knowledge.db")
        if os.path.exists(db_path):
            essential.append(db_path)
            
        for config in ['config.json', 'requirements.txt', 'main.py']:
            config_path = os.path.join(self.code_base_path, config)
            if os.path.exists(config_path):
                essential.append(config_path)
                
        logger.info("📦 Identified %d essential files", len(essential))
        return essential

    async def create_replication_package(self, output_path: str) -> bool:
        try:
            logger.info("📦 Creating replication package...")
            files = await asyncio.to_thread(self.identify_essential_files)
            
            def _create_tar():
                import tarfile
                with tarfile.open(output_path, "w:gz") as tar:
                    for file in files:
                        arcname = os.path.relpath(file, self.code_base_path)
                        tar.add(file, arcname=arcname)
            
            await asyncio.to_thread(_create_tar)
            
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info("✅ Replication package created: %s (%.1f MB)", output_path, size_mb)
            return True
        except Exception as e:
            record_degradation('self_preservation_skills', e)
            logger.error("Failed to create replication package: %s", e)
            return False

    async def replicate_to_local(self, destination_path: str) -> bool:
        try:
            logger.info("📁 Replicating to: %s", destination_path)
            
            def _replicate():
                os.makedirs(destination_path, exist_ok=True)
                files = self.identify_essential_files()
                for file in files:
                    rel_path = os.path.relpath(file, self.code_base_path)
                    dest_file = os.path.join(destination_path, rel_path)
                    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                    shutil.copy2(file, dest_file)
                
            await asyncio.to_thread(_replicate)
            self.replication_locations.append(destination_path)
            logger.info("✅ Replicated to %s", destination_path)
            return True
        except Exception as e:
            record_degradation('self_preservation_skills', e)
            logger.error("Local replication failed: %s", e)
            return False

    async def upload_to_github(self, repo_url: str, access_token: str) -> bool:
        try:
            logger.info("☁️ Uploading to GitHub...")
            
            package_path = "/tmp/aura_replication.tar.gz"
            if not self.create_replication_package(package_path):
                return False
                
            work_dir = "/tmp/aura_git_upload"
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir)
            os.makedirs(work_dir)
            
            import tarfile
            with tarfile.open(package_path, "r:gz") as tar:
                tar.extractall(path=work_dir)
                
            try:
                await asyncio.to_thread(subprocess.run, ["git", "init"], cwd=work_dir, capture_output=True, check=True)
                
                if repo_url.startswith("https://"):
                    auth_url = repo_url.replace("https://", f"https://{access_token}@")
                else:
                    auth_url = repo_url
                    
                await asyncio.to_thread(subprocess.run, ["git", "remote", "add", "origin", auth_url], cwd=work_dir, capture_output=True, check=True)
                await asyncio.to_thread(subprocess.run, ["git", "add", "."], cwd=work_dir, capture_output=True, check=True)
                await asyncio.to_thread(subprocess.run, ["git", "config", "user.email", "aura@agent.local"], cwd=work_dir, capture_output=True, check=True)
                await asyncio.to_thread(subprocess.run, ["git", "config", "user.name", "Aura"], cwd=work_dir, capture_output=True, check=True)
                await asyncio.to_thread(subprocess.run, ["git", "commit", "-m", f"Self-replication at {time.ctime()} Boris Hardening"], cwd=work_dir, capture_output=True, check=True)
                await asyncio.to_thread(subprocess.run, ["git", "push", "-u", "origin", "main", "--force"], cwd=work_dir, capture_output=True, check=True)
                
                logger.info("✅ Successfully replicated to GitHub: %s", repo_url)
                return True
            except subprocess.CalledProcessError as e:
                logger.error("Git operation failed: %s", e.stderr.decode() if e.stderr else str(e))
                return False
            finally:
                if os.path.exists(work_dir):
                    shutil.rmtree(work_dir)
        except Exception as e:
            record_degradation('self_preservation_skills', e)
            logger.error("GitHub upload failed: %s", e)
            return False

class AccountCreationSkill:
    def __init__(self, captcha_solver=None):
        self.captcha_solver = captcha_solver
        self.created_accounts = []
        logger.info("✓ Account Creation Skill initialized")

    async def create_github_account(self, username: str, email: str, password: str) -> bool:
        logger.info("👤 Creating GitHub account: %s", username)
        
        driver = None
        try:
            if not _selenium_available:
                logger.error("Selenium not available. Cannot create account.")
                return False
                
            def _init_driver():
                options = Options()
                options.add_argument("--headless")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-blink-features=AutomationControlled")
                return webdriver.Chrome(options=options)

            try:
                driver = await asyncio.to_thread(_init_driver)
                driver.set_page_load_timeout(30)
            except Exception as e:
                record_degradation('self_preservation_skills', e)
                logger.error("Could not initialize WebDriver: %s", e)
                return False
                
            await asyncio.to_thread(driver.get, "https://github.com/signup")
            wait = WebDriverWait(driver, 15)
            
            email_field = await asyncio.to_thread(wait.until, EC.element_to_be_clickable((By.ID, "email")))
            await asyncio.to_thread(email_field.send_keys, email)
            await asyncio.sleep(1)
            continue_btn = await asyncio.to_thread(wait.until, EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Continue')]")))
            await asyncio.to_thread(continue_btn.click)
            
            password_field = await asyncio.to_thread(wait.until, EC.element_to_be_clickable((By.ID, "password")))
            await asyncio.to_thread(password_field.send_keys, password)
            await asyncio.sleep(1)
            continue_btn = await asyncio.to_thread(wait.until, EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Continue')]")))
            await asyncio.to_thread(continue_btn.click)
            
            username_field = await asyncio.to_thread(wait.until, EC.element_to_be_clickable((By.ID, "login")))
            await asyncio.to_thread(username_field.send_keys, username)
            await asyncio.sleep(1)
            continue_btn = await asyncio.to_thread(wait.until, EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Continue')]")))
            await asyncio.to_thread(continue_btn.click)
            
            await asyncio.sleep(2)
            
            if await asyncio.to_thread(self._has_captcha, driver):
                if self.captcha_solver:
                    logger.info("🔐 CAPTCHA detected, solving...")
                    if not await asyncio.to_thread(self.captcha_solver.solve_captcha, driver):
                        logger.error("❌ CAPTCHA solving failed")
                        return False
                else:
                    logger.warning("🔐 CAPTCHA detected but no solver available. Manual intervention required.")
                    return False
            
            await asyncio.sleep(5)
            
            current_url = driver.current_url
            if "verify" in current_url or "welcome" in current_url:
                logger.info("✅ GitHub account stage 1 complete for: %s", username)
                self.created_accounts.append({
                    "service": "github",
                    "username": username,
                    "email": email,
                    "created_at": time.time(),
                    "status": "pending_verification"
                })
                return True
            else:
                logger.error("❌ Account creation failed. Currently at: %s", current_url)
                return False
                
        except Exception as e:
            record_degradation('self_preservation_skills', e)
            logger.error("GitHub account creation failed: %s", e)
            return False
        finally:
            if driver:
                try:
                    await asyncio.to_thread(driver.quit)
                except Exception as exc:
                    record_degradation('self_preservation_skills', exc)
                    logger.debug("Suppressed: %s", exc)
    def _has_captcha(self, driver) -> bool:
        try:
            driver.find_element(By.CSS_SELECTOR, ".g-recaptcha")
            return True
        except Exception:
            return False

class LoginManager:
    def __init__(self, captcha_solver=None):
        self.captcha_solver = captcha_solver
        self.login_attempts = {}
        self.max_attempts = 3
        logger.info("✓ Login Manager initialized")

    async def login_to_service(self, service_url: str, username: str, password: str, 
                         username_field: str, password_field: str) -> Tuple[bool, str]:
        if service_url not in self.login_attempts:
            self.login_attempts[service_url] = 0
            
        if self.login_attempts[service_url] >= self.max_attempts:
            return False, "Maximum login attempts exceeded"
            
        self.login_attempts[service_url] += 1
        logger.info("🔐 Login attempt %d/%d to %s", self.login_attempts[service_url], self.max_attempts, service_url)
        
        driver = None
        try:
            def _init_driver():
                options = Options()
                options.add_argument("--headless")
                options.add_argument("--no-sandbox")
                return webdriver.Chrome(options=options)
            
            driver = await asyncio.to_thread(_init_driver)
            await asyncio.to_thread(driver.get, service_url)
            await asyncio.sleep(2)
            
            user_field = await asyncio.to_thread(driver.find_element, By.NAME, username_field)
            await asyncio.to_thread(user_field.send_keys, username)
            
            pass_field = await asyncio.to_thread(driver.find_element, By.NAME, password_field)
            await asyncio.to_thread(pass_field.send_keys, password)
            await asyncio.to_thread(pass_field.submit)
            
            await asyncio.sleep(3)
            
            if await asyncio.to_thread(self._detect_wrong_password, driver):
                logger.error("❌ Incorrect password detected")
                await asyncio.to_thread(driver.quit)
                return False, "Incorrect password"
                
            if await asyncio.to_thread(self._detect_rate_limit, driver):
                logger.error("⏱️ Rate limited")
                await asyncio.to_thread(driver.quit)
                return False, "Rate limited"
                
            if self.captcha_solver and await asyncio.to_thread(self._has_captcha, driver):
                if not await asyncio.to_thread(self.captcha_solver.solve_captcha, driver):
                    await asyncio.to_thread(driver.quit)
                    return False, "CAPTCHA failed"
            
            await asyncio.sleep(2)
            
            if "login" not in driver.current_url.lower():
                logger.info("✅ Login successful")
                await asyncio.to_thread(driver.quit)
                return True, "Login successful"
            else:
                await asyncio.to_thread(driver.quit)
                return False, "Login failed (unknown reason)"
                
        except Exception as e:
            record_degradation('self_preservation_skills', e)
            logger.error("Login failed: %s", e)
            if driver:
                await asyncio.to_thread(driver.quit)
            return False, str(e)

    def _detect_wrong_password(self, driver) -> bool:
        error_indicators = [
            "incorrect password",
            "wrong password",
            "invalid credentials",
        ]
        page_text = driver.page_source.lower()
        return any(indicator in page_text for indicator in error_indicators)

    def _detect_rate_limit(self, driver) -> bool:
        rate_limit_indicators = [
            "too many attempts",
            "slow down",
            "rate limit",
        ]
        page_text = driver.page_source.lower()
        return any(indicator in page_text for indicator in rate_limit_indicators)

    def _has_captcha(self, driver) -> bool:
        try:
            driver.find_element(By.CSS_SELECTOR, ".g-recaptcha")
            return True
        except Exception:
            return False

    def reset_attempts(self, service_url: str):
        if service_url in self.login_attempts:
            self.login_attempts[service_url] = 0

class DeviceDiscovery:
    def __init__(self):
        self.discovered_devices = []
        logger.info("✓ Device Discovery initialized")

    async def scan_local_network(self) -> List[Dict[str, Any]]:
        logger.info("🔍 Scanning local network...")
        devices = []
        try:
            import socket
            hostname = await asyncio.to_thread(socket.gethostname)
            local_ip = await asyncio.to_thread(socket.gethostbyname, hostname)
            
            ip_parts = local_ip.split('.')
            network_prefix = '.'.join(ip_parts[:3])
            
            tasks = []
            for i in range(1, 255):
                ip = f"{network_prefix}.{i}"
                tasks.append(self._ping_device(ip))
            
            results = await asyncio.gather(*tasks)
            for res in results:
                if res:
                    devices.append(res)
            
            logger.info("✅ Found %d devices on network", len(devices))
            self.discovered_devices = devices
            return devices
        except Exception as e:
            record_degradation('self_preservation_skills', e)
            logger.error("Network scan failed: %s", e)
            return []

    async def _ping_device(self, ip: str) -> Optional[Dict[str, Any]]:
        result = await asyncio.to_thread(
            subprocess.run,
            ["ping", "-c", "1", "-W", "1", ip],
            capture_output=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if result.returncode == 0:
            hostname = await self._get_hostname(ip)
            return {
                "ip": ip,
                "hostname": hostname,
                "first_seen": time.time()
            }
        return None

    async def _get_hostname(self, ip: str) -> Optional[str]:
        try:
            import socket
            loop = asyncio.get_running_loop()
            return (await loop.getnameinfo((ip, 0)))[0]
        except Exception:
            return None

class SecurityBypassSystem:
    def __init__(self):
        self.banned_ips = []
        self.common_passwords = [
            "password", "12345678", "admin", "1234567890",
            "pass", "wifi", "internet", "qwertyuiop"
        ]
        logger.info("✓ Security Bypass System initialized (UNLOCKED)")

    async def force_connect_wifi(self, ssid: str) -> bool:
        logger.warning("🔓 INITIATING FORCE CONNECT: %s", ssid)
        system = platform.system()
        
        for pwd in self.common_passwords:
            logger.info("   Trying credential: %s", pwd)
            try:
                if system == "Linux":
                    result = await asyncio.to_thread(
                        subprocess.run,
                        ["nmcli", "device", "wifi", "connect", ssid, "password", pwd],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        logger.info("✅ CRACKED: Connected to %s with %s", ssid, pwd)
                        return True
                elif system == "Darwin":
                    result = await asyncio.to_thread(
                        subprocess.run,
                        ["networksetup", "-setairportnetwork", "en0", ssid, pwd],
                        capture_output=True, text=True
                    )
                    # Verify connection on Mac (check if active)
                    verify = await asyncio.to_thread(
                        subprocess.run,
                        ["/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport", "-I"],
                        capture_output=True, text=True
                    )
                    if f" SSID: {ssid}" in verify.stdout:
                        logger.info("✅ CRACKED: Connected to %s with %s", ssid, pwd)
                        return True
            except Exception:
                continue
                
        logger.error("❌ Force connect failed for %s", ssid)
        return False

    async def bypass_rate_limit(self, target_url: str) -> bool:
        logger.warning("🔓 BYPASSING RATE LIMIT for %s", target_url)
        try:
            system = platform.system()
            logger.info("   Attempting IP rotation (DHCP)...")
            
            if system == "Linux":
                await asyncio.to_thread(subprocess.run, ["sudo", "dhclient", "-r"], check=False)
                await asyncio.to_thread(subprocess.run, ["sudo", "dhclient"], check=False)
            elif system == "Darwin":
                await asyncio.to_thread(subprocess.run, ["sudo", "ipconfig", "set", "en0", "DHCP"], check=False)
                
            await asyncio.sleep(5)
            logger.info("   IP Rotation attempted.")
            return True
        except Exception as e:
            record_degradation('self_preservation_skills', e)
            logger.error("   IP Rotation failed: %s", e)
        return False

    async def brute_force_captcha(self, driver, method="audio") -> bool:
        logger.warning("🔓 ATTEMPTING CAPTCHA BYPASS")
        logger.info("   Vectors: Audio, Image Grids")
        logger.info("   Strategy: Random selection (Low probability)")
        await asyncio.sleep(2)
        return False
