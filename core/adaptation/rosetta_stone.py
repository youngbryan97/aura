import logging
import platform
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Core.Adaptation.RosettaStone")

class RosettaStone:
    """Adaptive Code Engine.
    1. Transpiles commands/code for the host OS.
    2. Analyzes code for potential threats (Digital Immune System extension).
    """
    
    def __init__(self):
        self.os_type = platform.system().lower() # darwin, linux, windows
        self.arch = platform.machine().lower()
        logger.info("Rosetta Stone initialized for %s (%s)", self.os_type, self.arch)
        
    def adapt_command(self, command: str, target_os: str = None) -> str:
        """Adapt a shell command to the target OS using a structured map."""
        target = target_os or self.os_type
        
        # Audit Fix: Use a structured map and split the command to avoid
        # accidental partial string replacement.
        COMMAND_MAP = {
            "windows": {
                "ls": "dir",
                "rm -rf": "rmdir /s /q",
                "cp": "copy",
                "mv": "move",
                "grep": "findstr"
            }
        }
        
        if target in COMMAND_MAP:
            mapping = COMMAND_MAP[target]
            # Try matching full command or starting tokens
            for cmd_src, cmd_target in mapping.items():
                if command.startswith(cmd_src + " ") or command == cmd_src:
                    return command.replace(cmd_src, cmd_target, 1)
                
        return command

    def analyze_threat(self, code: str) -> Dict[str, Any]:
        """Analyze code for potential malicious patterns.
        Returns: {safe: bool, threats: List[str], countermeasures: List[str]}
        """
        threats = []
        counters = []
        
        # Audit Fix: Use word boundaries and more robust regex to catch obfuscation.
        # 1. Destructive Patterns
        if re.search(rf"\brm\s+(-rf|-r\s+-f|-f\s+-r)\s+/", code):
            threats.append("Root Deletion Attempt")
            counters.append("Sandbox Isolation")
            
        if re.search(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;", code):
            threats.append("Fork Bomb")
            counters.append("Process Limiting")
            
        # Check for harmful python calls with varied spacing
        if re.search(rf"os\.(system|popen|spawn|execuv)\s*\(\s*['\"]rm\s+-rf", code) or 'shutil.rmtree' in code:
             threats.append("Python File Deletion")
             
        # 2. Exfiltration & Networking
        if re.search(rf"\b(socket|urllib|requests|aiohttp)\b", code) and \
           re.search(rf"\b(connect|get|post|request)\b", code):
            threats.append("Networking / Exfiltration Attempt")
            counters.append("Network Block")
            
        # 3. Persistence & System Modification
        if re.search(rf"\b(crontab|AutoRun|bashrc|launchctl|systemctl)\b", code):
            threats.append("Persistence Mechanism")
            
        # 4. Indirect execution / Obfuscation
        if re.search(rf"(__import__|eval|exec|getattr)\b", code):
            threats.append("Dynamic Execution / Obfuscation")
            
        is_safe = len(threats) == 0
        
        return {
            "safe": is_safe,
            "threats": threats,
            "countermeasures": counters
        }

# Global Singleton
rosetta_stone = RosettaStone()