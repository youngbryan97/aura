"""Diagnostics Agent
Responsible for deep system health checks, skill validation, and connectivity verification.
Run independently of the main loop to ensure resilience.
"""
from core.runtime.errors import record_degradation
import asyncio
import importlib
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import aiohttp

logger = logging.getLogger("Resilience.Diagnostics")

class DiagnosticsAgent:
    """Sub-agent for system self-diagnosis.
    """
    
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.base_path = Path(__file__).parent.parent.parent
        self.skills_path = self.base_path / "skills"
        
    async def run_full_diagnosis(self) -> Dict[str, Any]:
        """Run all diagnostic checks."""
        logger.info("Starting deep system diagnosis...")
        
        results = {
            "timestamp": asyncio.get_running_loop().time(),
            "skills": await self.check_skills(),
            "connectivity": await self.check_connectivity(),
            "integrity": self.check_integrity(),
            "system_status": "healthy"
        }
        
        # Aggregate status
        if not results["skills"]["all_valid"] or not results["connectivity"]["server_online"] or not results["integrity"]["critical_files_present"]:
            results["system_status"] = "degraded"
            
        logger.info("Diagnosis complete. Status: %s", results['system_status'])
        return results

    async def check_skills(self) -> Dict[str, Any]:
        """Verify all skills can be loaded and have valid signatures."""
        results = {
            "valid": [],
            "invalid": [],
            "all_valid": True
        }
        
        if not self.skills_path.exists():
            return {"error": "Skills directory missing", "all_valid": False}

        sys.path.append(str(self.base_path))
        
        for file in self.skills_path.glob("*.py"):
            if file.name == "__init__.py":
                continue
                
            skill_name = file.stem
            try:
                # Try to import
                module_spec = importlib.util.spec_from_file_location(skill_name, file)
                module = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(module)
                
                # Check for class with 'execute'
                valid_skill = False
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and hasattr(attr, 'execute'):
                        # Basic check passed
                        valid_skill = True
                        break
                
                if valid_skill:
                    results["valid"].append(skill_name)
                else:
                    results["invalid"].append({"name": skill_name, "reason": "No class with execute() found"})
                    
            except Exception as e:
                record_degradation('diagnostics_agent', e)
                results["invalid"].append({"name": skill_name, "reason": str(e)})
        
        if results["invalid"]:
            results["all_valid"] = False
            
        return results

    async def check_connectivity(self) -> Dict[str, Any]:
        """Check server and external connectivity."""
        results = {
            "server_online": False,
            "latency_ms": 0
        }
        
        try:
            start = asyncio.get_running_loop().time()
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8000/health", timeout=5) as resp:
                    if resp.status == 200:
                        results["server_online"] = True
                        results["latency_ms"] = (asyncio.get_running_loop().time() - start) * 1000
        except Exception:
            results["server_online"] = False
            
        return results

    def check_integrity(self) -> Dict[str, Any]:
        """Verify critical files exist."""
        critical_files = [
            "run_aura.py",
            "core/orchestrator.py",
            "core/config.py",
            "interface/static/index.html"
        ]
        
        results = {
            "missing": [],
            "critical_files_present": True
        }
        
        for f in critical_files:
            if not (self.base_path / f).exists():
                results["missing"].append(f)
        
        if results["missing"]:
            results["critical_files_present"] = False
            
        return results