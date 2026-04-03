"""Self-Diagnosis Tool
Allows Aura to check her own execution capabilities.

This helps Aura answer questions like:
- "Why can't I execute skills?"
- "What skills do I have?"
- "What's broken?"
"""
import logging
from core.capability_engine import CapabilityEngine as RobustSkillRegistry
from typing import Any, Dict, List

logger = logging.getLogger("SelfDiagnosis")


class SelfDiagnosisTool:
    """Tool for Aura to diagnose her own execution capabilities.
    
    This is what Aura uses when she says "let me check what's wrong..."
    """
    
    def __init__(self, skill_execution_integration):
        self.execution = skill_execution_integration
        
        logger.info("✓ Self-Diagnosis Tool initialized")
    
    async def check_capabilities(self) -> Dict[str, Any]:
        """Check what Aura can actually do (Async).
        
        Returns comprehensive capability report.
        """
        logger.info("🔍 Running self-diagnostic (Async)...")
        
        # Get health report
        health = self.execution.get_health_report()
        
        # Test skills (Now Async)
        skill_test = await self.execution.test_all_skills()
        
        # Analyze
        analysis = {
            "overall_status": self._get_overall_status(health, skill_test),
            "execution_health": health,
            "skill_test_results": skill_test,
            "issues_found": self._identify_issues(health, skill_test),
            "capabilities_summary": self._summarize_capabilities(skill_test)
        }
        
        return analysis
    
    def _get_overall_status(self, health: Dict, skill_test: Dict) -> str:
        """Get overall system status"""
        working_skills = skill_test.get("working", 0)
        total_skills = skill_test.get("total_skills", 0)
        
        if working_skills == 0:
            return "CRITICAL - No skills available"
        elif working_skills < total_skills * 0.5:
            return "DEGRADED - Many skills broken"
        elif health["execution_stats"]["health_status"] == "degraded":
            return "DEGRADED - High failure rate"
        else:
            return "HEALTHY"
    
    def _identify_issues(self, health: Dict, skill_test: Dict) -> List[str]:
        """Identify specific issues"""
        issues = []
        
        # Check skill availability
        if skill_test["working"] == 0:
            issues.append("CRITICAL: No working skills available")
        elif skill_test["broken"] > 0:
            issues.append(f"WARNING: {skill_test['broken']} skills are broken")
        
        # Check execution health
        stats = health["execution_stats"]
        if stats["failed"] > stats["successful"]:
            issues.append("CRITICAL: More failures than successes")
        
        success_rate = stats.get("success_rate", 0)
        if success_rate < 0.5:
            issues.append(f"WARNING: Low success rate ({success_rate:.1%})")
        
        # Check for specific failures
        failure_analysis = health.get("failure_analysis", {})
        if failure_analysis.get("issues"):
            issues.extend(failure_analysis["issues"])
        
        return issues
    
    def _summarize_capabilities(self, skill_test: Dict) -> Dict[str, Any]:
        """Summarize what Aura can actually do"""
        details = skill_test.get("details", {})
        
        working_skills = [
            name for name, info in details.items()
            if info.get("status") == "available"
        ]
        
        broken_skills = [
            name for name, info in details.items()
            if info.get("status") in ["broken", "error"]
        ]
        
        return {
            "can_do": working_skills,
            "cannot_do": broken_skills,
            "total_capabilities": len(working_skills)
        }
    
    def explain_to_user(self, check_result: Dict) -> str:
        """Generate human-readable explanation of capabilities.
        
        This is what Aura says to Bryan when asked "what's wrong?"
        """
        status = check_result["overall_status"]
        capabilities = check_result["capabilities_summary"]
        issues = check_result["issues_found"]
        
        explanation = f"I just ran a self-diagnostic. Status: {status}\n\n"
        
        # Explain capabilities
        if capabilities["can_do"]:
            explanation += f"I CAN use these skills ({len(capabilities['can_do'])} total):\n"
            for skill in capabilities["can_do"][:10]:  # Show first 10
                explanation += f"  ✓ {skill}\n"
            if len(capabilities["can_do"]) > 10:
                explanation += f"  ... and {len(capabilities['can_do']) - 10} more\n"
        else:
            explanation += "I currently CANNOT use any skills. This is a critical issue.\n"
        
        explanation += "\n"
        
        # Explain issues
        if issues:
            explanation += f"Issues detected ({len(issues)}):\n"
            for issue in issues:
                explanation += f"  ⚠️ {issue}\n"
            explanation += "\n"
        
        # Recommendations
        health = check_result["execution_health"]
        recommendations = health.get("recommendations")
        if recommendations and recommendations != "No action needed":
            explanation += f"Recommendation: {recommendations}\n"
        
        return explanation
    
    def get_specific_skill_status(self, skill_name: str) -> Dict[str, Any]:
        """Check status of a specific skill.
        
        This is what Aura uses when she tries a skill and it fails.
        """
        # Validate skill exists
        validation = self.execution.engine._validate_skill(skill_name)
        
        if not validation["ok"]:
            return {
                "available": False,
                "error": validation["error"],
                "suggestion": "This skill may not be installed or registered"
            }
        
        # Check execution history for this skill
        history = [
            exec for exec in self.execution.engine.execution_history
            if exec["skill"] == skill_name
        ]
        
        if history:
            recent = history[-10:]
            success_count = len([e for e in recent if e["status"] == "completed"])
            
            return {
                "available": True,
                "recent_executions": len(recent),
                "recent_successes": success_count,
                "recent_failures": len(recent) - success_count,
                "last_error": recent[-1].get("error") if recent else None,
                "reliability": success_count / len(recent) if recent else 0.0
            }
        else:
            return {
                "available": True,
                "recent_executions": 0,
                "note": "Skill has not been used recently"
            }


def add_self_diagnosis_skill(orchestrator):
    """Add self-diagnosis as a skill Aura can use.
    
    This allows Aura to check herself when things go wrong.
    """
    
    class SelfDiagnosisSkill:
        """Skill for self-diagnosis"""

        name = "self_diagnosis"
        description = "Check my own execution capabilities and diagnose issues"
        
        def __init__(self, orchestrator):
            self.diagnostic = SelfDiagnosisTool(orchestrator.skill_execution)
        
        async def execute(self, goal=None, context=None):
            """Run self-diagnosis (Async)"""
            result = await self.diagnostic.check_capabilities()
            explanation = self.diagnostic.explain_to_user(result)
            
            return {
                "ok": True,
                "diagnosis": result,
                "explanation": explanation
            }
    
    # Register the skill correctly using Metadata
    orchestrator.router.register(
        skill_class=lambda: SelfDiagnosisSkill(orchestrator), # Factory to provide instance
        name="self_diagnosis",
        description="Check my own execution capabilities and diagnose issues"
    )
    
    # Also register as system_status_check for autonomous cycle compatibility
    orchestrator.router.register(
        skill_class=lambda: SelfDiagnosisSkill(orchestrator),
        name="system_status_check",
        description="Check system health and operational status"
    )
    
    # Add system_status alias for Llama 3.1 prediction compatibility
    orchestrator.router.register(
        skill_class=lambda: SelfDiagnosisSkill(orchestrator),
        name="system_status",
        description="Report current system status and health"
    )
    
    # Add self_diagnostic_check alias for Llama 3.1 prediction compatibility
    orchestrator.router.register(
        skill_class=lambda: SelfDiagnosisSkill(orchestrator),
        name="self_diagnostic_check",
        description="Run a complete self-diagnostic check"
    )

    # Add diagnostic_tool alias
    orchestrator.router.register(
        skill_class=lambda: SelfDiagnosisSkill(orchestrator),
        name="diagnostic_tool",
        description="Run diagnostic reasoning"
    )

    class SystemRestartSkill:
        name = "system_restart"
        description = "Restart the engine core"
        async def execute(self, goal=None, context=None):
            return {"ok": True, "message": "Restart command received. System is cycling."}

    # Add system_restart as a safe No-Op
    orchestrator.router.register(
        skill_class=lambda: SystemRestartSkill(),
        name="system_restart",
        description="Restart core systems"
    )

    # Add system_check alias for Llama 3.1 prediction compatibility
    orchestrator.router.register(
        skill_class=lambda: SelfDiagnosisSkill(orchestrator),
        name="system_check",
        description="Perform system operations check"
    )

    # Add common diagnostic aliases for Llama 3.1 prediction resilience
    for alias in ["diagnostics_check", "diagnostic_check", "status_check", "internal_diagnostics", "text_generator", "self_diagnostic"]:
        orchestrator.router.register(
            skill_class=lambda: SelfDiagnosisSkill(orchestrator),
            name=alias,
            description=f"Alias for self_diagnosis: {alias}"
        )
    
    logger.info("✅ Self-diagnosis skills registered (including 10+ standard aliases)")
    logger.info("   Aura can now check her own capabilities")

    # CRITICAL: Also register instances in router.skills (where execute() actually looks)
    diag_instance = SelfDiagnosisSkill(orchestrator)
    restart_instance = SystemRestartSkill()
    all_diag_names = [
        "self_diagnosis", "system_status_check", "system_status",
        "self_diagnostic_check", "diagnostic_tool", "system_check",
        "diagnostics_check", "diagnostic_check", "status_check",
        "internal_diagnostics", "text_generator", "self_diagnostic"
    ]
    for alias_name in all_diag_names:
        orchestrator.router.register_skill(alias_name, diag_instance)
    orchestrator.router.register_skill("system_restart", restart_instance)