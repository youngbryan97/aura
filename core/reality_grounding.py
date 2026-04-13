import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger("RealityGrounding")

CapabilityProbe = Callable[[], tuple[bool, str]]

class ConfidenceLevel(Enum):
    CERTAIN = "certain"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

class InformationSource(Enum):
    VERIFIED_FILE = "verified_file"
    VERIFIED_CAPABILITY = "verified_capability"
    MEMORY = "memory"
    INFERENCE = "inference"
    ASSUMPTION = "assumption"
    UNKNOWN = "unknown"

@dataclass
class VerifiedClaim:
    claim: str
    is_true: bool
    confidence: ConfidenceLevel
    source: InformationSource
    evidence: str
    timestamp: float

class RealityVerifier:
    def __init__(self, orchestrator: Any) -> None:
        self.orchestrator = orchestrator
        self.verified_facts: dict[str, VerifiedClaim] = {}
    
    def verify_claim(self, claim: str) -> VerifiedClaim:
        claim_lower = claim.lower()
        if "i can" in claim_lower or "i have" in claim_lower:
            return self._verify_capability_claim(claim)
        return VerifiedClaim(claim=claim, is_true=False, confidence=ConfidenceLevel.UNKNOWN, 
                             source=InformationSource.UNKNOWN, evidence="No verification method", timestamp=0)

    def _verify_capability_claim(self, claim: str) -> VerifiedClaim:
        import time
        capability_keywords: dict[str, CapabilityProbe] = {
            "edit files": lambda: (hasattr(self.orchestrator, 'code_modification'), "code_modification engine status"),
            "search the web": lambda: ("web_search" in getattr(self.orchestrator.router, 'skills', {}), "web_search skill in router"),
            "execute code": lambda: (hasattr(self.orchestrator, 'skill_execution'), "skill execution engine status")
        }
        for kw, func in capability_keywords.items():
            if kw in claim.lower():
                ok, ev = func()
                return VerifiedClaim(claim, ok, ConfidenceLevel.CERTAIN, InformationSource.VERIFIED_CAPABILITY, ev, time.time())
        return VerifiedClaim(claim, False, ConfidenceLevel.UNKNOWN, InformationSource.UNKNOWN, "Unknown capability", time.time())

class RealityGroundingSystem:
    def __init__(self, orchestrator: Any) -> None:
        self.orchestrator = orchestrator
        self.verifier = RealityVerifier(orchestrator)
        logger.info("Reality Grounding System initialized")
    
    def ground_statement(self, statement: str) -> str:
        # Simple grounding: check for "I can" claims
        claims = re.findall(r'I can [^.!?]+', statement, re.IGNORECASE)
        for claim in claims:
            verified = self.verifier.verify_claim(claim)
            if not verified.is_true and verified.confidence != ConfidenceLevel.UNKNOWN:
                statement = statement.replace(claim, f"{claim} (unverified)")
        return statement

def integrate_reality_grounding(orchestrator: Any) -> None:
    orchestrator.reality_grounding = RealityGroundingSystem(orchestrator)
    logger.info("Reality Grounding integrated")
