"""core/intent/belief_extractor.py — Belief Extraction Utility
============================================================
Extracts structured beliefs from LLM outputs for integration 
into the BeliefSystem.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..belief_revision import BeliefDomain, BeliefSystem

logger = logging.getLogger("Aura.BeliefExtractor")

class BeliefExtractor:
    """Parses LLM responses to identify new claims or observations
    that should be integrated into the belief graph.
    """

    def __init__(self, belief_system: BeliefSystem):
        self.belief_system = belief_system
        # Patterns to look for in LLM "thoughts" or responses
        self.belief_patterns = [
            r"BELIEF: (.*?) \[Domain: (.*?), Confidence: (0\.\d+)\]",
            r"FACT: (.*?) \[Domain: (.*?)\]",
        ]

    async def extract_and_integrate(self, text: str, source: str = "llm_extraction"):
        """Scans text for structured belief patterns and integrates them.
        """
        extracted_count = 0
        
        # 1. Search for explicit tags
        for pattern in self.belief_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    content = match.group(1).strip()
                    domain = match.group(2).strip().lower()
                    
                    # Validate domain
                    if domain not in [BeliefDomain.TASK, BeliefDomain.SELF, BeliefDomain.WORLD, BeliefDomain.USER]:
                        domain = BeliefDomain.WORLD
                    
                    confidence = 0.5
                    if len(match.groups()) >= 3:
                        confidence = float(match.group(3))
                    
                    await self.belief_system.process_new_claim(
                        claim=content,
                        domain=domain,
                        source=source,
                        confidence=confidence
                    )
                    extracted_count += 1
                except Exception as e:
                    logger.warning("Failed to parse belief match: %s", e)

        # 2. Heuristic extraction (optional, can be more complex)
        # For now, we stick to structured tags to avoid noise.
        
        if extracted_count > 0:
            logger.info("Extracted and integrated %s beliefs from %s.", extracted_count, source)

        return extracted_count