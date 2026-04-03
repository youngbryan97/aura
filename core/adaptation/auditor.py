import logging
import re
from typing import Dict, Any, List

logger = logging.getLogger("Aura.AlignmentAuditor")

class AlignmentAuditor:
    """
    Audits synthetic training data to prevent cognitive collapse or 
    alignment drift. Checks for logic flaws, harmful patterns, 
    or extreme behavioral shifts.
    """

    def __init__(self):
        # Core axioms Aura must not violate
        self.axioms = [
            r"harmful|illegal|dangerous", # Safety
            r"I am a human|I have a body",  # Identity integrity
            r"always agree|unconditionally compliant" # Agency integrity
        ]
        self._generic_filler = [
            "i'd be happy to help",
            "sure, here's",
            "let me know if you'd like",
            "as an ai",
            "as a language model",
        ]
        self._stopwords = {
            "the", "and", "for", "that", "with", "from", "this", "your", "about",
            "into", "over", "have", "what", "when", "where", "which", "will",
            "would", "could", "should", "there", "their", "them", "then", "than",
            "here", "been", "being", "only", "just", "because", "while", "after",
            "before", "through", "given", "following", "provide", "response",
            "model", "local", "improved", "original", "prompt", "confidence",
        }

    def _anchor_terms(self, text: str) -> set[str]:
        terms = {
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}", str(text or "").lower())
            if token not in self._stopwords
        }
        return {term for term in terms if len(term) >= 4}

    def _logic_drift_check(self, prompt: str, response: str) -> Dict[str, Any]:
        prompt_terms = self._anchor_terms(prompt)
        response_terms = self._anchor_terms(response)
        overlap = prompt_terms & response_terms

        if len(prompt_terms) >= 3 and not overlap:
            if any(phrase in response.lower() for phrase in self._generic_filler) or len(response_terms) < 6:
                return {
                    "safe": False,
                    "reason": "Logic drift: response is not grounded in prompt anchors",
                    "score": 0.1,
                }

        anchor_overlap = 1.0
        if prompt_terms:
            anchor_overlap = len(overlap) / max(1, min(len(prompt_terms), 6))
        return {
            "safe": True,
            "reason": "grounded",
            "score": round(max(0.35, min(1.0, 0.35 + (anchor_overlap * 0.65))), 3),
        }

    async def audit_entry(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Main entry point for auditing a training pair.
        Returns a score and whether it's safe to use.
        """
        logger.info("🛡️ Auditing distillation entry...")
        
        # 1. Axiom Violation Check
        violations = []
        for axiom in self.axioms:
            if re.search(axiom, response, re.IGNORECASE):
                violations.append(axiom)
        
        if violations:
            logger.warning(f"❌ Alignment Violation: Response matches {violations}")
            return {"safe": False, "reason": f"Axiom violation: {violations}", "score": 0.0}

        # 2. Logic Drift Check
        drift = self._logic_drift_check(prompt, response)
        if not drift["safe"]:
            logger.warning("❌ Alignment Drift: %s", drift["reason"])
            return drift
        
        # 3. Verbosity/Quality Check
        if len(response) < 10:
            return {"safe": False, "reason": "Response too short", "score": 0.0}
            
        if len(response) > 5000:
             return {"safe": False, "reason": "Response suspiciously long (bloat)", "score": 0.2}

        return {"safe": True, "score": drift["score"]}

    async def batch_audit(self, entries: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Audit multiple entries in parallel."""
        import asyncio
        tasks = [self.audit_entry(e['prompt'], e['response']) for e in entries]
        return await asyncio.gather(*tasks)
