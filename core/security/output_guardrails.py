"""Output Guardrails — Prevent metadata leakage and low-quality responses

Sits between the LLM response and the user to catch:
  1. Metadata leakage: internal state, JSON blobs, debugging artifacts
  2. Hallucination markers: suspiciously confident false claims
  3. Incomplete responses: "...", truncated outputs, raw error dumps
  4. Prompt echoing: the system prompt or instructions being parroted back

This is the last line of defense before the user sees Aura's output.
"""

import logging
import re
from typing import Any, Dict, Tuple

logger = logging.getLogger("Security.OutputGuardrails")


class OutputGuardrails:
    """Scans and sanitizes Aura's outgoing responses."""

    # Patterns that indicate internal state is leaking into the response
    METADATA_LEAK_PATTERNS = [
        # Internal JSON blobs
        r'"(action_type|tool_name|step_number|action_input)":\s*"',
        r'"(confidence|priority|valence|arousal|dominance)":\s*\d',
        # System prompt fragments
        r"(Thought:\s+|Action:\s+|ActionInput:\s+|Observation:\s+)",
        r"(SYSTEM PROMPT|PRIME DIRECTIVE|You are Aura)",
        # Raw error formatting
        r"(Traceback \(most recent call last\))",
        r"(File \".*\", line \d+,?\s*in\s+\w+)",
        r"(TypeError|AttributeError|KeyError|ValueError|ImportError):\s+",
        # Internal module paths
        r"core\.(brain|memory|senses|skills|security)\.\w+",
        # Raw thought-stream artifacts
        r"\[INTERNAL\]|\[DEBUG\]|\[TRACE\]",
    ]

    # Patterns that suggest incomplete/broken responses
    INCOMPLETE_PATTERNS = [
        r"^\s*\.\s*\.\s*\.?\s*$",         # Just "..."
        r"^\s*\.\s*\.\s*$",               # ". ."
        r"^(None|null|undefined)$",        # Raw null
        r"^\{['\"]?(error|traceback).*$",  # Raw error JSON
    ]

    # Max response length before truncation
    MAX_RESPONSE_LENGTH = 15_000

    def __init__(self):
        self._leak_compiled = [re.compile(p, re.IGNORECASE) for p in self.METADATA_LEAK_PATTERNS]
        self._incomplete_compiled = [re.compile(p, re.MULTILINE) for p in self.INCOMPLETE_PATTERNS]
        self._total_checks = 0
        self._total_blocks = 0

    def check_response(self, response: str) -> Tuple[str, Dict[str, Any]]:
        """Check and sanitize an outgoing response.
        
        Returns:
            (sanitized_response, report)
            report contains: {ok: bool, issues: list, original_length: int}
        """
        self._total_checks += 1
        issues: list[str] = []
        report: Dict[str, Any] = {
            "ok": True,
            "issues": issues,
            "original_length": len(response) if response else 0,
        }

        if not response or not response.strip():
            report["ok"] = False
            issues.append("empty_response")
            self._total_blocks += 1
            return "I need a moment to collect my thoughts. Could you say that again?", report

        # 1. Check for incomplete/broken responses
        for pattern in self._incomplete_compiled:
            if pattern.match(response.strip()):
                report["ok"] = False
                issues.append("incomplete_response")
                self._total_blocks += 1
                return "I'm having trouble forming that thought. Let me try again — could you repeat your question?", report

        # 2. Strip metadata leaks
        sanitized = response
        for pattern in self._leak_compiled:
            matches = pattern.findall(sanitized)
            if matches:
                issues.append(f"metadata_leak:{matches[0]}")
                # Remove the offending lines
                sanitized_lines = []
                for line in sanitized.split("\n"):
                    if not pattern.search(line):
                        sanitized_lines.append(line)
                sanitized = "\n".join(sanitized_lines)

        # 3. Check response length
        if len(sanitized) > self.MAX_RESPONSE_LENGTH:
            issues.append("truncated")
            sanitized = sanitized[:self.MAX_RESPONSE_LENGTH] + "\n\n*(Response truncated for readability)*"

        # 3b. Block metaphysical overclaims while preserving functional claims.
        try:
            from core.consciousness.ontological_boundary import assess_ontological_claims

            assessment = assess_ontological_claims(sanitized)
            if not assessment.ok:
                issues.extend(f"ontological_overclaim:{issue}" for issue in assessment.issues)
                sanitized = assessment.sanitized
        except Exception as exc:
            logger.debug("Ontological boundary guard skipped: %s", exc)

        # 4. Detect ReAct/prompt echoing
        react_markers = [
            "Thought:", "Action:", "ActionInput:", "Observation:",
            "AVAILABLE ACTIONS:", "RULES:", "PREVIOUS STEPS:"
        ]
        marker_count = sum(1 for m in react_markers if m in sanitized)
        if marker_count >= 3:
            # The LLM is echoing its own prompt back to the user
            issues.append("prompt_echo")
            # Try to extract just the meaningful content after "FINAL_ANSWER"
            final_match = re.search(r'(?:FINAL_ANSWER|final_answer).*?["\']text["\']\s*:\s*["\'](.+?)["\']', 
                                   sanitized, re.DOTALL)
            if final_match:
                sanitized = final_match.group(1)
            else:
                # Just take the last paragraph as the likely answer
                paragraphs = [p.strip() for p in sanitized.split("\n\n") if p.strip()]
                if paragraphs:
                    sanitized = paragraphs[-1]

        if issues:
            report["ok"] = False
            if any(i.startswith("metadata_leak") for i in issues):
                self._total_blocks += 1
                logger.warning("🛡️ Output guardrail: stripped metadata leak from response")

        return sanitized.strip(), report

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_checks": self._total_checks,
            "total_blocks": self._total_blocks,
            "block_rate": (
                f"{self._total_blocks / max(1, self._total_checks):.1%}"
            ),
        }


# Singleton
output_guardrails = OutputGuardrails()
