"""
Tool Output Distillation Service — Ported from gemini-cli/toolDistillationService.ts

Multi-tier output handling:
  1. If output fits within budget → return as-is
  2. If oversized → save raw to disk + generate an LLM intent summary
  3. Proportional head/tail preservation for structural truncation fallback

Replaces all hard-coded [:2000] / [:5000] truncation across Aura's skills.
"""

from core.runtime.errors import record_degradation
import hashlib
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("Aura.ToolDistillation")

# ── Configuration ────────────────────────────────────────────────────────────

# Token thresholds for distillation triggers
DISTILL_TOKEN_THRESHOLD = 3000          # Distill if output exceeds this
SMALL_OUTPUT_MAX_CHARS = 12000          # ~3000 tokens — passed through without distillation
LARGE_OUTPUT_BUDGET_CHARS = 8000        # ~2000 tokens — target for distilled output
HEAD_FRACTION = 0.3                     # Keep 30% from head
TAIL_FRACTION = 0.7                     # Keep 70% from tail
SUMMARY_MAX_TOKENS = 1024              # Budget for LLM-generated summary

# Storage directory for full tool outputs
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/.aura_runtime/tool_outputs")


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content[:4096].encode()).hexdigest()[:12]


class ToolOutputDistillationService:
    """Intelligently distills large tool outputs while preserving key information.

    Replaces the naive [:N] truncation with multi-tier processing:
    1. Small outputs → pass through
    2. Medium outputs → structural truncation (head + tail + metadata)
    3. Large outputs → LLM-powered intent summary + raw saved to disk
    """

    def __init__(self, output_dir: str = None):
        self._output_dir = output_dir or DEFAULT_OUTPUT_DIR
        os.makedirs(self._output_dir, exist_ok=True)

    def save_raw_output(self, content: str, tool_name: str, command: str = "") -> str:
        """Save full output to disk for later retrieval.

        Returns: file path to the saved output.
        """
        timestamp = int(time.time())
        content_hash = _content_hash(content)
        filename = f"{tool_name}_{content_hash}_{timestamp}.txt"
        filepath = os.path.join(self._output_dir, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                if command:
                    f.write(f"# Command: {command}\n")
                    f.write(f"# Timestamp: {timestamp}\n")
                    f.write(f"# Output size: {len(content)} chars\n")
                    f.write("# " + "=" * 70 + "\n\n")
                f.write(content)
            return filepath
        except Exception as e:
            record_degradation('tool_distillation', e)
            logger.warning("Failed to save tool output: %s", e)
            return "(save failed)"

    def structural_truncate(
        self,
        content: str,
        budget_chars: int = LARGE_OUTPUT_BUDGET_CHARS,
        head_frac: float = HEAD_FRACTION,
        tail_frac: float = TAIL_FRACTION,
    ) -> str:
        """Proportional head+tail truncation preserving structure.

        Keeps the first N% and last M% of the output with a clear
        truncation marker in between.
        """
        if len(content) <= budget_chars:
            return content

        head_budget = int(budget_chars * head_frac)
        tail_budget = int(budget_chars * tail_frac)

        lines = content.split("\n")
        total_lines = len(lines)

        # Find the split points by character count
        head_chars = 0
        head_end = 0
        for i, line in enumerate(lines):
            head_chars += len(line) + 1
            if head_chars >= head_budget:
                head_end = i + 1
                break

        tail_chars = 0
        tail_start = total_lines
        for i in range(total_lines - 1, -1, -1):
            tail_chars += len(lines[i]) + 1
            if tail_chars >= tail_budget:
                tail_start = i
                break

        # Ensure no overlap
        if tail_start <= head_end:
            tail_start = head_end + 1

        omitted_lines = tail_start - head_end
        omitted_chars = sum(len(lines[i]) + 1 for i in range(head_end, tail_start))

        head_section = "\n".join(lines[:head_end])
        tail_section = "\n".join(lines[tail_start:])

        return (
            f"{head_section}\n\n"
            f"... [{omitted_lines} lines / {omitted_chars} chars omitted] ...\n\n"
            f"{tail_section}"
        )

    def _extract_key_signals(self, content: str, tool_name: str) -> str:
        """Extract critical signals from tool output (errors, paths, exit codes)."""
        signals = []

        # Extract error-like lines
        for line in content.split("\n"):
            line_lower = line.lower().strip()
            if any(kw in line_lower for kw in ["error", "traceback", "exception", "failed", "fatal", "permission denied"]):
                signals.append(f"  ⚠ {line.strip()}")
            elif any(kw in line_lower for kw in ["warning", "deprecated"]):
                signals.append(f"  ⚡ {line.strip()}")

        # Extract file paths
        import re
        paths = re.findall(r'(?:/[\w./-]+)+', content)
        unique_paths = list(dict.fromkeys(paths))[:10]
        if unique_paths:
            signals.append(f"  📁 Paths mentioned: {', '.join(unique_paths)}")

        # Extract exit codes
        exit_matches = re.findall(r'(?:exit|return|status)\s*(?:code)?\s*[:=]?\s*(\d+)', content, re.IGNORECASE)
        if exit_matches:
            signals.append(f"  🔢 Exit codes: {', '.join(exit_matches)}")

        return "\n".join(signals[:20]) if signals else ""

    async def distill(
        self,
        content: str,
        tool_name: str,
        command: str = "",
        brain: Any = None,
    ) -> str:
        """Distill tool output to fit within token budget.

        Args:
            content: Raw tool output
            tool_name: Name of the tool that produced the output
            command: Original command/action (for context)
            brain: Optional LocalBrain for LLM-powered summarization

        Returns:
            Distilled output string
        """
        if not content:
            return "(empty output)"

        # Tier 1: Small output — pass through
        if len(content) <= SMALL_OUTPUT_MAX_CHARS:
            return content

        # Save raw output to disk
        saved_path = self.save_raw_output(content, tool_name, command)

        # Extract key signals first
        key_signals = self._extract_key_signals(content, tool_name)

        # Tier 2: Try LLM-powered summary if brain is available
        if brain is not None:
            try:
                summary_prompt = (
                    f"Summarize the following {tool_name} output concisely. "
                    f"Focus on: results, errors, file paths created/modified, status, and key data points. "
                    f"Keep it under 500 words.\n\n"
                    f"Command: {command}\n"
                    f"Output ({len(content)} chars):\n"
                    f"{content[:8000]}"  # Feed first 8K to summarizer
                )
                result = await brain.generate(
                    summary_prompt,
                    options={"num_predict": SUMMARY_MAX_TOKENS, "temperature": 0.2}
                )
                summary = result.get("response", "").strip()

                if summary and len(summary) > 20:
                    header = (
                        f"📋 {tool_name} output distilled ({len(content)} chars → {len(summary)} chars)\n"
                        f"💾 Full output saved: {saved_path}\n"
                    )
                    if key_signals:
                        header += f"🔍 Key signals:\n{key_signals}\n"
                    return f"{header}\n{summary}"

            except Exception as e:
                record_degradation('tool_distillation', e)
                logger.warning("LLM distillation failed, falling back to structural truncation: %s", e)

        # Tier 3: Structural truncation fallback
        truncated = self.structural_truncate(content)
        header = (
            f"📋 {tool_name} output truncated ({len(content)} chars → {len(truncated)} chars)\n"
            f"💾 Full output saved: {saved_path}\n"
        )
        if key_signals:
            header += f"🔍 Key signals:\n{key_signals}\n"
        return f"{header}\n{truncated}"


# Global singleton for easy access
_default_service: Optional[ToolOutputDistillationService] = None


def get_distillation_service() -> ToolOutputDistillationService:
    """Get or create the global distillation service instance."""
    global _default_service
    if _default_service is None:
        _default_service = ToolOutputDistillationService()
    return _default_service
