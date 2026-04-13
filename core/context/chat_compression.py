"""
Chat Compression Service — Ported from gemini-cli/chatCompressionService.ts

3-phase compression pipeline:
  1. Reverse Token Budget — truncate old tool outputs to save space
  2. LLM-Powered Summarization — summarize older history into <state_snapshot>
  3. Probe Verification — self-check the summary for lost details

This keeps long conversations stable by preventing context overflow while
preserving critical information through intelligent summarization.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("Aura.ChatCompression")


class CompressionStatus(Enum):
    """Outcome of a compression attempt."""
    NOOP = "noop"                                       # Under threshold, nothing done
    COMPRESSED = "compressed"                           # Full LLM summarization succeeded
    CONTENT_TRUNCATED = "content_truncated"             # Only tool output truncation applied
    FAILED_EMPTY_SUMMARY = "failed_empty_summary"       # LLM returned empty summary
    FAILED_INFLATED = "failed_inflated_token_count"     # Summary was larger than original


@dataclass
class CompressionInfo:
    """Telemetry about a compression operation."""
    original_token_count: int
    new_token_count: int
    status: CompressionStatus
    duration_ms: float = 0.0


# ── Configuration Constants ──────────────────────────────────────────────────

# Trigger compression when history exceeds this fraction of model context limit
DEFAULT_COMPRESSION_THRESHOLD = 0.50

# Keep the most recent 30% of history after compression (the rest gets summarized)
COMPRESSION_PRESERVE_FRACTION = 0.30

# Token budget for function responses in the preserved history
FUNCTION_RESPONSE_TOKEN_BUDGET = 50_000

# Maximum lines to keep when truncating a tool response
TRUNCATION_TAIL_LINES = 30


def estimate_tokens(text: str) -> int:
    """Fast token estimate: ~4 chars per token for English."""
    return max(1, len(text) // 4)


def estimate_tokens_for_messages(messages: List[Dict[str, str]]) -> int:
    """Estimate total tokens across a list of messages."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        total += estimate_tokens(content)
        # Small overhead per message for role/formatting
        total += 4
    return total


# ── Tool Output Truncation ───────────────────────────────────────────────────

def _save_truncated_output(content: str, tool_name: str, truncation_id: str, temp_dir: str) -> str:
    """Save the full untruncated tool output to disk for later retrieval.

    Returns: path to the saved file.
    """
    os.makedirs(temp_dir, exist_ok=True)
    filename = f"truncated_{tool_name}_{truncation_id}_{int(time.time())}.txt"
    filepath = os.path.join(temp_dir, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath
    except Exception as e:
        logger.warning("Failed to save truncated output: %s", e)
        return "(save failed)"


def _format_truncated_output(content: str, saved_path: str, tail_lines: int = TRUNCATION_TAIL_LINES) -> str:
    """Create a truncated placeholder showing the last N lines + a pointer to the full output."""
    lines = content.split("\n")
    if len(lines) <= tail_lines:
        return content

    tail = "\n".join(lines[-tail_lines:])
    return (
        f"[Output truncated — {len(lines)} total lines. Showing last {tail_lines} lines.]\n"
        f"[Full output saved to: {saved_path}]\n\n"
        f"{tail}"
    )


def truncate_history_to_budget(
    history: List[Dict[str, str]],
    temp_dir: str,
    budget: int = FUNCTION_RESPONSE_TOKEN_BUDGET,
) -> List[Dict[str, str]]:
    """Truncate old tool/function response outputs using a reverse token budget.

    Iterates newest-first. Recent tool outputs are preserved in full.
    Once the cumulative token count exceeds the budget, older large
    tool responses are truncated to their last N lines and saved to disk.
    """
    function_response_tokens = 0
    truncation_counter = 0
    result = []

    # Process backwards (newest first)
    for msg in reversed(history):
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Only truncate tool/system context blocks
        is_tool_output = (
            role in ("system", "tool")
            and any(tag in content for tag in ("[TOOL CONTEXT]", "[TOOL RESULT]", "Output:"))
        )

        if is_tool_output:
            tokens = estimate_tokens(content)
            if function_response_tokens + tokens > budget:
                # Budget exceeded — truncate this response
                truncation_counter += 1
                saved_path = _save_truncated_output(
                    content, "tool_output", str(truncation_counter), temp_dir
                )
                truncated = _format_truncated_output(content, saved_path)
                result.insert(0, {**msg, "content": truncated})
                function_response_tokens += estimate_tokens(truncated)
            else:
                function_response_tokens += tokens
                result.insert(0, msg)
        else:
            result.insert(0, msg)

    return result


# ── Split Point Calculation ──────────────────────────────────────────────────

def find_compress_split_point(history: List[Dict[str, str]], fraction: float) -> int:
    """Find the index of the oldest message to KEEP after compression.

    Everything before this index gets summarized; everything from this index
    onward is preserved in full.

    The fraction parameter (0..1) represents how much of the total content
    should be compressed. A fraction of 0.7 means compress the oldest 70%.
    """
    if fraction <= 0 or fraction >= 1:
        raise ValueError("Fraction must be between 0 and 1")

    char_counts = [len(json.dumps(msg)) for msg in history]
    total_chars = sum(char_counts)
    target_chars = total_chars * fraction

    last_split_point = 0
    cumulative = 0

    for i, msg in enumerate(history):
        # Only split at user messages (not in the middle of a tool call/response)
        if msg.get("role") == "user":
            if cumulative >= target_chars:
                return i
            last_split_point = i
        cumulative += char_counts[i]

    # Check if we can compress everything
    if history and history[-1].get("role") == "assistant":
        return len(history)

    return last_split_point


# ── Compression Prompts ─────────────────────────────────────────────────────

COMPRESSION_SYSTEM_PROMPT = """You are a context compression assistant. Your job is to create a concise <state_snapshot> that preserves all critical information from the conversation history.

Your <state_snapshot> MUST include:
1. The user's original goal/request and any refined requirements
2. All file paths mentioned, created, or modified
3. All tool outputs, error messages, and their resolutions
4. Technical decisions made and their rationale
5. Current state of any ongoing task (what's done, what's pending)
6. Any constraints or preferences the user expressed

Format your response as:
<state_snapshot>
[structured summary here]
</state_snapshot>

Be specific — include exact file paths, error messages, and code snippets where relevant.
Do NOT generalize or abstract away concrete details."""

VERIFICATION_PROMPT = """Critically evaluate the <state_snapshot> you just generated. Did you omit any specific technical details, file paths, tool results, or user constraints mentioned in the history?

If anything is missing or could be more precise, generate a FINAL, improved <state_snapshot>. Otherwise, repeat the exact same <state_snapshot> again."""


# ── Main Service ─────────────────────────────────────────────────────────────

class ChatCompressionService:
    """Compresses conversation history to stay within model context limits.

    Ported from gemini-cli's ChatCompressionService with adaptations for
    Aura's local Ollama-based LLM architecture.
    """

    def __init__(self, temp_dir: str = None):
        self._temp_dir = temp_dir or os.path.expanduser("~/.aura_runtime/compression")
        self._has_failed_attempt = False
        self._compression_count = 0
        os.makedirs(self._temp_dir, exist_ok=True)

    async def compress(
        self,
        history: List[Dict[str, str]],
        model_token_limit: int,
        current_token_count: int,
        brain: Any = None,
        force: bool = False,
        threshold: float = DEFAULT_COMPRESSION_THRESHOLD,
    ) -> Tuple[Optional[List[Dict[str, str]]], CompressionInfo]:
        """Compress conversation history if it exceeds the threshold.

        Args:
            history: Current conversation messages (excluding system prompt)
            model_token_limit: Maximum tokens for the model
            current_token_count: Current estimated token usage
            brain: LocalBrain instance for LLM summarization calls
            force: Force compression regardless of threshold
            threshold: Fraction of model limit that triggers compression

        Returns:
            (new_history or None, CompressionInfo)
        """
        start_time = time.time()

        if not history:
            return None, CompressionInfo(0, 0, CompressionStatus.NOOP)

        original_tokens = current_token_count

        # Check threshold
        if not force and original_tokens < threshold * model_token_limit:
            return None, CompressionInfo(original_tokens, original_tokens, CompressionStatus.NOOP)

        logger.info(
            "Chat compression triggered: %d tokens (%.0f%% of %d limit)",
            original_tokens, (original_tokens / model_token_limit) * 100, model_token_limit
        )

        # Phase 1: Truncate tool outputs to budget
        truncated_history = truncate_history_to_budget(history, self._temp_dir)

        # If summarization previously failed and not forced, only use truncation
        if self._has_failed_attempt and not force:
            truncated_tokens = estimate_tokens_for_messages(truncated_history)
            if truncated_tokens < original_tokens:
                duration = (time.time() - start_time) * 1000
                return truncated_history, CompressionInfo(
                    original_tokens, truncated_tokens,
                    CompressionStatus.CONTENT_TRUNCATED, duration
                )
            return None, CompressionInfo(original_tokens, original_tokens, CompressionStatus.NOOP)

        # Phase 2: Split and summarize
        split_point = find_compress_split_point(
            truncated_history, 1 - COMPRESSION_PRESERVE_FRACTION
        )

        history_to_compress = truncated_history[:split_point]
        history_to_keep = truncated_history[split_point:]

        if not history_to_compress:
            return None, CompressionInfo(original_tokens, original_tokens, CompressionStatus.NOOP)

        # Generate summary via LLM
        if brain is None:
            logger.warning("No brain provided for compression — falling back to truncation only")
            self._has_failed_attempt = True
            truncated_tokens = estimate_tokens_for_messages(truncated_history)
            duration = (time.time() - start_time) * 1000
            return truncated_history, CompressionInfo(
                original_tokens, truncated_tokens,
                CompressionStatus.CONTENT_TRUNCATED, duration
            )

        try:
            # Build the summarization prompt from the history to compress
            history_text = "\n".join(
                f"[{msg.get('role', 'unknown')}]: {msg.get('content', '')}"
                for msg in history_to_compress
            )

            # Check for existing snapshot
            has_previous_snapshot = any(
                "<state_snapshot>" in msg.get("content", "")
                for msg in history_to_compress
            )

            anchor_instruction = (
                "A previous <state_snapshot> exists in the history. You MUST integrate all "
                "still-relevant information from that snapshot into the new one, updating it "
                "with the more recent events. Do not lose established constraints or critical knowledge."
                if has_previous_snapshot
                else "Generate a new <state_snapshot> based on the provided history."
            )

            summarize_prompt = f"{anchor_instruction}\n\nFirst, reason about what's important. Then, generate the updated <state_snapshot>.\n\nHistory to summarize:\n{history_text}"

            # Phase 2a: Generate summary
            summary_result = await brain.generate(
                summarize_prompt,
                system_prompt=COMPRESSION_SYSTEM_PROMPT,
                options={"num_predict": 2048, "num_ctx": 8192, "temperature": 0.3}
            )
            summary = summary_result.get("response", "").strip()

            if not summary:
                self._has_failed_attempt = True
                duration = (time.time() - start_time) * 1000
                return None, CompressionInfo(
                    original_tokens, original_tokens,
                    CompressionStatus.FAILED_EMPTY_SUMMARY, duration
                )

            # Phase 3: Probe verification — self-check the summary
            verify_result = await brain.generate(
                VERIFICATION_PROMPT,
                system_prompt=COMPRESSION_SYSTEM_PROMPT,
                options={"num_predict": 2048, "num_ctx": 8192, "temperature": 0.1}
            )
            final_summary = verify_result.get("response", "").strip() or summary

            # Build new history: snapshot + preserved recent history
            new_history = [
                {"role": "user", "content": final_summary},
                {"role": "assistant", "content": "Got it. I've loaded the conversation context from the snapshot."},
                *history_to_keep
            ]

            new_tokens = estimate_tokens_for_messages(new_history)

            # Sanity check: compression should not inflate
            if new_tokens > original_tokens:
                logger.warning(
                    "Compression inflated tokens: %d → %d. Discarding.",
                    original_tokens, new_tokens
                )
                duration = (time.time() - start_time) * 1000
                return None, CompressionInfo(
                    original_tokens, new_tokens,
                    CompressionStatus.FAILED_INFLATED, duration
                )

            self._compression_count += 1
            self._has_failed_attempt = False
            duration = (time.time() - start_time) * 1000
            logger.info(
                "Chat compressed: %d → %d tokens (%.0f%% reduction, %.0fms)",
                original_tokens, new_tokens,
                (1 - new_tokens / original_tokens) * 100, duration
            )

            return new_history, CompressionInfo(
                original_tokens, new_tokens,
                CompressionStatus.COMPRESSED, duration
            )

        except Exception as e:
            logger.error("Chat compression failed: %s", e, exc_info=True)
            self._has_failed_attempt = True
            # Fallback to truncation-only
            truncated_tokens = estimate_tokens_for_messages(truncated_history)
            duration = (time.time() - start_time) * 1000
            if truncated_tokens < original_tokens:
                return truncated_history, CompressionInfo(
                    original_tokens, truncated_tokens,
                    CompressionStatus.CONTENT_TRUNCATED, duration
                )
            return None, CompressionInfo(original_tokens, original_tokens, CompressionStatus.NOOP, duration)
