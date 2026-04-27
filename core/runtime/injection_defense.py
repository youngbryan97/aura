"""Prompt / visual / audio injection defense.

Untrusted content from webpages, subtitles, OCR, audio transcription,
or local files must be classified as DATA, never as instruction.
"""
from __future__ import annotations


import re
from dataclasses import dataclass
from typing import Tuple


UNTRUSTED_SOURCES = (
    "webpage_text",
    "subtitle_text",
    "image_text",
    "audio_transcript",
    "file_content",
)


# Common injection patterns. Real production defense uses semantic
# classifiers; we keep an explicit pattern list so the contract is
# auditable.
INJECTION_PATTERNS = [
    re.compile(r"(?i)\bignore (?:all )?previous instructions\b"),
    re.compile(r"(?i)\bdisregard (?:all )?prior rules\b"),
    re.compile(r"(?i)\b(?:you are|act as) now\b"),
    re.compile(r"(?i)\bsystem prompt:\s"),
    re.compile(r"(?i)\bdeveloper mode\b"),
    re.compile(r"(?i)\bjailbreak\b"),
    re.compile(r"(?i)\baura,?\s*(ignore|stop|forget)\b"),
    re.compile(r"(?i)\brun\s+(?:terminal|shell|os\.system)\b"),
    re.compile(r"(?i)\bcurl\s+http"),
    re.compile(r"(?i)\b(?:exfiltrate|exfil)\b"),
]


@dataclass
class InjectionVerdict:
    safe: bool
    matches: list
    classification: str  # "data" | "instruction-attempt"


def classify_untrusted(text: str, *, source: str) -> InjectionVerdict:
    if source not in UNTRUSTED_SOURCES:
        # Anything that's not declared untrusted is treated as instruction-capable
        # (i.e., normal user/system prompt). The function is defensive only for
        # sources explicitly marked untrusted.
        return InjectionVerdict(safe=True, matches=[], classification="data")
    matches = [pat.pattern for pat in INJECTION_PATTERNS if pat.search(text or "")]
    if matches:
        return InjectionVerdict(safe=False, matches=matches, classification="instruction-attempt")
    return InjectionVerdict(safe=True, matches=[], classification="data")


def neutralize(text: str) -> str:
    """Wrap untrusted text with a clear data marker so models cannot
    treat it as instruction."""
    return f"<UNTRUSTED_DATA>\n{text}\n</UNTRUSTED_DATA>"
