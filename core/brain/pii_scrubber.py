"""PII scrubbing for governed egress boundaries.

Aura's resident model does not require redaction. Network-capable skills such
as browsing, messaging, and web interlocution can still transmit text beyond
the host, so their shared egress boundary needs deterministic redaction and a
verifiable receipt independent of any model provider.
"""

import hashlib
import logging
import re

from core.runtime.errors import record_degradation


class PrivateNamesUnavailable(RuntimeError):
    """The private-name list could not be read.

    Its own exception type because the caller's correct response is specific:
    a scrub that cannot load the names it exists to remove must not report
    success. Reusing a generic error would let a broad `except` upstream
    treat "redaction is unavailable" the same as "there was nothing to
    redact" at a network egress boundary.
    """


logger = logging.getLogger("Aura.PIIScrubber")

__all__ = [
    "SCRUBBER_VERSION",
    "get_pii_patterns",
    "residual_pii_findings",
    "scrub_for_cloud_with_receipt",
    "scrub_for_egress_with_receipt",
    "scrub_pii_for_cloud",
    "scrub_pii_for_egress",
]

# Patterns that indicate PII-bearing content in system prompts
_PII_SECTION_MARKERS = (
    "CORE IDENTITY:",
    "SHARED HISTORY:",
    "KINSHIP:",
    "biography_private",
    "FamilyLegacy",
)

# Regex patterns for common PII structures in Aura's prompts
_PII_PATTERNS = [
    # Contact details and credentials. The scrubber's name list catches
    # "Bryan"; nothing caught an email address, a phone number, or an API key
    # pasted into outbound text. residual_pii_findings() checks that these were
    # actually removed, so a pattern that stops matching blocks the send
    # instead of quietly letting the data through.
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[REDACTED_EMAIL]"),
    (re.compile(r"\+?\d[\d\s().-]{8,}\d"), "[REDACTED_NUMBER]"),
    (
        re.compile(r"\b(?:sk|pk|api|key)[-_][A-Za-z0-9]{16,}\b", re.IGNORECASE),
        "[REDACTED_KEY]",
    ),
    # Trust scores: "trust": 0.92, trust=0.92, trust: 0.92
    (re.compile(r'"?trust"?\s*[:=]\s*\d+\.\d+', re.IGNORECASE), '"trust": [REDACTED]'),
    # Relationship labels: "relation": "Architect / Friend / Equal"
    (re.compile(r'"?relation"?\s*[:=]\s*"[^"]*"', re.IGNORECASE), '"relation": "[REDACTED]"'),
    # Known entities blocks: known_entities["name"] = {...}
    (re.compile(r'known_entities\[["\'][^"\']+["\']\]\s*=\s*\{[^}]*\}', re.IGNORECASE), 'known_entities["user"] = {[REDACTED]}'),
    # Relationship graph blocks
    (re.compile(r'relationship_graph\[["\'][^"\']+["\']\]\s*=\s*\{[^}]*\}', re.IGNORECASE), 'relationship_graph["user"] = {[REDACTED]}'),
    # "name: warm" style trust indicators in prompts
    (re.compile(r'\b\w+:\s*(?:warm|trusted|sovereign|friend|equal|architect)\b', re.IGNORECASE), '[user]: [REDACTED]'),
]


def _load_private_names() -> list[str]:
    """Load real names from biography_private.json for targeted redaction."""
    try:
        import json

        from core.config import config
        config_path = config.paths.home_dir / "biography_private.json"
        if config_path.exists():
            with open(config_path) as f:
                data = json.load(f)
            names = []
            creator_name = data.get("creator_name", "")
            if creator_name and len(creator_name) > 1:
                names.append(creator_name)
            for kin in data.get("kin", []):
                name = kin.get("name", "")
                if name and len(name) > 1:
                    names.append(name)
            return names
    except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as exc:
        # Silently returning [] disabled targeted redaction and let the
        # caller carry on scrubbing — so the owner's and his family's real
        # names crossed a network boundary unredacted, and the only signal was
        # an empty list that looks exactly like "this user named nobody".
        #
        # Raised rather than recorded-and-continued: the caller decides
        # whether to proceed without redaction, and it cannot decide what it
        # is never told.
        record_degradation(
            "pii_scrubber",
            exc,
            severity="critical",
            action="could not load the private-name list; targeted redaction is unavailable",
        )
        raise PrivateNamesUnavailable(
            "private-name list could not be loaded; refusing to report an "
            "empty redaction set"
        ) from exc
    return []


_cached_names: list[str] | None = None


def _get_private_names() -> list[str]:
    """Cached loader for private names. Raises PrivateNamesUnavailable."""
    global _cached_names
    if _cached_names is None:
        # Not cached on failure: a transient read error must not disable
        # targeted redaction for the life of the process.
        _cached_names = _load_private_names()
    return _cached_names


def scrub_pii_for_egress(text: str) -> str:
    """Remove personal identifiers before text crosses a governed boundary.

    Replaces:
    - Real names from biography_private.json with "the user"
    - Trust scores with [REDACTED]
    - Relationship labels with [REDACTED]
    - Known entity blocks with generic placeholders
    - Entire CORE IDENTITY / SHARED HISTORY / KINSHIP sections with a
      generic "You have a positive relationship with the user" line

    Args:
        text: The system prompt or message content to scrub.

    Returns:
        Scrubbed text suitable for residual verification before egress.
    """
    if not text:
        return text

    scrubbed = text

    # Replace real names with "the user".
    #
    # A failure here is NOT recoverable by continuing. The regex patterns
    # below catch shapes — emails, numbers — and cannot catch "Bryan",
    # which is exactly what this list is for. Scrubbing the rest and
    # returning would hand the caller text that looks scrubbed and still
    # carries the owner's and his family's names.
    try:
        private_names = _get_private_names()
    except PrivateNamesUnavailable:
        raise
    for name in private_names:
        if name in scrubbed:
            scrubbed = scrubbed.replace(name, "the user")
            # Also replace lowercase/title variants
            scrubbed = scrubbed.replace(name.lower(), "the user")
            scrubbed = scrubbed.replace(name.title(), "the user")

    # Apply regex patterns
    for pattern, replacement in _PII_PATTERNS:
        scrubbed = pattern.sub(replacement, scrubbed)

    # Replace entire PII sections with a generic summary
    for marker in _PII_SECTION_MARKERS:
        if marker in scrubbed:
            # Find the line containing the marker and replace it
            lines = scrubbed.split("\n")
            cleaned_lines = []
            skip_section = False
            for line in lines:
                if marker in line:
                    skip_section = True
                    cleaned_lines.append(
                        "CONTEXT: You have a positive working relationship with the user."
                    )
                    continue
                if skip_section and line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                    skip_section = False
                if not skip_section:
                    cleaned_lines.append(line)
            scrubbed = "\n".join(cleaned_lines)

    return scrubbed


def scrub_pii_for_cloud(text: str) -> str:
    """Compatibility wrapper for callers migrating to ``scrub_pii_for_egress``."""
    return scrub_pii_for_egress(text)


def get_pii_patterns() -> list[tuple[re.Pattern, str]]:
    """Return the PII patterns for external testing/validation."""
    return list(_PII_PATTERNS)


#: Bumped whenever the patterns, the section markers, or the name handling
#: change. A receipt without it cannot say WHICH scrubber produced the text,
#: and two calls in one send could otherwise have used different ones.
SCRUBBER_VERSION = "aura.pii_scrubber.v1"

#: Shapes that must not survive scrubbing. These are the residual scan, not
#: the scrubbing itself: the scrubber replaces, and this checks whether the
#: replacement actually happened before the text leaves the machine.
_RESIDUAL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("phone", re.compile(r"\+?\d[\d\s().-]{8,}\d")),
    ("api_key", re.compile(r"\b(?:sk|pk|api|key)[-_][A-Za-z0-9]{16,}\b", re.IGNORECASE)),
    ("trust_score", re.compile(r'"?trust"?\s*[:=]\s*\d+\.\d+', re.IGNORECASE)),
]


def residual_pii_findings(text: str) -> list[str]:
    """Kinds of personal data still present AFTER scrubbing.

    The original privacy claim was two scrubbed strings and a comment.
    Nothing checked whether the scrub worked, so a name the loader could not
    read, or a shape no pattern covered, left the machine with the claim
    intact. A non-empty result here blocks the send.
    """
    body = str(text or "")
    if not body:
        return []
    findings = [name for name, pattern in _RESIDUAL_PATTERNS if pattern.search(body)]
    try:
        for name in _get_private_names():
            if name and name in body:
                findings.append("private_name")
                break
    except PrivateNamesUnavailable:
        # Unable to check is not the same as clean. Say so; the caller blocks.
        findings.append("private_names_unreadable")
    return sorted(set(findings))


def scrub_for_egress_with_receipt(text: str) -> tuple[str, dict]:
    """Scrub, then record what the scrub actually did.

    The receipt is the difference between "we scrub before sending" as a
    comment and as a checkable claim: which scrubber, what went in (by hash),
    what came out (by hash), and whether anything recognisable survived.
    """
    source = str(text or "")
    scrubbed = scrub_pii_for_egress(source)
    scrubbed_text = "" if scrubbed is None else str(scrubbed)
    residual = residual_pii_findings(scrubbed_text)
    receipt = {
        "schema": "aura.egress.privacy_receipt.v1",
        "scrubber_version": SCRUBBER_VERSION,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "scrubbed_sha256": hashlib.sha256(scrubbed_text.encode("utf-8")).hexdigest(),
        "source_chars": len(source),
        "scrubbed_chars": len(scrubbed_text),
        "changed": scrubbed_text != source,
        "residual_findings": residual,
        "safe_to_send": not residual and bool(scrubbed_text or not source),
    }
    return scrubbed_text, receipt


def scrub_for_cloud_with_receipt(text: str) -> tuple[str, dict]:
    """Compatibility wrapper for the provider-neutral egress receipt API."""
    return scrub_for_egress_with_receipt(text)
