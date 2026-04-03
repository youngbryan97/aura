import re
import logging

logger = logging.getLogger(__name__)


def resolve_pronouns(text: str, session=None) -> str:
    """Lightweight pronoun resolver: replace direct-address 'you/your' with
    the registered user name for the session when available and it reads naturally.

    Heuristics:
    - If session has an attribute `name` or `identity`, use that.
    - Replace sentence-starting 'You' and occurrences of ' you' when followed by punctuation or end-of-sentence.
    - Only perform replacements when a name is present to avoid changing third-person uses.
    """
    if not text or session is None:
        return text

    # Attempt to extract a display name from session
    name = None
    for attr in ("name", "display_name", "identity", "username"):
        if hasattr(session, attr):
            val = getattr(session, attr)
            if val:
                name = str(val)
                break

    # Some session objects provide a summary() dict or method
    if not name:
        try:
            summary = getattr(session, "summary", None)
            if callable(summary):
                s = summary()
                if isinstance(s, dict) and s.get("name"):
                    name = s.get("name")
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)
    if not name:
        return text

    # Simple replacements: start-of-sentence You -> Name, and ' you' before punctuation/space/end
    def replace_start(match):
        lead = match.group(1)
        return f"{lead}{name}"

    # Replace capitalized 'You' at start of string or after sentence boundary
    text = re.sub(r'(^|[\.\?!]\s+)(You)(\b)', replace_start, text)

    # Conservative replacement: replace 'you' only when followed by punctuation or end-of-string
    # This avoids incorrect grammatical substitutions inside clauses.
    text = re.sub(r'(?i)\byou\b(?=(?:[\s]*[\?\!\.,]|$))', name, text)

    return text