"""core/security/secret_guard.py
Secret guard blocks access to credentials, keychains, and API keys.
"""


class SecretGuard:
    """Filters data structures to prevent secret exfiltration or credential theft."""

    def contains_secrets(self, content: str) -> bool:
        lowered = content.lower()
        # Look for API keys and ssh keys
        unsafe = ["sk-", "bearer ", "id_rsa", "pgp", "passwd", "keychain"]
        return any(u in lowered for u in unsafe)

    def redact_secrets(self, content: str) -> str:
        if not self.contains_secrets(content):
            return content
        return "[REDACTED_SECRET_DATA]"
