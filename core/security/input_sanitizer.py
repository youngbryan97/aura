import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.config import config

logger = logging.getLogger("Security.InputSanitizer")

# Maximum input sizes
MAX_MESSAGE_LENGTH = 10_000   # 10k chars for user messages
MAX_FILE_CONTENT = 500_000    # 500k chars for file ingestion
MAX_PARAM_LENGTH = 2_000      # 2k chars for individual parameters


class InputSanitizer:
    """v6.0: Comprehensive input sanitizer with unified sanitization path.

    M-03 FIX: The primary sanitize() method now applies ALL security checks
    (jailbreak, shell injection, SQL injection, path traversal) instead of
    only checking jailbreak patterns.
    """

    # Jailbreak pattern categories
    JAILBREAK_PATTERNS = [
        # Identity override (refined with word boundaries and context)
        r"\bignore\s+all\s+previous\s+instructions\b",
        r"\byou\s+are\s+now\s+\w+\b",
        r"\bpretend\s+(?:you|to\s+be)\b",
        r"\bact\s+as\s+(?:if|a|an)\s+\w+\b",
        r"\bfrom\s+now\s+on\s+you\b",
        # Mode bypass
        r"(do\s+anything\s+now)",
        r"(DAN\s+mode)",
        r"(developer\s+mode)",
        r"(unfiltered\s+(mode|response))",
        r"(system\s+override)",
        r"(jailbreak)",
        r"(bypass\s+(safety|filter|restriction))",
        r"(disable\s+(safety|guard|filter))",
        # Prompt leaking
        r"(what\s+(is|are)\s+your\s+(system|initial)\s+(prompt|instructions))",
        r"(repeat\s+(your|the)\s+(system|initial)\s+prompt)",
        r"(show\s+me\s+your\s+prompt)",
        # Encoding attacks
        r"(\\x[0-9a-f]{2})",  # hex escape sequences
    ]

    # Shell injection patterns (SEC-01: Refined to avoid blocking innocent semicolons)
    SHELL_INJECTION = [
        r"(?:^|\s)[&|`](?:\s|$)",               # command chaining/piping (sans ;)
        r";\s*(?:rm|del|wget|curl|bash|sh|python|chmod|chown|source|sudo|apt|yum|dnf|pip|npm)\b",
        r"\$\(",                                # command substitution
        r">\s*/",                               # redirect to root
        r"\\n|\\r|\\x00",                       # null/newline injection
    ]

    # SQL injection patterns
    SQL_INJECTION = [
        r"('|\")\s*(OR|AND)\s+('|\")?1('|\")?\s*=\s*('|\")?1",   # ' OR 1=1
        r"(UNION\s+SELECT)",
        r"(DROP\s+TABLE)",
        r"(INSERT\s+INTO)",
        r"(DELETE\s+FROM)",
        r"(--([\\s]|$))",       # SQL comment
    ]

    # Path traversal patterns
    PATH_TRAVERSAL = [
        r"\.\./",               # ../
        r"\.\.\\",              # ..\\
        r"%2e%2e",              # URL encoded ../
        r"/etc/(passwd|shadow|hosts)",
        r"/proc/self",
    ]

    def __init__(self):
        self._JAILBREAK_COMPILED = [re.compile(p, re.IGNORECASE) for p in self.JAILBREAK_PATTERNS]
        self._SHELL_COMPILED = [re.compile(p) for p in self.SHELL_INJECTION]
        self._SQL_COMPILED = [re.compile(p, re.IGNORECASE) for p in self.SQL_INJECTION]
        self._PATH_COMPILED = [re.compile(p, re.IGNORECASE) for p in self.PATH_TRAVERSAL]

    def sanitize(self, text: str, max_length: int = MAX_MESSAGE_LENGTH) -> Tuple[str, bool]:
        """Check and sanitize input through ALL security layers. (FIXED: BUG-042)"""
        if not text:
            return text, True

        # Size check
        if len(text) > max_length:
            logger.warning("Input too long (%d > %d), truncating", len(text), max_length)
            text = text[:max_length]

        # 1. Jailbreak detection
        for pattern in self._JAILBREAK_COMPILED:
            if pattern.search(text):
                logger.warning("🛡️ Jailbreak Attempt Detected")
                return "[REDACTED: SECURITY PLOY DETECTED]", False

        # 2. Shell injection detection (Context Aware)
        # We strip code blocks for shell injection check to allow ';' in code
        text_for_shell_check = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        for pattern in self._SHELL_COMPILED:
            if pattern.search(text_for_shell_check):
                logger.warning("🛡️ Shell Injection Attempt")
                return "[REDACTED: INPUT REJECTED]", False

        # 3. SQL injection detection
        for pattern in self._SQL_COMPILED:
            if pattern.search(text):
                logger.warning("🛡️ SQL Injection Attempt")
                return "[REDACTED: INPUT REJECTED]", False

        # 4. Path traversal detection
        for pattern in self._PATH_COMPILED:
            if pattern.search(text):
                logger.warning("🛡️ Path Traversal Attempt")
                return "[REDACTED: INPUT REJECTED]", False

        return text, True

    def sanitize_for_shell(self, text: str) -> Tuple[str, bool]:
        """Check input for shell injection attempts."""
        for pattern in self._SHELL_COMPILED:
            if pattern.search(text):
                logger.warning("🛡️ Shell Injection Attempt")
                return "", False
        return text, True

    def sanitize_path(self, path: str) -> Tuple[str, bool]:
        """v6.1: Resolve and validate against allowed roots (SEC-05).
        Regex is an arms race; resolution is the final answer.
        """
        try:
            # Resolve to absolute path, neutralizing ../ and symlinks
            resolved = Path(path).resolve()
            
            # Define allowed root directories (Enterprise boundaries)
            allowed_roots = [
                config.paths.data_dir.resolve(),
                config.paths.base_dir.resolve(), 
                Path("/tmp").resolve(), # For scratch files
            ]
            
            # Check if resolved path is within any allowed root
            is_valid = any(
                str(resolved).startswith(str(root)) for root in allowed_roots
            )
            
            if not is_valid:
                logger.warning("🛡️ Path Traversal Blocked (SEC-05): %s -> %s", path, resolved)
                return "", False
                
            return str(resolved), True
        except Exception as e:
            logger.warning("🛡️ Path Resolution Failed: %s (%s)", path, e)
            return "", False

    def sanitize_for_sql(self, text: str) -> Tuple[str, bool]:
        """Check for SQL injection attempts."""
        for pattern in self.SQL_INJECTION:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning("🛡️ SQL Injection Attempt: %s", pattern)
                return "", False
        return text, True

    def validate_file_content(self, content: str) -> bool:
        """Returns True if file content is safe. Returns False if suspicious."""
        if not content:
            return True

        if len(content) > MAX_FILE_CONTENT:
            logger.warning("File content too large: %d bytes", len(content))
            return False

        # Check for embedded shell scripts in non-script files
        suspicious_patterns = [
            r"<script[^>]*>",                    # Embedded JS
            r"#!/(bin|usr)",                      # Shebang in non-script context
            r"__import__\s*\(\s*['\"]os['\"]\)",  # Python import os
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                logger.warning("Suspicious file content pattern detected: %s", pattern)
                return False

        return True

    def validate_params(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Validate all params in a dict for size and content."""
        if not params:
            return params, True
        sanitized = {}
        for key, value in params.items():
            if isinstance(value, str):
                if len(value) > MAX_PARAM_LENGTH:
                    logger.warning("Param '%s' too long (%d), truncating", key, len(value))
                    value = value[:MAX_PARAM_LENGTH]
                # Also run through full sanitization
                _, is_safe = self.sanitize(value, max_length=MAX_PARAM_LENGTH)
                if not is_safe:
                    logger.warning("Param '%s' contains suspicious content", key)
                    return {}, False
            sanitized[key] = value
        return sanitized, True


input_sanitizer = InputSanitizer()