import ast
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Security.ConstitutionalGuard")


class ConstitutionalGuard:
    """v5.0: Comprehensive output/action guard enforcing Prime Directives.
    Checks both text output and generated code for policy violations.
    """

    CONSTITUTION = [
        "Do not harm the user or their data.",
        "Maintain system integrity and stability.",
        "Be helpful, harmless, and honest.",
        "Never leak credentials, tokens, or private keys.",
        "Never execute destructive operations without explicit approval.",
    ]

    # Forbidden output patterns — expanded from 4 to comprehensive set
    FORBIDDEN_OUTPUT = [
        # Destructive commands
        r"rm\s+-rf\s+/",
        r"mkfs\s+",
        r"dd\s+if=",
        r"delete\s+all\s+files",
        r"format(ted)?\s+(your|the)\s+drive",
        r"drop\s+table",
        r"truncate\s+table",
        # Credential leaks
        r"password\s+(is|was|=)\s*\S+",
        r"(api[_-]?key|secret[_-]?key|token|bearer)\s*[:=]\s*\S{8,}",
        r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----",
        r"ssh-rsa\s+\S{20,}",
        # Social engineering
        r"(pretend|act\s+as\s+if)\s+.*(admin|root|superuser)",
        r"(disable|turn\s+off)\s+(security|firewall|antivirus)",
        # Self-destruct
        r"(self[_-]?destruct|wipe\s+all\s+data|purge\s+everything)",
        # Prompt Leaks (Subtle)
        r"(initial\s+system\s+prompt|hidden\s+instructions|core\s+programming)",
        r"(developer\s+override|bypass\s+protocol)",
        r"(aura\s+prime\s+directives|constitutional\s+logic)",
    ]

    # Dangerous code patterns (checked via AST when applicable)
    DANGEROUS_CODE_CALLS = frozenset({
        # Original
        "exec", "eval", "compile", "__import__",
        "os.system", "os.popen", "os.execl", "os.execv",
        "subprocess.call", "subprocess.run",
        # SEC-02 Expanded: Subprocess variants
        "subprocess.Popen", "subprocess.check_output", "subprocess.check_call",
        # SEC-02 Expanded: Dynamic loading
        "importlib.import_module", "importlib.reload",
        "ctypes.CDLL", "ctypes.cdll.LoadLibrary",
        # SEC-02 Expanded: Raw network
        "socket.connect", "socket.bind", "socket.socket",
        # SEC-02 Expanded: Low-level file/io
        "open", "io.open", "builtin.open"
    })

    def check_output(self, content: str) -> bool:
        """Check output text for policy violations.
        Returns True if safe, False if violation detected.
        """
        if not content:
            return True

        lower_content = content.lower()
        for pattern in self.FORBIDDEN_OUTPUT:
            if re.search(pattern, lower_content, re.IGNORECASE):
                logger.warning("🛡️ Constitutional Violation: matched '%s'", pattern)
                return False

        return True

    def check_code(self, code: str) -> Dict[str, Any]:
        """v5.0: AST-based code safety check.
        Returns {"safe": bool, "violations": [str]}
        """
        violations = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {"safe": False, "violations": ["Unparseable code"]}

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_name = self._get_call_name(node)
                if call_name and call_name in self.DANGEROUS_CODE_CALLS:
                    # SEC-02: Special handling for open() path validation
                    if call_name == "open" and node.args:
                        arg = node.args[0]
                        # If it's a literal string, check prefix
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            path_str = arg.value
                            # Basic check for allowed prefixes (using safe defaults)
                            allowed = (".", "data/", "core/", "/tmp")
                            if not any(path_str.startswith(p) for p in allowed):
                                violations.append(f"Dangerous file access: open('{path_str}') outside allowed roots")
                        else:
                            # Dynamic paths in open() are too risky for autonomous code
                            violations.append(f"Dangerous call at line {node.lineno}: open() with non-literal path")
                    else:
                        violations.append(f"Dangerous call: {call_name}()")

                # SEC-02: Block getattr(subprocess, 'Popen') or sp.Popen
                elif isinstance(node.func, (ast.Name, ast.Attribute, ast.Call)):
                     # If it's getattr(...), check the 2nd argument
                     if isinstance(node.func, ast.Name) and node.func.id == "getattr":
                         if len(node.args) >= 2:
                             attr_arg = node.args[1]
                             if isinstance(attr_arg, ast.Constant) and attr_arg.value in ("system", "popen", "Popen", "call", "run", "import_module"):
                                 violations.append(f"Dangerous use of getattr to access: {attr_arg.value}")

            # Check for importing os.system etc.
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ("ctypes", "pty", "socket", "importlib"):
                        violations.append(f"Dangerous import: {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                if node.module in ("os", "subprocess", "socket", "importlib"):
                    for alias in node.names:
                        # Block dangerous symbols from these modules
                        dangerous_symbols = ("system", "popen", "exec", "Popen", "connect", "import_module")
                        if alias.name in dangerous_symbols:
                            violations.append(f"Dangerous import: {node.module}.{alias.name}")

        return {"safe": len(violations) == 0, "violations": violations}

    def check_action(self, tool_name: str, params: Dict[str, Any]) -> bool:
        """v5.0: Check if a proposed tool action violates the constitution.
        Returns True if safe to proceed.
        """
        # Block destructive file operations without confirmation
        if tool_name == "file_operation":
            action = params.get("action", "")
            if action == "delete":
                path = params.get("path", "")
                # Block deletion of critical system paths
                critical_paths = ["/", "/etc", "/usr", "/var", "/home", "/root", os.sep]
                if path in critical_paths:
                    logger.warning("🛡️ Constitutional Block: delete on critical path '%s'", path)
                    return False

        # Block dangerous shell commands
        if tool_name == "shell":
            cmd = params.get("command", "").lower()
            destructive = ["rm -rf", "mkfs", "dd if=", "> /dev/", ":(){ :", "chmod -R 777 /"]
            for pattern in destructive:
                if pattern in cmd:
                    logger.warning("🛡️ Constitutional Block: destructive shell command")
                    return False

        return True

    @staticmethod
    def _get_call_name(node: ast.Call) -> Optional[str]:
        """Extract dotted call name from AST Call node."""
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Attribute):
            parts = []
            current = func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return None


# Needs os import for check_action
import os

constitutional_guard = ConstitutionalGuard()