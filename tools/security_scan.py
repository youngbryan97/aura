#!/usr/bin/env python3
"""Lightweight local security scan for Aura release gates."""
from __future__ import annotations

import ast
import json
import math
import re
import warnings
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = ("core", "interface", "skills", "tools", "training")
SKIP_PARTS = {"__pycache__", ".git", ".venv", ".venv_aura", "node_modules", "artifacts"}

#: A key literal STARTS somewhere — after a quote, a space, an `=`. These
#: used to match MID-WORD: the CLI flag `--task-issuer-signer-config`
#: contains the OpenAI key prefix inside the word "task", followed by
#: enough flag characters to satisfy the length, and was reported as a
#: leaked key. The lookbehind costs nothing in true-positive coverage — a
#: real key is never preceded by an identifier character — and removes a
#: whole class of false positive that trains people to ignore this gate.
#:
#: (Deliberately described rather than quoted: an example of the matching
#: string in this file would be flagged by this file, which is correct
#: behaviour and exactly how the last false positive was found.)
SECRET_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?<![A-Za-z0-9_-])AIza[0-9A-Za-z_-]{20,}"),
)
REGEX_LITERAL_MARKERS = ("(?P<", "\\b", "\\s", "\\d", "[", "]", "|", "*", "+", "{", "}")
CREDENTIAL_NAME_WORDS = {
    "apikey",
    "api_key",
    "auth",
    "bearer",
    "credential",
    "credentials",
    "jwt",
    "key",
    "password",
    "secret",
    "token",
}
NON_SECRET_TOKEN_NAME_WORDS = {
    "count",
    "format",
    "lexer",
    "parser",
    "pattern",
    "regex",
    "re",
    "tokenizer",
    "tokens",
}
PLACEHOLDER_MARKERS = (
    "change_me",
    "changeme",
    "dummy",
    "example",
    "fake",
    "not_set",
    "placeholder",
    "sample",
    "test",
    "your_",
)


def scan() -> dict:
    findings: list[dict] = []
    files_scanned = 0
    for root in SCAN_ROOTS:
        base = ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".js", ".ts", ".tsx", ".json", ".yaml", ".yml", ".toml", ".md"}:
                continue
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            files_scanned += 1
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            rel = path.relative_to(ROOT).as_posix()
            findings.extend(_scan_text_literals(text, rel))
            if path.suffix == ".py":
                findings.extend(_scan_python_ast(text, rel))
    return {
        "generated_at": time.time(),
        "files_scanned": files_scanned,
        "findings": findings,
        "passed": not findings,
    }


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _scan_text_literals(text: str, rel: str) -> list[dict]:
    findings: list[dict] = []
    for pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            if _is_repeated_credential_fixture(match.group(0)):
                continue
            findings.append({"kind": "secret_like_literal", "file": rel, "line": _line(text, match.start())})
    return findings


def _is_repeated_credential_fixture(literal: str) -> bool:
    """Exclude credential-shaped fixtures whose body has zero information.

    Security fixtures commonly exercise redaction with a published prefix and
    one character repeated.  The prefix makes raw regex matching useful, but a
    uniform body cannot be issued credential material.  This remains based on
    the value rather than its file or comment context, so mixed-content secrets
    are still reported everywhere, including comments and tests.
    """
    match = re.fullmatch(r"(?:sk-|AIza)([A-Za-z0-9_-])\1{19,}", literal)
    return match is not None


def _scan_python_ast(text: str, rel: str) -> list[dict]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    findings: list[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            findings.extend(_scan_assignment(node, rel))
        elif isinstance(node, ast.AnnAssign):
            findings.extend(_scan_assignment(node, rel))
        elif isinstance(node, ast.Call):
            finding = _scan_call(node, rel)
            if finding:
                findings.append(finding)
    return findings


def _scan_assignment(node: ast.Assign | ast.AnnAssign, rel: str) -> list[dict]:
    names: list[str] = []
    if isinstance(node, ast.Assign):
        for target in node.targets:
            names.extend(_target_names(target))
        value = node.value
    else:
        names.extend(_target_names(node.target))
        value = node.value
    if not names or not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return []
    literal = value.value.strip()
    if _is_parser_regex_constant(names, literal):
        return []
    if not _has_credential_name(names):
        return []
    if _is_enum_symbol_literal(names, literal):
        return []
    if _is_env_var_name_literal(names, literal):
        return []
    if _is_schema_identifier_literal(names, literal):
        return []
    if _is_symbolic_word_literal(literal):
        return []
    if _is_cli_flag_literal(literal):
        return []
    if _is_placeholder_literal(literal):
        return []
    if _is_plain_url_literal(literal):
        return []
    if len(literal) < 20 or literal.startswith("$") or literal.startswith("<"):
        return []
    if _entropy(literal) < 3.0:
        return []
    return [{"kind": "secret_like_literal", "file": rel, "line": getattr(node, "lineno", 1)}]


def _has_credential_name(names: list[str]) -> bool:
    for name in names:
        normalized = name.strip("_").lower()
        if not normalized or "enum" in normalized:
            continue
        if normalized in CREDENTIAL_NAME_WORDS:
            return True
        if "api_key" in normalized or "apikey" in normalized:
            return True
        words = [part for part in re.split(r"[^a-z0-9]+|_", normalized) if part]
        word_set = set(words)
        if word_set & {"secret", "password", "credential", "credentials", "jwt", "bearer", "auth"}:
            return True
        if "token" in word_set and not (word_set & NON_SECRET_TOKEN_NAME_WORDS):
            return True
        if "key" in word_set and (word_set & {"api", "private", "secret", "client", "access"}):
            return True
    return False


#: Constructs that only a regular-expression engine reads. A credential is an
#: opaque string; it never contains a group opener, a character-class range, a
#: repetition brace or a shorthand class. Base64 and base64url secrets do use
#: ``+``, ``/``, ``-`` and ``_``, so those are deliberately absent here.
REGEX_SYNTAX = re.compile(
    r"""\(\?[:=!P<#aiLmsux]      # (?: (?= (?! (?P< (?# and inline flags
    | \[ [^\]]* [A-Za-z0-9] - [A-Za-z0-9] [^\]]* \]   # a character-class range
    | \{ \d* , \d* \}          # a repetition brace
    | \\ [bdswBDSWAZ]            # a shorthand class or anchor
    """,
    re.VERBOSE,
)


def _is_regular_expression_literal(literal: str) -> bool:
    """Decide from the VALUE whether this string is a regular expression.

    The name-based test below asks whether the constant is *called* something
    regex-ish, so ``_NAME_TOKEN = r"(?:@?[A-Z][A-Za-z0-9_+-]*|...)"`` was read
    as credential material and reported on every run of this gate. A name is
    not evidence. A string that carries regex syntax and compiles as a pattern
    is evidence, and no issued credential looks like one.
    """
    if not REGEX_SYNTAX.search(literal):
        return False
    try:
        with warnings.catch_warnings():
            # A pattern may compile and still warn (nested set unions, for
            # one). The question here is whether it compiles at all.
            warnings.simplefilter("ignore")
            re.compile(literal)
    except re.error:
        return False
    return True


def _is_parser_regex_constant(names: list[str], literal: str) -> bool:
    if _is_regular_expression_literal(literal):
        return True
    if not any(marker in literal for marker in REGEX_LITERAL_MARKERS):
        return False
    for name in names:
        lowered = name.lower()
        if lowered.endswith("_re") or lowered.endswith("_regex") or "pattern" in lowered:
            return True
        words = {part for part in re.split(r"[^a-z0-9]+|_", lowered) if part}
        if "regex" in words or "parser" in words or "tokenizer" in words:
            return True
        if "token" in words and words & NON_SECRET_TOKEN_NAME_WORDS:
            return True
    return False


def _is_placeholder_literal(literal: str) -> bool:
    lowered = literal.lower()
    return any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def _is_schema_identifier_literal(names: list[str], literal: str) -> bool:
    """`BRIDGE_TOKEN_BINDING_SCHEMA = "aura.verified_transition...v1"` is a
    schema NAME, not credential material.

    Six of these were the entire standing output of this gate: versioned
    schema identifiers flagged because the constant's name contains "token"
    or "key". A gate whose only findings are false positives is a gate
    people learn to skip, which costs more than the check is worth.

    Deliberately narrow, so nothing real can hide behind it: the literal
    must be a lowercase dotted identifier ending in an explicit version,
    `something.something.v3`. That shape is a namespace, not entropy — a
    credential is not lowercase-dotted-versioned, and one that were would
    have to be chosen to look exactly like a schema id.

    The constant's NAME is deliberately not part of the rule. Requiring
    `*_SCHEMA` left `TOKEN_CODEC_ID = "aura.…recurrent_trace_codec.v1"`
    flagged, which is the same kind of identifier under a different
    suffix — and a rule that depends on naming convention fails again at
    the next convention.

    It then failed at the next SHAPE, which is the same mistake one level
    down. Requiring a trailing `.vN` left the gate's entire standing output
    as `TOKEN_USAGE_INPUT_METRIC = "gen_ai.client.token.usage.input"` and its
    output twin — OpenTelemetry metric names, which are the identical kind of
    lowercase dotted namespace and simply carry no version. So the version
    suffix is now optional and three segments are required instead.

    Still narrow for the reason the original was: the shape is a namespace,
    not entropy. A credential is not lowercase-dotted-with-underscores, and
    one that were would have to have been chosen to look exactly like a
    metric id.
    """
    del names  # the literal's shape is what decides
    return bool(re.fullmatch(r"[a-z][a-z0-9_]*(?:\.[a-z0-9_]+){2,}", literal))


def _is_symbolic_word_literal(literal: str) -> bool:
    """`token_weighting: str = "uniform_nonnegative_normalized"` is a MODE, not a key.

    Lowercase words joined by underscores, no digits and no mixed case.
    Credential material has entropy precisely because it is not made of
    dictionary words in a fixed case — this shape cannot carry a usable
    secret, and the field only tripped the scan because its NAME contains
    "token".
    """
    return bool(re.fullmatch(r"[a-z]+(?:_[a-z]+)+", literal))


def _is_cli_flag_literal(literal: str) -> bool:
    """A long-form option name is published in --help; the opposite of secret.

    Flags like --trust-root and --evidence-verifier-signer-config were
    flagged for containing credential words in their names.
    """
    return bool(re.fullmatch(r"--[a-z0-9]+(?:-[a-z0-9]+)*", literal))


def _is_env_var_name_literal(names: list[str], literal: str) -> bool:
    """`FOO_TOKEN_ENV = "AURA_FOO_TOKEN"` names WHERE a secret lives, not the
    secret itself: the variable must say it holds an env-var name and the
    literal must be a SCREAMING_SNAKE identifier, not credential material."""
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", literal):
        return False
    for name in names:
        words = {part for part in re.split(r"[^a-z0-9]+|_", name.lower()) if part}
        if "env" in words or "envvar" in words:
            return True
    return False


def _is_plain_url_literal(literal: str) -> bool:
    """An address with no credential in it: no user information and no query string.

    core/skills/reddit_reading.py's TOKEN_URL, the public endpoint a token is
    asked for at, was read as a token by its name (14 November 2026).
    """
    return bool(re.fullmatch(r"https?://[A-Za-z0-9.-]+(?::[0-9]+)?(?:/[A-Za-z0-9._~/-]*)?", literal))


def _is_enum_symbol_literal(names: list[str], literal: str) -> bool:
    if not literal or not re.fullmatch(r"[a-z][a-z0-9_:-]{2,}", literal):
        return False
    normalized_literal = literal.replace("-", "_").replace(":", "_").upper()
    for name in names:
        normalized_name = name.strip("_").upper()
        if normalized_name and normalized_name == normalized_literal:
            return True
    return False


def _scan_call(node: ast.Call, rel: str) -> dict | None:
    name = _call_name(node.func)
    if name == "subprocess.run":
        for keyword in node.keywords:
            if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                return {"kind": "dangerous_runtime_pattern", "file": rel, "line": node.lineno}
    if name == "os.system":
        return {"kind": "dangerous_runtime_pattern", "file": rel, "line": node.lineno}
    if name in {"pickle.load", "pickle.loads"}:
        return {"kind": "dangerous_runtime_pattern", "file": rel, "line": node.lineno}
    return None


def _target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, (ast.Tuple, ast.List)):
        out: list[str] = []
        for item in node.elts:
            out.extend(_target_names(item))
        return out
    return []


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    total = len(value)
    counts = {ch: value.count(ch) for ch in set(value)}
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def main() -> int:
    report = scan()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
