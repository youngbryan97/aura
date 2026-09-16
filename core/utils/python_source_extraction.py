"""Python source out of a model's reply, without trusting the wrapper.

Pure text and AST work; no runtime, no model. Lives here so the packages
that ask a model for code — the reimplementation lab, self-repair — can
share one extractor without an import edge between them.
"""

from __future__ import annotations

import ast
import logging
import re

logger = logging.getLogger("Aura.PythonSourceExtraction")

#: Fenced Python blocks in a model reply.
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(?P<code>.*?)```", re.IGNORECASE | re.DOTALL)
#: Just the opening marker, for output that never got to close it.
_OPENING_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\r?\n?", re.IGNORECASE)


def _first_pythonish_line(text: str) -> int:
    starters = (
        "from ",
        "import ",
        "class ",
        "def ",
        "async def ",
        "@",
        '"""',
        "'''",
        "#",
        "__all__",
    )
    for idx, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith(starters):
            return idx
    return 0


def extract_python_code(text: str) -> str:
    """Extract Python source from a model response without trusting wrappers."""

    raw = str(text or "").strip()
    if not raw:
        return ""

    fenced = _FENCE_RE.findall(raw)
    if fenced:
        candidates = [candidate.strip() for candidate in fenced if candidate.strip()]
        if candidates:
            return max(candidates, key=len).strip()

    # An opening fence with no closing one. The pattern requires both, so a
    # generation that ran out of tokens mid-block matched nothing and the raw
    # text — fence marker and all — went to the parser, which reported
    # "invalid syntax" on line 1 and lost an otherwise usable implementation.
    # Truncation is ordinary; throwing the whole answer away for it is not.
    opening = _OPENING_FENCE_RE.search(raw)
    if opening:
        tail = raw[opening.end():]
        closing = tail.find("```")
        body = (tail[:closing] if closing >= 0 else tail).strip()
        if body:
            return body

    lines = raw.splitlines()
    start = _first_pythonish_line(raw)
    if start:
        raw = "\n".join(lines[start:]).strip()

    # Some models append a short explanatory tail after otherwise valid code.
    # Prefer the full response if it parses; otherwise progressively trim the
    # tail until the candidate is syntactically valid.
    try:
        ast.parse(raw)
        return raw
    except SyntaxError as _exc:
        logger.debug("Suppressed %s in core.brain.llm.code_generator: %s", type(_exc).__name__, _exc)

    trimmed = raw.splitlines()
    for end in range(len(trimmed) - 1, 0, -1):
        candidate = "\n".join(trimmed[:end]).rstrip()
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue

    return raw


__all__ = ["extract_python_code"]
