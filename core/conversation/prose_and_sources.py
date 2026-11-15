"""Two readings of a reply's text that the reliability checks share: where its authored prose runs, and whether
it carries a source.

Lifted from core/conversation/response_reliability.py whole, which re-exports them.
"""
from __future__ import annotations


def _authored_prose_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Return markdown spans that Aura is asserting as prose.

    Prompt-control words inside fenced code are data, not executable dialogue.
    The former detector ignored that distinction, so valid pseudocode
    containing ``state:`` was discarded as a prompt leak. Offsets are retained
    because transcript continuation must still be cut at the exact authored
    boundary.
    """

    spans: list[tuple[int, int]] = []
    offset = 0
    prose_start = 0
    fence_char = ""
    fence_width = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip(" \t")
        marker = ""
        if stripped.startswith("```"):
            marker = "`"
        elif stripped.startswith("~~~"):
            marker = "~"
        marker_width = len(stripped) - len(stripped.lstrip(marker)) if marker else 0
        is_fence = marker_width >= 3
        if fence_char:
            if is_fence and marker == fence_char and marker_width >= fence_width:
                fence_char = ""
                fence_width = 0
                prose_start = offset + len(line)
            offset += len(line)
            continue
        if is_fence:
            if prose_start < offset:
                spans.append((prose_start, offset))
            fence_char = marker
            fence_width = marker_width
            offset += len(line)
            continue
        offset += len(line)
    if not fence_char and prose_start < len(text):
        spans.append((prose_start, len(text)))
    return tuple(spans)


def _carries_source(raw: str) -> bool:
    """Whether the reply contains source code, established by parsing it.

    A word list could not see the strongest signal there is. The live
    fabrication was a Python function and the line "Output: 94867200.0", and
    the reply held no word from the context list — `print(` is not `printed`,
    and there is no reason it would be. Showing code and labelling what it
    produced is a claim that it ran, however it is worded, and whether a block
    of text is code is a question the parser answers.
    """
    try:
        import ast

        from core.brain.llm.code_generator import extract_python_code
    except ImportError:
        # not a failure: with no extractor, this text is not code as far as
        # anything here can tell, which is the answer.
        return False
    code = extract_python_code(raw)
    if not code or "\n" not in code.strip():
        # One line that happens to parse is any English sentence with a verb
        # in it; a program has more than one line, or a call in it at least.
        return False
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        # not a failure: text that will not parse is not a program, and the
        # docstring above says the parser is what answers that.
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom, ast.Call))
        for node in ast.walk(tree)
    )
