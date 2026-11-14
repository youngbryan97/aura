"""A re-answer is assembled for the person the first answer was for.

Live, 13 November 2026, 20:05: her first draft gave a stale energy figure, the runtime asked her again, and that second
pass ran with no principal bound ("relational memory withheld: no bound principal for user requests (hint was
'bryan')"): the re-answer to Bryan was assembled without what she knows of him. The call into the re-answer passed no
principal, and the re-answer defaults to none.
"""
from __future__ import annotations

import ast
from pathlib import Path

_INTO_A_TURN = {"_reanswer_when_the_runtime_contradicts_her", "_run_cognitive_engine_chat_turn"}


def test_every_call_into_her_chat_turn_names_the_person():
    unnamed = []
    for path in sorted(Path("interface/routes").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                keywords = {keyword.arg for keyword in node.keywords}
                if name in _INTO_A_TURN and "principal_id" not in keywords and None not in keywords:
                    unnamed.append(f"{path}:{node.lineno}")
    assert unnamed == []
