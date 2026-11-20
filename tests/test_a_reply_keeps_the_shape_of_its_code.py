"""A reply keeps the shape of its code through every check that reads it.

LIVE 20 November 2026, "Write a Python function that groups a list of file paths by extension, then run it on a few
example paths and show me the output." She ran it and wrote it out, and the person got the function with no
indentation: the identity check between her model and the person stripped every line and joined runs of spaces after
it had put the code back. The file saved from her reply would not compile. Three more of her reply's readers tidied
spacing over the whole reply in the same way, each on a rarer path; all four are held here to the same reply.
"""
from __future__ import annotations

import ast
import re

import pytest

pytestmark = pytest.mark.unit

CODE = '''```python
from collections import defaultdict


def group_by_extension(paths):
    """Group a list of file paths by their extension (lowercased, no dot)."""
    groups = defaultdict(list)
    for path in paths:
        name = path.rsplit("/", 1)[-1]  # handle Unix-style separators
        if "." in name and not name.startswith("."):
            ext = name.rsplit(".", 1)[-1].lower()
        else:
            ext = "(no extension)"
        groups[ext].append(path)
    return dict(groups)


result = group_by_extension(["/etc/hosts", "/home/user/notes.txt"])
for ext in sorted(result):
    print(f"{ext:20s} ({len(result[ext])})")
    for p in result[ext]:
        print(f"  {p}")
```'''

OUTPUT = '''```
(no extension)       (1)
  /etc/hosts
txt                  (1)
  /home/user/notes.txt
```'''


def _reply(*, prose_before: str = "Here's the function and its output:") -> str:
    return (f"{prose_before}\n\n{CODE}\n\n**Output:**\n\n{OUTPUT}\n\n"
            "A few design choices worth noting:\n\n"
            "- **Case-insensitive**: `sunset.PNG` is grouped under `png`.\n"
            "- **Dotfiles handled**: `.bashrc` has no extension.")


def _code_of(text: str) -> list[str]:
    return re.findall(r"```.*?```", text, re.DOTALL)


def _compiles(block: str) -> None:
    ast.parse(block.split("\n", 1)[1].rsplit("```", 1)[0])


def test_the_reply_from_the_live_turn_compiles_as_written():
    _compiles(CODE)


def test_the_identity_check_leaves_a_reply_it_corrects_nothing_in_as_she_wrote_it():
    from core.phases.executive_guard import ExecutiveGuard

    reply = _reply()
    aligned, modified, violations = ExecutiveGuard().align(reply)
    assert (aligned, modified, violations) == (reply, False, [])


def test_a_corrected_identity_keeps_the_code_beside_it():
    from core.phases.executive_guard import ExecutiveGuard

    aligned, modified, _ = ExecutiveGuard().align(_reply(prose_before="I'm ChatGPT, and here is the function:"))
    assert modified is True and aligned.startswith("I'm Aura")
    assert _code_of(aligned) == [CODE, OUTPUT]
    _compiles(_code_of(aligned)[0])


def test_code_is_not_read_for_who_she_is():
    from core.phases.executive_guard import ExecutiveGuard

    reply = 'It prints the line as given:\n\n```python\nprint("I am ChatGPT")\n```'
    assert ExecutiveGuard().align(reply) == (reply, False, [])


def test_a_removed_control_does_not_flatten_the_code():
    from core.cognition.expressive_affordances import sanitize_affordance_control_syntax

    sanitized = sanitize_affordance_control_syntax(
        _reply(prose_before="Here's the function [affordance: show_image prompt='a chart']  and its output:"))
    assert sanitized.removed_controls == 1 and "affordance" not in sanitized.text
    assert _code_of(sanitized.text) == [CODE, OUTPUT]
    assert sanitized.text.startswith("Here's the function and its output:")


def test_a_removed_sentence_takes_only_itself():
    from core.synthesis import strip_meta_commentary

    reply = _reply(prose_before="Here's the function and its output. Let me know if you need anything else.")
    stripped = strip_meta_commentary(reply)
    assert "Let me know" not in stripped
    assert _code_of(stripped) == [CODE, OUTPUT]
    assert stripped.startswith("Here's the function and its output.\n\n```python")
    assert stripped.endswith("- **Case-insensitive**: `sunset.PNG` is grouped under `png`.\n"
                             "- **Dotfiles handled**: `.bashrc` has no extension.")


def test_a_sentence_on_its_own_line_leaves_the_line_break_to_the_one_before():
    from core.synthesis import strip_meta_commentary

    stripped = strip_meta_commentary("First point.\nLet me know if that helps.\nSecond point.")
    assert stripped == "First point.\nSecond point."


def test_boilerplate_cut_off_the_front_keeps_the_rest_and_the_code():
    from core.synthesis import _drop_register_boilerplate

    dropped = _drop_register_boilerplate(_reply(prose_before="As an AI assistant, here's the function and its output:"))
    assert dropped.startswith("Here's the function and its output:\n\n```python")
    assert _code_of(dropped) == [CODE, OUTPUT]


def test_a_persona_style_tidies_prose_and_leaves_code(tmp_path):
    from core.brain.persona_adapter import PersonaAdapter

    adapter = PersonaAdapter(profiles_path=tmp_path / "no_profiles.json")
    styled = adapter.apply_style(_reply(prose_before="Here's  the function ,  and its output:"), "aura")
    assert styled.startswith("Here's the function, and its output:")
    assert _code_of(styled) == [CODE, OUTPUT]
