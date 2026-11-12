"""Asked for her code, she must show code that exists.

Live 2026-08-04 13:50. Asked for a snippet of her own source she found
interesting, she produced::

    def manage_load(self):
        current_load = self.get_current_load()
        ...

introduced as "a small part of my cognitive architecture". No file in this
repository contains it. Asked what module it came from she said she had
written it for the conversation — honest, one turn too late, after the
invention had already been served as her code.

The reader that answers this from real files existed the whole time. It ran
only on the DEGRADED path, after generation had already failed, so a healthy
turn never reached it and the question fell through to weights — and a model
asked for its own code will always produce something that looks like code.
"""
from __future__ import annotations

import re

from core.self.source_excerpt import source_evidence_brief, source_tree_is_readable


def test_the_brief_carries_real_files_with_line_numbers():
    brief = source_evidence_brief("show me a snippet of your code")
    assert brief
    # Every excerpt is attributed: a path with a line number.
    assert re.search(r"\b[\w/]+\.py:\d+", brief), brief[:400]


def test_the_excerpts_are_real_text_from_real_files():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    brief = source_evidence_brief("show me your code")
    cited = re.findall(r"([\w/]+\.py):(\d+)", brief)
    assert cited, "the brief cited no file at all"
    for relative, line in cited[:3]:
        path = root / relative
        assert path.is_file(), f"cited a file that does not exist: {relative}"
        assert int(line) >= 1


def test_the_excerpts_are_readings_with_their_place_and_no_orders():
    """The defect was a plausible invention, not a wrong file. What catches it is the check of what she shows against
    her tree (below); the brief says where each excerpt lives and gives no orders (F06)."""
    brief = source_evidence_brief("show me your code")
    lowered = brief.lower()
    assert "file and line" in lowered
    assert "do not" not in lowered and "must be" not in lowered


def test_nothing_is_attached_to_turns_that_are_not_about_her_code():
    from interface.routes.chat import _turn_may_concern_own_source

    for unrelated in (
        "how are you feeling?",
        "what's 17 times 4?",
        "write me a python module for sorting",
        "tell me a joke",
    ):
        assert not _turn_may_concern_own_source(unrelated), unrelated


def test_provenance_language_cannot_invent_a_missing_antecedent():
    from core.self.source_excerpt import forget_shown_excerpt
    from interface.routes.chat import (
        _turn_asks_where_that_came_from,
        _turn_may_concern_own_source,
    )

    forget_shown_excerpt()
    for unrelated in (
        "Did you make that up?",
        "Where did that come from?",
        "Was that... Was that humor?",
    ):
        assert not _turn_asks_where_that_came_from(unrelated), unrelated
        assert not _turn_may_concern_own_source(unrelated), unrelated


def test_asking_where_code_lives_counts_as_asking_about_her_source():
    """The follow-up that exposed the invention."""
    from interface.routes.chat import _turn_may_concern_own_source

    for question in (
        "What python module is that from",
        "where can it be found?",
        "Can you show me a snippet of your code and tell me where it can be found",
    ):
        assert _turn_may_concern_own_source(question), question


def test_an_unreadable_tree_says_so_rather_than_going_silent():
    """Silence would hand her the question with nothing but her weights."""
    import core.self.source_excerpt as module

    original = module.source_tree_is_readable
    module.source_tree_is_readable = lambda: False
    try:
        brief = module.source_evidence_brief("show me your code")
    finally:
        module.source_tree_is_readable = original
    assert brief, "an unreadable tree produced no note at all"
    assert "source tree: not readable" in brief.lower()  # a reading of what is so, not an instruction (F06)
    assert "nothing was read" in brief.lower()


def test_the_tree_is_readable_from_the_test_process():
    assert source_tree_is_readable()


# ─────────────────────────────────────────────────────────────────────────
# Notes can be overridden. A check cannot.
#
# The evidence brief reached the prompt on 2026-08-04 and she produced
# `retrieve_contextual_memory()` anyway — a function in no file here. So the
# claim is settled against the tree rather than trusted.
# ─────────────────────────────────────────────────────────────────────────

from core.self.source_excerpt import (  # noqa: E402
    code_blocks_in,
    reply_fabricates_own_code,
    snippet_verdict,
)

_FABRICATED = '''def retrieve_contextual_memory(self, context_key):
    if context_key in self.contextual_memories:
        return self.contextual_memories[context_key]
'''

_REAL = '''def source_tree_is_readable() -> bool:
    """Whether Aura can read her own source at all right now."""
'''


def test_the_live_fabrication_is_detected():
    verdict, _path = snippet_verdict(_FABRICATED)
    assert verdict == "absent"
    assert reply_fabricates_own_code(f"Sure:\n```python\n{_FABRICATED}```")


def test_real_code_is_found_and_attributed():
    verdict, path = snippet_verdict(_REAL)
    assert verdict == "found"
    assert path.endswith(".py")


def test_a_reply_with_no_code_is_not_accused():
    assert not reply_fabricates_own_code("I can't reach my source tree right now.")
    assert not reply_fabricates_own_code("")


def test_a_search_that_cannot_run_proves_nothing():
    """An unchecked verdict must never be read as fabrication."""
    import core.self.source_excerpt as module

    original = module._SOURCE_ROOT
    module._SOURCE_ROOT = original / "does-not-exist-anywhere"
    try:
        verdict, _ = module.snippet_verdict(_FABRICATED)
        assert verdict == "unchecked"
        assert not module.reply_fabricates_own_code(f"```python\n{_FABRICATED}```")
    finally:
        module._SOURCE_ROOT = original


def test_code_blocks_are_extracted_from_a_reply():
    blocks = code_blocks_in("text\n```python\nx = 1\n```\nmore\n```\ny = 2\n```")
    assert len(blocks) == 2


# ── denying a real excerpt is the same failure as inventing one ────────────


def test_what_she_showed_is_remembered_with_its_provenance():
    from core.self.source_excerpt import (
        forget_shown_excerpt,
        last_shown_excerpt,
        remember_shown_excerpt,
    )

    forget_shown_excerpt()
    assert last_shown_excerpt() == {}
    remembered = remember_shown_excerpt(
        "Here's a real piece of me:\n\ncore/mycelium.py:88 (_validate_route_pattern)"
        "\n\n```python\nx = 1\n```"
    )
    assert remembered
    assert remembered["relative_path"] == "core/mycelium.py"
    assert remembered["start_line"] == 88
    assert remembered["symbol"] == "_validate_route_pattern"
    forget_shown_excerpt()


def test_a_citation_to_a_file_that_does_not_exist_is_not_remembered():
    from core.self.source_excerpt import forget_shown_excerpt, remember_shown_excerpt

    forget_shown_excerpt()
    assert remember_shown_excerpt("see core/not_a_real_file_here.py:12") is None
    forget_shown_excerpt()


def test_the_follow_up_can_name_where_the_shown_code_lives():
    """Live 2026-08-04: she showed core/mycelium.py:88, then disowned it."""
    from core.self.source_excerpt import forget_shown_excerpt, remember_shown_excerpt

    forget_shown_excerpt()
    remember_shown_excerpt("core/mycelium.py:88 (_validate_route_pattern)")
    try:
        brief = source_evidence_brief("What python module is that from")
        assert "core/mycelium.py:88" in brief
        assert "read from the tree" in brief.lower()  # what is so; the instruction not to disown it is gone (F06)
    finally:
        forget_shown_excerpt()


# ── the check must not be defeated by how the code was formatted ──────────


def test_unfenced_code_is_checked_too():
    """The fabrication came back with no ``` around it and sailed through."""
    unfenced = (
        "Sure, here's a piece:\n\n"
        "def self_organize_modules(self, existing_module_data):\n"
        "    for module in existing_module_data:\n"
        "        module.reorganize()\n"
    )
    assert code_blocks_in(unfenced)
    assert reply_fabricates_own_code(unfenced)


def test_code_merely_QUOTED_in_prose_does_not_prove_it_exists():
    """This module's own comments quote the fabricated signature.

    A fixed-string search found it here and pronounced the invention
    genuine — the check has to distinguish a file CONTAINING a line from a
    file MENTIONING it.
    """
    verdict, _path = snippet_verdict(
        "def self_organize_modules(self, existing_module_data):\n"
        "    for module in existing_module_data:\n"
    )
    assert verdict == "absent"


def test_prose_with_no_code_is_never_flagged():
    assert not reply_fabricates_own_code(
        "I read it from core/mycelium.py — the routing validation logic."
    )


# ── the third form: denying she can, while holding the file ───────────────


def test_a_reply_that_shows_nothing_real_is_not_grounded():
    from core.self.source_excerpt import reply_is_grounded_in_source

    assert not reply_is_grounded_in_source(
        "I can't show you code files directly. However, I can describe the "
        "key components: the Cognitive Engine, the Memory System..."
    )
    assert not reply_is_grounded_in_source("")


def test_a_reply_that_cites_a_real_file_is_grounded():
    from core.self.source_excerpt import reply_is_grounded_in_source

    assert reply_is_grounded_in_source(
        "That came from core/mycelium.py:88 — the routing validation logic."
    )


def test_a_reply_citing_a_file_that_does_not_exist_is_not_grounded():
    from core.self.source_excerpt import reply_is_grounded_in_source

    assert not reply_is_grounded_in_source("It's in core/not_a_real_module.py:12.")


def test_a_real_excerpt_is_available_without_any_phrase_matching():
    from core.self.source_excerpt import grounded_excerpt_reply

    reply = grounded_excerpt_reply("show me how you're actually built")
    assert reply
    assert re.search(r"\b[\w/]+\.py:\d+", reply), reply[:200]
