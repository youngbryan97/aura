"""Every chat-template render prepares the transcript the same way.

Three adaptations stand between Aura's transcript and a model's chat template,
and each exists because a template raised rather than coped: one system block
at the front, the typed evidence role mapped to a wire role the template
distinguishes, tool arguments in the shape the template iterates. Composing
them by hand at each render site is how they drift apart, and they had:
`core/brain/llm/chat_format.py` composed all three at four sites, and the two
sites outside that module composed one of them.

LIVE, 2026-09-07: the latent-cortex engine called `system_first` and rendered
the result. `system_first` is where a second system message becomes a
`runtime_evidence` message, and the resident 27B's template raises "Unexpected
message role" on that role — so the call that prepared the transcript is the
one that made it unrenderable. Every boot failed warmup with
`warmup_readiness_no_text`, classified foreground_blocking, and the traceback
went to the log rather than to anybody.

So there is one preparer, `chat_format.for_this_template`, and this fails when
a render site passes anything else.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TREES = ("core", "interface", "skills", "llm", "executors", "security")

#: Render sites that do not carry an Aura transcript. Only ever shrinks.
#:
#: The vision worker renders through a multimodal processor whose template
#: takes image parts rather than a message list, and `chat_format` is written
#: for text transcripts.
ALLOWED = {
    "core/brain/llm/mlx_vision_worker.py",
}

#: The call, and enough of its first argument to see what was passed.
_RENDER = re.compile(r"\bapply_chat_template\(\s*([^,\n]*)", re.MULTILINE)


def _render_sites() -> list[tuple[str, int, str]]:
    sites = []
    for tree in TREES:
        base = ROOT / tree
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            body = path.read_text("utf-8", errors="replace")
            if "apply_chat_template(" not in body:
                continue
            relative = str(path.relative_to(ROOT))
            for match in _RENDER.finditer(body):
                first = match.group(1).strip()
                if not first or first.startswith(("*", "#", ")")):
                    # `apply_chat_template(*args, ...)` forwards somebody
                    # else's arguments and does not choose the transcript, and
                    # `...apply_chat_template()."""` is a docstring naming it.
                    continue
                line = body.count("\n", 0, match.start()) + 1
                sites.append((relative, line, first))
    return sites


def test_the_tree_still_renders_transcripts() -> None:
    """A gate over zero call sites reports green forever."""

    assert len(_render_sites()) >= 6


def test_every_render_prepares_the_transcript_the_same_way() -> None:
    offenders = [
        (path, line, first)
        for path, line, first in _render_sites()
        if path not in ALLOWED and not first.startswith("for_this_template(")
    ]
    assert not offenders, (
        "chat-template renders that prepare their own transcript: "
        + "; ".join(f"{path}:{line} -> {first}" for path, line, first in offenders)
        + " — pass core.brain.llm.chat_format.for_this_template(tokenizer, messages)"
    )


def test_the_allowlist_only_names_files_that_exist() -> None:
    missing = [name for name in sorted(ALLOWED) if not (ROOT / name).is_file()]
    assert not missing, f"allowlisted render sites that are gone: {missing}"


@pytest.mark.parametrize(
    "role", ["runtime_evidence", "developer", "model", "tool", "assistant", "user"]
)
def test_a_typed_role_reaches_a_template_that_refuses_it(role: str) -> None:
    """The preparer maps every role Aura uses onto one the template accepts."""

    from core.brain.llm.chat_format import for_this_template

    class _StrictTemplate:
        """A template that accepts what the resident 27B accepts, and no more."""

        chat_template = "{% for m in messages %}{{ m.role }}{% endfor %}"
        accepts = {"system", "user", "assistant", "tool"}

        def apply_chat_template(self, messages, **_kwargs):
            for message in messages:
                if message["role"] not in self.accepts:
                    raise ValueError(f"Unexpected message role: {message['role']}")
            return "".join(str(message["role"]) for message in messages)

    tokenizer = _StrictTemplate()
    prepared = for_this_template(
        tokenizer,
        [
            {"role": "system", "content": "authority"},
            {"role": "user", "content": "question"},
            {"role": role, "content": "body"},
        ],
    )
    tokenizer.apply_chat_template(prepared)


def test_a_second_system_message_does_not_become_an_unrenderable_role() -> None:
    """The live failure, as a test.

    `system_first` turns a second system message into a `runtime_evidence`
    message. A template that refuses that role then raises on a transcript the
    caller had just finished normalising.
    """

    from core.brain.llm.chat_format import for_this_template

    class _RefusesEvidence:
        chat_template = "{% for m in messages %}{{ m.role }}{% endfor %}"

        def apply_chat_template(self, messages, **_kwargs):
            for message in messages:
                if message["role"] not in {"system", "user", "assistant", "tool"}:
                    raise ValueError("Unexpected message role.")
            return "".join(str(message["role"]) for message in messages)

    tokenizer = _RefusesEvidence()
    prepared = for_this_template(
        tokenizer,
        [
            {"role": "system", "content": "authority"},
            {"role": "user", "content": "question"},
            {"role": "system", "content": "grounding that arrived late"},
        ],
    )
    tokenizer.apply_chat_template(prepared)
