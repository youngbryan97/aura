"""The sandbox bans `sys`, and sys.path is the only way to import a directory.

So "read the docs at this path, then use it" was impossible by construction:
every attempt came back "banned import or call" for doing the one thing the
request asked for. The ban is right — `sys` is the interpreter — and the fix is
not to weaken it but to make the capability a declared, checked thing instead of
something a program has to smuggle in.

The runner already hands the sandbox ready-made names it imported while real
imports still existed. A library the person named goes over the same way: the
path is checked before it gets there, the one line that touches the interpreter
is in the runner rather than inside a program nobody has read, and the sandboxed
code does not import anything at all.
"""

from __future__ import annotations

from pathlib import Path

from core.skills.code_repl import (
    _a_library_the_person_named,
    _library_path_is_allowed,
)

_KIT = (
    "/private/tmp/claude-501/-Users-bryan--aura-live-source/"
    "7a6cdc9e-da7f-47f7-8c38-8cfadf95a75e/scratchpad/ledgerkit"
)


def _the_kit_is_there() -> bool:
    """Whether the fixture library is actually present, not merely its folder.

    ``Path(_KIT).is_dir()`` was the guard, and the directory outlived its
    contents: another session cleaned the kit and left ``__pycache__`` behind,
    so the folder existed, the guard passed, and three tests ran against an
    empty library and failed on the sandbox returning "error". A skip that
    checks for the wrong thing is worse than no skip, because the failure it
    produces looks like a defect in the code under test.
    """
    root = Path(_KIT)
    return root.is_dir() and any(
        path.suffix == ".py" and path.name != "__init__.py"
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def test_a_directory_of_python_is_allowed() -> None:
    if not _the_kit_is_there():
        return
    allowed, why = _library_path_is_allowed(_KIT)
    assert allowed, why


def test_the_runtimes_own_source_is_not() -> None:
    """A sandbox meant to be separate from the runtime does not import it."""

    allowed, why = _library_path_is_allowed("core")
    assert not allowed
    assert "runtime" in why


def test_something_that_is_not_a_directory_of_python_is_not() -> None:
    import tempfile

    empty = tempfile.mkdtemp()
    allowed, why = _library_path_is_allowed(empty)
    assert not allowed and "no Python" in why

    allowed, why = _library_path_is_allowed("/nope/nothing/here")
    assert not allowed and "not a directory" in why


def test_the_path_is_taken_from_the_request_when_none_was_given() -> None:
    """A parameter only helps when the caller fills it.

    The path was in the request, this runtime resolves paths in requests for
    other reasons already, and "use the library at <path>" should not depend on
    the model remembering a field it has never seen.
    """

    if not _the_kit_is_there():
        return
    named_directly = _a_library_the_person_named(
        {"objective": f"There's a library at {_KIT} with an API.md. Use it."}
    )
    named_by_a_file = _a_library_the_person_named(
        {"objective": f"Read {_KIT}/API.md and then use the library."}
    )
    assert named_directly.endswith("ledgerkit")
    assert named_by_a_file.endswith("ledgerkit")

    assert _a_library_the_person_named({"objective": "what is 2 plus 2"}) == ""
    # The runtime's own source, built rather than spelled: a machine written
    # into a test is a test that only runs on that machine.
    own_source = Path(__file__).resolve().parents[1] / "core"
    assert _a_library_the_person_named({"objective": f"look at {own_source}"}) == ""


def test_the_runner_actually_runs_it() -> None:
    """End to end, in the sandbox, with no import in the sandboxed code."""

    if not _the_kit_is_there():
        return
    from core.sandbox.runner import run_untrusted

    out = run_untrusted(
        'book = Ledger("t")\n'
        'a = book.post("2026-08-28", "Accounts Receivable", "Revenue", 25000)\n'
        'b = book.post("2026-08-28", "Hosting Expense", "Accounts Payable", 4750)\n'
        "book.reverse(b)\n"
        "tb = book.trial_balance()\n"
        'print(sorted(tb.items()))\n'
        'print("sums to", sum(tb.values()))\n',
        timeout=20,
        library_root=_KIT,
    )
    assert out.get("status") == "ok", out.get("stderr")
    said = out.get("stdout") or ""
    assert "'Accounts Receivable', 25000" in said
    assert "'Revenue', -25000" in said
    assert "sums to 0" in said
