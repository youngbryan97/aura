"""The obvious code for a named library is the code that runs.

The sandbox already imports every module of the library it was given, while
real builtins still exist, and lays the names into the namespace. Anyone
writing code for that library still writes the import, because that is what
one writes, and got back ImportError('__import__ not found') — which reads as
the sandbox being broken rather than as the import being unnecessary.

The turn then fell through to a strategy with no library support at all, where
the only remaining way in was ``import sys``, which is banned. Live on
2026-08-28 the whole chain — read the docs, use the library, report the trial
balance — died three strategies deep on a refusal reading "banned import or
call", and the first strategy had already loaded the library.
"""

from __future__ import annotations

import json
from pathlib import Path

_LIBRARY = Path(
    "/private/tmp/claude-501/-Users-bryan--aura-live-source/"
    "7a6cdc9e-da7f-47f7-8c38-8cfadf95a75e/scratchpad/ledgerkit"
)


def _run(code: str, *, library_root: str) -> dict:
    from core.sandbox.runner import run_untrusted

    return run_untrusted(
        code, timeout=20, mem_bytes=512 * 1024 * 1024, library_root=library_root
    )


def test_the_import_anyone_would_write_works() -> None:
    if not _LIBRARY.exists():
        import pytest

        pytest.skip("the ledgerkit fixture is not on this machine")
    out = _run(
        "from ledgerkit import Ledger\n"
        "L = Ledger('acme')\n"
        "L.post('2026-03-01', debit='Accounts Receivable', credit='Revenue',"
        " amount_cents=25000)\n"
        "print(L.balance('Accounts Receivable'), L.balance('Revenue'))\n",
        library_root=str(_LIBRARY),
    )
    assert out["status"] == "ok", out
    assert out["stdout"].strip() == "25000 -25000"


def test_it_hands_back_nothing_it_has_not_already_loaded() -> None:
    """So this opens no door: the modules were imported before builtins went."""

    if not _LIBRARY.exists():
        import pytest

        pytest.skip("the ledgerkit fixture is not on this machine")
    out = _run("import os\nprint(os.getcwd())\n", library_root=str(_LIBRARY))
    assert out["status"] != "ok"
    assert "not part of the library" in json.dumps(out)


def test_without_a_library_nothing_changes() -> None:
    """A sandbox given no library keeps refusing imports outright."""

    out = _run("import json\nprint('x')\n", library_root="")
    assert out["status"] != "ok"


def test_runaway_code_still_stops() -> None:
    """The CPU limit is the real bound, and it is untouched.

    The wall-clock wait was the same number as the CPU limit plus two seconds,
    so a sandbox step could be killed for the machine being busy rather than
    for anything the code did — live on 2026-08-29 a ledger script that runs
    in 0.3 seconds was killed three times at 32.1 seconds while the resident
    27B was decoding. Loosening the wall must not loosen this.
    """

    import time

    from core.sandbox.runner import run_untrusted

    started = time.perf_counter()
    out = run_untrusted(
        "while True:\n    pass\n", timeout=3, mem_bytes=256 * 1024 * 1024
    )
    elapsed = time.perf_counter() - started
    assert out["status"] != "ok", out
    # Stopped by the CPU limit, not by the wall allowance six times larger.
    assert elapsed < 15.0, f"took {elapsed:.1f}s"


def test_ordinary_code_is_unaffected() -> None:
    from core.sandbox.runner import run_untrusted

    out = run_untrusted("print(6 * 7)", timeout=10, mem_bytes=256 * 1024 * 1024)
    assert out["status"] == "ok", out
    assert out["stdout"].strip() == "42"
