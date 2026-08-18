"""tests/test_forensics_root_agreement.py — the writers and the readers of
Aura's forensic record must name the same directory.

Measured on 2026-08-06, before this file existed:

    reader  core/soma/source_body.py     -> ~/.aura/data/error_logs/crash   (0 files)
    writer  aura_main._install_fault_forensics -> $CWD/data/error_logs/crash (5 files)
    writer  core/resilience/stall_watchdog     -> $CWD/data/error_logs/stalls (502 files)

Crash correlation therefore reported "the previous exit was clean" after every
real death, because it was reading a directory nothing had ever written to.
Nothing failed; nothing looked wrong; the answer was simply always no. That is
the failure mode this repository is least allowed to have — an absence of
evidence rendered as evidence of absence — and it is invisible to any test that
only checks that the reader runs.

So these tests assert the property that was violated, not the code that
violated it: one resolver answers "where do forensics live", every writer and
reader goes through it, and no module reconstructs the path from the working
directory.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest

from core.utils.paths import (
    forensics_dir,
    forensics_root,
    forensics_search_dirs,
    forensics_search_roots,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Every module that writes or reads a forensic artifact. A module added here
# without being routed through core.utils.paths fails the source scan below.
_FORENSICS_MODULES = (
    "aura_main.py",
    "core/introspection/self_forensics.py",
    "core/observability/incident_narrator.py",
    "core/resilience/memory_watchdog.py",
    "core/resilience/stall_watchdog.py",
    "core/runtime/flight_recorder.py",
    "core/soma/source_body.py",
)


def test_forensics_root_is_absolute_and_independent_of_cwd(monkeypatch, tmp_path):
    """The root must not move when the process does.

    This is the actual defect: a relative path resolves against wherever the
    launcher happened to start, so the same code produced two different roots
    for the same machine.
    """
    monkeypatch.delenv("AURA_LOG_DIR", raising=False)
    before = forensics_root()
    assert before.is_absolute(), f"forensics root is relative: {before}"

    monkeypatch.chdir(tmp_path)
    after = forensics_root()
    assert after == before, (
        "forensics root moved with the working directory: "
        f"{before} -> {after}. A dump written under one and read under the "
        "other is a dump nobody finds."
    )


def test_log_dir_override_redirects_the_whole_forensic_record(monkeypatch, tmp_path):
    """One switch moves every forensic writer, so hermetic runs stay hermetic."""
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path))
    root = forensics_root()
    assert tmp_path in root.parents or root.parent == tmp_path, (
        f"AURA_LOG_DIR={tmp_path} did not capture the forensics root ({root})"
    )
    for kind in ("crash", "stalls", "memory", "flight"):
        assert forensics_dir(kind).is_relative_to(tmp_path)


def test_crash_reader_searches_every_root_that_holds_evidence(monkeypatch, tmp_path):
    """The reader must see artifacts written under either convention.

    Canonicalising the root without this would have orphaned the crash history
    already on disk — correct going forward, blind about the past.
    """
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path))
    canonical = forensics_dir("crash")
    (canonical / "faulthandler.log").write_text("boot\n", encoding="utf-8")

    found = forensics_search_dirs("crash")
    assert canonical in found
    assert all(directory.is_absolute() for directory in forensics_search_roots())


def test_source_body_crash_correlation_reads_a_directory_that_exists(monkeypatch, tmp_path):
    """The organ that answers "did I die last time" must look somewhere real."""
    from core.soma.source_body import SourceBodyAwareness

    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path))
    dirs = SourceBodyAwareness._resolve_crash_dirs(None)
    assert dirs, "crash correlation resolved to no directory at all"
    assert any(directory.is_dir() for directory in dirs), (
        f"crash correlation points only at non-existent directories: {dirs}. "
        "It would answer 'no crash' for every awakening, forever."
    )


def test_source_body_sees_a_crash_written_by_the_canonical_writer(monkeypatch, tmp_path):
    """End to end: what the writer writes, the reader must detect.

    This is the test that would have caught the original defect. It does not
    inspect a path — it writes evidence the way the runtime writes it and asks
    the reader whether it noticed.
    """
    from core.soma.source_body import SourceBodyAwareness, SourceBodySnapshot

    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path))

    # The previous awakening is *now*: every artifact already on disk predates
    # it, so anything this test detects is something this test wrote.
    previous = SourceBodySnapshot(
        boot_id="prev",
        t=time.time(),
        commit_sha="0" * 40,
        branch="main",
        dirty_digest="",
        dirty_count=0,
    )
    organ = SourceBodyAwareness()
    assert organ._previous_exit_was_abrupt(previous) is False

    crash_file = forensics_dir("crash") / "faulthandler.log"
    # A real dump, not a boot header.
    #
    # This wrote "===== boot pid=1 =====" — which is what the fault sink
    # appends when it is ARMED at startup, on every healthy boot. Asserting
    # that it means a crash is asserting the defect measured live 2026-08-18:
    # every clean restart woke her with "my previous session ended abruptly",
    # because the sink's mtime is newer than the last awakening by
    # construction.
    #
    # The test is about ROOT AGREEMENT — that the reader looks where the
    # canonical writer writes — and that is unaffected by the payload, so it
    # uses content only a fault produces.
    crash_file.write_text(
        "===== boot pid=1 =====\nFatal Python error: Segmentation fault\n"
        "Current thread 0x00007f (most recent call first):\n",
        encoding="utf-8",
    )

    assert organ._previous_exit_was_abrupt(previous) is True, (
        "a crash dump written to the canonical crash directory was not seen by "
        "the crash-correlation reader"
    )


@pytest.mark.parametrize("relative_path", _FORENSICS_MODULES)
def test_no_forensics_module_rebuilds_the_path_from_the_working_directory(relative_path):
    """No module may construct a forensics path out of a relative literal.

    Scanned rather than reviewed: this defect is invisible in review precisely
    because ``Path("data/error_logs/crash")`` reads as if it names one place.
    """
    source_path = PROJECT_ROOT / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        text = node.value
        if "error_logs" not in text:
            continue
        if text.startswith("/") or text.startswith("~"):
            continue
        # A comment or docstring mentioning the legacy layout is documentation.
        # A bare string used as a path is the bug.
        if len(text) > 120 or "\n" in text:
            continue
        offenders.append((node.lineno, text))

    assert not offenders, (
        f"{relative_path} still builds forensics paths from relative literals "
        f"{offenders}. Route them through core.utils.paths.forensics_dir() so "
        "the reader and the writer cannot disagree."
    )
