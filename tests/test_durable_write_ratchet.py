"""Ratchet: no NEW file writes may bypass the atomic write gateway.

CLAUDE.md states the rule — "All consequential file writes go through
core/runtime/file_write_gateway.py" — and nothing enforced it.

A direct write is not atomic. The old contents are truncated the moment the
file is opened and the new contents land in pieces; a crash, an OOM kill or
a liveness-sentinel SIGKILL between those two moments leaves a truncated
file where durable state used to be. This runtime is killed that way often
enough that the window is not theoretical, and the failure is silent in the
worst way: the next boot reads a valid-looking short file and carries on
with half a state.

The measured answer was good news — core/ and interface/ carry exactly two
direct writes, both crash-time forensic dumps that must NOT use the gateway
(it allocates, locks and fsyncs, and those are precisely what hang when the
loop is already wedged or the allocator is already failing). So this is not
a cleanup; it is the gate that keeps it clean.

If this fails on code you just wrote: use get_file_write_gateway() (or the
*_async methods from async code), or write to a temp path and os.replace.
If you legitimately need a raw write while the process is dying, add it to
EXEMPT_CRASH_DUMPS with the reason.
"""
from __future__ import annotations

from core.runtime.durable_write_audit import (
    EXEMPT_CRASH_DUMPS,
    EXEMPT_FOREIGN_SCRATCH,
    DirectWrite,
    scan_direct_writes,
)


class TestNoUnexemptedBypasses:
    def test_no_write_bypasses_the_gateway(self):
        report = scan_direct_writes()
        offenders = [f"{w.path}:{w.line} ({w.call} in {w.function})" for w in report.enforceable]
        assert not offenders, (
            "these writes bypass the atomic gateway:\n  " + "\n  ".join(offenders)
        )

    def test_the_exemptions_still_exist(self):
        """An exemption for code that has moved is a stale allowance, and the
        next real bypass would hide behind it."""
        found = scan_direct_writes().keys
        stale = sorted((EXEMPT_CRASH_DUMPS | EXEMPT_FOREIGN_SCRATCH) - found)
        assert not stale, f"exempted writes no longer present: {stale}"

    def test_the_exemption_list_stays_small(self):
        """Exemptions are for dying-process forensics, not convenience."""
        assert len(EXEMPT_CRASH_DUMPS) <= 4

    def test_a_foreign_scratch_write_is_one_her_runtime_never_runs(self):
        """Each module exempted as foreign scratch imports nothing of hers at its top: it runs where her code is not
        importable, which is the whole reason it may not use her gateway."""
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        for key in EXEMPT_FOREIGN_SCRATCH:
            tree = ast.parse((root / key.split("::")[0]).read_text(encoding="utf-8"))
            top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
            names = [a.name for n in top if isinstance(n, ast.Import) for a in n.names]
            names += [n.module or "" for n in top if isinstance(n, ast.ImportFrom)]
            assert not [n for n in names if n.split(".")[0] in {"core", "interface", "skills"}], key


class TestTheScannerRecognisesCompliantWrites:
    """A scanner that cries wolf gets muted, which is worse than not having
    one. An earlier version reported 153 then 13 bypasses; every one of
    those was its own false positive."""

    def test_gateway_writes_are_not_flagged(self, tmp_path):
        source = tmp_path / "core" / "x.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "def save(d):\n"
            "    file_gateway = get_file_write_gateway()\n"
            "    file_gateway.write_text(d / 'a.json', '{}', source='x')\n",
            encoding="utf-8",
        )
        assert scan_direct_writes(roots=("core",), repo_root=tmp_path).writes == []

    def test_the_temp_then_replace_idiom_is_not_flagged(self, tmp_path):
        """Writing a temp file and os.replace-ing it IS the atomic write."""
        source = tmp_path / "core" / "y.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "def save(run_dir, payload):\n"
            "    manifest_tmp = run_dir / '.m.tmp'\n"
            "    manifest_tmp.write_text(payload)\n"
            "    os.replace(manifest_tmp, run_dir / 'm.json')\n",
            encoding="utf-8",
        )
        assert scan_direct_writes(roots=("core",), repo_root=tmp_path).writes == []

    def test_a_path_join_is_judged_by_its_base(self, tmp_path):
        source = tmp_path / "core" / "z.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "def save(gateway, d):\n"
            "    gateway.write_text(d / 'nested' / 'a.json', '{}')\n",
            encoding="utf-8",
        )
        assert scan_direct_writes(roots=("core",), repo_root=tmp_path).writes == []

    def test_reads_are_not_flagged(self, tmp_path):
        source = tmp_path / "core" / "r.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "def load(p):\n"
            "    with open(p) as fh:\n"
            "        return fh.read()\n"
            "def load2(p):\n"
            "    return p.read_text()\n",
            encoding="utf-8",
        )
        assert scan_direct_writes(roots=("core",), repo_root=tmp_path).writes == []


class TestTheScannerActuallyCatchesBypasses:
    """A gate that cannot fail is not a gate."""

    def test_a_raw_write_text_is_caught(self, tmp_path):
        source = tmp_path / "core" / "bad.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "def save(state_path, payload):\n"
            "    state_path.write_text(payload)\n",
            encoding="utf-8",
        )
        found = scan_direct_writes(roots=("core",), repo_root=tmp_path).writes
        assert [w.call for w in found] == ["write_text"]
        assert found[0].function == "save"

    def test_a_raw_open_for_writing_is_caught(self, tmp_path):
        source = tmp_path / "core" / "bad2.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "def save(p, payload):\n"
            "    with open(p, 'w') as fh:\n"
            "        fh.write(payload)\n",
            encoding="utf-8",
        )
        assert len(scan_direct_writes(roots=("core",), repo_root=tmp_path).writes) == 1

    def test_append_mode_is_caught(self, tmp_path):
        source = tmp_path / "core" / "bad3.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "def log(p, line):\n"
            "    with open(p, mode='a') as fh:\n"
            "        fh.write(line)\n",
            encoding="utf-8",
        )
        assert len(scan_direct_writes(roots=("core",), repo_root=tmp_path).writes) == 1

    def test_write_bytes_is_caught(self, tmp_path):
        source = tmp_path / "core" / "bad4.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "def save(state_path, blob):\n"
            "    state_path.write_bytes(blob)\n",
            encoding="utf-8",
        )
        assert len(scan_direct_writes(roots=("core",), repo_root=tmp_path).writes) == 1


class TestBaselineKeysAreStable:
    def test_a_key_ignores_the_line_number(self):
        """A write does not become a new defect because something above it
        grew by three lines."""
        first = DirectWrite("core/a.py", 10, "write_text", "Cls.save")
        second = DirectWrite("core/a.py", 99, "write_text", "Cls.save")
        assert first.key() == second.key()

    def test_a_key_separates_different_functions(self):
        first = DirectWrite("core/a.py", 10, "write_text", "Cls.save")
        second = DirectWrite("core/a.py", 10, "write_text", "Cls.other")
        assert first.key() != second.key()


class TestTheScanWorksInsideAWorktree:
    """The skip list must be matched against the RELATIVE path.

    A worktree lives under `.claude/worktrees/`, which is a skipped part. If
    the absolute path is tested, every file in a worktree is skipped, the
    scan finds nothing, and the ratchet reads that as "everything is clean" —
    a gate that silently disables itself exactly where most work happens.
    """

    def test_a_root_under_a_skipped_directory_is_still_scanned(self, tmp_path):
        root = tmp_path / ".claude" / "worktrees" / "wt"
        (root / "core").mkdir(parents=True)
        (root / "core" / "bad.py").write_text(
            "def save(p, payload):\n    p.write_text(payload)\n", encoding="utf-8",
        )
        found = scan_direct_writes(roots=("core",), repo_root=root).writes
        assert len(found) == 1, "the scan silently skipped a worktree"

    def test_skipped_directories_inside_the_root_are_still_skipped(self, tmp_path):
        (tmp_path / "core" / "artifacts").mkdir(parents=True)
        (tmp_path / "core" / "artifacts" / "gen.py").write_text(
            "def save(p, payload):\n    p.write_text(payload)\n", encoding="utf-8",
        )
        assert scan_direct_writes(roots=("core",), repo_root=tmp_path).writes == []


class TestExemptionsAreNarrow:
    """The scanner learned two new exemptions. A scanner that over-exempts is
    worse than none, so each is pinned together with the near-miss it must
    still catch."""

    @staticmethod
    def _flagged(source: str) -> bool:
        import ast
        import textwrap

        from core.runtime.durable_write_audit import _WriteVisitor

        visitor = _WriteVisitor("probe.py")
        visitor.visit(ast.parse(textwrap.dedent(source)))
        return bool(visitor.found)

    def test_plain_write_is_still_caught(self):
        assert self._flagged("def f():\n path.write_bytes(b'x')\n")
        assert self._flagged("def f():\n with open(p, 'wb') as h: h.write(b'x')\n")

    def test_exclusive_create_is_exempt_both_spellings(self):
        """O_CREAT|O_EXCL cannot truncate — it fails when the file exists —
        and the gateway's replace semantics would lose the exclusivity that
        key minting needs."""
        two_step = (
            "import os\n"
            "def f():\n"
            " fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)\n"
            " with open(fd, 'wb') as h: h.write(b'x')\n"
        )
        inline = (
            "import os\n"
            "def f():\n"
            " with open(os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as h:\n"
            "  h.write(b'x')\n"
        )
        assert not self._flagged(two_step)
        assert not self._flagged(inline)

    def test_os_open_without_o_excl_is_still_caught(self):
        """The exemption is O_EXCL, not os.open. Without it the open DOES
        truncate, which is the whole hazard."""
        assert self._flagged(
            "import os\n"
            "def f():\n"
            " fd = os.open(p, os.O_WRONLY | os.O_CREAT, 0o600)\n"
            " with open(fd, 'wb') as h: h.write(b'x')\n"
        )

    def test_exclusive_fd_exemption_does_not_leak_across_functions(self):
        """A name is only exempt in the scope that earned it."""
        assert self._flagged(
            "import os\n"
            "def a():\n"
            " fd = os.open(p, os.O_EXCL, 0o600)\n"
            "def b():\n"
            " with open(fd, 'wb') as h: h.write(b'x')\n"
        )

    def test_tempdir_derived_paths_are_exempt(self):
        assert not self._flagged(
            "import tempfile\n"
            "from pathlib import Path\n"
            "def f():\n"
            " with tempfile.TemporaryDirectory() as t:\n"
            "  q = Path(t) / 'a.bin'\n"
            "  q.write_bytes(b'x')\n"
        )

    def test_a_real_path_beside_a_tempdir_is_still_caught(self):
        """Opening a TemporaryDirectory must not bless every write in the
        function — only the ones actually derived from it."""
        assert self._flagged(
            "import tempfile\n"
            "from pathlib import Path\n"
            "def f():\n"
            " with tempfile.TemporaryDirectory() as t:\n"
            "  real = Path('/var/db') / 'a.bin'\n"
            "  real.write_bytes(b'x')\n"
        )
