"""A test gate that dies silently is indistinguishable from one that never ran.

``run_test_chunks`` promised in its own docstring that "a killed chunk is a
loud failure, never a silent pass". That covered the chunk and not the
parent. With a resident 32B holding ~18GB, ``--chunks 6`` — 353 files in one
pytest process — took the whole process group down, three times, and each
time the log held exactly one line: the chunk header. ``capture_output=True``
buffers the chunk's output in the parent, and the parent was gone.

The same run at ``--chunks 40`` (54 files) passes in 68s. Chunk size is a
memory budget, not a constant, and the runner should say what it has before
it gambles.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.run_test_chunks import (  # noqa: E402
    DEFAULT_PROGRESS_FILE,
    free_memory_gb,
    note_progress,
    run_chunk,
)


class TestTheMemoryPreflight:
    def test_it_reads_a_real_number(self):
        free = free_memory_gb()
        assert free is None or free > 0.0, (
            "vm_stat parsed to a non-positive number; the preflight would "
            "refuse every chunk or none of them"
        )

    def test_it_counts_reclaimable_not_just_free(self):
        """Free alone reads alarmingly low on a warm machine.

        macOS reclaims inactive pages under pressure, so a chunk can use
        them. Counting free only would make the preflight cry wolf on any
        host that has been up for an hour.
        """
        import inspect

        source = inspect.getsource(free_memory_gb)
        assert "Pages inactive" in source

    def test_a_chunk_is_refused_rather_than_gambled(self, tmp_path):
        """The whole point: say no instead of vanishing."""
        ok, message, ids = run_chunk(
            1,
            1,
            [tmp_path / "test_nothing.py"],
            marker="",
            timeout_s=5.0,
            python=sys.executable,
            extra_args=[],
            progress_file=tmp_path / "progress.log",
            # Nothing has this much; the refusal is what is under test.
            min_free_gb=10_000.0,
        )
        assert ok is False
        assert "REFUSED" in message
        assert "min-free-gb" in message
        assert ids == []

    def test_zero_means_warn_without_refusing(self, tmp_path):
        """The default must not block anyone's run on a busy machine."""
        probe = tmp_path / "test_probe.py"
        probe.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
        ok, message, _ = run_chunk(
            1,
            1,
            [probe],
            marker="",
            timeout_s=120.0,
            python=sys.executable,
            extra_args=["-p", "no:randomly"],
            progress_file=tmp_path / "progress.log",
            min_free_gb=0.0,
        )
        assert ok is True, message


class TestTheProgressNote:
    def test_a_chunk_records_start_and_end(self, tmp_path):
        progress = tmp_path / "progress.log"
        probe = tmp_path / "test_probe.py"
        probe.write_text("def test_ok():\n    assert True\n", encoding="utf-8")

        run_chunk(
            3,
            7,
            [probe],
            marker="",
            timeout_s=120.0,
            python=sys.executable,
            extra_args=["-p", "no:randomly"],
            progress_file=progress,
        )

        lines = progress.read_text(encoding="utf-8").splitlines()
        assert any("START chunk 3/7" in line for line in lines), (
            "nothing recorded before the chunk ran; a killed parent would "
            "leave no note again"
        )
        assert any("END chunk 3/7" in line for line in lines)

    def test_the_start_note_names_the_memory_it_had(self, tmp_path):
        """So a post-mortem can tell an OOM from a crash."""
        progress = tmp_path / "progress.log"
        probe = tmp_path / "test_probe.py"
        probe.write_text("def test_ok():\n    assert True\n", encoding="utf-8")

        run_chunk(
            1,
            1,
            [probe],
            marker="",
            timeout_s=120.0,
            python=sys.executable,
            extra_args=["-p", "no:randomly"],
            progress_file=progress,
        )
        start = next(
            line
            for line in progress.read_text(encoding="utf-8").splitlines()
            if "START" in line
        )
        assert "free" in start
        assert "files=1" in start

    def test_an_unwritable_progress_path_does_not_break_the_run(self, tmp_path):
        """A runner that crashed because its own logging failed is a poor joke."""
        note_progress(tmp_path / "no" / "such" / "dir" / "p.log", "hello")

    def test_none_is_accepted(self):
        note_progress(None, "hello")

    def test_the_default_path_is_outside_the_repo(self):
        """Scratch, not evidence: it must not land in the working tree."""
        repo = Path(__file__).resolve().parent.parent
        assert repo not in DEFAULT_PROGRESS_FILE.parents


class TestATimedOutChunkNamesWhatHung:
    """2026-09-16: chunk 2 of 40 timed out at forty minutes and the runner
    threw its output away, so nothing said which of the 99 files had hung."""

    def test_the_file_in_flight_is_named(self, tmp_path, capsys):
        progress = tmp_path / "progress.log"
        quick = tmp_path / "test_a_quick.py"
        quick.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
        slow = tmp_path / "test_b_hangs.py"
        slow.write_text(
            "import time\n\ndef test_hangs():\n    time.sleep(30)\n", encoding="utf-8"
        )

        ok, message, failed = run_chunk(
            1,
            1,
            [quick, slow],
            marker="",
            timeout_s=6.0,
            python=sys.executable,
            extra_args=["-p", "no:randomly"],
            progress_file=progress,
        )

        assert not ok and not failed
        assert "test_b_hangs.py" in message, message
        note = progress.read_text(encoding="utf-8")
        assert "TIMEOUT chunk 1/1 in_flight=" in note and "test_b_hangs.py" in note
        assert "in flight when killed:" in capsys.readouterr().out
