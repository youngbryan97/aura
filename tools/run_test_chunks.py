#!/usr/bin/env python3
"""Run the test suite in bounded process chunks.

Running all ~7,400 tests in a single pytest process accumulates enough
module/singleton/cache state that macOS OOM-kills the runner around the
83% mark (observed twice: SIGKILL, exit 137). One process cannot give
back what thousands of heavyweight test modules pile up.

This runner splits the test files into N chunks and runs each chunk in
its own pytest process, so memory is returned to the OS between chunks.
Failure output is preserved per chunk; the exit code is non-zero if any
chunk fails, times out, or dies on a signal — a killed chunk is a loud
failure, never a silent pass.

WHEN THE RUNNER ITSELF IS KILLED. That promise covered the chunk and not
the parent, and the gap was not theoretical: with a resident 32B holding
~18GB, ``--chunks 6`` (353 files per pytest process) took the whole process
group down. Observed three times, and each time the log held exactly one
line — the chunk header — because ``capture_output=True`` buffers the
chunk's output in the parent, and the parent was gone. A test gate that
dies silently is indistinguishable from one that never started.

Two things address that:

* A **progress file** is written before and after every chunk. If the parent
  disappears, the file names the chunk that was in flight and the memory
  that was free when it began.
* A **memory preflight** before each chunk. Marching into an OOM and leaving
  no note is the failure above; the runner now says what it has and what it
  is about to attempt, and ``--min-free-gb`` refuses rather than gambles.

Chunk size is a memory budget, not a constant. Six is right on an idle host;
it is wrong while something else holds 18GB, and the runner should say so
instead of vanishing.

The Git revision and tracked dirty marker are checked across chunks and
retries. Changed or unreadable source leaves the run failed; a retry on
another revision supplies no verdict about order dependence. The dirty
marker does not hash edits made while the checkout is already dirty.
Use a pinned worktree and leave its source unchanged during the run.
Chunks with no selected tests are excluded from passing counts. A run
whose entire selection is empty fails and records the empty chunk count.

Usage:
    python tools/run_test_chunks.py [--chunks N] [--marker "not live"]
    python tools/run_test_chunks.py --chunk-timeout 1800
    python tools/run_test_chunks.py --chunks 40 --min-free-gb 4
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_FAILED_NODE_RE = re.compile(r"^(?:FAILED|ERROR)\s+([^\s].*?)(?:\s+-\s+.*)?$")

#: Where the runner records what it is doing, so a killed parent leaves a
#: note. Deliberately outside the repo: this is scratch, not evidence.
DEFAULT_PROGRESS_FILE = Path("/tmp/aura_test_chunks_progress.log")


def free_memory_gb() -> float | None:
    """Free + inactive memory, in GB. None when it cannot be read.

    Inactive pages count: macOS reclaims them under pressure, so a chunk can
    use them. Free alone reads alarmingly low on a warm machine and would
    make the preflight cry wolf.
    """
    try:
        reading = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if reading.returncode != 0:
        return None
    page_size = 0
    pages = {}
    for line in reading.stdout.splitlines():
        if "page size of" in line:
            match = re.search(r"page size of (\d+)", line)
            if match:
                page_size = int(match.group(1))
        elif ":" in line:
            key, _, value = line.partition(":")
            digits = value.strip().rstrip(".")
            if digits.isdigit():
                pages[key.strip()] = int(digits)
    if page_size <= 0 or not {"Pages free", "Pages inactive"} <= pages.keys():
        return None
    reclaimable = pages["Pages free"] + pages["Pages inactive"]
    return reclaimable * page_size / (1024**3)


def memory_refusal(label: str, free_gb: float | None, min_free_gb: float) -> str:
    """An unreadable measurement cannot satisfy a requested memory floor."""
    if min_free_gb <= 0:
        return ""
    if free_gb is None or not math.isfinite(free_gb):
        reason = "reclaimable memory could not be measured"
    elif free_gb < min_free_gb:
        reason = f"{free_gb:.1f}GB reclaimable is below the requested floor"
    else:
        return ""
    return (
        f"{label} REFUSED: {reason} (--min-free-gb {min_free_gb:.1f}). "
        "Use more, smaller chunks or free memory; a chunk that OOMs takes this runner with it."
    )


def working_tree_revision() -> str:
    """The revision the tests are about to read, or "" when unknowable.

    Includes a dirty marker, because an uncommitted edit changes what runs
    just as surely as a commit does.
    """

    try:
        head = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if head.returncode != 0:
            return ""
        revision = head.stdout.strip()
        if not revision:
            return ""
        dirty = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if dirty.returncode != 0:
            return ""
        if dirty.stdout.strip():
            revision += "-dirty"
        return revision
    except (OSError, subprocess.SubprocessError, ValueError):
        return ""


def note_progress(path: Path | None, message: str) -> None:
    """Append one line to the progress file. Never raises.

    This is the only thing that survives the parent being killed, so it
    flushes immediately and tolerates every filesystem failure — a runner
    that crashed because its own logging failed would be a poor joke.
    """
    if path is None:
        return
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%H:%M:%S')} {message}\n")
            handle.flush()
    except OSError:
        pass


def _is_valid_pytest_node_id(value: str) -> bool:
    node_id = str(value or "").strip()
    if not node_id:
        return False
    if node_id in {"-", "=", "FAILED", "ERROR"}:
        return False
    if node_id.startswith(("=", "_", "short", "warnings", "summary")):
        return False
    return ".py" in node_id and not any(ch.isspace() for ch in node_id)


def parse_failed_node_ids(output: str) -> list[str]:
    """Extract concrete pytest node ids from summary lines.

    A malformed empty node id must never be retried: ``pytest ""`` is
    interpreted as a whole-suite invocation, which is unsafe on this repo.
    """

    ids: list[str] = []
    for line in str(output or "").splitlines():
        match = _FAILED_NODE_RE.match(line.strip())
        if not match:
            continue
        node_id = match.group(1).strip()
        if _is_valid_pytest_node_id(node_id):
            ids.append(node_id)
    return ids


_FILE_LINE_RE = re.compile(r"^(\S+\.py) ([.FEsxX]*)")
_MORE_RESULTS_RE = re.compile(r"^([.FEsxX]+)\s*(?:\[\s*\d+%\])?$")


def what_a_killed_chunk_reached(
    partial: str, files: list[Path]
) -> tuple[list[str], str, list[Path]]:
    """Read a killed chunk's output: files that failed, the file in flight, the files never started.

    pytest prints a file's path when its first test starts and a mark per test
    after it, wrapping long files onto lines of marks alone. A chunk killed
    at its timeout never reaches the ``-rfE`` summary, so these lines are the
    only record of what it found. On 14 November 2026 the runner kept the last
    twelve of them: chunk 1 of 6 ran 88% of 912 files in ninety minutes and
    every failure it saw was lost.

    A file has started when its line is printed, whatever its place in the
    list: pytest does not run the files in the order it is given them (that
    chunk printed its 75th file beside its 597th).
    """
    reached: list[str] = []
    failing: set[str] = set()
    for line in str(partial or "").splitlines():
        started = _FILE_LINE_RE.match(line)
        if started:
            reached.append(started.group(1))
            marks = started.group(2)
        elif reached and (more := _MORE_RESULTS_RE.match(line.strip())):
            marks = more.group(1)
        else:
            continue
        if "F" in marks or "E" in marks:
            failing.add(reached[-1])
    if not reached:
        return [], "unknown", []
    started = {_position_of(name, files) for name in reached}
    at = _position_of(reached[-1], files)
    in_flight = _runnable(files[at]) if at is not None else reached[-1]
    not_started = [path for position, path in enumerate(files) if position not in started]
    failed = []
    for name in dict.fromkeys(reached):
        if name in failing:
            where = _position_of(name, files)
            failed.append(_runnable(files[where]) if where is not None else name)
    return failed, in_flight, not_started


def _position_of(printed: str, files: list[Path]) -> int | None:
    """Which file pytest meant. It prints paths relative to where it ran."""
    meant = (ROOT / printed).resolve()
    for position, path in enumerate(files):
        if path.resolve() == meant:
            return position
    return None


def _runnable(path: Path) -> str:
    """A node id the retry pass can hand to pytest from the repository root."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def discover_test_files(tests_dir: Path) -> list[Path]:
    return sorted(p for p in tests_dir.rglob("test_*.py") if p.is_file())


def split_chunks(files: list[Path], chunks: int) -> list[list[Path]]:
    chunks = max(1, min(chunks, len(files) or 1))
    size = (len(files) + chunks - 1) // chunks
    return [files[i : i + size] for i in range(0, len(files), size)]


def run_chunk(
    index: int,
    total: int,
    files: list[Path],
    *,
    marker: str,
    timeout_s: float,
    python: str,
    extra_args: list[str],
    coverage: bool = False,
    progress_file: Path | None = None,
    min_free_gb: float = 0.0,
    part: int = 1,
) -> tuple[bool | None, str, list[str]]:
    """Return a verdict, message and failures; None means no selected tests."""
    label = f"chunk {index}/{total}" + (f" part {part}" if part > 1 else "")
    # Each chunk is a FRESH INTERPRETER launched with subprocess.run, so
    # wrapping this runner in `coverage run` measures the runner and nothing
    # else — `make coverage` reported 0.00% over 419,306 statements while
    # 27,494 tests passed underneath it. `parallel`/`concurrency` in
    # pyproject.toml do not help: they cover threads and the multiprocessing
    # module, not an arbitrary child interpreter.
    #
    # So the child runs coverage itself. --parallel-mode gives every chunk its
    # own data file for `coverage combine` to merge, which is also what makes
    # the six chunks safe to write concurrently.
    cmd = [python]
    if coverage:
        cmd += ["-m", "coverage", "run", "--parallel-mode"]
    # One line per file rather than -q's bare dots: when a chunk times out,
    # the last unfinished line names the file that hung. -rfE keeps the
    # FAILED lines the retry pass parses.
    cmd += [
        "-m",
        "pytest",
        *[str(f) for f in files],
        "-rfE",
        "-p",
        "no:cacheprovider",
    ]
    if marker:
        cmd.extend(["-m", marker])
    cmd.extend(extra_args)

    started = time.monotonic()
    # Preflight. Chunk size is a memory budget: 353 files per pytest process
    # is fine on an idle host and takes the whole process group down while a
    # resident 32B holds 18GB. Say what we have before attempting it, so a
    # kill is never the first news.
    free_gb = free_memory_gb()
    free_label = f"{free_gb:.1f}GB free" if free_gb is not None else "free memory unknown"
    print(
        f"━━ {label}: {len(files)} files ━━ ({free_label})", flush=True
    )
    note_progress(
        progress_file, f"START {label} files={len(files)} {free_label}"
    )
    if message := memory_refusal(label, free_gb, min_free_gb):
        note_progress(progress_file, message)
        return False, message, []
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, timeout=timeout_s, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        # The chunk's output up to the kill is the only evidence of which
        # file hung; a timeout that discarded it (2026-09-16, chunk 2 of 40)
        # left forty minutes of run with nothing to bisect.
        partial = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
        sys.stdout.write(partial.rstrip() + "\n")
        failing, in_flight, not_started = what_a_killed_chunk_reached(partial, files)
        print(f"    in flight when killed: {in_flight}", flush=True)
        note_progress(
            progress_file,
            f"TIMEOUT {label} in_flight={in_flight} "
            f"failing_files={len(failing)} not_started={len(not_started)}",
        )
        # The file in flight is judged alone by the retry pass, which says
        # whether it hangs by itself; the files after it still get a run.
        judged_alone = failing + ([in_flight] if in_flight != "unknown" else [])
        message = f"{label} TIMEOUT after {timeout_s:.0f}s (in flight: {in_flight})"
        if not not_started:
            return False, message, judged_alone
        print(f"    {len(not_started)} file(s) never started; running them in a fresh process", flush=True)
        _, rest, rest_ids = run_chunk(
            index,
            total,
            not_started,
            marker=marker,
            timeout_s=timeout_s,
            python=python,
            extra_args=extra_args,
            coverage=coverage,
            progress_file=progress_file,
            min_free_gb=min_free_gb,
            part=part + 1,
        )
        return False, f"{message}; then {rest}", judged_alone + rest_ids
    except OSError as exc:
        message = f"{label} could not start: {type(exc).__name__}: {exc}"
        note_progress(progress_file, f"LAUNCH ERROR {message}")
        return False, message, []
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    failed_ids = parse_failed_node_ids(proc.stdout)
    elapsed = time.monotonic() - started
    note_progress(
        progress_file,
        f"END {label} rc={proc.returncode} in {elapsed:.0f}s",
    )
    if proc.returncode == 0:
        return True, f"{label} passed in {elapsed:.0f}s", []
    if proc.returncode < 0 or proc.returncode in (137, 139, 143):
        return False, (
            f"{label} KILLED (exit {proc.returncode}) after "
            f"{elapsed:.0f}s — likely OOM; a killed chunk is a failure"
        ), failed_ids
    # Empty chunks can precede a nonempty one. They neither stop the run
    # nor supply passing evidence; an entirely empty selection must fail.
    if proc.returncode == 5:
        return None, f"{label} had no selected tests", []
    return False, f"{label} FAILED (exit {proc.returncode}) after {elapsed:.0f}s", failed_ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=6)
    parser.add_argument("--marker", default="not live")
    parser.add_argument("--chunk-timeout", type=float, default=2400.0)
    parser.add_argument(
        "--retry-timeout",
        type=float,
        default=900.0,
        help="seconds one failed test may take when re-run alone",
    )
    parser.add_argument("--tests-dir", type=Path, default=ROOT / "tests")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="measure each chunk with `coverage run --parallel-mode` (combine afterwards)",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="run later chunks after a failed chunk instead of stopping immediately",
    )
    parser.add_argument(
        "--only-chunks",
        default=None,
        help="comma-separated 1-based chunk indexes to run (e.g. 5,6) — "
        "resume a partial run without repeating chunks that already passed; "
        "an explicit selection must not be empty",
    )
    parser.add_argument(
        "--defect-register",
        default="",
        help="write a machine-readable JSON defect register (order-dependence + "
        "real failures) to this path for the self-repair backlog ingestor",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=DEFAULT_PROGRESS_FILE,
        help=(
            "append a line per chunk here, so a runner killed by the OS still "
            "names the chunk that was in flight"
        ),
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=0.0,
        help=(
            "refuse to start a chunk or isolated retry below this much reclaimable "
            "memory, or when memory cannot be measured. 0 warns without refusing."
        ),
    )
    parser.add_argument("extra", nargs="*", help="extra pytest args")
    args = parser.parse_args(argv)
    if args.chunks < 1:
        parser.error("--chunks must be a positive integer")
    for name, seconds in (("--chunk-timeout", args.chunk_timeout), ("--retry-timeout", args.retry_timeout)):
        if not math.isfinite(seconds) or seconds <= 0:
            parser.error(f"{name} must be finite and positive")
    if not math.isfinite(args.min_free_gb) or args.min_free_gb < 0:
        parser.error("--min-free-gb must be finite and nonnegative")
    if args.only_chunks is not None and not any(part.strip() for part in args.only_chunks.split(",")):
        parser.error("--only-chunks must select at least one chunk; omit it to run all chunks")

    files = discover_test_files(args.tests_dir)
    if not files:
        print(f"no test files found under {args.tests_dir}", file=sys.stderr)
        return 2

    chunk_lists = split_chunks(files, args.chunks)
    selected_indexes: set[int] | None = None
    if args.only_chunks is not None:
        try:
            selected_indexes = {
                int(part) for part in args.only_chunks.split(",") if part.strip()
            }
        except ValueError:
            print(f"invalid --only-chunks value: {args.only_chunks!r}", file=sys.stderr)
            return 2
        invalid = {i for i in selected_indexes if not 1 <= i <= len(chunk_lists)}
        if invalid:
            print(
                f"--only-chunks indexes out of range 1..{len(chunk_lists)}: {sorted(invalid)}",
                file=sys.stderr,
            )
            return 2
    # What tree is this? A full run takes hours, and this repository is a
    # shared checkout that several agents commit to — measured 2026-09-18,
    # 26 commits landed between the first chunk and the thirty-third. Chunks
    # that ran before and after a commit tested different code, so the
    # collected failures were a mixture across 26 trees rather than a
    # register of one, and a green chunk said nothing about the tree the red
    # chunk found. The revision is recorded per chunk, and a move is said
    # out loud while there is still a run to interpret.
    started_at_revision = working_tree_revision()
    note_progress(
        args.progress_file,
        f"RUN begins revision={started_at_revision or 'unknown'} "
        f"chunks={len(chunk_lists)} files={len(files)}",
    )
    moved_at: list[tuple[str, str]] = []
    source_errors: dict[str, str] = {}
    if not started_at_revision:
        source_errors["start"] = "source revision could not be measured"
        note_progress(args.progress_file, "SOURCE UNKNOWN start; no chunks started")

    def observe_revision(phase: str) -> str:
        revision = working_tree_revision()
        if started_at_revision and not revision:
            source_errors[phase] = "source revision could not be measured"
            note_progress(args.progress_file, f"SOURCE UNKNOWN {phase}")
        elif started_at_revision and revision != started_at_revision:
            if not moved_at or moved_at[-1][1] != revision:
                moved_at.append((phase, revision))
                message = (
                    f"MOVED {phase} tree is now {revision}, run began on "
                    f"{started_at_revision} — observations tested different code"
                )
                note_progress(args.progress_file, message)
                print(f"\n⚠️  {message}\n", flush=True)
        return revision

    results: list[tuple[bool | None, str, list[str]]] = []
    for i, chunk in enumerate(chunk_lists, start=1):
        if source_errors:
            break
        if selected_indexes is not None and i not in selected_indexes:
            continue
        observe_revision(f"chunk {i}/{len(chunk_lists)}")
        if source_errors:
            break
        result = run_chunk(
            i,
            len(chunk_lists),
            chunk,
            marker=args.marker,
            timeout_s=args.chunk_timeout,
            python=args.python,
            extra_args=list(args.extra),
            coverage=args.coverage,
            progress_file=args.progress_file,
            min_free_gb=args.min_free_gb,
        )
        results.append(result)
        if result[0] is False and not args.continue_on_failure:
            print(
                f"stopping after chunk {i}/{len(chunk_lists)} failure; "
                "rerun with --continue-on-failure to collect later chunk failures",
                flush=True,
            )
            break

    after_chunks_revision = observe_revision("after chunks")

    # Isolated retry: a test that fails in-chunk but passes alone is an
    # ORDER-DEPENDENCE defect. The retry classifies it; the original batch
    # failure still blocks the run.
    all_failed_ids = sorted({fid for _, _, ids in results for fid in ids})
    order_dependent: list[str] = []
    real_failures: list[str] = []
    timed_out_alone: list[str] = []
    retry_refusals: dict[str, str] = {}
    retry_errors: dict[str, str] = {}
    if all_failed_ids:
        print(f"\n━━ isolated retry of {len(all_failed_ids)} failed test(s) ━━", flush=True)
        for fid in all_failed_ids:
            retry_cmd = [args.python, "-m", "pytest", fid, "-q", "-p", "no:cacheprovider"]
            if args.marker:
                retry_cmd.extend(["-m", args.marker])
            retry_cmd.extend(list(args.extra))
            print(f"  retry: {fid}", flush=True)
            retry_revision = observe_revision(f"before retry {fid}")
            if moved_at or source_errors or not started_at_revision or not retry_revision:
                retry_errors[fid] = "source revision changed or is unknown; no isolated comparison"
                note_progress(args.progress_file, f"RETRY ERROR {fid}: {retry_errors[fid]}")
                continue
            free_gb = free_memory_gb()
            note_progress(args.progress_file, f"RETRY {fid} free_gb={free_gb}")
            if refusal := memory_refusal(f"retry {fid}", free_gb, args.min_free_gb):
                retry_refusals[fid] = refusal
                note_progress(args.progress_file, refusal)
                print(f"    {refusal}", flush=True)
                continue
            try:
                retry = subprocess.run(
                    retry_cmd,
                    cwd=ROOT,
                    timeout=args.retry_timeout,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            except subprocess.TimeoutExpired:
                # One slow test must not take the whole register with it. On
                # 2026-09-10 the pass reached its 120th retry, one boot-contract
                # test ran past ten minutes, the exception propagated, and
                # eight hours of chunk results ended in a traceback with no
                # order-dependence verdict for any of the 137.
                print(f"    timed out alone after {args.retry_timeout}s", flush=True)
                timed_out_alone.append(fid)
                continue
            except OSError as exc:
                retry_errors[fid] = f"retry could not start: {type(exc).__name__}: {exc}"
                note_progress(args.progress_file, f"RETRY ERROR {fid}: {retry_errors[fid]}")
                continue
            observe_revision(f"after retry {fid}")
            if moved_at or source_errors:
                retry_errors[fid] = "source revision changed or is unknown during retry; no isolated comparison"
                note_progress(args.progress_file, f"RETRY ERROR {fid}: {retry_errors[fid]}")
            elif retry.returncode == 0:
                order_dependent.append(fid)
            elif retry.returncode == 1:
                real_failures.append(fid)
            else:
                retry_errors[fid] = f"pytest exited {retry.returncode}; no isolated verdict"
                note_progress(args.progress_file, f"RETRY ERROR {fid}: {retry_errors[fid]}")

    finished_at_revision = observe_revision("after retries")
    chunk_outcomes = {
        "passed": sum(ok is True for ok, _, _ in results),
        "failed": sum(ok is False for ok, _, _ in results),
        "empty": sum(ok is None for ok, _, _ in results),
    }
    unisolated_chunks = [line for ok, line, ids in results if ok is False and not ids]
    register_error: str | None = None
    if args.defect_register:
        try:
            import json as _json

            register = {
                "schema": "aura.test_defect_register.v1",
                "generated_at_unix": time.time(),
                "order_dependent": order_dependent,
                "real_failures": real_failures,
                "timed_out_alone": timed_out_alone,
                "retry_refusals": retry_refusals,
                "retry_errors": retry_errors,
                "chunk_outcomes": chunk_outcomes,
                "unisolated_chunks": unisolated_chunks,
                "source_revision": {
                    "started": started_at_revision,
                    "after_chunks": after_chunks_revision,
                    "finished": finished_at_revision,
                    "changed": bool(moved_at),
                    "measured": bool(started_at_revision and finished_at_revision and not source_errors),
                    "changes": [{"phase": phase, "revision": revision} for phase, revision in moved_at],
                    "errors": source_errors,
                },
            }
            reg_path = Path(args.defect_register)
            reg_path.parent.mkdir(parents=True, exist_ok=True)
            reg_path.write_text(_json.dumps(register, indent=2) + "\n", encoding="utf-8")
            print(f"🗂  defect register → {reg_path}", flush=True)
        except OSError as _reg_exc:
            register_error = str(_reg_exc)
            print(f"⚠️  defect register write failed: {_reg_exc}", file=sys.stderr, flush=True)

    print("\n━━ chunk summary ━━")
    if moved_at:
        # Said here as well as when it happened, because a summary read
        # hours later is where the mixture would otherwise be invisible.
        print(
            f"\n⚠️  THE TREE MOVED {len(moved_at)} time(s) during this run: began on "
            f"{started_at_revision}, "
            + ", ".join(f"{phase} on {rev}" for phase, rev in moved_at)
        )
        print(
            "    Chunks either side of a move tested different code, so this is "
            "not one register. Re-run in a worktree pinned to a revision."
        )
    elif source_errors:
        print("\n❌ source revision became unreadable; this run has no source identity verdict")
    elif started_at_revision:
        print(f"revision {started_at_revision}, unchanged at every source check")
    for ok, line, _ in results:
        print(("⚪ " if ok is None else "✅ " if ok else "❌ ") + line)
    if order_dependent:
        print(f"\n⚠️  ORDER-DEPENDENCE register ({len(order_dependent)}) — fail in-chunk, pass alone:")
        for fid in order_dependent:
            print(f"  ⚠️  {fid}")
    if timed_out_alone:
        print(f"\n⏱  timed out alone ({len(timed_out_alone)}) — no verdict either way:")
        for fid in timed_out_alone:
            print(f"  ⏱  {fid}")
    if retry_refusals:
        print(f"\n❌ {len(retry_refusals)} retry/retries refused by the memory floor; unmeasured")
        for refusal in retry_refusals.values():
            print(f"  ❌ {refusal}")
    if retry_errors:
        print(f"\n❌ {len(retry_errors)} retry/retries produced no isolated verdict:")
        for fid, error in retry_errors.items():
            print(f"  ❌ {fid}: {error}")
    if register_error is not None:
        print("\n❌ the requested defect register could not be written")
        return 1
    if moved_at or source_errors:
        print("\n❌ source identity did not hold; re-run in a worktree pinned to a revision")
        return 1
    if real_failures:
        print(f"\n❌ real failures ({len(real_failures)}) — fail in-chunk AND alone:")
        for fid in real_failures:
            print(f"  ❌ {fid}")
        return 1
    if timed_out_alone or retry_refusals or retry_errors:
        print(f"\n❌ {len(timed_out_alone) + len(retry_refusals) + len(retry_errors)} test(s) could not be judged alone")
        return 1
    if unisolated_chunks:
        print(f"\n❌ {len(unisolated_chunks)} chunk(s) failed without isolatable test ids")
        return 1
    if order_dependent:
        print(f"\n❌ {len(order_dependent)} order-dependence defect(s) still fail the run")
        return 1
    if not chunk_outcomes["passed"]:
        print("\n❌ no selected tests ran; this run has no passing verdict")
        return 1
    print(f"\n✅ all {chunk_outcomes['passed']} chunks passed")
    if chunk_outcomes["empty"]:
        print(f"    {chunk_outcomes['empty']} empty chunk(s) excluded from the passing count")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
