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

Usage:
    python tools/run_test_chunks.py [--chunks N] [--marker "not live"]
    python tools/run_test_chunks.py --chunk-timeout 1800
    python tools/run_test_chunks.py --chunks 40 --min-free-gb 4
"""

from __future__ import annotations

import argparse
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
        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    page_size = 16384
    pages = {}
    for line in out.splitlines():
        if "page size of" in line:
            match = re.search(r"page size of (\d+)", line)
            if match:
                page_size = int(match.group(1))
        elif ":" in line:
            key, _, value = line.partition(":")
            digits = value.strip().rstrip(".")
            if digits.isdigit():
                pages[key.strip()] = int(digits)
    reclaimable = pages.get("Pages free", 0) + pages.get("Pages inactive", 0)
    if not reclaimable:
        return None
    return reclaimable * page_size / (1024**3)


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
) -> tuple[bool, str, list[str]]:
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
    cmd += [
        "-m",
        "pytest",
        *[str(f) for f in files],
        "-q",
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
        f"━━ chunk {index}/{total}: {len(files)} files ━━ ({free_label})", flush=True
    )
    note_progress(
        progress_file, f"START chunk {index}/{total} files={len(files)} {free_label}"
    )
    if min_free_gb > 0 and free_gb is not None and free_gb < min_free_gb:
        message = (
            f"chunk {index}/{total} REFUSED: {free_gb:.1f}GB reclaimable is "
            f"below --min-free-gb {min_free_gb:.1f}. Use more, smaller chunks "
            "or free memory; a chunk that OOMs takes this runner with it."
        )
        note_progress(progress_file, message)
        return False, message, []
    try:
        proc = subprocess.run(cmd, cwd=ROOT, timeout=timeout_s, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        note_progress(progress_file, f"TIMEOUT chunk {index}/{total}")
        return False, f"chunk {index}/{total} TIMEOUT after {timeout_s:.0f}s", []
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    failed_ids = parse_failed_node_ids(proc.stdout)
    elapsed = time.monotonic() - started
    note_progress(
        progress_file,
        f"END chunk {index}/{total} rc={proc.returncode} in {elapsed:.0f}s",
    )
    if proc.returncode == 0:
        return True, f"chunk {index}/{total} passed in {elapsed:.0f}s", []
    if proc.returncode < 0 or proc.returncode in (137, 139, 143):
        return False, (
            f"chunk {index}/{total} KILLED (exit {proc.returncode}) after "
            f"{elapsed:.0f}s — likely OOM; a killed chunk is a failure"
        ), failed_ids
    # pytest exit 5 == no tests collected in this chunk (e.g. everything
    # deselected by the marker) — that is not a failure.
    if proc.returncode == 5:
        return True, f"chunk {index}/{total} had no selected tests", []
    return False, f"chunk {index}/{total} FAILED (exit {proc.returncode}) after {elapsed:.0f}s", failed_ids


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
        default="",
        help="comma-separated 1-based chunk indexes to run (e.g. 5,6) — "
        "resume a partial run without repeating chunks that already passed",
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
            "refuse to start a chunk with less than this much reclaimable "
            "memory. 0 warns without refusing."
        ),
    )
    parser.add_argument("extra", nargs="*", help="extra pytest args")
    args = parser.parse_args(argv)

    files = discover_test_files(args.tests_dir)
    if not files:
        print(f"no test files found under {args.tests_dir}", file=sys.stderr)
        return 2

    chunk_lists = split_chunks(files, args.chunks)
    selected_indexes: set[int] | None = None
    if args.only_chunks.strip():
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
    results: list[tuple[bool, str, list[str]]] = []
    for i, chunk in enumerate(chunk_lists, start=1):
        if selected_indexes is not None and i not in selected_indexes:
            continue
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
        if not result[0] and not args.continue_on_failure:
            print(
                f"stopping after chunk {i}/{len(chunk_lists)} failure; "
                "rerun with --continue-on-failure to collect later chunk failures",
                flush=True,
            )
            break

    # Isolated retry: a test that fails in-chunk but passes alone is an
    # ORDER-DEPENDENCE defect — reported loudly in its own register, but
    # only both-ways failures block the run. Pollution stays visible
    # without rotating whack-a-mole on chunk composition.
    all_failed_ids = sorted({fid for _, _, ids in results for fid in ids})
    order_dependent: list[str] = []
    real_failures: list[str] = []
    timed_out_alone: list[str] = []
    if all_failed_ids:
        print(f"\n━━ isolated retry of {len(all_failed_ids)} failed test(s) ━━", flush=True)
        for fid in all_failed_ids:
            retry_cmd = [args.python, "-m", "pytest", fid, "-q", "-p", "no:cacheprovider"]
            if args.marker:
                retry_cmd.extend(["-m", args.marker])
            retry_cmd.extend(list(args.extra))
            print(f"  retry: {fid}", flush=True)
            try:
                retry = subprocess.run(
                    retry_cmd,
                    cwd=ROOT,
                    timeout=args.retry_timeout,
                    capture_output=True,
                    text=True,
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
            if retry.returncode == 0:
                order_dependent.append(fid)
            else:
                real_failures.append(fid)

    if args.defect_register:
        try:
            import json as _json

            register = {
                "schema": "aura.test_defect_register.v1",
                "generated_at_unix": time.time(),
                "order_dependent": order_dependent,
                "real_failures": real_failures,
                "timed_out_alone": timed_out_alone,
            }
            reg_path = Path(args.defect_register)
            reg_path.parent.mkdir(parents=True, exist_ok=True)
            reg_path.write_text(_json.dumps(register, indent=2) + "\n", encoding="utf-8")
            print(f"🗂  defect register → {reg_path}", flush=True)
        except OSError as _reg_exc:
            print(f"⚠️  defect register write failed: {_reg_exc}", file=sys.stderr, flush=True)

    print("\n━━ chunk summary ━━")
    chunk_failures = 0
    for ok, line, _ids in results:
        print(("✅ " if ok else "❌ ") + line)
        if not ok:
            chunk_failures += 1
    if order_dependent:
        print(f"\n⚠️  ORDER-DEPENDENCE register ({len(order_dependent)}) — fail in-chunk, pass alone:")
        for fid in order_dependent:
            print(f"  ⚠️  {fid}")
    if timed_out_alone:
        print(f"\n⏱  timed out alone ({len(timed_out_alone)}) — no verdict either way:")
        for fid in timed_out_alone:
            print(f"  ⏱  {fid}")
    if real_failures:
        print(f"\n❌ real failures ({len(real_failures)}) — fail in-chunk AND alone:")
        for fid in real_failures:
            print(f"  ❌ {fid}")
        return 1
    if timed_out_alone:
        print(f"\n❌ {len(timed_out_alone)} test(s) could not be judged alone")
        return 1
    if chunk_failures and not all_failed_ids:
        # Chunks died without parseable test ids (timeout/OOM): loud failure.
        print(f"\n❌ {chunk_failures} chunk(s) failed without isolatable test ids")
        return 1
    if order_dependent:
        print(f"\n✅ no real failures; {len(order_dependent)} order-dependence defect(s) registered")
        return 0
    print(f"\n✅ all {len(results)} chunks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
