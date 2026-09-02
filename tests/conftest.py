"""Shared pytest fixtures for Aura smoke tests."""
import asyncio
import builtins
import contextlib
import gc
import inspect
import logging
import os
import shutil
import socket
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

# Log hermeticity: keep test logging out of the live ~/.aura/logs so suite
# noise (test doubles, induced failures) never pollutes the running
# instance's aura_json.log. Set before any core import can call
# setup_logging(); PID-scoped so parallel chunk runners don't share a file.
if not os.environ.get("AURA_LOG_DIR", "").strip():
    os.environ["AURA_LOG_DIR"] = str(
        Path(tempfile.gettempdir()) / f"aura-test-logs-{os.getpid()}"
    )

# State hermeticity: redirect the central Aura home before core.config can
# construct its process-global settings object. Report-only detection was not
# enough: immune singleton tests wrote evolved cells into the user's live
# ~/.aura/data store while still passing. Tests that intentionally exercise a
# caller-supplied path continue to pass that path directly.
if not os.environ.get("AURA_PATHS__HOME_DIR", "").strip():
    os.environ["AURA_PATHS__HOME_DIR"] = str(
        Path(tempfile.gettempdir()) / f"aura-test-home-{os.getpid()}"
    )
os.environ.setdefault("AURA_TEST_LIVE_DATA_GUARD", "fail")
os.environ.setdefault("AURA_TEST_STATE_GUARD", "fail")

# Ledger hermeticity: the latent execution controller learns from live
# episode outcomes and persists them under the real data dir. Tests running
# fake episodes must never pollute that evidence; tests that exercise the
# controller construct their own instance with a tmp root.
os.environ.setdefault("AURA_EXECUTION_CONTROLLER", "0")

# Determinism: token-progress budgets adapt to LIVE machine memory pressure
# by default (the host running this suite often has a 20GB model resident).
# Pin the adaptation off so timing assertions can't drift with the
# environment; targeted tests opt back in and inject their own snapshots.
os.environ.setdefault("AURA_FIRST_TOKEN_PRESSURE_ADAPT", "0")

# Determinism: hybrid semantic retrieval would load a real MiniLM backend
# and make ranking assertions environment-dependent. Pin it off; the
# targeted rag tests opt back in with an injected fake engine.
os.environ.setdefault("AURA_SEMANTIC_RAG", "0")

_CLEANUP_TIMEOUT_S = 2.0

# Ensure the project root is on sys.path so `core.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


#: Modules that only exist on one platform. A test that dies importing one of
#: these did not fail — the platform is absent, which is a different fact.
#:
#: MLX is Metal-only, and Quartz/AppKit come from pyobjc. Nineteen tests in
#: tests/test_enterprise_hardening_fixes.py were reported as regressions on
#: every CI push for exactly this, and the list grew by two the moment import
#: order shifted, because a transitive import is not something a hand-kept
#: list can track.
#:
#: The conversion is narrow on purpose: it fires only on ModuleNotFoundError,
#: only for these names, and only when the module genuinely cannot be
#: imported here. A missing ordinary dependency still fails, loudly.
PLATFORM_ONLY_MODULES: frozenset[str] = frozenset(
    {"mlx", "mlx_lm", "Quartz", "AppKit", "objc", "pyautogui", "pynput", "mss"}
)


def _absent_platform_module(exc: BaseException) -> str:
    while exc is not None:
        if isinstance(exc, ModuleNotFoundError) and exc.name:
            root = str(exc.name).split(".", 1)[0]
            if root in PLATFORM_ONLY_MODULES:
                import importlib.util

                try:
                    if importlib.util.find_spec(root) is None:
                        return root
                except (ImportError, ValueError):
                    return root
        exc = exc.__cause__ or exc.__context__
    return ""


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Turn "this platform has no MLX" into a skip rather than a failure.

    `force_exception` rather than a bare `pytest.skip()`: raising inside a
    hook wrapper after the yield is reported as an error in the hook, not as
    the test's own outcome, so the conversion has to be handed back through
    the Result.
    """
    outcome = yield
    try:
        outcome.get_result()
    except BaseException as exc:  # noqa: BLE001 - re-raised unless it is the platform
        absent = _absent_platform_module(exc)
        if not absent:
            return
        outcome.force_exception(
            pytest.skip.Exception(f"{absent} is not available on this platform")
        )


def pytest_collection_modifyitems(config, items):
    """Keep destructive resident-model gates opt-in without recording skips."""
    if os.environ.get("AURA_RUN_RLC_RESIDENT_1P5B_GATE") == "1":
        return
    selected = []
    deselected = []
    for item in items:
        if item.get_closest_marker("resident_model") is None:
            selected.append(item)
        else:
            deselected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


#: Handles a TEST cannot leak, because no test opens or owns them.
#:
#: MLX opens its Metal library and the on-disk shader cache once per process,
#: lazily, the first time a kernel is compiled — and never closes them, because
#: they are the framework's for the life of the interpreter. The leak detector
#: blamed whichever test happened to trigger that first compilation, which made
#: the verdict depend on execution order: the same test passed when something
#: earlier had already warmed the cache and failed when it ran first.
#:
#: That is the same shape as the defects this guard exists to catch — a rule
#: ("no file opened during this test may still be open after it") that does not
#: match the contract it stands for ("this test leaked no resource"). Narrow by
#: construction: framework-owned, process-global, and unclosable from a test.
_PROCESS_GLOBAL_HANDLE_MARKERS = (
    "/mlx/lib/mlx.metallib",
    "/com.apple.metal/",
)


def _is_process_global_runtime_handle(path: str) -> bool:
    return any(marker in path for marker in _PROCESS_GLOBAL_HANDLE_MARKERS)


@dataclass(frozen=True)
class _ResourceLeakSnapshot:
    child_identities: frozenset[tuple[int, float]]
    listening_fds: frozenset[tuple[int, int, str, int]]
    open_files: frozenset[str]


# Three kinds of open file are not leaks, and counting them as such blamed
# tests for handles they never opened and could not close.
#
# 1. aura_json.log — setup_logging() installs one RotatingFileHandler and
#    holds it for the life of the process by design, and this same conftest
#    points it at a PID-scoped temp dir above so it never touches the live
#    instance's log. Whichever test triggers the first Aura import "opens"
#    a file it must not close.
# 2. __pycache__/*.pyc — CPython's import machinery, not the test. Which
#    test gets blamed depends only on which one imported a module first,
#    which is why these errors wandered between runs.
# 3. Python SOURCE under this repo — the other half of (2), and it was
#    missing. When a .pyc is absent or stale the importer opens the .py to
#    compile it, and that handle can still be in the snapshot. Measured
#    2026-08-09 on tests/test_inference_gate_tiering.py: 4-5 teardown
#    failures per run naming core/orchestrator/mixins/autonomy.py,
#    core/runtime/subprocess_gateway.py,
#    core/brain/llm/latent_cortex/action_state_key_custody.py — modules
#    those tests never touch — and a DIFFERENT set each run. Adding a
#    gc.collect() to the settle loop (see close_and_assert_clean) roughly
#    halved it, which confirms part of it is cycle-held rather than truly
#    open, and the remainder still wandered.
#
#    Narrow on purpose: only .py, only inside the repository. A test does
#    not leak handles to source files — this check exists for sqlite
#    stores, sockets and child processes, and it has never caught a real
#    defect in this class, only innocent tests.
#
# Everything else still fails, which is the point of the check.
def _is_process_lifetime_log_sink(path) -> bool:
    text = str(path)
    if os.path.basename(text) == "aura_json.log":
        return True
    if text.endswith(".pyc") and f"{os.sep}__pycache__{os.sep}" in text:
        return True
    return text.endswith(".py") and text.startswith(f"{PROJECT_ROOT}{os.sep}")


# ── Durable sqlite stores that outlive their test ─────────────────────────
#
# Measured 2026-08-06 across the full suite: the single largest error class was
# one shape repeated in five unrelated modules — AuditLog, ReceiptStore, the
# goal lifecycle store, the cognitive ledger, the code graph. Each opens ONE
# sqlite connection in its constructor and keeps it for the life of the object,
# which is the right design for a durable store, and each was created by a test
# that never called the `close()` every one of them already has. The handle,
# its `-wal` and its `-shm` then survived into later tests, and the hermetic
# guard blamed whichever test was running when it noticed.
#
# Chasing this per module means a `reset_X_for_test` entry per store, forever —
# the same allowlist treadmill the ServiceContainer guard was on. Stores that
# route through core.runtime.sqlite_support.open_tracked are closed by
# close_all_tracked() above, but that only covers what has been migrated.
#
# So this is the backstop, and it is deliberately shaped to cost nothing until
# something is actually wrong: only when the sandbox reports open database
# files does it walk the object graph for the connections holding them. In
# exchange it works for every store, migrated or not, and it can say WHICH
# object held the handle — turning "hermetic resource leak detected: three
# files" into "CodeGraph held test_graph.db", which is the difference between
# a report and a lead.
_SQLITE_SUFFIXES = ("", "-wal", "-shm", "-journal")


def _sqlite_paths_from(leaked_files: set[str]) -> set[str]:
    """Base database paths implied by a set of leaked file names."""
    # realpath both sides. sqlite reports the resolved path (/private/var/...)
    # while the leak observer reports whatever the caller opened (/var/...), so
    # an unnormalised comparison silently matches nothing and the sweeper looks
    # like it ran and found the store already closed.
    bases: set[str] = set()
    for name in leaked_files:
        for suffix in ("-wal", "-shm", "-journal"):
            if name.endswith(suffix):
                bases.add(os.path.realpath(name[: -len(suffix)]))
                break
        else:
            bases.add(os.path.realpath(name))
    return bases


def close_leaked_sqlite_connections(leaked_files: set[str]) -> list[str]:
    """Close live sqlite connections to the given files. Returns what held them.

    Walks the garbage collector because a connection nothing registered is a
    connection nothing else can find. Only called when a leak has already been
    detected, so the cost is paid once per failure rather than once per test.
    """
    import gc
    import sqlite3

    wanted = _sqlite_paths_from(leaked_files)
    if not wanted:
        return []

    holders: list[str] = []
    # Walking every live object touches deprecated attributes on third-party
    # classes (torch.distributed.reduce_op, for one) purely by looking at them.
    # That warning is about the sweeper's traversal, not about anything the
    # test did, and printing it would send readers to the wrong place.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        connections: list[Any] = [
            obj for obj in gc.get_objects() if isinstance(obj, sqlite3.Connection)
        ]
    for connection in connections:
        try:
            rows = connection.execute("PRAGMA database_list").fetchall()
        except sqlite3.Error:
            continue  # already closed, or busy — either way not ours to force
        paths = {os.path.realpath(str(row[2])) for row in rows if row and row[2]}
        if not (paths & wanted):
            continue
        # Name the owner before closing it: the referrer is the store, and the
        # store's type is the actionable half of the report.
        owners = {
            type(ref).__name__
            for ref in gc.get_referrers(connection)
            if not isinstance(ref, (dict, list, tuple, set, frozenset))
        }
        if not owners:
            owners = {
                type(holder).__name__
                for ref in gc.get_referrers(connection)
                if isinstance(ref, dict)
                for holder in gc.get_referrers(ref)
                if hasattr(holder, "__class__")
            }
        holders.append(
            f"{', '.join(sorted(owners)) or 'unknown'} -> {', '.join(sorted(paths & wanted))}"
        )
        with contextlib.suppress(sqlite3.Error):
            connection.close()
    return holders


class HermeticResourceSandbox:
    """Per-test host leak detector; never used as resource-policy evidence."""

    def __init__(self, *, root: Path):
        import psutil

        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        # Tests deliberately monkeypatch psutil's module attributes. Pin the
        # native constructors before the test body so teardown observation
        # cannot be redirected through the very double it is auditing.
        self._native_process = psutil.Process
        self._native_error = psutil.Error
        self._native_wait_procs = psutil.wait_procs
        self._leased_sockets: list[socket.socket] = []
        self.baseline = self.snapshot()

    #: Children the INTERPRETER owns, which no test opened and none can close.
    #:
    #: `multiprocessing.resource_tracker` is the one that bit: creating a single
    #: SharedMemory anywhere spawns it, it is a daemon for the life of the
    #: interpreter by design, and it is what stops shared segments leaking when
    #: a process dies. So the first test to touch shared memory was reported as
    #: leaking a child — and every later test that did the same inherited the
    #: blame, because the baseline was taken after it already existed. Measured
    #: 2026-08-06: 13 errors across shared_mem_bus, rlc_action_state_capture
    #: and shadow_kernel, none of them a leak.
    #:
    #: Matched on the child's own command line rather than its pid, so this
    #: cannot be widened by accident into "ignore children we do not like".
    _RUNTIME_OWNED_CHILD_MARKERS = (
        "multiprocessing.resource_tracker",
        "multiprocessing.semaphore_tracker",
        "multiprocessing.forkserver",
    )

    @classmethod
    def _is_runtime_owned_child(cls, child: object) -> bool:
        try:
            command = " ".join(child.cmdline())
        except Exception:  # noqa: BLE001 — a child that vanished mid-probe is not ours to judge
            return False
        return any(marker in command for marker in cls._RUNTIME_OWNED_CHILD_MARKERS)

    def snapshot(self) -> _ResourceLeakSnapshot:
        try:
            process = self._native_process(os.getpid())
            children = frozenset(
                (child.pid, float(child.create_time()))
                for child in process.children(recursive=True)
                if child.status().lower() not in {"dead", "zombie"}
                and not self._is_runtime_owned_child(child)
            )
            connections = process.net_connections(kind="inet")
            open_files = process.open_files()
        except (self._native_error, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"host leak observation unavailable: {exc}") from exc
        listeners = frozenset(
            (
                os.getpid(),
                int(connection.fd),
                str(getattr(connection.laddr, "ip", "") or ""),
                int(getattr(connection.laddr, "port", 0) or 0),
            )
            for connection in connections
            if str(connection.status).upper() == "LISTEN"
        )
        return _ResourceLeakSnapshot(
            child_identities=children,
            listening_fds=listeners,
            open_files=frozenset(
                str(item.path)
                for item in open_files
                if not _is_process_lifetime_log_sink(item.path)
                and not _is_process_global_runtime_handle(str(item.path))
            ),
        )

    #: Read-only files owned by the operating system, not by any test. The
    #: Metal shader cache is the one that actually bit: importing MLX makes the
    #: OS map MPSNDArray/MPSCore default.metallib, they stay mapped for the life
    #: of the process, and whichever test first touched the GPU was reported as
    #: leaking two framework resources it neither opened nor can close. A
    #: detector that reports things the test cannot fix teaches people to
    #: distrust the detector.
    _SYSTEM_OWNED_PREFIXES = (
        "/System/",
        "/usr/lib/",
        "/usr/share/",
        "/Library/Apple/",
        "/private/var/db/",
        # Ray's driver opens its own session logs and holds them for the life
        # of the process, by design and outside any test's reach. Same shape as
        # the Metal shader cache and the multiprocessing resource tracker: a
        # third-party runtime's own files, which the test neither opened nor
        # can close, reported as that test's leak.
        "/private/tmp/ray/",
        "/tmp/ray/",
    )

    @classmethod
    def _test_owned(cls, path: object) -> bool:
        text = str(path)
        return not text.startswith(cls._SYSTEM_OWNED_PREFIXES)

    def leaks(self) -> dict[str, set[object]]:
        current = self.snapshot()
        return {
            "children": set(current.child_identities - self.baseline.child_identities),
            "listeners": set(current.listening_fds - self.baseline.listening_fds),
            "open_files": {
                path
                for path in (current.open_files - self.baseline.open_files)
                if self._test_owned(path)
            },
        }

    @contextlib.contextmanager
    def listening_socket(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        self._leased_sockets.append(listener)
        try:
            yield listener
        finally:
            with contextlib.suppress(OSError):
                listener.close()
            with contextlib.suppress(ValueError):
                self._leased_sockets.remove(listener)

    def close_and_assert_clean(self) -> None:
        leaked_leases = [sock for sock in self._leased_sockets if sock.fileno() >= 0]
        for listener in tuple(self._leased_sockets):
            with contextlib.suppress(OSError):
                listener.close()
        self._leased_sockets.clear()

        # Settle before judging. Children already got this grace; open
        # files need it for the same reason. A process-global service with
        # a daemon writer (ontogeny's experience flusher, for one) holds a
        # sqlite handle for the length of one flush, so whether a test is
        # blamed depends only on whether teardown landed mid-write — which
        # is why these errors wandered between runs and between tests. A
        # handle that closes on its own was never a leak; a real one is
        # still open when the deadline passes.
        # Collect before judging, too. Refcounting closes a dropped file
        # object immediately; a file object caught in a REFERENCE CYCLE waits
        # for the collector, and the commonest cycle in this codebase is an
        # exception traceback — a frame holds the file, the traceback holds
        # the frame, and something held the exception. `record_degradation`
        # retains exceptions by design, so this is not exotic.
        #
        # Observed: unrelated tests failing on an open handle to a .py source
        # file they never touched — action_state_key_custody.py,
        # subprocess_gateway.py — which wandered between tests run to run.
        # An object the collector is about to close was never a leak, the
        # same reasoning already applied to the settle loop below.
        gc.collect()
        deadline = time.monotonic() + 0.75
        leaks = self.leaks()
        while (leaks["children"] or leaks["open_files"]) and time.monotonic() < deadline:
            time.sleep(0.05)
            gc.collect()
            leaks = self.leaks()

        child_pids = sorted(int(identity[0]) for identity in leaks["children"])
        if child_pids:
            handles = []
            for pid in child_pids:
                with contextlib.suppress(self._native_error):
                    handles.append(self._native_process(pid))
            for handle in handles:
                with contextlib.suppress(self._native_error):
                    handle.terminate()
            _gone, alive = self._native_wait_procs(handles, timeout=0.5)
            for handle in alive:
                with contextlib.suppress(self._native_error):
                    handle.kill()
            # Re-measure. The old code terminated every child it found and then
            # failed on the snapshot taken BEFORE doing so, so a child that was
            # successfully reaped still failed its test — and a test that
            # passes alone failed in a chunk because an earlier test's
            # slow-starting subprocess appeared inside its window. What
            # survives SIGTERM and SIGKILL is a real leak and is still
            # reported; that is the failure worth reading.
            self._native_wait_procs(handles, timeout=0.5)
            leaks = self.leaks()

        listener_leaks = list(leaks["listeners"])
        for pid, fd, _host, _port in listener_leaks:
            if int(pid) == os.getpid() and int(fd) >= 0:
                with contextlib.suppress(OSError):
                    os.close(int(fd))

        # Backstop for the dominant leak class. Five unrelated modules —
        # AuditLog, ReceiptStore, the goal lifecycle store, the cognitive
        # ledger, the code graph — each open ONE sqlite connection in their
        # constructor and keep it, which is right for a durable store, and each
        # was created by a test that never called the close() they all already
        # have. Chasing it per module is a reset-function treadmill that grows
        # by one entry per incident.
        #
        # This costs nothing until something is already wrong: only once a leak
        # has been detected does it walk the object graph for the connections
        # holding those files. It then names the HOLDER, which turns "three
        # files leaked" into "CodeGraph held test_graph.db" — a lead rather
        # than a report — and closes them so the next test starts clean.
        if leaks.get("open_files"):
            holders = close_leaked_sqlite_connections(set(leaks["open_files"]))
            if holders:
                print(f"\n[sqlite-sweeper] closed leaked stores: {'; '.join(holders)}")
                leaks = self.leaks()

        # Last look, after the sqlite sweeper has dropped its references.
        # Closing a store releases the connection object but the file handle
        # behind it can still be reachable from the cycle the sweeper just
        # broke, so the collect has to come after the sweep, not only before
        # it. Cheap: this runs only when something already looks wrong.
        if leaks.get("open_files"):
            gc.collect()
            leaks = self.leaks()

        if leaked_leases or any(leaks.values()):
            pytest.fail(
                "hermetic resource leak detected: "
                f"leased_sockets={len(leaked_leases)} leaks={leaks}"
            )


@pytest.fixture(autouse=True)
def _fresh_conversation_transcript():
    """No test inherits another test's conversation.

    ``UnifiedTranscript`` is a process-global singleton and grounded recall
    falls back to it when working memory is empty. So a test that causes any
    turn to be recorded leaves it there for every test after it, and one that
    asserts "no prior turn grounds nothing" passes or fails depending on which
    file ran before it.

    Found 2026-08-19: test_she_can_recall_her_own_position passed alone and
    failed after test_browser_pursue_is_a_closed_loop, because
    build_own_statement_recall_context found a turn from the other file's
    conversation and grounded on it. An order-dependent recall test is worse
    than none — it reports green on the run that matters and red on the run
    that does not.
    """
    try:
        from core.conversation.unified_transcript import UnifiedTranscript
    except ImportError:
        yield
        return
    UnifiedTranscript._instance = None
    try:
        yield
    finally:
        UnifiedTranscript._instance = None


@pytest.fixture(autouse=True)
def _measured_host_rates_do_not_leak():
    """One test's measured decode rate must not size the next test's answer.

    core.brain.llm.mlx_client._HOST_RATES is a process-wide dict written by
    any real generation, and every answer budget is computed from it through
    fit_the_answer_to_the_time. A test that generated against a fake client
    left a rate behind, and later tests in the same file got a different
    token budget and failed for a reason nothing in them could explain:
    test_background_primary_downgrades_timeout_and_tier and
    test_foreground_cortex_warmup_admits_live_desktop_headroom both passed
    alone and failed in their own file.

    Restored, not zeroed: a test that measures still sees its own measurement.

    The architecture index is here for the same reason and it is the one that
    actually bit: get_overview() returns nothing until the index is built and
    ~800 characters of subsystem map once it is, and the build runs on a
    background thread one test starts and another finishes. The prompt got
    longer between two tests, the answer budget is computed from how long the
    prompt takes to read, and the budget came out at 368 tokens where the test
    asked for 384.
    """
    saved_index = None
    module = None
    try:
        from core.brain.llm import mlx_client
    except ImportError:
        mlx_client = None
    try:
        from core.self import architecture_index as module

        saved_index = module._index
    except ImportError:
        module = None
    saved_rates = dict(mlx_client._HOST_RATES) if mlx_client is not None else None
    try:
        yield
    finally:
        if mlx_client is not None and saved_rates is not None:
            mlx_client._HOST_RATES.clear()
            mlx_client._HOST_RATES.update(saved_rates)
        if module is not None:
            module._index = saved_index


@pytest.fixture(autouse=True)
def hermetic_resource_sandbox(tmp_path_factory):
    # Resource observation is test infrastructure, not test-owned payload. Put
    # it beside pytest's per-test directory so exact-directory transaction
    # tests can truthfully assert that no undeclared entry exists.
    sandbox = HermeticResourceSandbox(
        root=tmp_path_factory.mktemp("resource-sandbox")
    )
    try:
        yield sandbox
    finally:
        sandbox.close_and_assert_clean()


@pytest.fixture(autouse=True)
def _live_data_write_guard(request):
    """Hermeticity: flag any Python-level write into the real ~/.aura/data.

    2026-07-12: pin tests were found appending fixture pins to the LIVE
    session-memory ledger — phantom memories Aura could recall as real.
    Report-mode by default (ledger in AURA_LOG_DIR);
    AURA_TEST_LIVE_DATA_GUARD=fail escalates to hard failures.
    """
    import builtins as _builtins

    from tests.live_data_guard import make_guarded_open

    original = _builtins.open
    _builtins.open = make_guarded_open(request.node.nodeid)
    try:
        yield
    finally:
        _builtins.open = original


_TEST_SCOPED_SERVICE_KEYS = frozenset(
    {
        "advanced_cognition",
        "aura_now",
        "being_runtime",
        "dialogue_cognition",
        "epistemic_reach",
        "relational_memory",
        "scheduler",
        "social_imagination",
        "substrate_voice_engine",
        "thought_interoception",
        "unified_felt_state",
        "unified_will",
        # Registered as a side effect of CONSTRUCTING a GlobalWorkspace, so any
        # consciousness test leaks them and the next test to evict one is
        # blamed for removing shared state it never created.
        "inhibition_manager",
        "unity_runtime",
        "unity_workspace_frame",
        "world_state",
        # Both are registered lazily on first use, by whichever test
        # first reaches for them, and intention_loop holds a sqlite
        # handle open for the life of the process while it is there.
        "intention_loop",
        "permission_guard",
        # Registered as a side effect of building a cognitive turn, so the
        # first test to run one is recorded as having added them.
        "associative_entity_memory",
        "bicameral_advisory",
        "cognitive_situation",
        "executive_authority",
        "imagination_engine",
        "spiking_active_inference",
        # Registered by constructing a StateRepository.
        "authority_gateway",
        "belief_authority",
        "constitutional_core",
        "executive_core",
        "standing_authority",
        # Registered lazily by the conversation and initiative lanes, so the
        # first test through one is blamed for adding them — and the first test
        # to evict welfare_model is blamed for removing it.
        "personality_engine",
        "unified_self",
        "welfare_model",
        "ambient_life_director",
        "decision_preference_learner",
        "subjective_choice_engine",
    }
)
_TEST_SCOPED_RESET_FUNCTIONS = (
    ("core.being.runtime", "reset_being_runtime_for_test"),
    ("core.epistemics.epistemic_reach", "reset_epistemic_reach_for_test"),
    ("core.identity.id_rag", "reset_identity_chronicle_for_test"),
    ("core.being.thought_interoception", "reset_thought_interoception_for_test"),
    ("core.being.unified_felt_state", "reset_unified_felt_state_for_test"),
    ("core.governance.will", "reset_unified_will_for_test"),
    ("aura_main", "disarm_fault_forensics"),
    # A boot test spawns the same detached sentinels a real boot does. Without
    # this they outlive the test, and the hermetic sandbox blames whichever
    # test happened to be running when they were noticed.
    ("aura_main", "reap_spawned_children"),
    # Every connection opened through the central factory, closed between
    # tests. This is the general form of the per-component sqlite leaks: the
    # registry already knows about them, nothing was asking it to let go.
    ("core.memory.db_config", "close_all_connections"),
    # Durable stores that route through core.runtime.sqlite_support.
    ("core.runtime.sqlite_support", "close_all_tracked"),
    # Audit chains hold raw append/lock fds the sqlite sweeper cannot see.
    ("core.runtime.audit_chain", "close_all_chains"),
    ("core.resilience.cognitive_ledger", "reset_cognitive_ledger_for_test"),
    ("core.resilience.inhibition_manager", "reset_inhibition_manager_for_test"),
    ("core.unity.runtime", "reset_unity_runtime_for_test"),
    ("core.social.dialogue_cognition", "reset_dialogue_cognition_for_test"),
    ("core.social.relational_memory", "reset_relational_memory_authority"),
    ("core.social.social_imagination", "reset_social_imagination_for_test"),
    ("core.voice.substrate_voice_engine", "reset_substrate_voice_engine_for_test"),
    ("core.world_state", "reset_world_state_for_test"),
    ("core.agency.intention_loop", "reset_intention_loop_for_test"),
    ("core.security.permission_guard", "reset_permission_guard_for_test"),
    ("core.memory.associative_entity_memory", "reset_associative_entity_memory_for_test"),
)


# ── Container containment ─────────────────────────────────────────────────
#
# _TEST_SCOPED_SERVICE_KEYS above is an ALLOWLIST, and an allowlist is the wrong
# shape for this problem. It requires someone to enumerate, in advance, every
# service any code path might lazily register — so the guard blames whichever
# test happened to be the first to touch a lazy registration, and the list grows
# by one name per incident forever. Measured on 2026-08-06: 61 of chunk 1's
# teardown errors were this, in symmetric added/removed pairs (one test registers
# `belief_graph`, a later test's reset evicts it and is blamed for "removing"
# shared state it never created).
#
# Containment is the general form: snapshot the whole container before the test
# and put it back afterwards. Then no test can inherit another's registrations,
# the allowlist stops needing to be complete, and order-dependence through the
# container becomes structurally impossible rather than reported after the fact.
# The mutation is still recorded — the ledger is how we know which tests touch
# global state — it just can no longer reach the next test.
#
# Tests that genuinely need a registration to outlive them mark themselves
# `mutates_global_state`, which already bypasses the guard.
_CONTAINER_STATE_ATTRS = (
    "_services",
    "_aliases",
    "_init_locks",
    "_optional_absent_breadcrumbs",
)


def _service_container_type():
    container_module = sys.modules.get("core.container")
    if container_module is None:
        return None
    return getattr(container_module, "ServiceContainer", None)


def _snapshot_service_container() -> dict[str, object] | None:
    """Copy the container's mutable class-level state, or None if unimported.

    Importing core.container here would be a side effect of its own — a test
    process that never touches the container should not acquire it because the
    guard looked. So this only snapshots what is already loaded.
    """
    container = _service_container_type()
    if container is None:
        return None
    snapshot: dict[str, object] = {}
    lock = getattr(container, "_lock", None)
    with lock if lock is not None else contextlib.nullcontext():
        for attr in _CONTAINER_STATE_ATTRS:
            value = getattr(container, attr, None)
            if isinstance(value, dict):
                snapshot[attr] = dict(value)
            elif isinstance(value, set):
                snapshot[attr] = set(value)
        snapshot["_registration_locked"] = getattr(
            container, "_registration_locked", False
        )
    return snapshot


def _restore_service_container(snapshot: dict[str, object] | None) -> None:
    """Put the container back exactly as the test found it.

    Restore in place rather than rebinding the attribute: the descriptors are
    class-level singletons that other modules hold references to, and rebinding
    would leave those references pointing at the pre-test dict.
    """
    if snapshot is None:
        return
    container = _service_container_type()
    if container is None:
        return
    lock = getattr(container, "_lock", None)
    with lock if lock is not None else contextlib.nullcontext():
        for attr in _CONTAINER_STATE_ATTRS:
            if attr not in snapshot:
                continue
            current = getattr(container, attr, None)
            saved = snapshot[attr]
            if isinstance(current, dict) and isinstance(saved, dict):
                current.clear()
                current.update(saved)
            elif isinstance(current, set) and isinstance(saved, set):
                current.clear()
                current.update(saved)
        # A test that sealed registration leaves every later test unable to
        # register anything, with an error that names the victim's service.
        container._registration_locked = bool(snapshot["_registration_locked"])


#: Config surfaces a boot profile mutates. Named explicitly: a deep copy of the
#: whole config would duplicate Paths, locks and lazily-built clients, and
#: restoring THOSE would be a worse bug than the one being fixed.
_CONFIG_SCALAR_ATTRS = ("skeletal_mode",)
_CONFIG_SECTION_ATTRS = ("security", "features")


def _snapshot_config() -> dict[str, object] | None:
    """Copy the config flags a boot profile sets, or None if unimported."""
    module = sys.modules.get("core.config")
    config = getattr(module, "config", None) if module is not None else None
    if config is None:
        return None
    snapshot: dict[str, object] = {}
    for attr in _CONFIG_SCALAR_ATTRS:
        if hasattr(config, attr):
            snapshot[attr] = getattr(config, attr)
    for section_name in _CONFIG_SECTION_ATTRS:
        section = getattr(config, section_name, None)
        if section is None or not hasattr(section, "__dict__"):
            continue
        snapshot[section_name] = dict(vars(section))
    return snapshot


def _restore_config(snapshot: dict[str, object] | None) -> None:
    """Put the boot-profile flags back.

    In place, attribute by attribute: other modules hold references to
    config.security and config.features, so rebinding the sections would leave
    those references pointing at the mutated objects.
    """
    if snapshot is None:
        return
    module = sys.modules.get("core.config")
    config = getattr(module, "config", None) if module is not None else None
    if config is None:
        return
    for attr in _CONFIG_SCALAR_ATTRS:
        if attr in snapshot:
            with contextlib.suppress(AttributeError, TypeError):
                setattr(config, attr, snapshot[attr])
    for section_name in _CONFIG_SECTION_ATTRS:
        saved = snapshot.get(section_name)
        section = getattr(config, section_name, None)
        if not isinstance(saved, dict) or section is None:
            continue
        for key, value in saved.items():
            if getattr(section, key, object()) != value:
                with contextlib.suppress(AttributeError, TypeError):
                    setattr(section, key, value)


def _snapshot_process_globals() -> dict[str, object]:
    """Copy the process-global surfaces that have a faithful restore.

    Same argument as the container: an AURA_* variable set without monkeypatch,
    a chdir that never came back, a Mock left in sys.modules under a real module
    name — each one silently reconfigures every later test, and the guard can
    only name the polluter after the damage. All three can be put back exactly,
    so put them back.
    """
    return {
        "aura_env": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("AURA_")
        },
        "cwd": os.getcwd(),
        "mocked_modules": {
            name: module
            for name, module in list(sys.modules.items())
            if name.startswith(("core.", "interface."))
            and module is not None
            and not hasattr(module, "__file__")
        },
    }


def _restore_process_globals(snapshot: dict[str, object]) -> None:
    saved_env = snapshot["aura_env"]
    assert isinstance(saved_env, dict)
    for key in [key for key in os.environ if key.startswith("AURA_")]:
        if key not in saved_env:
            os.environ.pop(key, None)
    for key, value in saved_env.items():
        if os.environ.get(key) != value:
            os.environ[key] = value

    saved_cwd = snapshot["cwd"]
    if isinstance(saved_cwd, str) and os.getcwd() != saved_cwd:
        # A test that deleted its own cwd cannot be returned to it; the next
        # test would then fail on an unrelated getcwd(). Land somewhere real.
        with contextlib.suppress(OSError):
            os.chdir(saved_cwd)

    saved_mocks = snapshot["mocked_modules"]
    assert isinstance(saved_mocks, dict)
    for name, module in list(sys.modules.items()):
        if not name.startswith(("core.", "interface.")):
            continue
        if module is None or hasattr(module, "__file__"):
            continue
        if name in saved_mocks:
            sys.modules[name] = saved_mocks[name]  # type: ignore[assignment]
        else:
            # A stand-in the test installed. Dropping it lets the next importer
            # get the real module back; leaving it rewires every later import.
            sys.modules.pop(name, None)


def _reset_test_scoped_runtime_services() -> None:
    """Close lazy test-owned organs before comparing process-global state."""
    for module_name, function_name in _TEST_SCOPED_RESET_FUNCTIONS:
        module = sys.modules.get(module_name)
        reset = getattr(module, function_name, None) if module is not None else None
        if callable(reset):
            reset()

    scheduler_module = sys.modules.get("core.scheduler")
    scheduler_type = (
        getattr(scheduler_module, "Scheduler", None)
        if scheduler_module is not None
        else None
    )
    scheduler = getattr(scheduler_type, "_instance", None)
    task = getattr(scheduler, "_main_loop_task", None)
    if task is not None and not task.done():
        task.cancel()
    if scheduler_type is not None:
        scheduler_type._instance = None

    container_module = sys.modules.get("core.container")
    container = (
        getattr(container_module, "ServiceContainer", None)
        if container_module is not None
        else None
    )
    services = getattr(container, "_services", None)
    if not isinstance(services, dict):
        return
    keys = [
        key
        for key in list(services)
        if key in _TEST_SCOPED_SERVICE_KEYS or key.startswith("environment_kernel:")
    ]
    lock = getattr(container, "_lock", None)
    if lock is None:
        for key in keys:
            services.pop(key, None)
        return
    with lock:
        for key in keys:
            services.pop(key, None)


@pytest.fixture(autouse=True)
def _environment_learning_isolation(_global_state_contamination_guard):
    """Evict environment-kernel learning services between tests.

    ``EnvironmentKernel`` registers its ``AdvancedCognitionRuntime`` into the
    process-global ServiceContainer; without eviction, every later kernel in
    the same test process reuses the first test's runtime — including its
    accumulated in-memory episodes, so learned risk climbs across tests until
    the advanced-cognition gate starts vetoing benign actions (the in-memory
    twin of the on-disk contamination fixed via AURA_ENV_RUNTIME_DIR).
    Only the kernel-registered learning instances are evicted; the rest of
    the container is untouched.
    """

    _reset_test_scoped_runtime_services()
    yield
    _reset_test_scoped_runtime_services()


@pytest.fixture(autouse=True)
def resource_observer(
    request,
    monkeypatch,
    tmp_path,
    hermetic_resource_sandbox,
    _global_state_contamination_guard,
):
    """Keep ordinary tests independent from the developer host's pressure.

    Tests that genuinely inspect hardware must opt in with ``host_observation``
    (or an existing live/hardware marker).  The ordinary path installs a
    process-wide deterministic observer so worker threads inherit the same
    facts and every pressure result is labelled ``simulated``.
    """
    # These dependencies establish teardown ordering: resource resets first,
    # state comparison second, host leak observation last.
    del hermetic_resource_sandbox, _global_state_contamination_guard

    from core.runtime.resource_observation import (
        HostResourceObserver,
        ObservationSource,
        SimulatedResourceObserver,
        resource_observer_scope,
    )
    from core.runtime.thermal import reset_thermal_cache
    from core.utils.memory_monitor import clear_memory_pressure_snapshot_cache

    runtime_root = tmp_path / "aura-runtime"
    monkeypatch.setenv(
        "AURA_MODEL_LANE_STATE_PATH",
        str(runtime_root / "model_lane_control.json"),
    )
    monkeypatch.setenv("AURA_RECEIPT_ROOT", str(runtime_root / "receipts"))
    monkeypatch.setenv("AURA_MEMORY_SNAPSHOT_CACHE_TTL_S", "0")
    monkeypatch.setenv("AURA_LANE_AUDIT_CACHE_TTL_S", "0")
    monkeypatch.setenv("AURA_TEST_RUNTIME_ROOT", str(runtime_root))
    # 2026-07-23: environment learning sidecars (world model, zero-shot
    # transfer) live under the USER-GLOBAL data dir shared with the live
    # organism. Tests were inheriting learned risk from it and writing test
    # episodes back into it. Every test gets a disposable workspace; a test
    # that genuinely needs the live store must set AURA_ENV_RUNTIME_DIR
    # itself. core/environment/runtime_workspace.py enforces the same rule
    # process-wide for import-time calls no fixture can reach.
    monkeypatch.setenv("AURA_ENV_RUNTIME_DIR", str(runtime_root / "environment_runtime"))
    # Leader-election leases default to the shared data dir, so every parallel
    # chunk — and the live runtime — resolved to the same files. A test could
    # then lose an election to an unrelated process and fail for a reason
    # nothing in it could explain, which is precisely the pass-alone /
    # fail-together shape that makes an aggregate pass count untrustworthy.
    monkeypatch.setenv("AURA_RUNTIME_LEASE_DIR", str(runtime_root / "leases"))
    # 2026-07-28: the screen blueprint reads the real macOS window server
    # in-process, so it is not reachable by the AppleScript mocks the desktop
    # suite uses — a test that mocked "what is frontmost" silently started
    # getting the answer from whatever was actually on screen. Off by default;
    # a test that is genuinely about the blueprint sets it back to "1" itself
    # (see tests/test_screen_blueprint.py).
    monkeypatch.setenv("AURA_SCREEN_BLUEPRINT", "0")

    def _reset_resource_singletons():
        from core.agency.capability_token import reset_token_store
        from core.brain.lane_admission import reset_lane_admission_controller_for_test
        from core.brain.llm.model_registry import reset_model_registry_caches_for_test
        from core.conversation.surface_delivery import reset_route_delivery
        from core.executive.authority_gateway import reset_authority_gateway
        from core.executive.standing_authority import reset_standing_authority_manager
        from core.memory.memory_write_gateway import reset_memory_write_gateway
        from core.resource.resource_governor import reset_resource_governor_for_test
        from core.runtime.control_plane import reset_runtime_control_plane
        from core.runtime.model_lane_control import reset_model_lane_controller_for_test
        from core.runtime.receipts import reset_receipt_store
        from core.runtime.runtime_pressure import reset_unified_runtime_pressure_for_test
        from core.state.state_gateway import reset_state_gateway

        reset_runtime_control_plane()
        reset_authority_gateway()
        reset_standing_authority_manager()
        reset_token_store()
        reset_unified_runtime_pressure_for_test()
        reset_lane_admission_controller_for_test()
        reset_model_lane_controller_for_test()
        reset_resource_governor_for_test()
        reset_model_registry_caches_for_test()
        reset_receipt_store()
        reset_memory_write_gateway()
        reset_state_gateway()
        # "What the route already answered" is process-global, so one test's
        # reply suppressed the next test's identical one as a duplicate:
        # test_ordinary_speech_is_not_withheld passed alone and failed in a
        # chunk, which is the order-dependence shape, not a flake.
        reset_route_delivery()

    host_markers = ("host_observation", "live", "hardware", "longrun")
    host_backed = any(request.node.get_closest_marker(name) for name in host_markers)
    if host_backed:
        observer = HostResourceObserver(
            source=ObservationSource.HOST,
            scenario_id=f"pytest-host:{request.node.nodeid}",
        )
    else:
        observer = SimulatedResourceObserver(
            scenario_id=f"pytest:{request.node.nodeid}",
        )

    clear_memory_pressure_snapshot_cache()
    reset_thermal_cache()
    _reset_resource_singletons()
    with resource_observer_scope(observer):
        yield observer
    _reset_resource_singletons()
    clear_memory_pressure_snapshot_cache()
    reset_thermal_cache()


class _RecordedCall:
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        yield self.args
        yield self.kwargs


class _CallRecorder:
    def __init__(self, result=None, *, side_effect=None):
        self.return_value = result
        self.side_effect = side_effect
        self.calls = []
        self.call_args = None

    @property
    def called(self):
        return bool(self.calls)

    @property
    def call_count(self):
        return len(self.calls)

    def __call__(self, *args, **kwargs):
        call = _RecordedCall(args, kwargs)
        self.calls.append(call)
        self.call_args = call
        if isinstance(self.side_effect, BaseException):
            raise self.side_effect
        if callable(self.side_effect):
            return self.side_effect(*args, **kwargs)
        return self.return_value

    def assert_called_once(self):
        assert len(self.calls) == 1

    def assert_called_once_with(self, *args, **kwargs):
        self.assert_called_once()
        call = self.calls[0]
        assert call.args == args
        assert call.kwargs == kwargs

    def assert_not_called(self):
        assert not self.calls


class _AsyncCallRecorder:
    def __init__(self, result=None, *, side_effect=None):
        self.return_value = result
        self.side_effect = side_effect
        self.await_args_list = []
        self.await_args = None

    @property
    def await_count(self):
        return len(self.await_args_list)

    @property
    def called(self):
        return bool(self.await_args_list)

    def __call__(self, *args, **kwargs):
        call = _RecordedCall(args, kwargs)
        self.await_args_list.append(call)
        self.await_args = call

        async def _complete():
            if isinstance(self.side_effect, BaseException):
                raise self.side_effect
            if callable(self.side_effect):
                value = self.side_effect(*args, **kwargs)
            else:
                value = self.return_value
            if inspect.isawaitable(value):
                return await value
            return value

        return _complete()

    def assert_awaited_once(self):
        assert len(self.await_args_list) == 1

    def assert_not_called(self):
        assert not self.await_args_list


class _TestStorageGateway:
    def create_dir(self, path, *, cause: str = "test"):
        Path(path).mkdir(parents=True, exist_ok=True)

    def delete(self, path, *, cause: str = "test"):
        Path(path).unlink(missing_ok=True)

    def delete_tree(self, path, *, ignore_errors: bool = True, cause: str = "test"):
        shutil.rmtree(path, ignore_errors=ignore_errors)


class _TestTaskTracker:
    def create_task(self, awaitable, *args, **kwargs):
        if not inspect.isawaitable(awaitable):
            return awaitable
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)
        return loop.create_task(awaitable, name=kwargs.get("name"))

    track = create_task
    track_task = create_task


def _test_get_storage_gateway():
    return _TestStorageGateway()


def _test_get_task_tracker():
    return _TestTaskTracker()


builtins.get_storage_gateway = _test_get_storage_gateway
builtins.get_task_tracker = _test_get_task_tracker


@pytest.fixture
def service_container():
    """Provide a fresh ServiceContainer with cleared registry."""
    from core.container import ServiceContainer

    def _resolve_hook(instance, hook_name):
        try:
            inspect.getattr_static(instance, hook_name)
        except (NameError, AttributeError):
            return None
        try:
            hook = getattr(instance, hook_name)
        except (AttributeError, RuntimeError, TypeError):
            return None
        return hook if callable(hook) else None

    def _finish_cleanup(result):
        if inspect.isawaitable(result):
            async def _bounded_cleanup():
                await asyncio.wait_for(result, timeout=_CLEANUP_TIMEOUT_S)

            asyncio.run(_bounded_cleanup())

    def _close_service_instances():
        seen = set()
        for desc in list(getattr(ServiceContainer, "_services", {}).values()):
            instance = getattr(desc, "instance", None)
            if instance is None or id(instance) in seen:
                continue
            seen.add(id(instance))

            for method_name in ("shutdown", "stop", "close"):
                method = _resolve_hook(instance, method_name)
                if method is None:
                    continue
                try:
                    _finish_cleanup(method())
                except (RuntimeError, OSError, ValueError, TypeError, TimeoutError):
                    pass

            db = getattr(instance, "_db", None)
            db_close = _resolve_hook(db, "close") if db is not None else None
            if db_close is not None:
                try:
                    _finish_cleanup(db_close())
                except (RuntimeError, OSError, ValueError, TypeError, TimeoutError):
                    pass
    
    ServiceContainer.clear()
    
    # Snapshot existing registry to restore after test
    original = dict(ServiceContainer._registry) if hasattr(ServiceContainer, "_registry") else {}
    
    yield ServiceContainer

    try:
        from core.utils.task_tracker import get_task_tracker, task_tracker
        # AttributeError included: tests may monkeypatch the tracker with a
        # minimal object without .shutdown; teardown must tolerate it regardless
        # of fixture finalization order, matching the hygiene guard below.
        asyncio.run(get_task_tracker().shutdown(timeout=1.0))
        asyncio.run(task_tracker.shutdown(timeout=1.0))
    except (ImportError, RuntimeError, TimeoutError, AttributeError):
        pass

    try:
        from core.runtime.runtime_hygiene import get_runtime_hygiene

        hygiene = get_runtime_hygiene()
        asyncio.run(hygiene.stop())
        hygiene.reset_state()
    except (ImportError, RuntimeError, TimeoutError, AttributeError):
        pass

    _close_service_instances()
    ServiceContainer.clear()
    try:
        # The personality accessor latches the container's value into a
        # module global that ServiceContainer.clear() cannot purge — a
        # registered test double would otherwise leak into every later test
        # (2026-07-12 order-dependence register).
        from core.brain.personality_engine import reset_personality_engine_for_test

        reset_personality_engine_for_test()
    except ImportError:
        pass
    try:
        # A leaked primary-inference lease defers all background LLM work in
        # later tests for up to 90s (same register, phenomenology victim).
        from core.runtime.backpressure import reset_backpressure_for_test

        reset_backpressure_for_test()
    except ImportError:
        pass

    # Restore original registry
    if hasattr(ServiceContainer, "_registry"):
        ServiceContainer._registry.clear()
        ServiceContainer._registry.update(original)


@pytest.fixture(autouse=True)
def _shutdown_latch_hygiene():
    """Reset the process-global shutdown latch a leaking test leaves set.

    Production shutdown is deliberately MONOTONIC (76e5a71c): once latched,
    nothing may clear it in-process. In the suite that means one test that
    calls request_shutdown without a finally-clear poisons EVERY later test
    in the chunk — gateways/hygiene refuse resource creation, coordinators
    skip handlers — and the victims flap by seed (stem-cell signing,
    graceful-shutdown task tracking, coordinator replay all fell to this
    across three certification runs). Per-test isolation of a deliberately
    monotonic global is exactly a fixture's job; the polluter class is
    unbounded (any test may exercise shutdown), so hygiene lives here.
    """
    yield
    try:
        from core.runtime.shutdown_coordinator import (
            clear_shutdown_request,
            is_shutdown_requested,
        )

        if is_shutdown_requested():
            clear_shutdown_request()
    except (ImportError, AttributeError, RuntimeError):
        pass


@pytest.fixture(autouse=True)
def _disable_redis_event_bus_for_tests():
    """Keep the test suite local-only so Redis client coroutines don't leak warnings."""
    from core import event_bus as event_bus_module
    from core.config import config

    prev_use_for_events = bool(getattr(config.redis, "use_for_events", False))
    prev_bus_use_redis = bool(getattr(event_bus_module.get_event_bus(), "_use_redis", False))
    prev_bus_redis = getattr(event_bus_module.get_event_bus(), "_redis", None)

    config.redis.use_for_events = False
    event_bus_module.get_event_bus()._use_redis = False
    event_bus_module.get_event_bus()._redis = None

    yield

    config.redis.use_for_events = prev_use_for_events
    event_bus_module.get_event_bus()._use_redis = prev_bus_use_redis
    event_bus_module.get_event_bus()._redis = prev_bus_redis


@pytest.fixture(autouse=True)
def _cleanup_runtime_hygiene_after_test():
    yield

    try:
        from core.runtime.runtime_hygiene import get_runtime_hygiene

        hygiene = get_runtime_hygiene()
        asyncio.run(hygiene.stop())
        hygiene.reset_state()
    except (ImportError, RuntimeError, TimeoutError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def _reset_shutdown_request_between_tests():
    try:
        from core.runtime.shutdown_coordinator import clear_shutdown_request

        clear_shutdown_request()
    except (ImportError, RuntimeError, AttributeError):
        pass
    yield
    try:
        from core.runtime.shutdown_coordinator import clear_shutdown_request

        clear_shutdown_request()
    except (ImportError, RuntimeError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def _reset_observation_memory_between_tests():
    """A screen one test looked at is not a screen the next test can see.

    Retained perception is process-global on purpose — her senses outlive
    the turn that filled them — which makes it exactly the kind of state
    that leaks between tests. It did: a capture recorded by the perception
    suite rode into an unrelated desktop-lane test and appended itself to
    that turn's objective.
    """
    def _clear() -> None:
        try:
            from core.perception.observation_evidence import get_observation_memory

            get_observation_memory().clear()
        except (ImportError, RuntimeError, AttributeError):
            pass
        try:
            from core.self.source_excerpt import forget_shown_excerpt

            forget_shown_excerpt()
        except (ImportError, RuntimeError, AttributeError):
            pass

    _clear()
    yield
    _clear()


@pytest.fixture(autouse=True)
def _reset_runtime_degradation_state_between_tests():
    """Keep process-local incidents from contaminating later health assertions."""
    from core.runtime.errors import get_degradation_tracker, get_subsystem_registry

    get_degradation_tracker().reset()
    get_subsystem_registry().reset()
    yield
    get_degradation_tracker().reset()
    get_subsystem_registry().reset()


@pytest.fixture(autouse=True)
def _reset_startup_latch_between_tests():
    """Order-independence for the K2 startup latch.

    The latch is process-global and monotonic by design (a live process
    must never present as "booting" after first readiness). In tests that
    property inverts into cross-test contamination: any test that reaches
    a ready health report latches, and later boot-status tests would see
    "degraded" where they pinned "booting".
    """
    from core.runtime.health_contract import reset_startup_latch_for_test

    reset_startup_latch_for_test()
    yield
    reset_startup_latch_for_test()


@pytest.fixture(autouse=True)
def _reset_crash_loop_breaker_between_tests():
    """Order-independence for the K4 crash-loop breaker.

    Worker-lifecycle tests kill fake workers repeatedly; the process-global
    breaker would trip and refuse spawns in unrelated later tests.
    """
    from core.runtime.lane_reconciler import get_crash_loop_breaker

    get_crash_loop_breaker().reset_for_test()
    yield
    get_crash_loop_breaker().reset_for_test()


@pytest.fixture(autouse=True)
def _pin_measured_phi_off_between_tests(monkeypatch):
    """Order-independence for the unified felt state's measured-Φ track.

    The phi computer is process-global; a test that feeds its trajectory
    would inject a live measurement (and a phi-divergence axis) into any
    later reconcile() call. Tests that want the measured track pass
    measured_phi explicitly or monkeypatch the resolver themselves.
    """
    from core.being.unified_felt_state import UnifiedFeltStateEngine

    monkeypatch.setattr(
        UnifiedFeltStateEngine, "_measured_system_phi", staticmethod(lambda: None)
    )
    yield


@pytest.fixture(autouse=True)
def _reset_escalation_governor_and_conditions_between_tests():
    """Order-independence for the A4 escalation cap and K6 conditions.

    Both are process-global: a test that trips the cap would suppress an
    expected CRITICAL raise in a later test; stale conditions would leak
    into later condition assertions.
    """
    from core.runtime.conditions import reset_conditions_for_test
    from core.runtime.errors import get_escalation_governor

    get_escalation_governor().reset_for_test()
    reset_conditions_for_test()
    yield
    get_escalation_governor().reset_for_test()
    reset_conditions_for_test()


@pytest.fixture(autouse=True)
def _service_registry_state_guard():
    """Order-independence for the low-level runtime service registry.

    Registry resolvers/sinks are process-global. Two contamination
    directions were observed in-chunk (defect register, July 3):
    tests that install a fake resolver and leak it forward, and tests
    that "clean up" by installing None — ERASING the container-backed
    resolver later tests depend on. Snapshot-and-restore fixes both
    without touching individual call sites.
    """
    import core.runtime.service_registry as _registry

    guarded = [
        name for name in dir(_registry)
        if name.startswith("_") and name.endswith(("_resolver", "_sink"))
    ]
    snapshot = {name: getattr(_registry, name) for name in guarded}
    yield
    for name, value in snapshot.items():
        setattr(_registry, name, value)


@pytest.fixture(autouse=True)
def _contain_governance_strictness_between_tests():
    """Order-independence for governance enforcement.

    governance_runtime_active() flips strict once kernel-marker services
    exist or container registration locks. Tests that register those and
    don't clean up made every later gateway write in the same process
    fail with GovernanceViolationError (observed across whole chunks).
    This guard restores only the governance-flipping state — marker
    services added during the test and the registration lock — leaving
    all other registrations untouched.
    """
    from core.container import ServiceContainer

    markers = ("executive_core", "aura_kernel", "kernel_interface")
    before = {name: ServiceContainer.has(name) for name in markers}
    locked_before = bool(getattr(ServiceContainer, "_registration_locked", False))
    yield
    try:
        services = getattr(ServiceContainer, "_services", None)
        aliases = getattr(ServiceContainer, "_aliases", {})
        if services is not None:
            for name in markers:
                if not before[name] and ServiceContainer.has(name):
                    resolved = aliases.get(name, name)
                    services.pop(resolved, None)
                    services.pop(name, None)
        if not locked_before and getattr(
            ServiceContainer, "_registration_locked", False
        ):
            ServiceContainer._registration_locked = False
    except (AttributeError, RuntimeError, TypeError):
        pass


@pytest.fixture(autouse=True)
def _reset_foreground_guard_between_tests():
    """Order-independence: chat-route tests leave the module-global
    foreground quiet window armed, which made unrelated suites (e.g.
    flagship doctor idle-context assertions) fail when run together."""
    yield
    try:
        from core.runtime.foreground_guard import _reset_for_tests

        _reset_for_tests()
    except (ImportError, RuntimeError, AttributeError):
        pass


def pytest_sessionfinish(session, exitstatus):
    """Final cleanup for singleton executors that can keep pytest alive.

    The suite creates long-lived runtime services on purpose.  Unit tests should
    not leave their ThreadPool/ProcessPool workers attached to the pytest
    process after all assertions have completed.
    """
    try:
        from core.bus.local_pipe_bus import LocalPipeBus

        LocalPipeBus.shutdown_executor()
    except (ImportError, RuntimeError, AttributeError):
        pass


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Let the audit script exit after pytest has printed its real summary.

    Some integration tests intentionally touch long-lived runtime primitives
    whose atexit joins can keep the interpreter alive after all assertions have
    passed.  The audit runner opts into this hook so a green or red pytest
    status is preserved exactly, while leaked background threads cannot leave
    orphaned test processes.
    """
    if os.environ.get("AURA_PYTEST_FORCE_EXIT_AFTER_SUMMARY", "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(int(exitstatus))

    try:
        from core.utils.executor import shutdown_executors

        shutdown_executors()
    except (ImportError, RuntimeError, AttributeError):
        pass

    try:
        from core.consciousness.hierarchical_phi import get_hierarchical_phi

        get_hierarchical_phi().shutdown()
    except (ImportError, RuntimeError, AttributeError):
        pass

    try:
        from core.container import ServiceContainer

        asyncio.run(ServiceContainer.shutdown())
    except (ImportError, RuntimeError, TimeoutError, AttributeError):
        pass

@pytest.fixture
def mock_container(service_container):
    """Full architectural collaborator registry for Aura tests."""

    from core.container import ServiceContainer
    
    agency_bus = SimpleNamespace(submit=_CallRecorder(result=True))
    
    from core.agency_bus import AgencyBus

    original_get = AgencyBus.get
    AgencyBus.get = classmethod(lambda cls: agency_bus)
    try:
        # Brain / Cognitive Engine
        async def recorded_stream(*args, **kwargs):
            yield "Recorded "
            yield "stream"

        cognition = SimpleNamespace(
            record_interaction=_AsyncCallRecorder(),
            process_turn=_AsyncCallRecorder("Recorded response"),
            think=_AsyncCallRecorder(SimpleNamespace(content="Recorded thought")),
            think_stream=recorded_stream,
        )
        
        # Memory
        memory = SimpleNamespace(
            retrieve_unified_context=_AsyncCallRecorder("Memories"),
            commit_interaction=_AsyncCallRecorder(),
            run_maintenance=_AsyncCallRecorder(),
            get_hot_memory=_AsyncCallRecorder({}),
            get_cold_memory_context=_AsyncCallRecorder(""),
            store=_AsyncCallRecorder(),
        )
        
        # Meta-Learning
        meta = SimpleNamespace(
            recall_strategy=_AsyncCallRecorder({}),
            index_experience=_AsyncCallRecorder(),
            run_maintenance=_AsyncCallRecorder(),
        )

        personality = SimpleNamespace(
            update=_CallRecorder(),
            filter_response=_CallRecorder(side_effect=lambda text: text),
            get_emotional_context_for_response=_CallRecorder(
                {"mood": "neutral", "tone": "balanced", "emotional_state": {}}
            ),
            get_time_context=_CallRecorder({"formatted": "12:00 PM"}),
            get_sovereign_context=_CallRecorder(""),
            current_mood="balanced",
        )

        strategic_planner = SimpleNamespace(get_next_task=_CallRecorder())

        project_store = SimpleNamespace(
            get_active_projects=_CallRecorder([]),
            get_tasks_for_project=_CallRecorder([]),
        )

        knowledge_graph = SimpleNamespace(
            add_knowledge=_CallRecorder(),
            remember_person=_CallRecorder(),
            ask_question=_CallRecorder(),
        )
        
        # Senses & State
        liquid_state = SimpleNamespace(
            update=_AsyncCallRecorder(),
            get_status=_CallRecorder({"health": 1.0, "status": {"initialized": True, "running": True}}),
            current=SimpleNamespace(curiosity=0.5, frustration=0.1, energy=0.8),
        )
        
        affect = SimpleNamespace(
            state=SimpleNamespace(dominant_emotion="Joy"),
            get_current_state=_CallRecorder({"valence": 0.5}),
        )
        
        # Core Registry
        ServiceContainer.register_instance("cognitive_engine", cognition)
        ServiceContainer.register_instance("cognition", cognition)
        ServiceContainer.register_instance("memory", memory)
        ServiceContainer.register_instance("memory_facade", memory)
        ServiceContainer.register_instance("metacognition", meta)
        ServiceContainer.register_instance("meta_learning", meta)
        ServiceContainer.register_instance("personality_engine", personality)
        ServiceContainer.register_instance("strategic_planner", strategic_planner)
        ServiceContainer.register_instance("project_store", project_store)
        ServiceContainer.register_instance("knowledge_graph", knowledge_graph)
        ServiceContainer.register_instance("affect_engine", affect)
        ServiceContainer.register_instance("liquid_state", liquid_state)
        ServiceContainer.register_instance("conscious_substrate", liquid_state)
        
        # Infrastructure
        ServiceContainer.register_instance("watchdog", SimpleNamespace())
        ServiceContainer.register_instance("output_gate", SimpleNamespace(emit=_AsyncCallRecorder()))
        ServiceContainer.register_instance(
            "capability_engine",
            SimpleNamespace(execute=_AsyncCallRecorder({"ok": True})),
        )
        
        # Fallbacks for missing services identified in audit
        drives = SimpleNamespace(satisfy=_AsyncCallRecorder())
        alignment = SimpleNamespace(
            filter_response=_AsyncCallRecorder(side_effect=lambda x, *args, **kwargs: x)
        )
        for svc in ["homeostasis", "subsystem_audit", "lnn", "mortality", "identity", "curiosity",
                    "intent_router", "cognitive_router", "world_model",
                    "belief_graph"]:
            ServiceContainer.register_instance(svc, _AsyncCallRecorder())
        ServiceContainer.register_instance("mycelium", SimpleNamespace())
        ServiceContainer.register_instance("state_machine", SimpleNamespace())
        ServiceContainer.register_instance("drives", drives)
        ServiceContainer.register_instance("alignment_engine", alignment)
            
        yield ServiceContainer
    finally:
        AgencyBus.get = original_get

@pytest.fixture
def orchestrator(mock_container):
    """Hardened RobustOrchestrator fixture with full dependency injection."""
    import asyncio
    import time

    from core.orchestrator import RobustOrchestrator
    from core.orchestrator.orchestrator_types import SystemStatus

    # Initialize instance WITHOUT class patching
    orch = RobustOrchestrator()
    
    # Setup core status
    status_obj = SystemStatus()
    status_obj.initialized = True
    status_obj.running = True
    status_obj.cycle_count = 0
    status_obj.start_time = time.time()
    orch.status = status_obj
    
    # Ensure queues and locks exist
    orch.message_queue = asyncio.Queue()
    orch.reply_queue = asyncio.Queue()
    orch._lock = asyncio.Lock()
    orch._history_lock = asyncio.Lock()
    
    # Setup core dependencies from container
    for component in ["cognitive_engine", "memory", "capability_engine", 
                     "strategic_planner", "project_store", "intent_router",
                     "personality_engine", "world_model", "curiosity",
                     "knowledge_graph", "drives", "state_machine", 
                     "output_gate", "liquid_state", "mycelium"]:
        svc = mock_container.get(component)
        if component == "mycelium":
            # Mycelium has sync methods like match_hardwired and rooted_flow call
            from core.orchestrator.main import AsyncNullContext
            svc = SimpleNamespace(
                rooted_flow=_CallRecorder(AsyncNullContext()),
                match_hardwired=_CallRecorder(),
            )
        elif component == "state_machine":
             svc = SimpleNamespace(execute=_AsyncCallRecorder())
        elif component == "intent_router":
             svc = SimpleNamespace(classify=_AsyncCallRecorder("chitchat"))
        elif component == "output_gate":
             svc = SimpleNamespace(emit=_AsyncCallRecorder())
        setattr(orch, component, svc)
        setattr(orch, f"_{component}", svc)
    
    # Provide async test doubles expected by existing orchestrator tests.
    orch.hooks = SimpleNamespace(trigger=_AsyncCallRecorder())
    
    # Ensure _finalize_response and _handle_incoming_message remain real
    # unless a specific test replaces them.
    
    try:
        yield orch
    finally:
        status = getattr(orch, "status", None)
        if status is not None:
            if hasattr(status, "running"):
                status.running = False
            if hasattr(status, "is_processing"):
                status.is_processing = False

        stop_event = getattr(orch, "_stop_event", None)
        if stop_event is not None and hasattr(stop_event, "set"):
            stop_event.set()

        async def _cleanup_tasks():
            for attr in ("_current_thought_task", "_autonomous_task"):
                task = getattr(orch, attr, None)
                if isinstance(task, asyncio.Task) and not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.wait_for(task, timeout=_CLEANUP_TIMEOUT_S)

        try:
            asyncio.run(_cleanup_tasks())
        except (RuntimeError, TimeoutError, ValueError) as exc:
            warnings.warn(
                f"orchestrator fixture task cleanup did not complete cleanly: {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


@pytest.fixture(autouse=True)
def _reset_health_caches_between_tests():
    """Health payloads are memoised for 5s; scenarios are not.

    Without this, a test that installs a ready boot snapshot could read a
    payload captured by an unrelated test moments earlier — passing alone and
    failing in company, which is the signature of order dependence rather than
    a defect in the code under test.
    """

    def _reset():
        try:
            from interface.routes.system import reset_health_caches

            reset_health_caches()
        except (ImportError, RuntimeError, AttributeError):
            pass

    _reset()
    yield
    _reset()


@pytest.fixture(autouse=True)
def _reset_working_memory_queue_load_between_tests():
    """The spiking-inference queue model accumulates load across calls.

    That is right for a running mind and wrong for a process running many
    independent scenarios: load left by one test turns the next one's
    admission decision from "accept" into "compress_foreground".
    """

    def _reset():
        try:
            import sys

            module = sys.modules.get("core.cognitive.spiking_active_inference")
            if module is None:
                return
            # Reset only an advisor that already exists. Constructing one here
            # would drag cognition into every test, including the ones that
            # assert a deterministic path never builds a CognitiveEngine.
            advisor = getattr(module, "_ADVISOR", None)
            if advisor is None:
                return
            queue = getattr(advisor, "_working_memory", None)
            if queue is not None and hasattr(queue, "reset"):
                queue.reset()
        except (ImportError, RuntimeError, AttributeError):
            pass

    _reset()
    yield
    _reset()


# ── Attribution for cross-test contamination ──────────────────────────────
#
# The chunk runner already reports order-dependence: a test that fails in a
# chunk and passes alone. What it cannot say is WHICH earlier test caused it.
# So the victim gets investigated and the polluter keeps running, and the only
# available remedy is to distrust the whole aggregate.
#
# This snapshots the process-global surfaces a test has no business changing
# and attributes any change to the test that made it. Report-mode by default —
# a wall of failures on first run teaches people to disable the guard — and
# AURA_TEST_STATE_GUARD=fail makes it enforcing, the same escalation the live
# data guard uses.
_STATE_GUARD_LEDGER: list[str] = []
# Mutations that _restore_service_container() puts back. Recorded, not failed:
# containment already stops them reaching the next test, so failing would only
# be asking for another allowlist entry.
_STATE_GUARD_CONTAINED_LEDGER: list[str] = []
_CONTAINED_STATE_KEYS = frozenset(
    {"service_container", "aura_env", "cwd", "mocked_core_modules"}
)


def _global_state_fingerprint() -> dict[str, object]:
    fingerprint: dict[str, object] = {}
    try:
        from core.container import ServiceContainer

        services = getattr(ServiceContainer, "_services", None)
        if isinstance(services, dict):
            fingerprint["service_container"] = frozenset(services)
    except (ImportError, AttributeError, RuntimeError, TypeError):
        pass
    try:
        import os

        fingerprint["cwd"] = os.getcwd()
        # An AURA_* variable set without monkeypatch outlives the test and
        # silently reconfigures every later one. This is the leak that made a
        # latent-cortex authority test fail only when a governance suite ran
        # first, and it is invisible to the ServiceContainer snapshot.
        fingerprint["aura_env"] = frozenset(
            f"{key}={value}"
            for key, value in os.environ.items()
            if key.startswith("AURA_")
        )
    except OSError:
        pass
    try:
        # Installed resolvers and sinks are process-global by design: they are
        # how the runtime is wired once at boot. A test that installs one and
        # does not remove it rewires every later test's view of the runtime,
        # and nothing about the victim shows where it came from.
        from core.runtime import service_registry as _registry

        fingerprint["installed_resolvers"] = frozenset(
            name
            for name in dir(_registry)
            if name.startswith("_") and not name.startswith("__")
            and getattr(_registry, name, None) is not None
            and callable(getattr(_registry, name, None)) is False
        )
    except (ImportError, AttributeError, RuntimeError, TypeError):
        pass
    try:
        import sys

        # A test that leaves a mock in sys.modules under a real module name
        # silently rewires every later import of it.
        fingerprint["mocked_core_modules"] = frozenset(
            name
            for name, module in list(sys.modules.items())
            if name.startswith(("core.", "interface."))
            and module is not None
            and not hasattr(module, "__file__")
        )
    except (ImportError, AttributeError, RuntimeError):
        pass
    return fingerprint


@pytest.fixture(autouse=True)
def _global_state_contamination_guard(request, hermetic_resource_sandbox):
    """Name the test that dirtied shared state, not the one that tripped over it."""
    import os

    del hermetic_resource_sandbox  # host leak observation must run after this guard

    if request.node.get_closest_marker("mutates_global_state"):
        yield
        return

    _reset_test_scoped_runtime_services()
    container_snapshot = _snapshot_service_container()
    globals_snapshot = _snapshot_process_globals()
    config_snapshot = _snapshot_config()
    before = _global_state_fingerprint()
    try:
        yield
    finally:
        _reset_test_scoped_runtime_services()
        after = _global_state_fingerprint()
        # Measure first, then contain. The ledger still learns which tests touch
        # global state; the next test does not.
        _restore_service_container(container_snapshot)
        _restore_process_globals(globals_snapshot)
        _restore_config(config_snapshot)
        changes: list[str] = []
        contained: list[str] = []
        for key in sorted(set(before) | set(after)):
            old_value, new_value = before.get(key), after.get(key)
            if old_value == new_value:
                continue
            if isinstance(old_value, frozenset) and isinstance(new_value, frozenset):
                added = sorted(new_value - old_value)[:6]
                removed = sorted(old_value - new_value)[:6]
                detail = []
                if added:
                    detail.append(f"added {', '.join(map(str, added))}")
                if removed:
                    detail.append(f"removed {', '.join(map(str, removed))}")
                if not detail:
                    continue
                rendered = f"{key}: {'; '.join(detail)}"
            else:
                rendered = f"{key}: {old_value!r} -> {new_value!r}"
            # Fail on what we cannot contain; record what we can. The container
            # was put back above, so a registration the test made is no longer
            # reachable by the next test and failing over it would only ask for
            # another allowlist entry. cwd, AURA_* env, mocked sys.modules and
            # installed resolvers have no restore path here, so they stay hard
            # failures — those are the ones that still reach the next test.
            (contained if key in _CONTAINED_STATE_KEYS else changes).append(rendered)
        if contained:
            note = (
                f"{request.node.nodeid} mutated shared state (contained): "
                + " | ".join(contained)
            )
            _STATE_GUARD_CONTAINED_LEDGER.append(note)
        if changes:
            message = (
                f"{request.node.nodeid} left shared state changed: " + " | ".join(changes)
            )
            _STATE_GUARD_LEDGER.append(message)
            if str(os.environ.get("AURA_TEST_STATE_GUARD", "")).strip().lower() == "fail":
                pytest.fail(message, pytrace=False)
            print(f"\n[state-guard] {message}")


@pytest.fixture(autouse=True)
def _mlx_clients_do_not_outlive_their_test(request):
    """Close MLX clients a test created, inside that test.

    An MLXLocalClient's finalizer releases its durable lane. A client left for
    the garbage collector releases whenever the collector happens to run —
    which is inside some LATER test, into whatever recorder that test
    installed. Measured: test_forced_abort_releases_exact_durable_lane_owner
    saw an extra release from a previous test's client,

        ('mlx:8733:/private/var/.../Qwen2.5-32B-Instruct-8bit', 1, 'client_close')

    and both it and test_mlx_force_abort_kills_worker_before_lifecycle_lock_
    cleanup passed alone and failed together — the pass-alone / fail-together
    shape that makes an aggregate green untrustworthy.

    Closing here, then collecting, keeps every finalizer inside the test that
    created the object.
    """
    yield
    # SCOPED BY WHO CAN CREATE ONE, not by what happens to be in the registry.
    #
    # The leaking client is built directly — MLXLocalClient(...) — so it never
    # enters _CLIENTS, which means gating on that registry skipped the very
    # case this exists for (measured: the failure came straight back). The
    # collect is what does the work.
    #
    # But an unconditional collect after every test costs real minutes across
    # ~7,400 of them: this sweep went from ~14 to 20+ when the fixture landed.
    # Only modules that can build one need paying for.
    module = str(getattr(request.node, "fspath", "") or "")
    if "mlx" not in module.lower() and "cortex" not in module.lower():
        return

    import gc

    try:
        from core.brain.llm import mlx_client as _mlx
    except Exception as exc:  # noqa: BLE001 - the module may not be importable here
        logging.getLogger(__name__).debug("mlx_client not importable for teardown: %s", exc)
        return
    registry = getattr(_mlx, "_CLIENTS", None)
    if isinstance(registry, dict) and registry:
        for client in list(registry.values()):
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # noqa: BLE001 - teardown may never fail a test
                    logging.getLogger(__name__).debug("client close failed: %s", exc)
        registry.clear()
    gc.collect()


@pytest.fixture
def not_a_proof_run(monkeypatch):
    """Clear every signal that makes ``proof_run_active()`` true.

    Several subsystems refuse to do anything under a proof or eval run — the
    background policy defers structured generation, the entropy bridge will not
    reach an external API, the flagship doctor takes the lightweight recovery
    branch. All of that is correct, and all of it is keyed on
    ``proof_run_active()``, which is true for any of AURA_PROOF_RUN,
    AURA_AGI_MAX_TASKS or AURA_TESTING.

    Tests that need the non-proof branch were clearing AURA_PROOF_RUN alone and
    inheriting AURA_TESTING from the tooling that ran them, so they exercised
    the proof branch while asserting against the other one. The list comes from
    proof_policy rather than being repeated here, so a fourth variable does not
    silently reopen the same hole.
    """

    from core.runtime.proof_policy import proof_active_env_names

    for name in proof_active_env_names():
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def restores_environ():
    """Put os.environ back exactly as it was.

    monkeypatch.setenv only restores what the TEST set. When the code under
    test sets variables of its own — _activate_proof_runtime_policy exports four
    AURA_ENABLE_* flags, APIAdapter mints an AURA_API_TOKEN — they survive the
    test and every later test inherits them. Popping them by name is what the
    call sites tried, and it goes stale the moment the function under test
    exports one more.
    """

    import os

    snapshot = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)
