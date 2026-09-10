"""Who owns a piece of state, and which runtime is allowed to touch it.

Aura accumulated persistent stores, registries, singletons and learning
systems faster than it accumulated boundaries between them. The result was
spooky action at a distance — behaviour changing because another checkout
ran, bugs that vanished when reproduced alone.

The two live cases this exists to make impossible:

* Terminal-grid TESTS read and wrote the live user-global learning
  directory. Learned risk from live state reached 1.0 and vetoed a benign
  move; test episodes were written back into the organism's real learning
  memory. A test run permanently changed what Aura believed.
* A fixture left a fail-closed Mycelium instance registered after its test
  finished. Dozens of later tests failed only inside the full suite,
  because they inherited global state nobody had declared they owned.

Three ideas, and the third is the one that does the work:

1. **Identity.** Every runtime has an immutable ``runtime_instance_id``.
   Every persistent record names the runtime and model that produced it,
   so a store can always answer "who wrote this".
2. **Roots.** All state lives under one explicitly resolved state root.
   There is no second way to find it and no ambient default buried in a
   subsystem.
3. **Structural separation.** Live, test, bench and dev roots are
   different directories, and a non-live runtime that reaches for the live
   root does not get a warning — it raises. A boundary that merely logs is
   a boundary that has already been crossed.

The profile is DERIVED, not declared by the code being tested. A test
cannot opt itself back into the live root by setting a flag, because the
tests that did this damage did not know they were doing it.
"""
from __future__ import annotations

import enum
import functools
import os
import platform
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "RuntimeProfile",
    "StateOwnershipViolation",
    "BootstrapFlagSpec",
    "bootstrap_flag_specs",
    "bootstrap_flag_value",
    "runtime_instance_id",
    "runtime_profile",
    "state_root",
    "state_root_override",
    "live_state_root",
    "is_live_state_path",
    "assert_state_path_allowed",
    "stamp_record",
    "runtime_identity",
    "shared_asset_root",
]


@dataclass(frozen=True)
class BootstrapFlagSpec:
    """A process-bootstrap flag that cannot depend on persisted settings.

    Resolving the state root through the normal flag stack creates a dependency
    cycle: persisted settings need the state root before they can be read.  The
    flag registry imports these declarations for observability, while this
    module remains a standard-library-only root of the runtime graph.
    """

    name: str
    default: str
    description: str
    owner: str = "core.runtime.state_ownership"
    kind: str = "string"


_BOOTSTRAP_FLAGS = {
    spec.name: spec
    for spec in (
        BootstrapFlagSpec("AURA_HOME", "", "Override the process home root"),
        BootstrapFlagSpec(
            "AURA_LIVE_STATE_ROOT",
            "",
            "Inherited identity of the live runtime state root",
        ),
        BootstrapFlagSpec("AURA_BENCH_RUN", "", "Mark a benchmark runtime"),
        BootstrapFlagSpec("AURA_BENCHMARK", "", "Mark a benchmark runtime"),
        BootstrapFlagSpec("AURA_TESTING", "", "Mark a hermetic test runtime"),
        BootstrapFlagSpec("AURA_PROOF_RUN", "", "Mark a hermetic proof runtime"),
        BootstrapFlagSpec("AURA_DEV_RUNTIME", "", "Mark a development runtime"),
        BootstrapFlagSpec(
            "AURA_STATE_ROOT", "", "Override root for durable state stores"
        ),
        BootstrapFlagSpec(
            "AURA_ASSET_ROOT", "", "Override root for shared immutable assets"
        ),
        BootstrapFlagSpec(
            "AURA_ALLOW_LIVE_STATE_WRITE",
            "",
            "Allow an explicitly launched maintenance process to write live state",
        ),
    )
}


def bootstrap_flag_specs() -> dict[str, BootstrapFlagSpec]:
    """Return the immutable bootstrap declarations for flag introspection."""

    return dict(_BOOTSTRAP_FLAGS)


def bootstrap_flag_value(name: str) -> tuple[str, str]:
    """Resolve one bootstrap flag without consulting state-backed settings."""

    spec = _BOOTSTRAP_FLAGS.get(name)
    if spec is None:
        raise KeyError(f"unknown state-ownership bootstrap flag: {name}")
    value = os.environ.get(name)
    if value is None:
        return spec.default, "default"
    return str(value), "env"


def _bootstrap_env(name: str) -> str:
    return bootstrap_flag_value(name)[0]


class StateOwnershipViolation(RuntimeError):  # noqa: N818
    """A runtime reached for state it does not own.

    Deliberately not a warning. The failures this prevents were silent:
    nothing raised, a test wrote into live learning memory, and the damage
    was found weeks later in behaviour rather than in a log.
    """


class RuntimeProfile(enum.StrEnum):
    """Which kind of runtime this process is. Derived, never self-declared."""

    LIVE = "live"
    TEST = "test"
    BENCH = "bench"
    DEV = "dev"

    @property
    def may_touch_live_state(self) -> bool:
        return self is RuntimeProfile.LIVE


#: The real user-global root. Never used directly outside this module —
#: everything goes through ``state_root()`` so the profile can intercept.
_LIVE_ROOT_NAME = ".aura"

_INSTANCE_NONCE = uuid.uuid4().hex[:16]
_STARTED_AT = time.time()


def _home() -> Path:
    override = _bootstrap_env("AURA_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home().expanduser().resolve()


#: The home this process started with, captured before anything can redirect
#: it. THE reference point: "live state" means the real user's ~/.aura, and a
#: test that repoints HOME must not be able to move what that phrase denotes
#: — otherwise the guard would happily let a run write to whatever it had
#: just declared to be its own live root.
_ORIGINAL_HOME: Path = _home()

#: Publish that reference point so it survives a spawn. Without this, every
#: child re-infers "the live instance" from the HOME it was handed, and a child
#: given a sandbox HOME concludes the sandbox is the live instance — making the
#: ownership guard refuse the child's own legitimate state writes. Exported
#: once, at the moment the true value is still knowable, it propagates through
#: ordinary environment inheritance to every descendant with no per-spawn code.
#:
#: setdefault, never overwrite: if this process was itself told the answer by a
#: parent, that parent was closer to the truth than this process is.
if not _bootstrap_env("AURA_LIVE_STATE_ROOT"):
    os.environ["AURA_LIVE_STATE_ROOT"] = str(_ORIGINAL_HOME / _LIVE_ROOT_NAME)


def live_state_root() -> Path:
    """The REAL instance's state root. Fixed for the life of the process.

    Deliberately not recomputed from the current ``HOME``: this is the thing
    being protected, and a protected resource whose identity follows a
    mutable environment variable is not protected.

    The one exception is inheritance, and it exists because the rule above
    breaks down across a spawn. ``_ORIGINAL_HOME`` is "the home this process
    started with", which is only the same as "the real user's home" for a
    process that was not handed a redirected one at exec. A child spawned by a
    test already has the fake ``HOME`` at import, so it concludes the sandbox
    IS the live instance and refuses to write its own state — the guard firing
    on the thing it was meant to protect. The parent knows the true answer, so
    it passes it down (see ``subprocess_gateway``), and a child prefers what it
    was told over what it can infer.

    This does not weaken the guard: anything able to set this variable can
    already set ``AURA_ALLOW_LIVE_STATE_WRITE``. It is inherited trust from the
    parent, not a self-declaration — the protection this module provides is
    against accident, and an accident cannot forge a parent.
    """
    inherited = _bootstrap_env("AURA_LIVE_STATE_ROOT")
    if inherited:
        try:
            return Path(inherited).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            # An unusable value must fall back to inference rather than
            # disabling the guard.
            pass
    return _ORIGINAL_HOME / _LIVE_ROOT_NAME


def _derive_profile() -> RuntimeProfile:
    """Work out what this process is from evidence, not from a declaration.

    Order matters. A bench run inside pytest is a BENCH run — it has its
    own root and its own numbers, and folding it into the test root would
    let one overwrite the other.
    """
    if _bootstrap_env("AURA_BENCH_RUN") or _bootstrap_env("AURA_BENCHMARK"):
        return RuntimeProfile.BENCH
    # Is THIS process running pytest? `pytest` in sys.modules answers that
    # and the environment does not: a subprocess a test spawns inherits
    # PYTEST_VERSION and PYTEST_CURRENT_TEST but is an ordinary program.
    #
    # That distinction is load-bearing. A delegated child launched with its
    # own HOME derives its state root from that HOME — which is what the
    # parent arranged — and env-based detection sent it to a `-test` sibling
    # the parent never created, breaking a lane delegation that worked.
    #
    # AURA_TESTING and AURA_PROOF_RUN stay environmental because they are
    # DECLARED by a harness rather than leaked by one: a caller that sets
    # them means it, including for a child.
    if (
        "pytest" in sys.modules
        or _bootstrap_env("AURA_TESTING")
        or _bootstrap_env("AURA_PROOF_RUN")
    ):
        return RuntimeProfile.TEST
    if _bootstrap_env("AURA_DEV_RUNTIME"):
        return RuntimeProfile.DEV
    return RuntimeProfile.LIVE


@functools.lru_cache(maxsize=1)
def runtime_profile() -> RuntimeProfile:
    """This process's profile, resolved once.

    Cached deliberately: a profile that could change mid-process would let
    a runtime write half its state to one root and half to another, which
    is worse than either root being wrong.
    """
    return _derive_profile()


def runtime_instance_id() -> str:
    """Immutable identity for this runtime, stable for the process lifetime.

    Carries the profile in the id itself, so a record found in the wrong
    place names its own origin without a lookup.
    """
    return f"{runtime_profile().value}-{_INSTANCE_NONCE}"


def state_root_override() -> str:
    """Return the one canonical declaration of the explicit state root."""

    return _bootstrap_env("AURA_STATE_ROOT")


@functools.lru_cache(maxsize=32)
def _resolved_state_root(
    injected: str,
    current_home: str,
    profile: RuntimeProfile,
    live_root: str,
) -> Path:
    if injected:
        return Path(injected).expanduser().resolve()
    current = Path(current_home) / _LIVE_ROOT_NAME
    if profile is RuntimeProfile.LIVE or current != Path(live_root):
        return current
    return current.with_name(f"{current.name}-{profile.value}")


def state_root() -> Path:
    """THE state root. The only way to find where state lives.

    Resolution order:

    1. ``AURA_STATE_ROOT`` — an explicit injection, honoured for every
       profile. This is how a harness gives a run its own world.
    2. The profile's default. LIVE gets the real root; TEST, BENCH and DEV
       get siblings that are structurally distinct directories, not
       subdirectories of the live root — a subdirectory is one ``..`` away
       from the thing it was supposed to be separate from.
    3. When ``HOME`` has been redirected away from the home this process
       started with — the standard way a test isolates itself — then
       ``$HOME/.aura`` is ALREADY not the live instance's state, and it is
       used as-is. Appending a ``-test`` suffix there would send the run to
       a directory its own fixture never created, which is a broken test
       rather than a safer one.

    The cache is keyed on the inputs rather than set once, so a test that
    repoints ``HOME`` per-test gets the root it just arranged. A live
    runtime never changes either input, so it resolves once and stays.
    """
    return _resolved_state_root(
        state_root_override(),
        str(_home()),
        runtime_profile(),
        str(live_state_root()),
    )


def shared_asset_root() -> Path:
    """Where immutable, downloaded artifacts live. Always the real root.

    Model weights, checkpoints and tokenizers are NOT per-runtime state.
    They are large, content-addressed, read-only, and shared by every
    runtime on the host — a test run reading them changes nothing, while
    diverting it to an empty ``-test`` sibling breaks it for no benefit
    (measured: the Auto-AVSR checkpoint went `manifest_missing` the moment
    assets followed the state root).

    The rule this encodes: separate state by what a runtime WRITES, not by
    everything it touches. Anything that lives here must be genuinely
    immutable — if a runtime mutates it, it is state and belongs under
    ``state_root()``.
    """
    override = _bootstrap_env("AURA_ASSET_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return live_state_root()


def is_live_state_path(path: Path | str) -> bool:
    """Whether a path lands inside the real instance's state.

    Resolved, not string-compared: ``~/.aura-test/../.aura/data`` is the
    live root, and a check that missed that would be decorative. Uses
    ``strict=False`` so a path that does not exist yet — the interesting
    case, since writes create things — still resolves.
    """
    try:
        candidate = Path(path).expanduser().resolve(strict=False)
        live = live_state_root()
    except (OSError, RuntimeError, ValueError):
        # Fail CLOSED. This used to return False, which reads as "not live
        # state" and lets the write through — but a path that cannot be
        # resolved has not been shown to be outside the live root, it has
        # only refused to answer. Treating "I could not check" as "it is
        # safe" is the exact inversion this module exists to prevent, and
        # nothing noticed it because no test exercised the branch (found by
        # tools/guard_mutation.py). A pathological path losing a write is a
        # far cheaper outcome than a non-live runtime reaching live state.
        return True
    if candidate == live:
        return True
    if live not in candidate.parents:
        return False

    # Inside the live root — but the SOURCE CHECKOUT lives there too. This
    # repo sits at ~/.aura/live-source/, so every repo-relative path
    # (<repo>/data/..., <repo>/artifacts/...) resolved as "the live
    # instance's state" and the guard refused ordinary writes into the
    # working tree. That is not what it protects: the live instance's state
    # is ~/.aura/data, ~/.aura/run, ~/.aura/logs — not the code, and not a
    # worktree under it.
    #
    # A checkout is identified by its own location rather than by name, so a
    # worktree beneath .claude/worktrees/ is covered without listing it.
    for root in _source_roots():
        if candidate == root or root in candidate.parents:
            return False
    return True



@functools.lru_cache(maxsize=1)
def _source_roots() -> tuple[Path, ...]:
    """Every checkout of this repository, main worktree and linked ones.

    The running module names one of them: the directory three levels above
    this file. That is enough when a process only ever writes inside its own
    checkout, and it is not enough for the subject-core battery, which runs
    from a worktree pinned to the commit under test and writes its artifacts
    into the one campaign directory in the main checkout. That write landed
    inside the live root, outside the running checkout, and was refused as
    live instance state.

    A linked worktree's ``.git`` is a file naming its git directory, whose
    grandparent is the main checkout, so the pair is read off disk rather
    than guessed. Cached because the answer cannot change within a process
    and this sits on the write path.
    """
    try:
        here = Path(__file__).resolve().parent.parent.parent
    except (OSError, RuntimeError, ValueError):
        return ()
    roots = [here]
    marker = here / ".git"
    try:
        if marker.is_file():
            text = marker.read_text(encoding="utf-8", errors="replace").strip()
            if text.startswith("gitdir:"):
                gitdir = Path(text.split(":", 1)[1].strip()).expanduser()
                if not gitdir.is_absolute():
                    gitdir = (here / gitdir).resolve(strict=False)
                # <main>/.git/worktrees/<name> -> <main>
                for parent in gitdir.resolve(strict=False).parents:
                    if parent.name == ".git":
                        roots.append(parent.parent)
                        break
    except (OSError, RuntimeError, ValueError, UnicodeDecodeError):
        # A checkout whose .git cannot be read still protects itself; it
        # simply does not learn about its siblings.
        pass
    return tuple(dict.fromkeys(roots))


def assert_state_path_allowed(path: Path | str, *, source: str = "unknown") -> None:
    """Refuse a write into live state from a runtime that does not own it.

    The enforcement point. Called from the file-write gateway, so every
    consequential write in the codebase inherits it without each subsystem
    remembering to ask.

    A LIVE runtime is unaffected. Everything else raises on contact with
    the live root, which is what turns "tests wrote into Aura's real
    learning memory" from a thing that happened into a thing that cannot.
    """
    profile = runtime_profile()
    if profile.may_touch_live_state:
        return
    if not is_live_state_path(path):
        return
    if _bootstrap_env("AURA_ALLOW_LIVE_STATE_WRITE") == "1":
        # A deliberate, named escape for the rare tool that really does
        # maintain the live instance. Env-only: nothing in the codebase can
        # set it for itself mid-run without an operator having done so.
        return
    raise StateOwnershipViolation(
        f"{profile.value} runtime tried to write live instance state at {path} "
        f"(source={source}). Live state belongs to the live runtime. Use "
        f"AURA_STATE_ROOT to give this run its own root, or set "
        f"AURA_ALLOW_LIVE_STATE_WRITE=1 if maintaining the live instance is "
        f"genuinely the intent."
    )


def runtime_identity() -> dict[str, Any]:
    """Who this runtime is, for stamping onto anything it persists."""
    return {
        "runtime_instance_id": runtime_instance_id(),
        "runtime_profile": runtime_profile().value,
        "state_root": str(state_root()),
        "started_at": _STARTED_AT,
        "pid": os.getpid(),
        "host": platform.node(),
    }


def stamp_record(
    payload: Mapping[str, Any], *, model_identity: str | None = None
) -> dict[str, Any]:
    """Name the runtime (and model) that produced a persistent record.

    A record that does not say who wrote it cannot be attributed after the
    fact, and unattributable state is how a test's episodes ended up
    indistinguishable from the organism's real ones.

    Non-mutating: returns a new mapping. Never overwrites an existing
    stamp, so a record forwarded between runtimes keeps its true origin.
    """
    stamped = dict(payload or {})
    if "_runtime" not in stamped:
        identity = runtime_identity()
        if model_identity:
            identity["model_identity"] = str(model_identity)
        stamped["_runtime"] = identity
    return stamped


def reset_for_testing() -> None:
    """Clear the cached resolution. For tests OF this module only.

    Named loudly on purpose: anything else calling this is defeating the
    caching that keeps a runtime's state in one place.
    """
    global _INSTANCE_NONCE
    _INSTANCE_NONCE = uuid.uuid4().hex[:16]
    runtime_profile.cache_clear()
    _resolved_state_root.cache_clear()
