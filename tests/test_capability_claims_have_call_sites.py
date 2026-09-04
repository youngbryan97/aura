"""A capability needs a call site, not a class.

Two subsystems were found defined, registered, tested in isolation, and
never invoked by anything that runs:

* ``core/consciousness/latent_bridge.py`` — the backward path from model
  hidden state into the substrate. ``attach_latent_bridge()`` has no caller
  anywhere. The consciousness layer status said ``deferred``, which reads as
  "will happen shortly" and had said so since it was written.
* ``core/being/closed_loop_controller.py`` — ``build_main15_closed_loop()``
  is called from its own docstring and its own tests.

Neither is a bug in the code they contain. Both are correct in isolation.
The defect is the CLAIM: substantial, tested and uninvoked reads exactly
like working from the outside.

This file pins the honest statements so they cannot silently drift back into
implied liveness — and, more usefully, fails the moment someone wires one
up without updating what the system says about itself.
"""
from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Directories that are the live system. A call from tests/, tools/ or an
#: archive is not a production call site.
_PRODUCTION_ROOTS = ("core", "interface")

#: The boot entrypoint lives at the repo root, not under a package. Excluding
#: it made a capability wired ONLY from boot look uncalled — which is the exact
#: mistake this file exists to catch, in the direction that deletes live code.
_PRODUCTION_ENTRYPOINTS = ("aura_main.py",)

_SKIP_PARTS = frozenset(
    {".git", ".venv", "__pycache__", "node_modules", ".claude", "artifacts", "archive"}
)


@lru_cache(maxsize=1)
def _call_index() -> dict[str, frozenset[str]]:
    """{called name: files that call it}, over production sources, built once.

    One pass. Re-parsing the whole tree per entry point is what a
    per-subsystem test could afford; the general rule below asks the same
    question a dozen times and could not.
    """
    index: dict[str, set[str]] = {}
    candidates: list[Path] = [
        ROOT / name for name in _PRODUCTION_ENTRYPOINTS if (ROOT / name).is_file()
    ]
    for root in _PRODUCTION_ROOTS:
        base = ROOT / root
        if not base.is_dir():
            continue
        candidates.extend(base.rglob("*.py"))
    for path in candidates:
        rel = path.relative_to(ROOT)
        if _SKIP_PARTS.intersection(rel.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else ""
            )
            if name:
                index.setdefault(name, set()).add(str(rel))
    return {name: frozenset(files) for name, files in index.items()}


def _production_call_sites(function_name: str, defining_file: str) -> list[str]:
    """Production files that CALL ``function_name``.

    The defining file is excluded: a function referenced only inside the
    module that defines it has no caller in any sense that matters.
    """
    return sorted(_call_index().get(function_name, frozenset()) - {defining_file})


def test_latent_bridge_status_matches_its_wiring():
    """The status must track reality in whichever direction reality moves."""
    callers = _production_call_sites(
        "attach_latent_bridge", "core/consciousness/latent_bridge.py"
    )
    system = (ROOT / "core" / "consciousness" / "system.py").read_text(encoding="utf-8")

    if callers:
        assert '"latent_bridge"] = "unwired"' not in system, (
            "attach_latent_bridge() now has production callers "
            f"({callers}) — the consciousness layer status still reports it "
            "as unwired. Update the status to match."
        )
    else:
        assert '"latent_bridge"] = "unwired"' in system, (
            "attach_latent_bridge() has no production caller, so the backward "
            "hidden-state path is not live. The layer status must say so — it "
            "previously said 'deferred', which reads as a promise that "
            "something will attach it."
        )
        assert '"latent_bridge"] = "deferred"' not in system, (
            "'deferred' is a claim about a future event. Nothing redeems it "
            "here, so it must not be used."
        )


#: Capabilities that are BUILT, correct, and deliberately not wired, each
#: with a reason that is about the runtime rather than the code. Being on
#: this list is a tracked state, not an excuse: the test below fails the
#: moment a module joins it without being added here.
_TRACKED_UNWIRED: dict[str, str] = {
    "core/brain/cross_tier_verifier.py": (
        "weak generator / strong verifier needs the 72B tier, and the MLX "
        "hot-swap makes loading it expensive enough that no live path calls "
        "it. The technique is real and the implementation is sound; what is "
        "missing is a model lane, not code."
    ),
    "core/self_modification/lineage_enclosure.py": (
        "the boundary whole-agent lineage runs inside: caps on generations, "
        "population, disk and wall clock; writes kept out of the live state "
        "root; identity-bearing configuration refused rather than stripped. "
        "Started by a person, never by a running Aura, which is why it has "
        "no caller."
    ),
    "core/self_modification/lineage.py": (
        "heritable variation + selection at the ORGANISM level. Reachable "
        "only through `Enclosure.manager`, which caps what a run may spend, "
        "keeps its writes out of the live state root, and refuses to let a "
        "child inherit an identity. Nothing in the runtime starts one — "
        "organism reproduction is a decision about what Aura is, not a "
        "wiring task."
    ),
}


def test_no_module_is_unwired_without_being_tracked():
    """The invariant is not "nothing is unwired" — it is "nothing is unwired
    QUIETLY".

    This replaces a test that named ONE module, which could only ever catch
    that one. core/being/closed_loop_controller.py was deleted along with
    activation_coupler.py and continuum_adapter.py, which existed only to
    serve it: all three duplicated capabilities that ARE live elsewhere —
    residual steering in mlx_worker, action policy in policy_coupler — so
    the cluster made the system look like it had two closed loops when it
    had one.

    What survives is different in kind. A capability blocked on a model lane
    is parked, not dead, and deleting it would lose real work to make a test
    green. So the rule is that being unwired must be DECLARED in the module
    and ENUMERATED here with a reason — anything else is accumulation.
    """
    declaring = set(_modules_declaring_unwired())
    untracked = sorted(declaring - set(_TRACKED_UNWIRED))

    assert not untracked, (
        f"these modules declare themselves unwired and are not tracked: "
        f"{untracked}. Wire it, delete it, or add it to _TRACKED_UNWIRED "
        "with the reason — substantial, tested and uninvoked reads exactly "
        "like working from the outside."
    )


def test_every_tracked_unwired_module_still_exists():
    """A stale entry hides the fact that the register stopped being real."""
    missing = [name for name in _TRACKED_UNWIRED if not (ROOT / name).exists()]

    assert not missing, (
        f"_TRACKED_UNWIRED names modules that are gone: {missing}. Remove the "
        "entry; a register listing things that do not exist cannot be trusted "
        "about the things that do."
    )


def test_every_tracked_unwired_module_says_so_in_its_own_docstring():
    """The register and the module must agree.

    If only the register says it, a reader opening the file learns nothing.
    If only the module says it, nobody is counting.
    """
    declaring = set(_modules_declaring_unwired())
    silent = sorted(set(_TRACKED_UNWIRED) - declaring)

    assert not silent, (
        f"tracked as unwired but no longer declaring it: {silent}. Either it "
        "got wired — say so and remove it here — or the honesty was deleted."
    )


def test_the_call_site_scanner_actually_finds_calls():
    """A scanner that finds nothing would pass both tests above vacuously."""
    # record_degradation is called all over core/; if this comes back empty the
    # scanner is broken and the assertions above prove nothing.
    hits = _production_call_sites("record_degradation", "core/runtime/errors.py")
    assert len(hits) > 20, f"scanner found only {len(hits)} callers; it is broken"

def test_verifier_curriculum_declares_that_it_is_not_wired():
    """Found by the residue sweep, same shape as the two above."""
    callers = _production_call_sites(
        "boot_verifier_curriculum", "core/brain/verifier_curriculum.py"
    ) + _production_call_sites(
        "get_verifier_curriculum", "core/brain/verifier_curriculum.py"
    )
    module = (ROOT / "core" / "brain" / "verifier_curriculum.py").read_text(
        encoding="utf-8"
    )
    if callers:
        assert "NOT WIRED INTO THE LIVE RUNTIME" not in module, (
            f"verifier_curriculum now has production callers ({callers}); the "
            "module still declares itself unwired."
        )
    else:
        assert "NOT WIRED INTO THE LIVE RUNTIME" in module, (
            "Neither boot_verifier_curriculum() nor get_verifier_curriculum() "
            "has a production caller and nothing reads the ServiceContainer "
            "key it registers. The module must say so."
        )


def _key_readers(key: str) -> list[str]:
    """Production files that resolve a ServiceContainer key by name."""
    hits: list[str] = []
    for path in (ROOT / "core").rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if f'"{key}"' in text and (
            "optional_service(" in text or "ServiceContainer.get(" in text
        ):
            hits.append(str(path))
    return hits


def test_the_verifier_foundry_is_live_and_not_mislabelled():
    """The contrast case: the foundry IS wired, so it must not be declared dead.

    latent_cortex_service calls get_verifier_foundry() directly, so
    'delete the uncalled thing' applied bluntly would have taken a live
    capability with it.

    This test used to also assert that ``boot_verifier_foundry`` stayed
    deleted, on the grounds that nothing read the ``verifier_foundry``
    container key. Both halves of that premise have since become false:
    aura_main imports the wrapper (and logged an ImportError on every boot
    while it was missing), and procedural_memory and verifier_curriculum both
    resolve the key. So the assertion now runs the other way — the wrapper must
    exist BECAUSE it is imported, and the key must have readers.
    """
    callers = _production_call_sites(
        "get_verifier_foundry", "core/brain/verifiers/foundry.py"
    )
    assert callers, "get_verifier_foundry lost its production callers"
    module = (ROOT / "core" / "brain" / "verifiers" / "foundry.py").read_text(
        encoding="utf-8"
    )
    assert "NOT WIRED" not in module

    boot_callers = _production_call_sites(
        "boot_verifier_foundry", "core/brain/verifiers/foundry.py"
    )
    if "def boot_verifier_foundry" in module:
        assert boot_callers, (
            "boot_verifier_foundry exists with no production caller — either "
            "wire it or delete it"
        )
        assert _key_readers("verifier_foundry"), (
            "the wrapper registers a container key nothing reads"
        )
    else:
        assert not boot_callers, (
            "boot_verifier_foundry is imported but not defined; every boot "
            "will log an ImportError and run without the foundry"
        )


def test_cross_tier_verifier_declares_that_it_is_not_wired():
    """Found by the CP126 pass, same shape as the three above.

    Its docstring claimed it "wires to the live Solver tier in production".
    That was a claim about a call site that does not exist.
    """
    callers = _production_call_sites(
        "get_cross_tier_verifier", "core/brain/cross_tier_verifier.py"
    ) + _production_call_sites(
        "CrossTierVerifier", "core/brain/cross_tier_verifier.py"
    )
    module = (ROOT / "core" / "brain" / "cross_tier_verifier.py").read_text("utf-8")
    if callers:
        assert "NOT WIRED INTO THE LIVE RESPONSE PATH" not in module, (
            f"cross-tier verification now has production callers ({callers}); "
            "the module still declares itself unwired."
        )
    else:
        assert "NOT WIRED INTO THE LIVE RESPONSE PATH" in module, (
            "neither get_cross_tier_verifier() nor CrossTierVerifier has a "
            "production caller, so cross-tier verification is not live."
        )


# ── the general rule ─────────────────────────────────────────────────────
#
# The tests above are hand-written, one per subsystem, which means a module
# that declares itself unwired and never gets a test is unchecked — and
# LineageManager was exactly that: whole-agent reproduction, modelled and
# tested, no normal-runtime caller, nothing verifying either half.
#
# This makes the declaration self-checking for every module, present and
# future. The convention it enforces is small: a module that says it is not
# wired must name its entry points in double backticks in the docstring, and
# every one of those that this module actually defines must have no
# production caller.

#: Phrases a module uses to declare it has no production caller. Matched
#: against the module docstring only — a comment deep in a function is not a
#: declaration anybody reads.
_UNWIRED_MARKERS = (
    "NOT WIRED",
    "UNWIRED",
    "no production caller",
    "has no caller",
)

#: Modules whose "not wired" text is about something other than themselves.
_NOT_A_SELF_DECLARATION = frozenset(
    {
        # Describes the wiring status of OTHER subsystems it inventories.
        "core/consciousness/candidate_gate_inventory.py",
    }
)


def _modules_declaring_unwired() -> dict[str, tuple[str, set[str]]]:
    """{path: (docstring, entry-point names it defines and names in backticks)}."""
    found: dict[str, tuple[str, set[str]]] = {}
    for root in _PRODUCTION_ROOTS:
        base = ROOT / root
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            rel = str(path.relative_to(ROOT))
            if _SKIP_PARTS.intersection(Path(rel).parts) or rel in _NOT_A_SELF_DECLARATION:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            doc = ast.get_docstring(tree) or ""
            if not any(marker in doc for marker in _UNWIRED_MARKERS):
                continue
            defined = {
                node.name
                for node in tree.body
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
            }
            quoted = set(re.findall(r"``([A-Za-z_][A-Za-z0-9_]*)(?:\(\))?``", doc))
            # A module explaining that SOMETHING ELSE was unwired is not
            # declaring itself dead. core/skills/skill_retrieval.py exists
            # because SkillLibrary.get_available_skills_prompt() had no
            # callers; it says so in its first paragraph, names that function
            # in backticks, and defines none of it. Reading that as a
            # self-declaration made the module that FIXED a dead capability
            # look like a dead capability.
            marker_lines = [
                line
                for line in doc.splitlines()
                if any(marker in line for marker in _UNWIRED_MARKERS)
            ]
            near_marker = set()
            for line in marker_lines:
                near_marker.update(
                    re.findall(r"``([A-Za-z_][A-Za-z0-9_]*)(?:\(\))?``", line)
                )
            if near_marker and not (near_marker & defined):
                continue
            found[rel] = (doc, defined & quoted)
    return found


def test_every_module_declaring_itself_unwired_names_an_entry_point():
    """A declaration with nothing to check is prose, and prose drifts."""
    unnamed = [
        path for path, (_doc, entries) in _modules_declaring_unwired().items()
        if not entries
    ]
    assert not unnamed, (
        "these modules declare they are not wired but name no entry point this "
        f"file can verify: {unnamed}. Put the function or class in double "
        "backticks in the docstring so the claim is checkable."
    )


def test_no_module_declaring_itself_unwired_actually_has_a_caller():
    """The direction that catches someone wiring it and not saying so.

    Reach is transitive, so a call site inside a module that is itself
    unreached does not make the callee reached. Without that, enclosing an
    unwired mechanism behind a boundary — which is more careful, not less —
    would read here as having wired it.
    """
    declaring = _modules_declaring_unwired()
    unreached = set(declaring)
    # Fixpoint: a declaring module stays unreached while every call site of
    # its entry points lies in another module that is itself unreached.
    changed = True
    while changed:
        changed = False
        for path in sorted(unreached):
            _doc, entries = declaring[path]
            reached_by = {
                caller
                for entry in entries
                for caller in _production_call_sites(entry, path)
                if caller not in unreached
            }
            if reached_by:
                unreached.discard(path)
                changed = True

    wired_but_denying: dict[str, dict[str, list[str]]] = {}
    for path, (_doc, entries) in declaring.items():
        if path in unreached:
            continue
        callers = {
            entry: [c for c in _production_call_sites(entry, path) if c not in unreached]
            for entry in sorted(entries)
        }
        real = {entry: hits for entry, hits in callers.items() if hits}
        if real:
            wired_but_denying[path] = real
    assert not wired_but_denying, (
        "these modules declare they have no production caller and do: "
        f"{wired_but_denying}. Either the wiring is accidental, or the module "
        "is live and its docstring is now the misleading part."
    )


def test_the_declarations_cover_the_subsystems_that_were_found_unwired():
    """A regression guard on the sweep itself.

    If one of these stops declaring its status, it is because someone either
    wired it (fine — the test above will say so) or deleted the honesty
    (not fine).
    """
    declaring = set(_modules_declaring_unwired())
    for expected in _TRACKED_UNWIRED:
        assert expected in declaring, (
            f"{expected} no longer declares its wiring status. It was found "
            "substantial, tested and uninvoked; if that changed, say so."
        )


ENCLOSURE = "core/self_modification/lineage_enclosure.py"


def test_the_enclosure_is_the_only_way_to_a_lineage_manager():
    """Whole-agent reproduction runs inside a boundary or it does not run.

    The interesting half is what it lets someone claim. Aura's ALife
    substrate IS causal — the pattern replicator mutates real neural-mesh
    weight matrices in place. Whole-agent reproduction is that other module,
    and it is reachable only through an enclosure that caps what a run may
    spend, keeps its writes out of the live state root, and refuses to let a
    child inherit an identity. "ALife cognitive agent" is supportable;
    "self-reproducing digital organism" is not, and the difference is a
    boundary somebody has to cross on purpose.
    """
    module_path = "core/self_modification/lineage.py"
    constructors = _production_call_sites("LineageManager", module_path)
    assert constructors == [ENCLOSURE], (
        f"LineageManager is constructed by {constructors}. The enclosure must "
        "be the only way in: every other construction site is a lineage with "
        "no resource, authority or identity boundary on it."
    )
    assert not _key_readers("lineage_manager"), (
        "something resolves the lineage_manager service key again. The "
        "registration was removed because a registered service with no reader "
        "reads as integration, which for whole-agent reproduction is the one "
        "place that misreading matters."
    )


def test_nothing_in_the_live_runtime_starts_a_lineage():
    """An enclosure is started by a person, not by a running Aura."""
    callers = _production_call_sites("Enclosure", ENCLOSURE)
    assert not callers, (
        f"{callers} construct a lineage Enclosure. Nothing in the runtime may "
        "start whole-agent reproduction on its own; that is a decision about "
        "what Aura is, taken by someone, in a call they can see."
    )


def test_the_lineage_default_path_stays_out_of_live_state():
    """The default was the live data directory, in a module documented unwired."""
    enclosure = (ROOT / ENCLOSURE).read_text(encoding="utf-8")
    assert "db_path=self._root" in enclosure, (
        "the enclosure no longer passes an explicit database path, so a run "
        "falls back to LineageManager's own default, which is the live data "
        "directory"
    )


def test_a_registered_service_nobody_reads_is_not_evidence_of_wiring():
    """The scanner half of the test above, proven not to be vacuous."""
    # A key that IS read, so an always-empty _key_readers would be caught.
    assert _key_readers("memory_facade"), (
        "_key_readers found no reader for a key that is definitely read; "
        "the helper is broken and the assertions above prove nothing."
    )
