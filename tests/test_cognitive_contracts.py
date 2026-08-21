"""Cognition declares itself, and the declaration is checked against behaviour.

The criticism this answers: Aura has explicit criteria everywhere — throttles,
idle windows, φ-scaled thresholds, eight-dimension score vectors, hysteresis
windows — and no universal, versioned schema that states any of it in one
auditable language. ``PhaseSpec`` carried a name, an attribute and a class.

A declaration nothing verifies is documentation with a dataclass around it, so
these tests hold the mechanism to the standard that makes it worth having:

* the baseline of undeclared phases only shrinks;
* a contract's thresholds are the LIVE constants, not copies of them;
* a phase marked ``thresholds_exhaustive`` has no bare numeric comparison left;
* an undeclared write is detected by measurement, not by self-report;
* the provenance record answers "why" from receipts rather than from prose.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap

import pytest

from core.runtime.cognitive_contract import (
    UNCONTRACTED_PHASES,
    BranchSpec,
    CognitiveTransformContract,
    all_contracts,
    contract_coverage_report,
    contract_for,
    register_contract,
    watched_fields,
)
from core.runtime.cognitive_provenance import (
    begin_transformation,
    open_tick,
    recent_graphs,
    recording_tick,
    reset_provenance_for_test,
    why_field_changed,
    write_profile,
)
from core.runtime.pipeline_blueprint import (
    kernel_phase_attribute_order,
    phase_class_for_attribute,
)

#: The size of the baseline when the mechanism landed. It only shrinks.
# Was 28 — one phase declared itself and twenty-eight did not. Ten more were
# written from measurement (tools/observe_phase_writes.py runs the real phase
# against a real AuraState and reports what it moved), so the ceiling moves
# with them. It only ever moves down.
_BASELINE_CEILING = 18


class _Affect:
    def __init__(self) -> None:
        self.curiosity = 0.1
        self.social_hunger = 0.1
        self.arousal = 0.5


class _Cognition:
    def __init__(self) -> None:
        self.discourse_depth = 0
        self.conversation_energy = 0.5
        self.working_memory: list[dict[str, str]] = []
        self.pending_initiatives: list[dict[str, str]] = []


class _State:
    """The smallest object the digest walker needs. Not an AuraState.

    Deliberately structural: the contract machinery resolves dotted paths by
    attribute and mapping lookup, so anything shaped like a state works, and a
    test that needed a real AuraState would be testing AuraState.
    """

    def __init__(self) -> None:
        self.state_id = "s-test"
        self.version = 1
        self.updated_at = 1.0
        self.affect = _Affect()
        self.cognition = _Cognition()
        self.response_modifiers: dict[str, float] = {}


@pytest.fixture(autouse=True)
def _clean_provenance():
    reset_provenance_for_test()
    yield
    reset_provenance_for_test()


# ── The ratchet ────────────────────────────────────────────────────────────


def test_uncontracted_baseline_only_shrinks() -> None:
    assert len(UNCONTRACTED_PHASES) <= _BASELINE_CEILING, (
        "a phase was added to UNCONTRACTED_PHASES. The baseline only shrinks — "
        "a new phase ships with a contract, or the mechanism decays into a "
        "list of exemptions."
    )


def test_every_pipeline_phase_is_contracted_or_baselined() -> None:
    """No phase may be silently outside the scheme."""

    pipeline = {
        phase_class_for_attribute(attribute) for attribute in kernel_phase_attribute_order()
    }
    contracted = set(all_contracts())
    unaccounted = sorted(pipeline - contracted - set(UNCONTRACTED_PHASES))
    assert unaccounted == [], (
        f"{unaccounted} run in the pipeline with neither a contract nor a "
        "baseline entry"
    )


def test_baseline_names_are_real_phases() -> None:
    """A baseline that accumulates dead names stops meaning anything."""

    pipeline = {
        phase_class_for_attribute(attribute) for attribute in kernel_phase_attribute_order()
    }
    stale = sorted(name for name in UNCONTRACTED_PHASES if name not in pipeline)
    assert stale == [], f"{stale} are baselined but no longer in the pipeline"


def test_contracted_phases_are_not_also_baselined() -> None:
    overlap = sorted(set(all_contracts()) & set(UNCONTRACTED_PHASES))
    assert overlap == [], f"{overlap} have a contract and a baseline entry"


def test_coverage_report_is_derived_from_the_pipeline() -> None:
    report = contract_coverage_report()
    assert report["pipeline_phases"] == len(kernel_phase_attribute_order())
    assert report["contracted_count"] == len(report["contracted"])


def test_memory_consolidation_declares_every_live_state_write() -> None:
    contract = contract_for("MemoryConsolidationPhase")
    assert contract is not None
    assert {
        "cognition.coherence_score",
        "cognition.fragmentation_score",
        "cognition.long_term_memory",
        "cognition.modifiers",
        "cognition.working_memory",
        "health",
        "response_modifiers",
        "transition_cause",
    } <= set(contract.writes)


# ── Thresholds are the live constants, not copies ──────────────────────────


def test_initiative_generation_has_an_exhaustive_contract() -> None:
    contract = contract_for("InitiativeGenerationPhase")
    assert contract is not None, "the exemplar contract must exist"
    assert contract.thresholds_exhaustive is True
    assert contract.thresholds, "an exhaustive contract with no thresholds says nothing"
    assert "transition_cause" in contract.writes


@pytest.mark.parametrize(
    "name",
    sorted(
        name
        for name, contract in all_contracts().items()
        if contract.thresholds
    ),
)
def test_declared_thresholds_are_the_modules_own_constants(name: str) -> None:
    """A contract that copies a number is a second source of truth.

    And the copy is the one that goes stale. Identity against the module
    constant is what makes the declaration causal rather than descriptive.
    """

    contract = all_contracts()[name]
    module = importlib.import_module(contract.module)
    for key, value in contract.thresholds.items():
        assert hasattr(module, key), (
            f"{contract.module} does not define {key}, so the contract's "
            "threshold is a literal nobody is using"
        )
        assert getattr(module, key) == value, (
            f"{key} disagrees between {contract.module} and its contract"
        )


def _numeric_comparison_literals(func) -> list[float]:
    """Numeric literals used in comparisons inside a function body."""

    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    found: list[float] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for operand in [node.left, *node.comparators]:
            if isinstance(operand, ast.Constant) and isinstance(
                operand.value, (int, float)
            ) and not isinstance(operand.value, bool):
                found.append(operand.value)
    return found


@pytest.mark.parametrize(
    "name",
    sorted(
        name
        for name, contract in all_contracts().items()
        if contract.thresholds_exhaustive
    ),
)
def test_exhaustive_contracts_leave_no_bare_comparison_literal(name: str) -> None:
    """``thresholds_exhaustive`` is a claim, so it is checked.

    A phase that says it declares every constant it branches on, and then
    compares against 0.8 inline, has a criterion nothing can read — which is
    the state the whole contract layer exists to leave behind.
    """

    module = importlib.import_module(all_contracts()[name].module)
    phase_cls = getattr(module, name)
    literals = _numeric_comparison_literals(phase_cls.execute)
    # 0 and 1 are structural (empty checks, index bounds), not policy.
    policy_literals = [value for value in literals if value not in (0, 1)]
    assert policy_literals == [], (
        f"{name}.execute compares against {policy_literals} inline while its "
        "contract claims thresholds_exhaustive. Name the constant and "
        "reference it from both places."
    )


def test_contract_hash_changes_when_a_threshold_changes() -> None:
    """Receipts must not claim provenance from a contract that has moved."""

    base = CognitiveTransformContract(
        name="_probe", version="1.0", purpose="p", thresholds={"T": 0.5}
    )
    moved = CognitiveTransformContract(
        name="_probe", version="1.0", purpose="p", thresholds={"T": 0.6}
    )
    assert base.content_hash != moved.content_hash


def test_two_different_contracts_for_one_phase_are_refused() -> None:
    register_contract(
        CognitiveTransformContract(name="_dup", version="1.0", purpose="a")
    )
    with pytest.raises(ValueError):
        register_contract(
            CognitiveTransformContract(name="_dup", version="2.0", purpose="b")
        )


# ── Undeclared writes are detected by measurement ──────────────────────────


def test_undeclared_write_is_caught_without_the_phase_reporting_it() -> None:
    register_contract(
        CognitiveTransformContract(
            name="_narrow",
            version="1.0",
            purpose="only allowed to touch curiosity",
            reads=("affect.curiosity",),
            writes=("affect.curiosity",),
        )
    )
    state = _State()
    with recording_tick(objective="probe") as graph:
        transformation = begin_transformation("_narrow", state)
        # The phase does something it never declared.
        state.affect.curiosity = 0.9
        state.affect.arousal = 0.05
        transformation.complete(state)

    receipt = graph.receipts[-1]
    assert "affect.curiosity" in receipt.observed_writes
    assert receipt.undeclared_writes == ("affect.arousal",)
    assert receipt.honoured_contract is False
    assert graph.contract_violations


def test_negative_validation_can_measure_without_filing_runtime_incident(
    monkeypatch,
) -> None:
    """A manufactured boot probe is evidence, not a live phase failure."""

    from core.runtime import cognitive_provenance

    register_contract(
        CognitiveTransformContract(
            name="_isolated_negative_probe",
            version="1.0",
            purpose="measure an expected undeclared write",
            writes=("affect.curiosity",),
        )
    )
    reported = []
    monkeypatch.setattr(
        cognitive_provenance,
        "_report_violation",
        lambda name, fields: reported.append((name, fields)),
    )
    state = _State()
    with recording_tick(objective="isolated validation") as graph:
        transformation = begin_transformation("_isolated_negative_probe", state)
        state.affect.arousal = 0.05
        transformation.complete(state, publish_violation=False)

    assert graph.receipts[-1].undeclared_writes == ("affect.arousal",)
    assert reported == []


def test_declared_write_is_not_a_violation() -> None:
    register_contract(
        CognitiveTransformContract(
            name="_wide",
            version="1.0",
            purpose="allowed both",
            writes=("affect.curiosity", "affect.arousal"),
        )
    )
    state = _State()
    with recording_tick() as graph:
        transformation = begin_transformation("_wide", state)
        state.affect.curiosity = 0.9
        state.affect.arousal = 0.05
        transformation.complete(state)
    assert graph.receipts[-1].honoured_contract is True


def test_uncontracted_phase_still_reports_what_it_wrote() -> None:
    """The productive half of the ratchet: measurement before declaration."""

    state = _State()
    with recording_tick() as graph:
        transformation = begin_transformation("_unknown_phase", state)
        state.affect.arousal = 0.05
        transformation.complete(state)
    receipt = graph.receipts[-1]
    assert receipt.contract_hash == ""
    assert "affect.arousal" in receipt.observed_writes
    assert receipt.undeclared_writes == ()

    profile = write_profile()
    assert profile["_unknown_phase"]["observed_writes"] == ["affect.arousal"]
    assert profile["_unknown_phase"]["contracted"] is False


# ── The record answers "why" ───────────────────────────────────────────────


def test_provenance_answers_why_a_field_moved() -> None:
    register_contract(
        CognitiveTransformContract(
            name="_decider",
            version="1.0",
            purpose="move curiosity on a named branch",
            writes=("affect.curiosity",),
            branches=(BranchSpec("fired", "curiosity > threshold", "decay it"),),
        )
    )
    state = _State()
    with recording_tick(objective="why probe"):
        transformation = begin_transformation("_decider", state)
        transformation.note_branch("fired", threshold=0.8, curiosity=0.91)
        state.affect.curiosity = 0.51
        transformation.complete(state)

    answer = why_field_changed("affect.curiosity")
    assert answer["found"] is True
    assert answer["transform"] == "_decider"
    assert answer["branch"] == "fired"
    assert answer["criteria"]["threshold"] == 0.8


def test_why_reports_absence_rather_than_inventing_an_answer() -> None:
    answer = why_field_changed("affect.curiosity")
    assert answer["found"] is False
    assert "searched_ticks" in answer


def test_narration_is_built_from_receipts() -> None:
    state = _State()
    with recording_tick(objective="narrate probe", priority=True) as graph:
        transformation = begin_transformation("_decider", state)
        transformation.note_branch("fired", threshold=0.8)
        state.affect.curiosity = 0.4
        transformation.complete(state)
    text = graph.narrate()
    assert "_decider" in text
    assert "branch: fired" in text
    assert "affect.curiosity" in text
    assert "(foreground)" in text


def test_open_tick_records_a_tick_that_never_finished() -> None:
    """A tick that dies mid-phase is the one whose record is worth the most."""

    state = _State()
    open_tick(objective="crashed tick")
    transformation = begin_transformation("_decider", state)
    state.affect.curiosity = 0.2
    transformation.complete(state)
    # No close_tick — the process "died" here.
    graphs = recent_graphs(4)
    assert graphs and graphs[-1].objective == "crashed tick"
    assert graphs[-1].receipts


def test_kernel_measures_provenance_around_every_phase() -> None:
    """The wiring, asserted against the kernel's own source.

    ``_execute_phase_with_timing`` is the single seam every kernel phase passes
    through. If provenance moves out of it, some phases stop being recorded and
    the graph silently becomes partial.
    """

    from core.kernel.aura_kernel import AuraKernel

    source = inspect.getsource(AuraKernel._execute_phase_with_timing)
    assert "begin_transformation" in source
    assert "transformation.complete" in source

    # Follows self.<helper>() delegation, so extracting part of tick into
    # _tick_body does not break a contract about the tick pipeline.
    from tools.find_extraction_seam import implementation_source

    tick_source = implementation_source(AuraKernel, "tick")
    assert "open_tick" in tick_source


# ── The bootstrap ──────────────────────────────────────────────────────────
#
# `watched_fields()` is derived FROM the contracts. That made discovery
# circular: an uncontracted phase could only be seen touching fields some
# already-written contract happened to name, so `write_profile` — documented
# as "the productive end of the ratchet" — reported nearly nothing for the
# phases it exists to describe. The method was right and could not run.


def test_an_uncontracted_phase_is_watched_beyond_the_declared_fields() -> None:
    from core.runtime.cognitive_contract import discovery_paths
    from core.state.aura_state import AuraState

    state = AuraState()
    discovery = set(discovery_paths(state))
    declared = set(watched_fields())

    assert discovery - declared, (
        "discovery adds nothing beyond the declared fields, so an uncontracted "
        "phase can only be observed touching what other phases declared — the "
        "ratchet has no productive end again"
    )


def test_discovery_sees_a_write_no_contract_declares() -> None:
    """The concrete failure, reproduced.

    A phase moving a field outside every contract must still show up, or its
    contract can never be grounded in measurement.
    """
    from core.runtime.cognitive_contract import (
        diff_digests,
        discovery_paths,
        observed_field_digest,
    )
    from core.state.aura_state import AuraState

    state = AuraState()
    undeclared = next(
        (path for path in discovery_paths(state) if path not in set(watched_fields())),
        "",
    )
    assert undeclared, "no undeclared path available to test with"

    before = observed_field_digest(state, discover=True)
    root, _, leaf = undeclared.partition(".")
    target = getattr(state, root)
    if leaf:
        setattr(target, leaf, {"probe": "changed"})
    else:
        setattr(state, root, {"probe": "changed"})
    after = observed_field_digest(state, discover=True)

    assert undeclared in diff_digests(before, after)
    assert undeclared not in diff_digests(
        observed_field_digest(state, paths=tuple(watched_fields())),
        observed_field_digest(state, paths=tuple(watched_fields())),
    )


def test_a_contracted_phase_narrows_back_to_its_declared_fields() -> None:
    """Discovery is the bootstrap, not the steady state.

    Widening enforcement for contracted phases would digest the whole state
    surface on the foreground path for every phase, every tick.
    """
    from core.runtime.cognitive_contract import discovery_paths, observed_field_digest
    from core.state.aura_state import AuraState

    state = AuraState()

    narrow = observed_field_digest(state, discover=False)
    wide = observed_field_digest(state, discover=True)

    assert len(wide) > len(narrow)
    assert set(narrow) <= set(wide)
    assert set(discovery_paths(state)) <= set(wide)


def test_the_observation_tool_exists_and_names_its_own_limits() -> None:
    """Contracts written from a probe must say the probe is a floor.

    Phases whose interesting branch needs a model report only their no-model
    path here, and a contract that presented that as the whole write set
    would be the same overclaim in a new place.
    """
    from pathlib import Path

    tool = Path(__file__).resolve().parents[1] / "tools" / "observe_phase_writes.py"

    assert tool.exists()
    body = tool.read_text("utf-8")
    assert "floor, not a ceiling" in body


def test_every_module_declaring_a_contract_is_in_the_registry() -> None:
    """A contract that registers only by luck understates coverage.

    `register_contract` runs at import time, so a declaration in a module
    nobody imported is invisible — and `contract_coverage_report` would report
    that phase as uncontracted, which reads as a measurement and is not one.
    """
    from pathlib import Path

    from core.runtime.phase_contract_registry import CONTRACT_MODULES

    root = Path(__file__).resolve().parents[1]
    declaring: set[str] = set()
    for path in (root / "core").rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        if path.name in {"cognitive_contract.py", "phase_contract_registry.py"}:
            continue
        body = path.read_text("utf-8", errors="ignore")
        if "register_contract(" not in body:
            continue
        # MODULE-LEVEL only. A probe that registers a throwaway contract
        # inside a function is not a declaration and must not be pulled into
        # the import list — importing it would prove nothing and the list is
        # supposed to name real phases.
        try:
            tree = ast.parse(body)
        except SyntaxError:
            continue
        module_level = any(
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", "") == "register_contract"
            for node in tree.body
        )
        if module_level:
            declaring.add(
                str(path.relative_to(root)).removesuffix(".py").replace("/", ".")
            )

    missing = sorted(declaring - set(CONTRACT_MODULES))
    assert missing == [], (
        f"{missing} declare a contract but are not in CONTRACT_MODULES, so it "
        "registers only if something else happens to import them"
    )


def test_the_coverage_report_is_the_same_from_a_cold_registry() -> None:
    """The number must not depend on the caller's import graph."""
    report = contract_coverage_report()

    assert report["unloadable_contract_modules"] == []
    assert report["contracted_count"] >= 11, (
        "coverage regressed below the contracts written from measurement"
    )


# ── Declared reads must be honest ──────────────────────────────────────────
#
# `writes` is measured. `reads` is not — nothing observes them — so a reads
# tuple is exactly the kind of declaration that drifts into describing what
# the author believed. It did, immediately: the first ConsciousnessPhase
# contract written this session declared affect and soma reads because that
# is what a phase called "consciousness" ought to consult. The phase reads no
# state at all; it pulls from two services.


def test_declared_reads_are_either_visible_or_declared_indirect() -> None:
    """A read must appear in the phase's module, or the contract must say why not.

    Delegation is legitimate — a phase that reads through its engine really
    does read those fields. Silent fabrication is not, and the two are
    indistinguishable unless the contract states which it is.
    """
    from pathlib import Path

    offenders: list[str] = []
    for name, contract in sorted(all_contracts().items()):
        if not contract.reads or not contract.module:
            continue
        module = importlib.import_module(contract.module)
        source = Path(module.__file__).read_text("utf-8")
        # Exclude the contract declaration itself, or every read is trivially
        # "present" because the contract names it.
        marker = "register_contract("
        code = source[: source.index(marker)] if marker in source else source

        unseen = [path for path in contract.reads if path.split(".")[-1] not in code]
        if unseen and "delegate" not in contract.calibration_source:
            offenders.append(f"{name}: {unseen}")

    assert offenders == [], (
        "these contracts declare reads that do not appear in their own module "
        f"and do not say the reads happen through a delegate: {offenders}"
    )


def test_a_contract_that_reads_nothing_says_so_rather_than_guessing() -> None:
    """The corrected case, pinned.

    ConsciousnessPhase consults the consciousness integration singleton and
    the causal world model service. Declaring affect/soma reads made the
    contract describe an intuition about the name.
    """
    contract = contract_for("ConsciousnessPhase")

    assert contract is not None
    assert contract.reads == (), (
        "ConsciousnessPhase declares AuraState reads again; it reads services, "
        "not state"
    )
    assert contract.side_effects, "the services it does consult must be declared"


def test_calibration_source_is_a_string_not_an_accidental_tuple() -> None:
    """A trailing comma inside the parens makes it a one-element tuple.

    Which happened, and it silently broke the delegate check above: every
    substring test against a tuple of one long string is False, so contracts
    that DID declare their delegation read as if they had not.
    """
    wrong = {
        name: type(contract.calibration_source).__name__
        for name, contract in all_contracts().items()
        if not isinstance(contract.calibration_source, str)
    }

    assert wrong == {}, f"calibration_source is not a string: {wrong}"


def test_every_contract_states_where_its_thresholds_came_from() -> None:
    """"judgement" is an honest answer; silence is not."""
    silent = sorted(
        name
        for name, contract in all_contracts().items()
        if not str(contract.calibration_source or "").strip()
    )

    assert silent == [], f"{silent} declare thresholds with no stated provenance"
