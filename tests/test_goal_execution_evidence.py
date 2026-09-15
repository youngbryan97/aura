"""Real execution distinguishes useful results, checked effects and failed work."""

from types import SimpleNamespace

import pytest

from core.agency.goal_planner import GoalPlanner
from core.agency.goal_pursuit import GoalPursuitEngine
from core.brain.reasoning_amplifier import AmplifiedResult, VerifierOutcome, amplify
from core.capabilities.post_action_verifier import VerificationResult
from core.skills.fluid_executor import ExecutionReceipt, FluidExecutor, Step, StepActionResult, StepResult


async def no_wait(_seconds):
    return None


class Allowed:
    def approve(self, name):
        return True


def executor(**kwargs):
    return FluidExecutor(gateway=Allowed(), sleep=no_wait, **kwargs)


@pytest.mark.asyncio
async def test_omitted_check_allows_work_without_verified_progress():
    async def act():
        return StepActionResult(True)

    receipt = await executor().run("produce", [Step("produce", act)])
    assert receipt.completed
    assert receipt.verified_progress == 0
    assert not receipt.verification_complete
    assert receipt.steps[0].action_completed
    assert receipt.steps[0].verification_outcome == "not_requested"
    assert receipt.to_dict()["verification_complete"] is False


@pytest.mark.asyncio
async def test_explicit_action_failure_cannot_pass_unconditional_check():
    async def act():
        return StepActionResult(False, "no result")

    receipt = await executor().run("produce", [Step("produce", act, max_retries=0)])
    assert not receipt.completed
    assert not receipt.steps[0].action_completed
    assert receipt.steps[0].detail == "no result"


@pytest.mark.asyncio
async def test_unavailable_observation_retries_only_observer():
    actions = []
    observations = []

    async def act():
        actions.append("effect")

    class Verifier:
        async def verify(self, predicate, args):
            observations.append("read")
            if len(observations) < 3:
                return VerificationResult.unavailable(predicate, args, "temporarily offline")
            return VerificationResult(predicate, args, True, evidence="effect observed")

    receipt = await executor(verifier=Verifier()).run("effect", [Step("effect", act, verify="measured")])
    assert receipt.completed and receipt.verification_complete
    assert actions == ["effect"] and len(observations) == 3
    assert receipt.steps[0].attempts == 1


@pytest.mark.asyncio
async def test_missing_observation_cannot_replan_and_repeat_effect():
    actions = []
    replans = []

    async def act():
        actions.append("effect")

    class Verifier:
        async def verify(self, predicate, args):
            return VerificationResult.unavailable(predicate, args, "offline")

    def replan(receipt):
        replans.append(receipt)
        return [Step("repeat", act)]

    outcome = await GoalPursuitEngine(executor=executor(verifier=Verifier())).pursue(
        "effect", [Step("effect", act, verify="measured")], replan=replan,
    )
    assert outcome.deferred and not outcome.completed and not outcome.verified
    assert actions == ["effect"] and not replans
    assert outcome.receipts[0].outcome == "awaiting_verification"
    assert outcome.receipts[0].steps[0].to_dict()["awaiting_verification"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["raises", "malformed"])
async def test_unavailable_and_malformed_verifiers_do_not_repeat_actions(kind):
    actions = []

    async def act():
        actions.append(1)

    class Verifier:
        async def verify(self, predicate, args):
            if kind == "raises":
                raise RuntimeError("offline")
            return {"success": True}

    receipt = await executor(verifier=Verifier()).run("effect", [Step("effect", act, verify="measured")])
    assert not receipt.completed and actions == [1]
    assert receipt.steps[0].awaiting_verification


@pytest.mark.asyncio
async def test_required_effect_check_cannot_be_overridden_by_action_assessment():
    async def act():
        return StepActionResult(True, assessment=VerificationResult("internal", {}, True))

    class Verifier:
        async def verify(self, predicate, args):
            return VerificationResult(predicate, args, False, evidence="effect absent")

    receipt = await executor(verifier=Verifier()).run(
        "effect", [Step("effect", act, verify="external", max_retries=0)],
    )
    assert not receipt.completed
    assert receipt.steps[0].verification_outcome == "measured_mismatch"


@pytest.mark.asyncio
@pytest.mark.parametrize("ok,answer", [(False, ""), (True, "")])
async def test_unsolved_computation_does_not_complete_goal(monkeypatch, ok, answer):
    from core.brain import tool_augmented_reasoning

    monkeypatch.setattr(tool_augmented_reasoning, "solve_exact", lambda _: SimpleNamespace(
        ok=ok, answer=answer, detail="unsupported computation",
    ))
    step = GoalPlanner()._compute_step("compute")
    step.max_retries = 0
    result = await executor().run_step(step)
    assert not result.ok and not result.verified


@pytest.mark.asyncio
async def test_real_exact_computation_completes_without_invented_assessment():
    planner = GoalPlanner()
    receipt = await executor().run("compute", [planner._compute_step("47 * 89")])
    assert receipt.completed and "4183" in planner.last.answer
    assert not receipt.verification_complete


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict,expected_ok,expected_verified", [
    (VerifierOutcome.PASS, True, True),
    (VerifierOutcome.UNKNOWN, True, False),
    (VerifierOutcome.FAIL, False, False),
])
async def test_reasoning_assessment_survives_planning(monkeypatch, verdict, expected_ok, expected_verified):
    from core.brain.reasoning_amplifier import DeliberationEngine

    async def deliberate(*args, **kwargs):
        return AmplifiedResult("produced answer", 0.8, 1, int(verdict == VerifierOutcome.PASS), 1.0,
            verified=verdict == VerifierOutcome.PASS, answer_verdict=verdict)

    monkeypatch.setattr(DeliberationEngine, "adaptive_deliberate", deliberate)
    planner = GoalPlanner()
    step = planner._reason_step("reason")
    step.max_retries = 0
    receipt = await executor().run("reason", [step])
    assert receipt.completed is expected_ok
    assert receipt.verification_complete is expected_verified
    assert planner.last.answer == "produced answer"


@pytest.mark.asyncio
async def test_amplifier_marks_selected_unknown_distinct_from_other_failed_candidate():
    async def verify(answer):
        return (VerifierOutcome.FAIL if answer == "bad" else VerifierOutcome.UNKNOWN), []

    result = await amplify(["bad", "unchecked"], verify=verify)
    assert result.answer == "unchecked" and result.verifier_checked
    assert result.answer_verdict == VerifierOutcome.UNKNOWN
    rejected = await amplify(["bad"], verify=verify)
    assert rejected.answer_verdict == VerifierOutcome.FAIL
    assert rejected.to_dict()["answer_verdict"] == VerifierOutcome.FAIL.value


@pytest.mark.asyncio
async def test_denied_reach_does_not_complete_goal():
    class Reach:
        async def get(self, url):
            return SimpleNamespace(ok=False, reason="denied", body_preview="")

    step = GoalPlanner(reach=Reach())._reach_step("fetch https://example.com")
    step.max_retries = 0
    result = await executor().run_step(step)
    assert not result.ok and not result.verified


@pytest.mark.asyncio
async def test_pursuit_preserves_unverified_completion():
    async def act():
        return StepActionResult(True)

    outcome = await GoalPursuitEngine(executor=executor()).pursue("produce", [Step("produce", act)])
    assert outcome.completed and not outcome.verified
    assert outcome.to_dict()["verified"] is False


def test_replanner_targets_failure_not_earlier_unchecked_success():
    from core.agency.replanning import failed_step

    receipt = ExecutionReceipt("goal", False, steps=[
        StepResult("read", ok=True, verified=False),
        StepResult("compute", ok=False, verified=False),
    ])
    assert failed_step(receipt) == "compute"


def test_completion_evidence_invariant():
    from core.skills.fluid_executor import _action_evidence_invariant

    assert _action_evidence_invariant() == ()


@pytest.mark.asyncio
async def test_empty_recombination_cannot_transfer_its_verdict_to_a_subanswer():
    from core.brain.reasoning_amplifier import DeliberationEngine

    async def generate(question, temperature):
        return "partial subanswer"

    async def decompose(question):
        return ["part"]

    async def recombine(question, parts):
        return ""

    async def verify(answer):
        return VerifierOutcome.PASS, []

    result = await DeliberationEngine(n_samples=1).decompose_and_solve(
        "whole", generate, decompose, recombine=recombine, verify=verify,
    )
    assert result.answer == "partial subanswer"
    assert not result.verified
    assert result.answer_verdict == VerifierOutcome.UNKNOWN
