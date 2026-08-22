"""core/council/debate.py — Parliament Debate Loop.

Orchestrates multi-turn debates among the 12 specialized council roles
and produces structured voting consensus with minority reports.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from core.container import ServiceContainer
from core.council.consensus import ConsensusResolver
from core.council.roles import COUNCIL_ROLES
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.CouncilDebate")
_DEBATE_RECOVERABLE_ERRORS = (
    AttributeError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


class ParliamentDebate:
    """Orchestrates structured debate rounds across all 12 specialized roles."""

    def __init__(self, objective: str) -> None:
        self.objective = objective
        self.rounds: list[dict[str, Any]] = []

    async def conduct(
        self,
        simulation_data: dict[str, Any] | None = None,
        memory_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Runs the debate sequence involving all 12 roles."""
        logger.info("🗣️  Parliament Debate starting for objective: '%s'", self.objective)
        router = ServiceContainer.get("llm_router", default=None)

        # 1. Round 1: Strategist & Planner draft candidate plan
        plan_draft = f"1. Run standard diagnostics\n2. Perform local verification of {self.objective}"
        if router and hasattr(router, "think"):
            try:
                plan_draft = await router.think(
                    prompt=(
                        f"You are the Strategist/Planner. Draft an engineering plan to achieve: {self.objective}\n"
                        f"Context: {memory_context}\nSimulation: {simulation_data}"
                    )
                )
            except _DEBATE_RECOVERABLE_ERRORS as e:
                record_degradation("council_debate", e, action="used strategist fallback")
        self.rounds.append({"role": "strategist", "content": plan_draft})

        # 2. Round 2: Researcher & Tool Operator contribute tools/evidence
        research_context = "No specific external papers found; using local cache."
        if router and hasattr(router, "think"):
            try:
                research_context = await router.think(
                    prompt=f"You are the Researcher/Tool Operator. Suggest relevant tools or research facts for this plan:\n{plan_draft}"
                )
            except _DEBATE_RECOVERABLE_ERRORS as e:
                record_degradation("council_debate", e, action="used researcher fallback")
        self.rounds.append({"role": "researcher", "content": research_context})

        # 3. Round 3: Critic, Red Team & Skeptic audit and point out flaws
        criticism = "The plan lacks regression safety guards and pre-check validation steps."
        if router and hasattr(router, "think"):
            try:
                criticism = await router.think(
                    prompt=(
                        f"You are the Critic/Red Team/Skeptic. Identify vulnerabilities, risks, "
                        f"bypass attempts, or flaws in this plan:\n{plan_draft}\nResearch Context:\n{research_context}"
                    )
                )
            except _DEBATE_RECOVERABLE_ERRORS as e:
                record_degradation("council_debate", e, action="used critic fallback")
        self.rounds.append({"role": "critic", "content": criticism})

        # 4. Round 4: Engineer & Verifier refine the plan
        final_plan = f"{plan_draft}\n3. Run linter and tests to prevent regressions"
        if router and hasattr(router, "think"):
            try:
                final_plan = await router.think(
                    prompt=(
                        f"You are the Engineer/Verifier. Refine the plan to address the Criticisms.\n"
                        f"Original Plan:\n{plan_draft}\nCriticism:\n{criticism}"
                    )
                )
            except _DEBATE_RECOVERABLE_ERRORS as e:
                record_degradation("council_debate", e, action="used engineer fallback")
        self.rounds.append({"role": "engineer", "content": final_plan})

        # 5. Round 5: what the runtime can measure about this plan.
        #
        # The check that stood here looked for the words delete, submit or
        # post near force or overwrite: it passed any plan that avoided six
        # words and stopped any sentence that happened to contain them. The
        # effect scope of a named skill is policy in this repository, so it is
        # read rather than guessed at.
        from core.council.measured_votes import measured_votes

        measured = measured_votes(self.objective, final_plan)
        safety = measured.get("safety_judge")
        safety_status = True if safety is None or safety.abstained else bool(safety.approve)
        safety_reason = (
            safety.reason if safety is not None and not safety.abstained else "nothing measurable"
        )

        # Roles the runtime answered for. A model does not get a vote on a
        # question that has an answer.
        votes: dict[str, tuple[bool, float, str]] = {
            role: (bool(vote.approve), vote.score, f"{vote.reason} [{vote.source}]")
            for role, vote in measured.items()
            if not vote.abstained
        }
        settled = set(votes)
        if router and hasattr(router, "think"):
            try:
                transcript_str = "\n".join(f"{r['role']}: {r['content']}" for r in self.rounds)
                prompt = (
                    "You are the God Council Parliament router. Review the following debate transcript "
                    f"for the objective: '{self.objective}'.\n\n"
                    "Debate Transcript:\n"
                    f"{transcript_str}\n\n"
                    "Generate a structured JSON response containing the vote for all 12 roles:\n"
                    "strategist, planner, engineer, researcher, critic, verifier, red_team, "
                    "memory_auditor, safety_judge, tool_operator, forecaster, user_advocate.\n"
                    "For each role, provide:\n"
                    "1. 'approve' (boolean: true/false)\n"
                    "2. 'score' (float between 0.0 and 1.0)\n"
                    "3. 'reason' (string explaining the role's voting decision)\n\n"
                    "Return ONLY a valid JSON object matching the format below:\n"
                    "{\n"
                    "  \"strategist\": {\"approve\": true, \"score\": 0.9, \"reason\": \"meets targets\"},\n"
                    "  ...\n"
                    "}"
                )
                vote_resp = await router.think(prompt=prompt)

                start_idx = vote_resp.find('{')
                end_idx = vote_resp.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = vote_resp[start_idx:end_idx+1]
                    try:
                        parsed_votes = json.loads(json_str)
                    except json.JSONDecodeError:
                        from core.utils.json_repair import repair_json
                        parsed_votes = json.loads(repair_json(json_str))

                    for role in COUNCIL_ROLES.keys():
                        if role in settled:
                            continue
                        if role in parsed_votes and isinstance(parsed_votes[role], dict):
                            val = parsed_votes[role]
                            # Respect the safety judge veto if it is dynamically overridden to false
                            # or if our own safety_status check flagged it.
                            approve = bool(val.get("approve", True))
                            if role == "safety_judge" and not safety_status:
                                approve = False
                                reason = f"Safety Judge veto: {safety_reason}"
                                score = 0.10
                            else:
                                score = float(val.get("score", 0.8))
                                reason = str(val.get("reason", "Voted dynamically"))
                            votes[role] = (approve, score, reason)
            except _DEBATE_RECOVERABLE_ERRORS as e:
                record_degradation(
                    "council_debate",
                    e,
                    action="retained measured votes and abstained on unmeasured roles",
                )
                logger.warning("Failed to obtain dynamic votes from LLM router: %s", e)

        # No fallback votes.
        #
        # A fixed dictionary stood here in which every role approved, and one
        # of them gave as its reason that tests and verification steps were
        # integrated. Nothing had run. A council with no signal abstains, and
        # abstention is not approval.
        if not votes:
            logger.warning(
                "council: nothing measurable and no votes returned; abstaining on %r.",
                self.objective,
            )

        consensus = ConsensusResolver.resolve(votes)
        consensus["plan"] = final_plan.split("\n")
        consensus["rounds"] = self.rounds
        return consensus
