"""Whole-graph revision before committing a representation.

Each candidate passes the same binder, temporal constraints and typed floor.
Later available evidence can change the entire choice. Executable predictions
remain predictions, and a representation commit never authorizes a tool act.
"""

from dataclasses import dataclass

from core.learning.semantic_semasiographic import (
    MeaningDiagram,
    MeaningFrame,
    digest,
    execute_meaning_diagram,
    resolve_meaning_diagram,
)
from core.runtime.gateways import StateGateway


@dataclass(frozen=True)
class DiagramProposal:
    identity: str
    diagram: object
    frames: object
    observations: object
    inputs: object
    output: tuple
    baseline_costs: object = None
    conditions: tuple = ()
    language_cost: float = 0.


@dataclass(frozen=True)
class PublicDiagramCondition:
    """A supplied endpoint premise, distinguished from a desired goal."""
    lower: int | None
    upper: int | None
    source: str
    available_at: int
    basis: str

    def __post_init__(self):
        if (self.lower is None and self.upper is None
                or any(v is not None and type(v) is not int for v in (self.lower, self.upper))
                or self.lower is not None and self.upper is not None and self.lower > self.upper
                or not isinstance(self.source, str) or not self.source
                or type(self.available_at) is not int or self.available_at < 0
                or self.basis not in {"observed", "public_goal", "assumption"}):
            raise ValueError("endpoint condition requires a public source and declared basis")

    def admits(self, result):
        return (type(result) is int and (self.lower is None or result >= self.lower)
                and (self.upper is None or result <= self.upper))


@dataclass(frozen=True)
class DiagramWorkingState:
    version: int
    cutoff: int
    candidates: tuple
    selected: str | None
    margin: float | None
    status: str

    @property
    def identity(self):
        return digest((self.version, self.cutoff, self.candidates, self.selected, self.margin, self.status))

    def to_dict(self):
        body = {"schema": "aura.diagram_working_state.v1", "version": self.version,
                "cutoff": self.cutoff, "candidates": self.candidates, "selected": self.selected,
                "margin": self.margin, "status": self.status}
        return {**body, "content_sha256": digest(body)}

    @classmethod
    def from_dict(cls, payload):
        import math

        body = dict(payload)
        expected = body.pop("content_sha256", None)
        if (body.get("schema") != "aura.diagram_working_state.v1" or digest(body) != expected
                or type(body["version"]) is not int or body["version"] < 1
                or type(body["cutoff"]) is not int or body["cutoff"] < 0
                or body["margin"] is not None and (not math.isfinite(body["margin"]) or body["margin"] < 0)
                or body["status"] not in {"proposed", "ambiguous", "unresolved"}
                or not 0 < len(body["candidates"]) <= 64
                or len({r["id"] for r in body["candidates"]}) != len(body["candidates"])):
            raise ValueError("diagram workspace storage custody differs")
        state = cls(body["version"], body["cutoff"], tuple(body["candidates"]), body["selected"],
                    body["margin"], body["status"])
        selected = next((r for r in state.candidates if r["id"] == state.selected), None)
        if ((state.status == "proposed") != (selected is not None)
                or selected is not None and selected["status"] != "executable_prediction"
                or digest(state.to_dict()) != digest(payload)):
            raise ValueError("diagram workspace selection differs from its alternatives")
        return state


def revise_diagram_workspace(engine, proposals, *, cutoff, previous=None, minimum_margin=0., world="actual"):
    """Evaluate a bounded candidate family; no claim of exhaustive discovery."""
    import math

    proposals = tuple(proposals)
    if (not proposals or len(proposals) > 64 or len({p.identity for p in proposals}) != len(proposals)
            or any(not isinstance(p.identity, str) or not p.identity for p in proposals)
            or not math.isfinite(minimum_margin) or minimum_margin < 0
            or type(cutoff) is not int or cutoff < 0
            or previous is not None and cutoff < previous.cutoff):
        raise ValueError("invalid bounded diagram revision")
    rows, eligible = [], []
    for proposal in proposals:
        try:
            if (not math.isfinite(proposal.language_cost)
                    or any(not isinstance(c, PublicDiagramCondition) for c in proposal.conditions)):
                raise ValueError("diagram proposal costs or public conditions are invalid")
            conditions = tuple(c for c in proposal.conditions if c.available_at <= cutoff)
            executions = {}

            def diagram_accepts(candidate, proposal=proposal, conditions=conditions, executions=executions):
                try:
                    execution = execute_meaning_diagram(candidate, proposal.frames, proposal.inputs,
                        output=proposal.output, cutoff=cutoff, world=world)
                except ValueError:
                    # not a failure: a rejected complete graph receives a solver exclusion cut.
                    return False
                if any(not c.admits(execution.result) for c in conditions):
                    return False
                executions[candidate.custody_sha256] = execution
                return True

            resolved, binding, receipt = resolve_meaning_diagram(engine, proposal.diagram, proposal.frames,
                proposal.observations, cutoff=cutoff, world=world, baseline_costs=proposal.baseline_costs,
                diagram_accepts=diagram_accepts)
            if resolved is None:
                rows.append({"id": proposal.identity, "status": binding.status, "receipt": receipt,
                    "conditions": [(c.source, c.basis) for c in conditions]})
                continue
            execution = executions[resolved.custody_sha256]
            energy = binding.energy + proposal.language_cost
            rows.append({"id": proposal.identity, "status": "executable_prediction",
                "energy": energy, "diagram": resolved.to_dict(), "receipt": receipt,
                "conditions": [(c.source, c.basis) for c in conditions],
                "world": world, "output": proposal.output,
                "input_values": tuple(sorted(dict(proposal.inputs).items())),
                "frames": [{"symbol": f.symbol, "roles": f.roles, "output_type": f.output_type,
                            "operation": f.operation, "world": f.world}
                           for _, f in sorted(proposal.frames.items())],
                "predicted_result": execution.result, "execution": execution.receipt})
            eligible.append((energy, proposal.identity))
        except ValueError as exc:
            rows.append({"id": proposal.identity, "status": "inadmissible", "reason": str(exc)})
    eligible.sort()
    margin = eligible[1][0] - eligible[0][0] if len(eligible) > 1 else None
    selected = eligible[0][1] if eligible and (len(eligible) == 1 or margin > minimum_margin) else None
    return DiagramWorkingState(1 if previous is None else previous.version + 1, cutoff, tuple(rows), selected,
        margin, "proposed" if selected is not None else "ambiguous" if eligible else "unresolved")


async def retain_diagram_choice(state, gateway: StateGateway, *, expected_identity):
    """Commit one resolved representation snapshot, never an external action."""
    from core.runtime.gateways import StateMutationRequest

    if state.identity != expected_identity or state.selected is None or state.status != "proposed":
        raise ValueError("diagram commit is stale or unresolved")
    row = next(row for row in state.candidates if row["id"] == state.selected)
    body = {"schema": "aura.diagram_choice.v2", "state_identity": state.identity,
            "version": state.version, "cutoff": state.cutoff, "selected": row,
            "working_state": state.to_dict(), "action_authority": False, "semantic_truth_certified": False}
    return await gateway.mutate(StateMutationRequest(key=state.identity,
        new_value={**body, "content_sha256": digest(body)},
        domain="semantic_diagram_choices", cause="retain whole interpreted graph after bounded revision"))


def replay_diagram_choice(payload):
    """Reload a retained graph and independently execute its declared denotation."""
    from core.learning.semantic_program_ir import normalize_semantic_value

    body = dict(payload)
    expected = body.pop("content_sha256", None)
    if (body.get("schema") != "aura.diagram_choice.v2" or digest(body) != expected
            or body.get("action_authority") is not False or body.get("semantic_truth_certified") is not False):
        raise ValueError("retained diagram choice custody or authority differs")
    state = DiagramWorkingState.from_dict(body["working_state"])
    row = next((r for r in state.candidates if r["id"] == state.selected), None)
    if (state.identity != body["state_identity"] or row is None or digest(row) != digest(body["selected"])
            or state.version != body["version"] or state.cutoff != body["cutoff"]):
        raise ValueError("retained diagram choice differs from its exact workspace")
    diagram = MeaningDiagram.from_dict(row["diagram"])
    frames = tuple(MeaningFrame(**{**f, "roles": tuple(tuple(r) for r in f["roles"])}) for f in row["frames"])
    execution = execute_meaning_diagram(diagram, {(f.world, f.symbol): f for f in frames},
        {tuple(k): v for k, v in row["input_values"]}, output=tuple(row["output"]),
        cutoff=state.cutoff, world=row["world"])
    if (execution.result != normalize_semantic_value(row["predicted_result"])
            or digest(execution.receipt) != digest(row["execution"])):
        raise ValueError("retained diagram denotation differs from independent floor execution")
    return state, execution
