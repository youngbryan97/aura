"""Standing invariants for loaded local-model runtime assignments."""

from __future__ import annotations

from collections.abc import Iterator

from core.verify.invariants import Violation, invariant

_OWNER = "core/brain/llm/model_runtime_invariants.py"


@invariant(
    "models.loaded_clients_have_immutable_assignments",
    scope="cognition",
    owner=_OWNER,
    description="every loaded model client carries a valid artifact-bound role and QoS assignment",
)
def loaded_model_clients_have_immutable_assignments() -> Iterator[Violation]:
    from core.brain.llm.mlx_client import clients_snapshot
    from core.runtime.model_runtime_assignment import ModelRuntimeAssignment

    for registry_path, client in clients_snapshot():
        assignment = getattr(client, "runtime_assignment", None)
        if not isinstance(assignment, ModelRuntimeAssignment):
            yield Violation(
                subject=str(registry_path),
                message="loaded MLX client has no immutable model runtime assignment",
                remedy="construct the client through the registry and carry its assignment",
            )
            continue
        try:
            restored = ModelRuntimeAssignment.from_dict(assignment.to_dict())
            assignment.assert_bound_to(
                model_path=str(getattr(client, "model_path", "")),
                purpose="serve",
            )
            if restored != assignment:
                raise ValueError("model_runtime_assignment_round_trip_mismatch")
        except (TypeError, ValueError) as exc:
            yield Violation(
                subject=str(registry_path),
                message=f"loaded MLX client assignment is invalid: {exc}",
                remedy="reissue the assignment from exact registry and artifact evidence",
            )


__all__ = ["loaded_model_clients_have_immutable_assignments"]
