"""Which worker this is, and whether it is the one that was launched.

A worker process is not trusted because it answered. It is trusted because its
capture origin attests to the supervisor key that started it, its identity
transitions are ones this client agreed to, and its mycelial binding is one it
keeps pulsing. Every refusal here is a receipt rather than a boolean, so what
was wrong with an identity can be read back later.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("LLM.MLX")

import copy
import os
import time
from collections.abc import Mapping
from typing import Any


class _KnowsWhichWorkerItIsTalkingTo:
    """Lifted whole from MLXLocalClient; see mlx_client.py."""

    def _accept_worker_identity_transition(
        self,
        raw_identity: Any,
    ) -> dict[str, Any]:
        """Validate and re-attest the sole allowed live identity transition.

        A hot expert-adapter swap may change only the measured adapter list and
        its digest. Model, tokenizer, quantization, process, source, steering,
        and capture key must remain exactly the initialized worker's values.
        """

        from core.brain.llm.latent_cortex.runtime_identity import (
            worker_identity_errors,
        )

        if not isinstance(raw_identity, Mapping):
            raise ValueError("expert adapter response omitted worker identity")
        errors = worker_identity_errors(raw_identity)
        current = getattr(self, "_worker_identity", {})
        if not isinstance(current, Mapping) or not current:
            errors.append("parent_worker_identity_unavailable")
        immutable_fields = (
            "schema",
            "worker_boot_id",
            "worker_pid",
            "worker_model_path",
            "worker_model_parameter_count",
            "worker_model_stored_parameter_element_count",
            "worker_model_parameter_count_basis",
            "worker_source_sha256",
            "worker_affective_steering_active",
            "worker_affective_steering_alpha",
            "worker_action_capture_identity",
            "worker_tokenizer",
            "worker_runtime_tokenizer",
            "worker_quantization",
            "worker_stack_identity_gaps",
        )
        if isinstance(current, Mapping):
            errors.extend(
                f"{field}_changed_during_adapter_swap"
                for field in immutable_fields
                if raw_identity.get(field) != current.get(field)
            )
        if errors:
            raise ValueError(",".join(sorted(set(errors))))
        attested = self._attest_worker_capture_origin(dict(raw_identity))
        attested_errors = worker_identity_errors(attested)
        if attested_errors:
            raise ValueError(
                "reattested_worker_identity_invalid:"
                + ",".join(sorted(set(attested_errors)))
            )
        return attested

    def _init_receipt_errors(self, res: dict[str, Any]) -> list[str]:
        """Every reason this init receipt must NOT be trusted as READY.

        Checks the exact model/worker identity and the recurrent-depth
        invariants the lane declares it needs. Returns an empty list only when
        the receipt positively establishes both.
        """
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .mlx_client import (
            _expected_recurrent_loops_from_model_path,
            _note_recurrent_depth_basis_disagreement,
            _real_model_path,
            _record_mlx_degradation,
        )

        errors: list[str] = []

        try:
            from core.brain.llm.token_budget_evidence import calibration_batch_errors

            errors.extend(
                calibration_batch_errors(res.get("token_budget_calibration"))
            )
        except ImportError as exc:
            _record_mlx_degradation(
                exc,
                action="token-budget calibration validator unavailable during handshake",
                severity="error",
            )
            errors.append("token_budget_calibration_validator_unavailable")

        identity = res.get("worker_identity")
        try:
            from core.brain.llm.latent_cortex.runtime_identity import (
                worker_identity_errors,
            )

            errors.extend(worker_identity_errors(identity))
        except ImportError as exc:
            # No validator means no proof of identity. Absence of a check is
            # not a passed check.
            _record_mlx_degradation(
                exc,
                action="worker identity validator unavailable during handshake",
                severity="error",
            )
            errors.append("worker_identity_validator_unavailable")

        # The worker must be serving the model THIS client asked for.
        if isinstance(identity, dict):
            reported_path = str(identity.get("worker_model_path") or "")
            if reported_path and _real_model_path(reported_path) != _real_model_path(
                self.model_path
            ):
                errors.append("worker_model_path_mismatch")

        # Recurrence: if this lane requires depth, the receipt must prove it.
        required_loops = _expected_recurrent_loops_from_model_path(self.model_path)
        _note_recurrent_depth_basis_disagreement(
            self.model_path, getattr(self, "_expert_adapter_path", None), required_loops
        )
        recurrent_status = res.get("recurrent_depth")
        if required_loops > 1:
            if not isinstance(recurrent_status, dict):
                errors.append("missing_recurrent_depth_receipt")
            else:
                if not bool(recurrent_status.get("active")):
                    errors.append("recurrent_depth_inactive")
                reported_loops = recurrent_status.get("loops")
                if isinstance(reported_loops, int) and reported_loops != required_loops:
                    errors.append(f"recurrent_depth_mismatch:{reported_loops}!={required_loops}")

        # The adapter-activation receipt: whether the recurrent adapter this
        # worker was supposed to load actually loaded. The worker states it in
        # its signed identity; the handshake must also carry it as a separate
        # top-level receipt, and the two must agree. Without this the client
        # takes the worker's word for what weights it is running — which is
        # exactly how a trained adapter that silently failed to attach reads
        # as a live one.
        if isinstance(identity, dict):
            declared_activation = identity.get("worker_recurrent_adapter_activation")
            if isinstance(declared_activation, dict):
                reported_activation = res.get("recurrent_adapter_activation")
                if not isinstance(reported_activation, dict):
                    errors.append("missing_recurrent_adapter_activation_receipt")
                elif reported_activation != declared_activation:
                    errors.append("recurrent_adapter_activation_receipt_mismatch")

        # Optional shadow tissue still requires an explicit inactive receipt.
        # Otherwise a configured load failure and intentional absence are
        # indistinguishable. This pure-data validator cannot initialize MLX in
        # the parent process.
        try:
            from core.brain.llm.unified_recurrent_shadow_contract import (
                shadow_load_receipt_errors,
            )

            errors.extend(
                shadow_load_receipt_errors(res.get("unified_recurrent_shadow"))
            )
        except ImportError as exc:
            _record_mlx_degradation(
                exc,
                action="unified recurrent shadow receipt validator unavailable",
                severity="error",
            )
            errors.append("unified_recurrent_shadow_validator_unavailable")
        try:
            from core.brain.llm.unified_recurrent_qualified_activation import (
                activation_matches_shadow_receipt,
                qualified_activation_load_receipt_errors,
            )

            qualified_receipt = res.get(
                "unified_recurrent_qualified_activation"
            )
            errors.extend(
                qualified_activation_load_receipt_errors(qualified_receipt)
            )
            if (
                isinstance(qualified_receipt, Mapping)
                and qualified_receipt.get("loaded") is True
                and not activation_matches_shadow_receipt(
                    qualified_receipt.get("activation", {}),
                    res.get("unified_recurrent_shadow", {}),
                )
            ):
                errors.append("qualified_activation_shadow_receipt_differs")
        except ImportError as exc:
            _record_mlx_degradation(
                exc,
                action="qualified recurrent activation validator unavailable",
                severity="error",
            )
            errors.append("qualified_activation_validator_unavailable")
        return errors

    def _attest_worker_capture_origin(
        self,
        worker_identity: Mapping[str, Any],
        *,
        attested_at_unix: int | None = None,
    ) -> dict[str, Any]:
        """Bind the worker's boot key to this parent-owned spawn authority."""

        from core.brain.llm.latent_cortex.worker_capture_identity import (
            build_worker_capture_origin_binding,
            validate_worker_capture_origin_binding,
        )

        authority = self._worker_capture_launch_authority
        process = self._process
        expected_pid = getattr(process, "pid", None)
        if authority is None or type(expected_pid) is not int or expected_pid <= 0:
            raise RuntimeError("worker_capture_launch_authority_unavailable")
        capture_identity = worker_identity.get("worker_action_capture_identity")
        bootstrap_binding = getattr(self, "_worker_capture_origin_binding", {})
        if isinstance(bootstrap_binding, Mapping) and bootstrap_binding:
            validated = validate_worker_capture_origin_binding(
                bootstrap_binding,
                expected_supervisor_public_key=authority.private_key.public_key(),
            )
            if validated.get("worker_identity") != capture_identity:
                raise ValueError("worker_capture_bootstrap_ready_identity_mismatch")
            if validated["worker_identity"].get("worker_pid") != expected_pid:
                raise ValueError("worker_capture_bootstrap_process_mismatch")
            return {
                **dict(worker_identity),
                "worker_action_capture_origin_binding": copy.deepcopy(validated),
            }
        binding = build_worker_capture_origin_binding(
            authority,
            capture_identity,
            attested_at_unix=(
                int(time.time()) if attested_at_unix is None else attested_at_unix
            ),
            expected_worker_pid=expected_pid,
        )
        return {
            **dict(worker_identity),
            "worker_action_capture_origin_binding": binding,
        }

    def _accept_worker_capture_bootstrap(
        self,
        worker_capture_identity: Mapping[str, Any],
        *,
        attested_at_unix: int | None = None,
    ) -> dict[str, Any]:
        """Attest the child's capture key while its launch challenge is live."""

        from core.brain.llm.latent_cortex.worker_capture_identity import (
            build_worker_capture_origin_binding,
        )

        authority = self._worker_capture_launch_authority
        process = self._process
        expected_pid = getattr(process, "pid", None)
        if authority is None or type(expected_pid) is not int or expected_pid <= 0:
            raise RuntimeError("worker_capture_launch_authority_unavailable")
        existing = getattr(self, "_worker_capture_origin_binding", {})
        if isinstance(existing, Mapping) and existing:
            if existing.get("worker_identity") != worker_capture_identity:
                raise ValueError("worker_capture_bootstrap_identity_changed")
            return copy.deepcopy(dict(existing))
        binding = build_worker_capture_origin_binding(
            authority,
            worker_capture_identity,
            attested_at_unix=(
                int(time.time()) if attested_at_unix is None else attested_at_unix
            ),
            expected_worker_pid=expected_pid,
        )
        self._worker_capture_origin_binding = copy.deepcopy(binding)
        return copy.deepcopy(binding)

    def get_worker_capture_supervisor_public_key(self) -> bytes:
        """Return the parent key expected by independent capture verification."""

        import base64
        import binascii

        identity = self.get_worker_identity_snapshot()
        binding = identity.get("worker_action_capture_origin_binding")
        if not isinstance(binding, Mapping):
            return b""
        challenge = binding.get("launch_challenge")
        if not isinstance(challenge, Mapping):
            return b""
        try:
            raw = base64.b64decode(
                challenge.get("supervisor_public_key_b64"),
                validate=True,
            )
        except (binascii.Error, TypeError, ValueError):
            return b""
        return raw if len(raw) == 32 else b""

    def get_worker_identity_snapshot(self) -> dict[str, Any]:
        """Return immutable identity evidence for resident-scale policy decisions.

        CP126 375fc058: this promised immutability and returned ``dict(...)``,
        a SHALLOW copy. Every nested dict and list stayed shared with the
        client's authoritative record, so a consumer holding a "snapshot"
        could mutate the identity that later policy, admission and proof
        decisions read — and the evidence would still look like evidence.
        """
        identity = getattr(self, "_worker_identity", None)
        if not isinstance(identity, dict):
            return {}
        return copy.deepcopy(identity)

    def _attest_mycelial_worker(self, init_receipt: Mapping[str, Any]) -> None:
        """Publish the accepted worker identity to Mycelium after READY validation."""
        identity = self.get_worker_identity_snapshot()
        boot_id = str(identity.get("worker_boot_id") or "").strip()
        worker_pid = identity.get("worker_pid")
        device = str(init_receipt.get("device") or "").strip().lower()
        if not boot_id or not isinstance(worker_pid, int) or worker_pid <= 0 or not device:
            raise ValueError("validated worker receipt lacks root attestation identity")
        from core.container import ServiceContainer

        mycelium = ServiceContainer.get("mycelial_network", default=None)
        if mycelium is None or not hasattr(mycelium, "attest_neural_root"):
            return
        worker_target = f"{boot_id}:{worker_pid}"
        hardware_target = f"mlx:{device}"
        shared_evidence = {
            "worker_boot_id": boot_id,
            "worker_pid": worker_pid,
            "worker_model_path": str(identity.get("worker_model_path") or self.model_path),
            "worker_device": device,
            "worker_source_sha256": str(identity.get("worker_source_sha256") or ""),
        }
        mycelium.attest_neural_root(
            "llm",
            root_kind="worker",
            target_id=worker_target,
            owner_generation=boot_id,
            evidence=shared_evidence,
            liveness_contract="heartbeat",
            stale_after_s=10.0,
        )
        mycelium.attest_neural_root(
            f"worker:{worker_target}",
            root_kind="hardware",
            target_id=hardware_target,
            owner_generation=boot_id,
            evidence=shared_evidence,
            liveness_contract="heartbeat",
            stale_after_s=10.0,
        )
        self._mycelial_root_refs = [
            {
                "source": "llm",
                "root_kind": "worker",
                "target_id": worker_target,
                "owner_generation": boot_id,
            },
            {
                "source": f"worker:{worker_target}",
                "root_kind": "hardware",
                "target_id": hardware_target,
                "owner_generation": boot_id,
            },
        ]

    def _pulse_mycelial_worker(self, heartbeat: Mapping[str, Any]) -> None:
        refs = list(getattr(self, "_mycelial_root_refs", ()) or ())
        if not refs:
            return
        identity = self.get_worker_identity_snapshot()
        if (
            str(heartbeat.get("worker_boot_id") or "")
            != str(identity.get("worker_boot_id") or "")
            or heartbeat.get("worker_pid") != identity.get("worker_pid")
        ):
            return
        from core.container import ServiceContainer

        mycelium = ServiceContainer.get("mycelial_network", default=None)
        if mycelium is None or not hasattr(mycelium, "pulse_neural_root"):
            return
        evidence = {
            "active_job": bool(heartbeat.get("active_job")),
            "ipc_backlog": int(heartbeat.get("ipc_backlog") or 0),
            "ipc_broken": bool(heartbeat.get("ipc_broken")),
            "loop_stalled": bool(heartbeat.get("loop_stalled")),
        }
        # A heartbeat with a generation-progress alarm still proves this
        # process and its IPC path are live. Generation health is handled by
        # the request watchdog; only broken IPC invalidates the root probe.
        success = not evidence["ipc_broken"]
        for ref in refs:
            mycelium.pulse_neural_root(
                ref["source"],
                root_kind=ref["root_kind"],
                target_id=ref["target_id"],
                owner_generation=ref["owner_generation"],
                success=success,
                evidence=evidence,
            )

    def _unbind_mycelial_worker(self) -> None:
        refs = list(getattr(self, "_mycelial_root_refs", ()) or ())
        self._mycelial_root_refs = []
        if not refs:
            return
        try:
            from core.container import ServiceContainer

            mycelium = ServiceContainer.get("mycelial_network", default=None)
            if mycelium is None or not hasattr(mycelium, "unbind_neural_roots"):
                return
            for ref in refs:
                mycelium.unbind_neural_roots(
                    ref["source"],
                    owner_generation=ref["owner_generation"],
                )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            logger.debug(
                "Mycelial worker-root retirement unavailable for %s.",
                os.path.basename(self.model_path),
            )
