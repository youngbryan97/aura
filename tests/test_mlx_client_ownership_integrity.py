"""CP126 mlx_client: ownership, fencing, and cancellation binding.

* ``35eefee4`` — a failed durable owner release recorded a critical
  degradation, then marked the lane cold and returned success. The retained
  fencing token blocked later admission, so a terminal recovery dependency
  was reported as a clean abort.
* ``4cb6a1a0`` — a new caller's timeout, normalized to as little as five
  seconds, was used as ``stale_after`` when acquiring the process-global
  foreground owner, so a short request could declare a legitimately-working
  owner stale and steal foreground authority.
* ``07d62d51`` — the clean-cancellation acknowledgement accepted a reason
  string and two worker-supplied booleans bound to nothing, so a stale or
  replayed response could certify that parameters were unchanged and
  ephemeral weights erased for the wrong episode.
"""
from __future__ import annotations

import inspect
import threading

import pytest

from core.brain.llm import mlx_client


class TestFailedOwnerReleaseIsNotSuccess:
    def _client(self):
        client = mlx_client.MLXLocalClient.__new__(mlx_client.MLXLocalClient)
        client.model_path = "/models/aura-32b"
        client._model_lane_state_lock = threading.RLock()
        client._model_lane_owner_id = "owner-abc"
        client._model_lane_fencing_token = 77
        client._lane_release_failure = None
        client._lane_state = "ready"
        client._set_lane_state = lambda state, reason="": setattr(
            client, "_lane_state", state,
        )
        client._record_degraded_event = lambda *a, **k: None
        return client

    def test_no_failure_means_no_recovery_needed(self):
        assert self._client().lane_recovery_required() is None

    def test_a_failed_release_names_the_owner_and_token(self):
        client = self._client()
        client._note_lane_release_failure(
            RuntimeError("controller down"), reason="forced_abort",
        )
        pending = client.lane_recovery_required()
        assert pending["owner_id"] == "owner-abc"
        assert pending["fencing_token"] == 77
        assert pending["reason"] == "forced_abort"
        assert "controller down" in pending["error"]

    def test_the_lane_is_left_in_a_named_fenced_state(self):
        client = self._client()
        client._note_lane_release_failure(RuntimeError("x"), reason="forced_abort")
        assert client._lane_state == "fenced"

    def test_the_receipt_is_a_copy(self):
        client = self._client()
        client._note_lane_release_failure(RuntimeError("x"), reason="r")
        first = client.lane_recovery_required()
        first["owner_id"] = "tampered"
        assert client.lane_recovery_required()["owner_id"] == "owner-abc"

    def test_a_confirmed_release_retires_the_fence(self):
        client = self._client()
        client._note_lane_release_failure(RuntimeError("x"), reason="r")
        client._clear_lane_release_failure()
        assert client.lane_recovery_required() is None

    def test_the_abort_path_no_longer_marks_a_stranded_lane_cold(self):
        source = inspect.getsource(mlx_client)
        block = source.split("self._release_durable_model_lane_owner_sync(reason=reason)", 1)[1][:1500]
        assert "_note_lane_release_failure" in block
        # The failure branch returns BEFORE the cold transition, so a
        # stranded fence never presents as a cleanly cooled lane.
        after_note = block.split("_note_lane_release_failure(exc, reason=reason)", 1)[1]
        assert after_note.index("return True") < after_note.index('_set_lane_state("cold"')


class TestForegroundEvictionHonoursTheHolder:
    @pytest.fixture(autouse=True)
    def _reset(self):
        original = mlx_client._FOREGROUND_OWNER_STALE_AFTER
        yield
        mlx_client._FOREGROUND_OWNER_STALE_AFTER = original

    def test_an_undeclared_holder_is_never_evicted_on_age(self):
        mlx_client._FOREGROUND_OWNER_STALE_AFTER = None
        assert mlx_client._foreground_owner_eviction_after() is None

    def test_a_short_declaration_is_floored(self):
        mlx_client._FOREGROUND_OWNER_STALE_AFTER = 5.0
        assert (
            mlx_client._foreground_owner_eviction_after()
            == mlx_client._FOREGROUND_OWNER_MIN_EVICTION_S
        )

    def test_a_long_declaration_is_honoured(self):
        mlx_client._FOREGROUND_OWNER_STALE_AFTER = 600.0
        assert mlx_client._foreground_owner_eviction_after() == 600.0

    def test_the_floor_is_meaningful(self):
        assert mlx_client._FOREGROUND_OWNER_MIN_EVICTION_S >= 10.0

    def test_eviction_no_longer_reads_the_newcomers_budget(self):
        source = inspect.getsource(mlx_client._foreground_owner_context)
        assert "eviction_after = _foreground_owner_eviction_after()" in source
        assert "holder_age > stale_after" not in source

    def test_acquisition_records_the_holders_own_budget(self):
        source = inspect.getsource(mlx_client._foreground_owner_context)
        assert "_FOREGROUND_OWNER_STALE_AFTER = stale_after" in source

    def test_release_clears_the_recorded_budget(self):
        source = inspect.getsource(mlx_client._foreground_owner_context)
        # Both the locked release and the last-resort self-clear.
        assert source.count("_FOREGROUND_OWNER_STALE_AFTER = None") >= 3


class TestCancellationAckIsBound:
    @pytest.fixture(autouse=True)
    def _runtime_integrity_verifier(self, monkeypatch):
        from core.brain.llm.latent_cortex import runtime_integrity

        def measured_safe(value, **expected):
            return bool(
                value == {"fixture": "measured-safe"}
                and expected
                == {
                    "require_worker": True,
                    "expected_episode_id": "episode-1",
                    "expected_input_tokens_sha256": "c" * 64,
                    "expected_worker_identity": self._identity(),
                    "expected_fast_weights_applied": False,
                    "expected_fast_weights_attach_attempted": False,
                    "expected_checkpoint_fingerprint": "",
                    "expected_checkpoint_method": "",
                    "expected_checkpoint_file_count": 0,
                }
            )

        monkeypatch.setattr(
            runtime_integrity,
            "runtime_integrity_safe",
            measured_safe,
        )

    def _client(self, identity=None):
        client = mlx_client.MLXLocalClient.__new__(mlx_client.MLXLocalClient)
        client.model_path = "/models/aura-32b"
        client._worker_identity = identity if identity is not None else {}
        return client

    def _ack(self, **receipt_overrides):
        receipt = {
            "params_unchanged": True,
            "request_payload_sha256": "sha-this-request",
            "worker_identity": {
                **self._identity(),
                "worker_model_path": "/models/aura-32b",
            },
            "runtime_integrity": {"fixture": "measured-safe"},
            "episode_id": "episode-1",
            "input_tokens_sha256": "c" * 64,
            "fast_weights_applied": False,
            "checkpoint_fingerprint": "",
            "checkpoint_fingerprint_method": "",
            "checkpoint_file_count": 0,
        }
        worker_overrides = {
            key: receipt_overrides.pop(key)
            for key in (
                "worker_boot_id",
                "worker_pid",
                "worker_model_path",
            )
            if key in receipt_overrides
        }
        receipt["worker_identity"].update(worker_overrides)
        receipt.update(receipt_overrides)
        return {"id": "req-1", "message": "soft_cancelled", "receipt": receipt}

    def _identity(self):
        return {"worker_boot_id": "b" * 32, "worker_pid": 4242}

    def test_a_fully_bound_ack_is_clean(self):
        client = self._client(self._identity())
        assert client._clean_latent_cancel_ack(
            self._ack(),
            expected_request_id="req-1",
            expected_request_sha256="sha-this-request",
        ) is True

    def test_a_bare_shaped_dict_no_longer_certifies(self):
        client = self._client(self._identity())
        forged = {"message": "soft_cancelled", "receipt": {"params_unchanged": True}}
        assert client._clean_latent_cancel_ack(
            forged,
            expected_request_id="req-1",
            expected_request_sha256="sha-this-request",
        ) is False

    def test_another_requests_ack_is_refused(self):
        client = self._client(self._identity())
        assert client._clean_latent_cancel_ack(
            self._ack(),
            expected_request_id="req-2",
            expected_request_sha256="sha-this-request",
        ) is False

    def test_a_replayed_payload_digest_is_refused(self):
        client = self._client(self._identity())
        assert client._clean_latent_cancel_ack(
            self._ack(request_payload_sha256="sha-a-previous-episode"),
            expected_request_id="req-1",
            expected_request_sha256="sha-this-request",
        ) is False

    def test_a_previous_boot_is_refused(self):
        client = self._client(self._identity())
        assert client._clean_latent_cancel_ack(
            self._ack(worker_boot_id="a" * 32),
            expected_request_id="req-1",
            expected_request_sha256="sha-this-request",
        ) is False

    def test_a_different_worker_pid_is_refused(self):
        client = self._client(self._identity())
        assert client._clean_latent_cancel_ack(
            self._ack(worker_pid=9999),
            expected_request_id="req-1",
            expected_request_sha256="sha-this-request",
        ) is False

    def test_a_different_model_is_refused(self):
        client = self._client(self._identity())
        assert client._clean_latent_cancel_ack(
            self._ack(worker_model_path="/models/some-other-model"),
            expected_request_id="req-1",
            expected_request_sha256="sha-this-request",
        ) is False

    def test_unproven_runtime_integrity_still_fails(self):
        client = self._client(self._identity())
        assert client._clean_latent_cancel_ack(
            self._ack(runtime_integrity=None),
            expected_request_id="req-1",
            expected_request_sha256="sha-this-request",
        ) is False

    def test_applied_but_unerased_fast_weights_still_fail(self):
        client = self._client(self._identity())
        assert client._clean_latent_cancel_ack(
            self._ack(fast_weights_applied=True),
            expected_request_id="req-1",
            expected_request_sha256="sha-this-request",
        ) is False

    def test_a_wrong_reason_is_refused(self):
        client = self._client(self._identity())
        ack = self._ack()
        ack["message"] = "finished"
        assert client._clean_latent_cancel_ack(
            ack,
            expected_request_id="req-1",
            expected_request_sha256="sha-this-request",
        ) is False

    def test_non_dict_input_is_refused(self):
        client = self._client()
        assert client._clean_latent_cancel_ack(None) is False
        assert client._clean_latent_cancel_ack("soft_cancelled") is False

    def test_the_call_site_passes_the_binding(self):
        source = inspect.getsource(mlx_client)
        block = source.split("if self._clean_latent_cancel_ack(", 1)[1][:300]
        assert "expected_request_id=req_id" in block
        assert "expected_request_sha256=expected_request_sha256" in block
