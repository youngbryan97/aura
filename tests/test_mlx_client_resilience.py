import asyncio
import contextlib
import copy
import importlib
import os
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.brain.llm.latent_cortex.runtime_integrity import canonical_sha256
from core.brain.llm.latent_cortex.worker_capture_identity import (
    build_worker_capture_identity,
    build_worker_capture_launch_authority,
)
from core.brain.llm.mlx_client import MLXLocalClient
from core.brain.llm.mlx_vision_client import MLXVisionClient
from core.brain.llm.mlx_worker import (
    IPCWriterThread,
    WorkerMemorySentinel,
    _apply_surface_generation_controls,
    _build_operator_evidence_prompt,
    _build_prefill_progress_callback,
    _merge_stop_sequences,
    _operator_evidence_fragment_incomplete,
    _prefill_step_size_for_model,
    _prompt_cache_entry_budget_for_model,
    _restore_surface_generation_controls,
    _should_emit_generation_progress,
    _trim_complete_operator_evidence,
)
from core.brain.llm.token_budget_evidence import (
    CALIBRATION_SCHEMA,
)
from core.brain.llm.token_budget_evidence import (
    MIN_OBSERVATIONS as MIN_CALIBRATION_OBSERVATIONS,
)
from core.brain.llm.unified_recurrent_qualified_activation import (
    seal_qualified_activation_load_receipt,
)
from core.brain.llm.unified_recurrent_shadow_contract import (
    LOAD_SCHEMA as UNIFIED_RECURRENT_SHADOW_LOAD_SCHEMA,
)
from core.brain.llm.unified_recurrent_shadow_contract import (
    seal_shadow_load_receipt,
)
from core.utils.deadlines import get_deadline
from tests.fixtures.rlc_runtime_integrity import complete_serving_stack

TMP_ROOT = Path(tempfile.gettempdir())
QWEN32_MODEL = str(TMP_ROOT / "Qwen2.5-32B-Instruct-8bit")
TEST_MODEL = str(TMP_ROOT / "test-model")

def ready_init_receipt(
    model_path: str = TEST_MODEL,
    *,
    client: MLXLocalClient | None = None,
    process=None,
    **overrides,
) -> dict:
    """A handshake receipt that satisfies the READINESS-IS-EARNED contract.

    CP126 25ca1c12 made ``status: ok`` insufficient on its own: a worker is
    only READY once its receipt positively establishes worker identity (boot
    id, pid, model path, parameter counts and their basis) and, on lanes that
    require it, recurrent depth. Fixtures that predate that contract used a
    bare ``{"status": "ok", "action": "init"}`` and now correctly fail the
    handshake, so the shared valid receipt lives here.
    """
    lowered = str(model_path).lower()
    loops = 2 if any(token in lowered for token in ("32b", "cortex", "zenith")) else 1
    capture_identity = None
    if client is not None or process is not None:
        if client is None or process is None or type(getattr(process, "pid", None)) is not int:
            raise ValueError("synthetic worker receipt requires client and process")
        authority = build_worker_capture_launch_authority()
        client._worker_capture_launch_authority = authority
        capture_identity = build_worker_capture_identity(
            worker_boot_id="1" * 32,
            worker_pid=process.pid,
            private_key=None,
            launch_challenge=authority.challenge,
        ).public_identity
    receipt = {
        "status": "ok",
        "action": "init",
        # The worker measures a fixed prompt mix with its resident tokenizer
        # and the handshake refuses a receipt without it. This fixture predated
        # that requirement, so nine tests were asserting readiness against a
        # receipt the running system would reject.
        "token_budget_calibration": {
            "schema": CALIBRATION_SCHEMA,
            "sample_set": "aura-runtime-mixed-v1",
            "observations": [
                {"label": f"sample_{index}", "chars": 120 * (index + 1), "tokens": 30 * (index + 1)}
                for index in range(MIN_CALIBRATION_OBSERVATIONS)
            ],
        },
        "worker_identity": {
            "schema": (
                "aura.latent_cortex.worker_identity.v2"
                if capture_identity is not None
                else "aura.latent_cortex.worker_identity.v1"
            ),
            "worker_boot_id": "1" * 32,
            "worker_pid": process.pid if process is not None else 4242,
            "worker_model_path": os.path.realpath(model_path),
            "worker_model_parameter_count": 1_000_000,
            "worker_model_stored_parameter_element_count": 1_000_000,
            "worker_model_parameter_count_basis": "stored_tensor_elements",
            "worker_source_sha256": "2" * 64,
            "worker_affective_steering_active": True,
            "worker_affective_steering_alpha": 0.30,
            **(
                {"worker_action_capture_identity": capture_identity}
                if capture_identity is not None
                else {}
            ),
            **complete_serving_stack(),
        },
        "recurrent_depth": {
            "active": True,
            "config": {"n_loops": loops},
            "loops": loops,
            "expected_loops": loops,
            "required": loops > 1,
        },
        "unified_recurrent_shadow": seal_shadow_load_receipt(
            {
                "schema": UNIFIED_RECURRENT_SHADOW_LOAD_SCHEMA,
                "configured": False,
                "loaded": False,
                "reason": "not_configured",
                "package_id": "",
                "manifest_sha256": "",
                "checkpoint_sha256": "",
                "controller_sha256": "",
                "families": [],
                "task_depths": [],
                "recurrence_depth": 0,
                "model_identity_strength": "none",
                "mode": "shadow_only",
                "serving_authority": False,
            }
        ),
        "unified_recurrent_qualified_activation": (
            seal_qualified_activation_load_receipt(
                configured=False,
                loaded=False,
                reason="not_configured",
                activation=None,
            )
        ),
    }
    receipt.update(overrides)
    return receipt


def test_worker_memory_sentinel_requires_explicit_child_exit_authority(monkeypatch):
    writer = SimpleNamespace(put=lambda _payload: None)
    sentinel = WorkerMemorySentinel(writer, QWEN32_MODEL)

    assert sentinel._exit_for_memory_fuse("unit-test-pressure") is False
    assert sentinel._stop_event.is_set()

    exit_codes: list[int] = []
    monkeypatch.setattr(
        "core.brain.llm.mlx_worker.os._exit",
        lambda code: exit_codes.append(code),
    )
    authorized = WorkerMemorySentinel(
        writer,
        QWEN32_MODEL,
        hard_exit_allowed=True,
    )

    assert authorized._exit_for_memory_fuse("unit-test-pressure") is None
    assert exit_codes == [137]


class ReplaceAttr:
    def __init__(self, obj, name, value):
        self.obj = obj
        self.name = name
        self.value = value
        self.missing = object()
        self.old_value = self.missing

    def __enter__(self):
        self.old_value = getattr(self.obj, self.name, self.missing)
        setattr(self.obj, self.name, self.value)
        return self.value

    def __exit__(self, *_exc):
        if self.old_value is self.missing:
            delattr(self.obj, self.name)
        else:
            setattr(self.obj, self.name, self.old_value)
        return False


def replace_dotted(dotted_name: str, value):
    parts = dotted_name.split(".")
    import_error = None
    for idx in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:idx])
        try:
            target = importlib.import_module(module_name)
            attrs = parts[idx:]
            break
        except ModuleNotFoundError as exc:
            import_error = exc
    else:
        raise import_error or ModuleNotFoundError(dotted_name)

    for attr_name in attrs[:-1]:
        target = getattr(target, attr_name)
    return ReplaceAttr(target, attrs[-1], value)


class ProcessProbe:
    _next_pid = 900_000
    _registry = {}

    def __init__(self, alive=True):
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.name = f"ProcessProbe-{self.pid}"
        self.alive = alive
        self.kill_calls = 0
        self.join_calls = []
        type(self)._registry[self.pid] = self

    def is_alive(self):
        return self.alive

    def kill(self):
        self.kill_calls += 1
        self.alive = False

    def join(self, timeout=None):
        self.join_calls.append(SimpleNamespace(timeout=timeout))

    def assert_killed_once(self):
        assert self.kill_calls == 1

    def assert_not_killed(self):
        assert self.kill_calls == 0

    def assert_joined_with(self, *, timeout):
        assert self.join_calls
        assert self.join_calls[-1].timeout == timeout


def test_expert_adapter_identity_transition_is_narrow_and_reattested():
    client = MLXLocalClient(model_path=TEST_MODEL)
    process = ProcessProbe(alive=True)
    raw = ready_init_receipt(
        client=client,
        process=process,
    )["worker_identity"]
    client._process = process
    client._worker_identity = client._attest_worker_capture_origin(raw)

    transitioned = copy.deepcopy(raw)
    transitioned["worker_adapters"] = [
        {
            "name": "layers.1.self_attn.o_proj",
            "type": "LoRALinear",
            "rank": 8,
            "scale": 1.0,
            "parameter_sha256": "a" * 64,
            "parameter_scope": "adapter_owned_excluding_wrapped_base_v1",
        }
    ]
    transitioned["worker_adapter_stack_sha256"] = canonical_sha256(
        transitioned["worker_adapters"]
    )
    accepted = client._accept_worker_identity_transition(transitioned)
    assert accepted["worker_adapters"] == transitioned["worker_adapters"]
    assert accepted["worker_action_capture_origin_binding"]

    tokenizer_swap = copy.deepcopy(transitioned)
    tokenizer_swap["worker_runtime_tokenizer"]["vocab_size"] = 256
    with pytest.raises(
        ValueError,
        match="worker_runtime_tokenizer_changed_during_adapter_swap",
    ):
        client._accept_worker_identity_transition(tokenizer_swap)


def test_validated_worker_identity_owns_generation_bound_mycelial_roots(monkeypatch):
    client = MLXLocalClient.__new__(MLXLocalClient)
    client.model_path = TEST_MODEL
    client._worker_identity = {
        "worker_boot_id": "boot-1",
        "worker_pid": 4242,
        "worker_model_path": TEST_MODEL,
        "worker_source_sha256": "a" * 64,
    }
    client._mycelial_root_refs = []

    class MyceliumRecorder:
        def __init__(self):
            self.attestations = []
            self.pulses = []
            self.unbound = []

        def attest_neural_root(self, source, **kwargs):
            self.attestations.append((source, kwargs))

        def pulse_neural_root(self, source, **kwargs):
            self.pulses.append((source, kwargs))
            return True

        def unbind_neural_roots(self, source, **kwargs):
            self.unbound.append((source, kwargs))
            return 1

    mycelium = MyceliumRecorder()
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: mycelium if name == "mycelial_network" else default,
    )

    client._attest_mycelial_worker({"device": "gpu"})
    assert [item[0] for item in mycelium.attestations] == [
        "llm",
        "worker:boot-1:4242",
    ]
    assert all(
        item[1]["owner_generation"] == "boot-1"
        for item in mycelium.attestations
    )
    assert mycelium.attestations[1][1]["target_id"] == "mlx:gpu"

    client._pulse_mycelial_worker(
        {
            "worker_boot_id": "boot-1",
            "worker_pid": 4242,
            "active_job": False,
            "ipc_backlog": 0,
            "ipc_broken": False,
        }
    )
    assert len(mycelium.pulses) == 2
    assert all(item[1]["success"] is True for item in mycelium.pulses)

    client._pulse_mycelial_worker(
        {
            "worker_boot_id": "retired-worker",
            "worker_pid": 9999,
            "ipc_broken": False,
        }
    )
    assert len(mycelium.pulses) == 2

    client._unbind_mycelial_worker()
    assert len(mycelium.unbound) == 2
    assert client._mycelial_root_refs == []


@pytest.fixture(autouse=True)
def _isolated_model_lane_controller(monkeypatch, tmp_path):
    """Give synthetic worker processes a hermetic durable-lane identity."""

    from core.runtime import model_lane_control
    from core.runtime.receipts import ReceiptStore

    original_identity = model_lane_control.process_identity_for_pid

    def _identity(pid, *, observer=None):
        probe = ProcessProbe._registry.get(int(pid))
        if probe is not None:
            return model_lane_control.ProcessIdentity(int(pid), float(pid))
        return original_identity(pid, observer=observer)

    def _alive(identity):
        probe = ProcessProbe._registry.get(int(identity.pid))
        if probe is not None:
            return bool(probe.alive and identity.started_at == float(identity.pid))
        current = original_identity(identity.pid)
        return bool(
            current.started_at > 0.0
            and abs(current.started_at - identity.started_at) <= 0.5
        )

    controller = model_lane_control.ModelLaneController(
        state_path=tmp_path / "model_lane_control.json",
        receipt_store=ReceiptStore(tmp_path / "receipts"),
        process_alive=_alive,
        process_discovery=None,
    )
    monkeypatch.setattr(model_lane_control, "process_identity_for_pid", _identity)
    monkeypatch.setattr(model_lane_control, "get_model_lane_controller", lambda: controller)
    yield
    ProcessProbe._registry.clear()


class AsyncCallProbe:
    def __init__(self, return_value=None, side_effect=None):
        self.return_value = return_value
        self.side_effect = side_effect
        self.await_args_list = []
        self.await_args = None

    async def __call__(self, *args, **kwargs):
        call = SimpleNamespace(args=args, kwargs=kwargs)
        self.await_args_list.append(call)
        self.await_args = call
        if isinstance(self.side_effect, list):
            result = self.side_effect.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        if isinstance(self.side_effect, BaseException):
            raise self.side_effect
        if self.side_effect is not None:
            result = self.side_effect(*args, **kwargs)
            if hasattr(result, "__await__"):
                return await result
            return result
        return self.return_value

    def assert_not_awaited(self):
        assert not self.await_args_list

    def assert_awaited_once(self):
        assert len(self.await_args_list) == 1

    def assert_awaited_once_with(self, *args, **kwargs):
        self.assert_awaited_once()
        call = self.await_args_list[0]
        assert call.args == args
        assert call.kwargs == kwargs

    def assert_any_await(self, *args, **kwargs):
        assert any(call.args == args and call.kwargs == kwargs for call in self.await_args_list)


class SyncCallProbe:
    def __init__(self, return_value=None, side_effect=None):
        self.return_value = return_value
        self.side_effect = side_effect
        self.call_args_list = []
        self.call_args = None

    def __call__(self, *args, **kwargs):
        call = SimpleNamespace(args=args, kwargs=kwargs)
        self.call_args_list.append(call)
        self.call_args = call
        if isinstance(self.side_effect, list):
            result = self.side_effect.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        if isinstance(self.side_effect, BaseException):
            raise self.side_effect
        if self.side_effect is not None:
            return self.side_effect(*args, **kwargs)
        return self.return_value

    def assert_called_once_with(self, *args, **kwargs):
        assert len(self.call_args_list) == 1
        call = self.call_args_list[0]
        assert call.args == args
        assert call.kwargs == kwargs

    def assert_not_called(self):
        assert not self.call_args_list


class MPContextProbe:
    def __init__(self, *queues):
        self.queues = list(queues)

    def Queue(self, **_kwargs):  # noqa: N802 - mirrors multiprocessing context API.
        if not self.queues:
            raise AssertionError("No queued IPC probe available")
        return self.queues.pop(0)


class LockProbe:
    def __init__(self, *, acquire_result=True, release_error: BaseException | None = None):
        self.acquire_result = acquire_result
        self.release_error = release_error
        self.acquire_calls = []
        self.release_calls = 0

    def acquire(self, *args, **kwargs):
        self.acquire_calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        return self.acquire_result

    def release(self):
        self.release_calls += 1
        if self.release_error is not None:
            raise self.release_error

    def assert_released(self):
        assert self.release_calls > 0


class TestMLXClientResilience(unittest.IsolatedAsyncioTestCase):
    class _FakeQueue:
        def __init__(self):
            self.closed = False
            self.joined = False

        def empty(self):
            return True

        def close(self):
            self.closed = True

        def join_thread(self):
            self.joined = True

    def _attach_local_ipc_queues(self, client):
        client._req_q = queue.Queue()
        client._res_q = queue.Queue()

    def test_close_releases_worker_and_ipc_queues(self):
        client = MLXLocalClient(model_path=TEST_MODEL)
        req_q = self._FakeQueue()
        res_q = self._FakeQueue()
        proc = ProcessProbe(alive=True)
        client._req_q = req_q
        client._res_q = res_q
        client._process = proc
        client._init_done = True

        client.close()

        proc.assert_killed_once()
        proc.assert_joined_with(timeout=2.0)
        self.assertTrue(req_q.closed)
        self.assertTrue(req_q.joined)
        self.assertTrue(res_q.closed)
        self.assertTrue(res_q.joined)
        self.assertIsNone(client._req_q)
        self.assertIsNone(client._res_q)
        self.assertFalse(client._init_done)
        self.assertEqual(client.get_lane_status()["state"], "closed")

    def test_client_constructor_defers_ipc_queue_allocation(self):
        client = MLXLocalClient(model_path=TEST_MODEL)

        self.assertIsNone(client._req_q)
        self.assertIsNone(client._res_q)
        self.assertFalse(hasattr(client._substrate_mem, "get_lock"))
        self.assertFalse(hasattr(client._steering_active, "get_lock"))

    def test_ready_lane_without_progress_is_not_conversation_ready(self):
        client = MLXLocalClient(model_path=TEST_MODEL)
        client._process = ProcessProbe(alive=True)
        client._init_done = True
        client._set_lane_state("ready")

        with replace_dotted("core.brain.llm.mlx_client.time.time", lambda: 1000.0):
            lane = client.get_lane_status()

        self.assertEqual(lane["state"], "recovering")
        self.assertFalse(lane["conversation_ready"])
        self.assertEqual(lane["last_error"], "worker_progress_stale")
        self.assertIn("no_worker_progress", lane["readiness_blockers"])
        self.assertIn("lane_recovering", lane["readiness_blockers"])

    def test_heartbeat_alone_does_not_make_conversation_ready(self):
        client = MLXLocalClient(model_path=TEST_MODEL)
        client._process = ProcessProbe(alive=True)
        client._init_done = True
        client._set_lane_state("ready")
        client._last_heartbeat = 999.5

        with replace_dotted("core.brain.llm.mlx_client.time.time", lambda: 1000.0):
            lane = client.get_lane_status()

        self.assertEqual(lane["state"], "recovering")
        self.assertFalse(lane["conversation_ready"])
        self.assertIn("no_worker_progress", lane["readiness_blockers"])
        self.assertLess(lane["heartbeat_age_s"], 1.0)
        self.assertIsNone(lane["progress_age_s"])

    def test_ready_lane_with_stale_progress_is_not_conversation_ready(self):
        client = MLXLocalClient(model_path=TEST_MODEL)
        client._process = ProcessProbe(alive=True)
        client._init_done = True
        client._set_lane_state("ready")
        client._last_heartbeat = 900.0
        client._last_progress_at = 900.0
        client._last_ready_at = 900.0

        with replace_dotted("core.brain.llm.mlx_client.time.time", lambda: 1000.0):
            lane = client.get_lane_status()

        self.assertEqual(lane["state"], "recovering")
        self.assertFalse(lane["conversation_ready"])
        self.assertEqual(lane["last_error"], "worker_progress_stale")
        self.assertIn("worker_progress_stale", lane["readiness_blockers"])
        self.assertGreaterEqual(lane["heartbeat_age_s"], 100.0)

    def test_ready_lane_with_warmup_foreground_owner_is_not_conversation_ready(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        client._process = ProcessProbe(alive=True)
        client._init_done = True
        client._set_lane_state("ready")
        client._last_heartbeat = 1000.0
        client._last_progress_at = 1000.0
        client._last_ready_at = 1000.0
        client._warmup_in_flight = True
        client._active_generations = 1

        with replace_dotted("core.brain.llm.mlx_client.time.time", lambda: 1001.0):
            with replace_dotted("core.brain.llm.mlx_client._foreground_owner_active", lambda: True):
                with replace_dotted("core.brain.llm.mlx_client._FOREGROUND_OWNER_NAME", "warmup:test"):
                    lane = client.get_lane_status()

        self.assertFalse(lane["conversation_ready"])
        self.assertIn("warmup_in_flight", lane["readiness_blockers"])
        self.assertIn("warmup_foreground_owner", lane["readiness_blockers"])

    def test_stale_foreground_owner_clear_is_nonblocking_when_lock_is_busy(self):
        import core.brain.llm.mlx_client as mlx_module

        old_owner = mlx_module._FOREGROUND_OWNER_NAME
        old_owned_at = mlx_module._FOREGROUND_OWNER_ACQUIRED_AT
        lock = mlx_module._FOREGROUND_OWNER_LOCK
        self.assertTrue(lock.acquire(False))
        result: list[str | None] = []

        def _status_clear() -> None:
            result.append(mlx_module._clear_stale_foreground_owner(max_age_s=0.0))

        thread = threading.Thread(target=_status_clear, daemon=True)
        try:
            mlx_module._FOREGROUND_OWNER_NAME = "chat_api:test"
            mlx_module._stamp_foreground_owner(time.time() - 500.0)
            thread.start()
            thread.join(0.2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(result, [None])
            self.assertEqual(mlx_module._FOREGROUND_OWNER_NAME, "chat_api:test")
        finally:
            lock.release()
            thread.join(1.0)
            mlx_module._FOREGROUND_OWNER_NAME = old_owner
            mlx_module._stamp_foreground_owner(old_owned_at)

    def test_replace_ipc_queues_closes_previous_queues_before_recreation(self):
        client = MLXLocalClient(model_path=TEST_MODEL)
        old_req_q = self._FakeQueue()
        old_res_q = self._FakeQueue()
        new_req_q = self._FakeQueue()
        new_res_q = self._FakeQueue()
        client._req_q = old_req_q
        client._res_q = old_res_q
        client._mp_context = MPContextProbe(new_req_q, new_res_q)

        client._replace_ipc_queues()

        self.assertTrue(old_req_q.closed)
        self.assertTrue(old_req_q.joined)
        self.assertTrue(old_res_q.closed)
        self.assertTrue(old_res_q.joined)
        self.assertIs(client._req_q, new_req_q)
        self.assertIs(client._res_q, new_res_q)
        self.assertFalse(client._closed)

    def test_vision_client_releases_worker_and_ipc_queues(self):
        client = MLXVisionClient(model_path=TEST_MODEL)
        req_q = self._FakeQueue()
        res_q = self._FakeQueue()
        proc = ProcessProbe(alive=False)
        client._req_q = req_q
        client._res_q = res_q
        client._process = proc
        client._init_done = True

        client.stop()

        proc.assert_joined_with(timeout=3.0)
        self.assertTrue(req_q.closed)
        self.assertTrue(req_q.joined)
        self.assertTrue(res_q.closed)
        self.assertTrue(res_q.joined)
        self.assertIsNone(client._req_q)
        self.assertIsNone(client._res_q)
        self.assertFalse(client._init_done)

    async def test_worker_stop_sequences_are_role_boundary_safe(self):
        stops = _merge_stop_sequences(["Assistant:", "Aura:", "user:", "\nHuman:"])

        self.assertNotIn("Assistant:", stops)
        self.assertNotIn("Aura:", stops)
        self.assertNotIn("user:", stops)
        self.assertIn("\nAssistant:", stops)
        self.assertIn("\nuser:", stops)
        self.assertIn("\nHuman:", stops)

    async def test_operator_evidence_contract_prompt_is_complete_and_bounded(self):
        prompt, prefix = _build_operator_evidence_prompt(
            [
                {"role": "system", "content": "Return one paragraph."},
                {
                    "role": "user",
                    "content": (
                        "What objective, governed tool use, receipt, trace, stop condition, "
                        "and personhood boundary should Aura use?"
                    ),
                },
            ],
            "",
        )

        self.assertIn("bounded software-operator evidence lane", prompt)
        self.assertIn("objective", prompt)
        self.assertIn("personhood boundary", prompt)
        self.assertEqual(
            prefix,
            "Operationally, Aura should set an objective, use governed tool actions, "
            "keep each receipt and trace, stop when blocked or unsafe, and treat the "
            "result as evidence of bounded software operation rather than personhood proof. ",
        )
        self.assertFalse(
            _operator_evidence_fragment_incomplete(
                "Aura should pursue a bounded objective, use governed tool calls with a "
                "receipt and trace, stop when governance or evidence fails, and treat "
                "that as operational evidence rather than proof of literal personhood."
            )
        )
        self.assertTrue(
            _operator_evidence_fragment_incomplete(
                "I feel like a person who chooses things in a shining field."
            )
        )
        self.assertTrue(
            _operator_evidence_fragment_incomplete(
                "Aura should pursue a bounded objective, use governed tool calls with a "
                "receipt and trace, stop when governance or evidence fails, and treat "
                "that as operational evidence rather than proof of literal personhood. "
                "That's one paragraph as requested."
            )
        )

    async def test_operator_evidence_trims_clipped_tail_to_complete_sentences(self):
        clipped = (
            "Operationally, Aura should set an objective, use governed tool actions, "
            "keep each receipt and trace, stop when blocked or unsafe, and treat the "
            "result as evidence of bounded software operation rather than personhood proof. "
            "Receipts and traces show tool use was governed. Stopping when blocked or "
            "unsafe shows boundedness. The result is evidence of software ope"
        )

        trimmed = _trim_complete_operator_evidence(clipped)

        self.assertEqual(
            trimmed,
            "Operationally, Aura should set an objective, use governed tool actions, "
            "keep each receipt and trace, stop when blocked or unsafe, and treat the "
            "result as evidence of bounded software operation rather than personhood proof. "
            "Receipts and traces show tool use was governed. Stopping when blocked or "
            "unsafe shows boundedness.",
        )
        self.assertFalse(_operator_evidence_fragment_incomplete(trimmed))

    async def test_surface_generation_controls_clamp_steering_and_recurrent_depth(self):
        class _Hook:
            _alpha = 5.0

        class _Engine:
            _alpha = 5.0
            _surface_alpha_override = None

            def __init__(self):
                self._hooks = [_Hook()]

            def set_surface_alpha_override(self, alpha):
                self._surface_alpha_override = alpha
                if alpha is not None:
                    for hook in self._hooks:
                        hook._alpha = min(hook._alpha, alpha)

        class _Layer:
            pass

        class _Inner:
            _recurrent_depth_config = {"n_loops": 2}
            # resolve_model_layers identifies the forward owner by its layer
            # stack and refuses to guess from arbitrary attributes. Without one
            # the clamp had nothing to apply to, and the test read that as the
            # clamp not happening.
            layers = [_Layer(), _Layer()]
            embed_tokens = object()
            norm = object()

        class _Model:
            model = _Inner()

        engine = _Engine()
        state = _apply_surface_generation_controls(
            engine,
            _Model(),
            {"clean_user_surface_contract": True},
        )

        self.assertLessEqual(engine._surface_alpha_override, 0.35)
        self.assertLessEqual(engine._hooks[0]._alpha, 0.35)
        self.assertEqual(_Model.model._recurrent_depth_runtime_loops, 1)

        _restore_surface_generation_controls(state)

        self.assertIsNone(engine._surface_alpha_override)
        self.assertFalse(hasattr(_Model.model, "_recurrent_depth_runtime_loops"))

        health_engine = _Engine()
        health_state = _apply_surface_generation_controls(
            health_engine,
            _Model(),
            {"health_probe": True},
        )
        try:
            self.assertEqual(_Model.model._recurrent_depth_runtime_loops, 1)
            self.assertLessEqual(health_engine._surface_alpha_override, 0.35)
        finally:
            _restore_surface_generation_controls(health_state)

    async def test_foreground_request_lock_timeout_is_bounded_for_live_chat(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)

        self.assertEqual(
            client._request_lock_timeout(deadline=None, foreground_request=True),
            12.0,
        )

    async def test_foreground_request_lock_timeout_preempts_wedged_holder(self):
        from core.brain.llm.mlx_client import _new_shared_future

        client = MLXLocalClient(model_path=QWEN32_MODEL)
        stuck = _new_shared_future()
        client._request_lock.acquire()
        client._request_lock_owner_label = "Cortex"
        client._request_lock_acquired_at = time.time() - 2.0
        client._current_gen_future = stuck
        client._last_heartbeat = 0.0

        try:
            with ReplaceAttr(client, "_request_lock_timeout", lambda *_args, **_kwargs: 0.05), ReplaceAttr(
                client, "_first_token_sla", lambda *_args, **_kwargs: 0.01
            ):
                acquired = await client._acquire_request_lock(
                    owner_label="live_chat",
                    deadline=None,
                    foreground_request=True,
                )

            self.assertFalse(acquired)
            self.assertEqual(
                client._deferred_reboot_reason,
                "foreground_preemption_wedged_holder",
            )
            self.assertTrue(stuck.done())
        finally:
            client._current_gen_future = None
            with contextlib.suppress(RuntimeError):
                client._request_lock.release()

    async def test_force_abort_completes_waiting_generation_future(self):
        from core.brain.llm.mlx_client import _await_shared_future, _new_shared_future

        client = MLXLocalClient(model_path=QWEN32_MODEL)
        future = _new_shared_future()
        client._pending_generations["probe-req"] = future
        client._current_gen_future = future
        client._current_request_id = "probe-req"
        client._active_generations = 1
        try:
            aborted = client.force_abort_active_generation("unit_test_generation_timeout")
            payload = await _await_shared_future(future, timeout_s=0.1)
        finally:
            client.close()

        self.assertTrue(aborted)
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["action"], "generate")
        self.assertEqual(payload["id"], "probe-req")
        self.assertEqual(payload["message"], "unit_test_generation_timeout")
        self.assertTrue(payload["force_aborted"])
        self.assertFalse(future.cancelled())
        self.assertEqual(client._last_generation_completed_at, 0.0)
        self.assertEqual(client._first_token_sla(foreground_request=True), 120.0)

    async def test_worker_sanitizer_finishes_current_request_for_caller_recovery(self):
        """A sanitized draft answers the caller; it does not abandon the request.

        This asserted on a literal spelling in the worker source and on the
        next occurrence of `ipc_writer.put({` after it. The sanitizer moved
        and the marker survived only inside a comment, so the search raised
        ValueError instead of reporting anything about behaviour. What matters
        is the property: whatever the sanitizer rejects, the caller gets a
        reply it can recover from rather than a request that never returns.
        """
        from core.brain.llm.mlx_worker import _route_telemetry_sanitizer_draft

        unspeakable = "PROCEEDING TOOL_ACTION CONVERGE_UNION MySelfEpsilon ExistenceHash"

        # With a repair lane that owns the draft, the caller receives it as
        # rejected evidence and can recover.
        carried, carried_reasons = _route_telemetry_sanitizer_draft(
            unspeakable, is_proof=False, authored_surface_repair_available=True
        )
        self.assertEqual(carried, unspeakable)
        self.assertTrue(carried_reasons)

        # With no owner, it is withheld — but the reasons still come back, so
        # the caller is told what happened instead of being left waiting.
        withheld, withheld_reasons = _route_telemetry_sanitizer_draft(
            unspeakable, is_proof=False, authored_surface_repair_available=False
        )
        self.assertEqual(withheld, "")
        self.assertTrue(withheld_reasons)

        kept, kept_reasons = _route_telemetry_sanitizer_draft(
            "The bridge takes seventeen minutes if the two fastest carry the torch back.",
            is_proof=False,
            authored_surface_repair_available=False,
        )
        self.assertIn("seventeen minutes", kept)
        self.assertFalse(list(kept_reasons))

    async def test_heavy_model_hotswap_reboots_other_heavy_client_before_spawn(self):
        import core.brain.llm.mlx_client as mlx_module

        primary_path = "/models/32B"
        deep_path = "/models/72B"

        primary = MLXLocalClient(model_path=primary_path)
        solver = MLXLocalClient(model_path=deep_path)

        primary_proc = ProcessProbe(alive=True)
        primary._process = primary_proc
        primary._init_done = True

        async def _reboot_primary(*_args, **_kwargs):
            primary_proc.alive = False
            primary._init_done = False

        primary_reboot = AsyncCallProbe(side_effect=_reboot_primary)
        primary.reboot_worker = primary_reboot

        solver_proc = ProcessProbe(alive=True)

        async def _spawn_solver():
            solver._init_future.set_result(
                ready_init_receipt(deep_path, client=solver, process=solver_proc)
            )
            return solver_proc

        old_clients = dict(mlx_module._CLIENTS)
        old_last_heavy = mlx_module._GLOBAL_LAST_HEAVY_MODEL
        old_last_swap = mlx_module._GLOBAL_LAST_SWAP_TIME
        mlx_module._CLIENTS = {
            primary_path: primary,
            deep_path: solver,
        }
        mlx_module._GLOBAL_LAST_HEAVY_MODEL = ""
        mlx_module._GLOBAL_LAST_SWAP_TIME = 0.0

        try:
            with replace_dotted("core.brain.llm.model_registry.ACTIVE_MODEL", "Qwen2.5-32B-Instruct-8bit"), \
                 replace_dotted("core.brain.llm.model_registry.get_deep_model_path", lambda: deep_path), \
                 replace_dotted("core.brain.llm.model_registry.get_model_path", lambda name=None: primary_path if "32B" in str(name) or name is None else deep_path), \
                 replace_dotted("core.brain.llm.mlx_client.os.path.realpath", lambda path, *_args, **_kwargs: path), \
                 replace_dotted("core.brain.llm.mlx_client._declared_mlx_worker_footprint_gb", lambda _path: 40.0), \
                 replace_dotted("core.brain.llm.mlx_client._reclaim_model_lane_capacity", lambda _claim: True), \
                 ReplaceAttr(solver, "_spawn_worker", AsyncCallProbe(side_effect=_spawn_solver)):
                await solver._ensure_worker_alive(foreground_request=True)
        finally:
            mlx_module._CLIENTS = old_clients
            mlx_module._GLOBAL_LAST_HEAVY_MODEL = old_last_heavy
            mlx_module._GLOBAL_LAST_SWAP_TIME = old_last_swap

        primary_reboot.assert_awaited_once()
        self.assertTrue(solver._init_done)

    async def test_model_lane_compensation_receipts_only_confirm_ready_owner(self):
        import core.brain.llm.mlx_client as mlx_module

        target = MLXLocalClient(model_path=TEST_MODEL)
        target._model_lane_owner_id = "mlx:test:compensation"
        warmup = AsyncCallProbe(return_value=True)
        target.warmup = warmup
        target.is_alive = lambda: True
        previous_clients = dict(mlx_module._CLIENTS)
        mlx_module._CLIENTS = {TEST_MODEL: target}
        try:
            restored = await mlx_module._compensate_model_lane_owner(
                SimpleNamespace(
                    owner_id=target._model_lane_owner_id,
                    model_path=TEST_MODEL,
                ),
                "unit-test-candidate-failure",
            )
        finally:
            mlx_module._CLIENTS = previous_clients

        self.assertTrue(restored)
        warmup.assert_awaited_once_with(skip_swap_cooldown=True)

    async def test_ensure_worker_sets_init_future_before_spawn(self):
        client = MLXLocalClient(model_path=TEST_MODEL)

        async def spawn_side_effect():
            self.assertIsNotNone(client._init_future)
            self.assertFalse(client._init_future.done())
            process = ProcessProbe(alive=True)
            client._init_future.set_result(
                ready_init_receipt(client=client, process=process)
            )
            return process

        with ReplaceAttr(client, "_spawn_worker", AsyncCallProbe(side_effect=spawn_side_effect)):
            await client._ensure_worker_alive()

        self.assertTrue(client._init_done)
        self.assertTrue(client.is_alive())

    async def test_ensure_worker_reuses_existing_handshake_future(self):
        client = MLXLocalClient(model_path=TEST_MODEL)

        live_process = ProcessProbe(alive=True)
        client._process = live_process
        client._init_done = False
        import asyncio
        real_future = asyncio.get_running_loop().create_future()
        real_future.set_result(
            ready_init_receipt(client=client, process=live_process)
        )
        client._init_future = real_future

        spawn_probe = AsyncCallProbe()
        with ReplaceAttr(client, "_spawn_worker", spawn_probe):
            await client._ensure_worker_alive()

        spawn_probe.assert_not_awaited()
        live_process.assert_not_killed()
        self.assertTrue(client._init_done)

    async def test_ensure_worker_reuses_cross_loop_handshake_future(self):
        client = MLXLocalClient(model_path=TEST_MODEL)

        live_process = ProcessProbe(alive=True)
        client._process = live_process
        client._init_done = False

        holder = {}
        ready = threading.Event()

        def _loop_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            future = loop.create_future()
            holder["future"] = future
            ready.set()

            async def _complete():
                await asyncio.sleep(0.05)
                future.set_result(
                    ready_init_receipt(client=client, process=live_process)
                )
                await asyncio.sleep(0.05)

            loop.run_until_complete(_complete())
            loop.close()

        thread = threading.Thread(target=_loop_thread, name="mlx-cross-loop-init", daemon=True)
        thread.start()
        ready.wait(timeout=1.0)
        client._init_future = holder["future"]

        try:
            spawn_probe = AsyncCallProbe()
            with ReplaceAttr(client, "_spawn_worker", spawn_probe):
                await client._ensure_worker_alive()
        finally:
            thread.join(timeout=1.0)

        spawn_probe.assert_not_awaited()
        self.assertTrue(client._init_done)
        live_process.assert_not_killed()

    async def test_cancelled_generation_preserves_healthy_worker(self):
        client = MLXLocalClient(model_path=TEST_MODEL)
        proc = ProcessProbe(alive=True)
        client._process = proc
        client._init_done = True
        self._attach_local_ipc_queues(client)
        client._set_lane_state("ready")
        client._last_heartbeat = client._last_progress_at = client._last_ready_at = 10_000.0

        cancelled_calls = []

        async def _cancelled(*args, **kwargs):
            cancelled_calls.append((args, kwargs))
            raise asyncio.CancelledError

        reboot_probe = AsyncCallProbe()
        with ReplaceAttr(client, "_ensure_worker_alive", AsyncCallProbe(return_value=True)):
            with ReplaceAttr(client, "_wait_for_generation_result", AsyncCallProbe(side_effect=_cancelled)):
                with ReplaceAttr(client, "reboot_worker", reboot_probe):
                    with replace_dotted("time.time", lambda: 10_001.0):
                        with self.assertRaises(asyncio.CancelledError):
                            await client._generate_inner("hello", foreground_request=True)

        reboot_probe.assert_not_awaited()
        self.assertEqual(len(cancelled_calls), 1)

    async def test_expected_cancelled_generation_does_not_mark_worker_unhealthy(self):
        client = MLXLocalClient(model_path=TEST_MODEL)
        self._attach_local_ipc_queues(client)

        cancelled_calls = []

        async def _cancelled(*args, **kwargs):
            cancelled_calls.append((args, kwargs))
            # What a planned reboot does: name the requests that are actually
            # in flight when it decides to cancel them.
            client._note_expected_generation_cancellation(
                "yield_to_Qwen2.5-72B-Instruct-4bit",
                request_ids=[client._current_request_id],
            )
            raise asyncio.CancelledError

        with ReplaceAttr(client, "_ensure_worker_alive", AsyncCallProbe(return_value=True)):
            with ReplaceAttr(client, "_wait_for_generation_result", AsyncCallProbe(side_effect=_cancelled)):
                with self.assertRaises(asyncio.CancelledError):
                    await client._generate_inner("hello", foreground_request=True)

        self.assertIsNone(client._deferred_reboot_reason)
        self.assertEqual(len(cancelled_calls), 1)
        # The claim was spent by the request it was filed against.
        self.assertEqual(client._expected_cancels, {})

    async def test_one_requests_planned_cancellation_does_not_excuse_another(self):
        """A claim is bound to the request it was filed against.

        The accounting used to be a client-wide credit counter, so whichever
        cancellation arrived first spent the credit. An unrelated cancellation
        could be logged as routine reboot cleanup while the request the reboot
        actually killed was reported as a surprise — the two conclusions the
        runtime uses to decide whether a worker stopped responding.
        """

        client = MLXLocalClient(model_path=TEST_MODEL)
        client._note_expected_generation_cancellation(
            "planned_reboot", request_ids=["req-planned"]
        )

        # An unrelated request must not be able to claim it.
        self.assertEqual(client._consume_expected_generation_cancellation("req-other"), "")
        # And the planned one still can, afterwards.
        self.assertEqual(
            client._consume_expected_generation_cancellation("req-planned"), "planned_reboot"
        )
        # Claims are single-use.
        self.assertEqual(client._consume_expected_generation_cancellation("req-planned"), "")

    async def test_a_stale_claim_is_not_attributable(self):
        """Past the TTL the cancellation is no longer the plan's doing."""

        client = MLXLocalClient(model_path=TEST_MODEL)
        client._note_expected_generation_cancellation(
            "planned_reboot", request_ids=["req-old"]
        )
        client._expected_cancels["req-old"] = (
            "planned_reboot",
            time.time() - (client._EXPECTED_CANCEL_TTL_S + 1.0),
        )
        self.assertEqual(client._consume_expected_generation_cancellation("req-old"), "")
        self.assertEqual(client._expected_cancels, {})

    async def test_generate_times_out_waiting_for_foreground_owner(self):
        import core.brain.llm.mlx_client as mlx_module

        client = MLXLocalClient(model_path=QWEN32_MODEL)
        old_owner = mlx_module._FOREGROUND_OWNER_NAME
        old_owned_at = mlx_module._FOREGROUND_OWNER_ACQUIRED_AT
        mlx_module._FOREGROUND_OWNER_NAME = "warmup:cortex"
        mlx_module._stamp_foreground_owner(time.time())

        try:
            acquire_probe = AsyncCallProbe(return_value=True)
            inner = AsyncCallProbe()
            with ReplaceAttr(client, "_acquire_request_lock", acquire_probe):
                with ReplaceAttr(client, "_generate_inner", inner):
                    with replace_dotted("core.brain.llm.mlx_client._foreground_owner_wait_budget", lambda *_args, **_kwargs: 0.0):
                        result = await client.generate(
                            "hello",
                            foreground_request=True,
                            owner_label="test",
                            deadline=get_deadline(30.0),
                        )
        finally:
            mlx_module._FOREGROUND_OWNER_NAME = old_owner
            mlx_module._stamp_foreground_owner(old_owned_at)

        self.assertIsNone(result)
        inner.assert_not_awaited()

    async def test_generate_clears_stale_foreground_owner_and_continues(self):
        import core.brain.llm.mlx_client as mlx_module

        client = MLXLocalClient(model_path=QWEN32_MODEL)
        old_owner = mlx_module._FOREGROUND_OWNER_NAME
        old_owned_at = mlx_module._FOREGROUND_OWNER_ACQUIRED_AT
        old_stale_after = mlx_module._FOREGROUND_OWNER_STALE_AFTER
        mlx_module._FOREGROUND_OWNER_NAME = "warmup:cortex"
        mlx_module._stamp_foreground_owner(time.time() - 120.0)
        # CP126 4cb6a1a0: an owner is stale only by ITS OWN declared budget —
        # a holder that never declared one is not evictable on age alone (a
        # short newcomer used to be able to declare a working owner stale).
        # A real holder always declares this in _foreground_owner_context, so
        # the simulated holder must too, or it is simply not a stale owner.
        mlx_module._FOREGROUND_OWNER_STALE_AFTER = 30.0

        try:
            with ReplaceAttr(client, "_acquire_request_lock", AsyncCallProbe(return_value=True)):
                inner = AsyncCallProbe(return_value="ok")
                with ReplaceAttr(client, "_generate_inner", inner):
                    result = await client.generate(
                        "hello",
                        foreground_request=True,
                        owner_label="test",
                        deadline=get_deadline(30.0),
                    )
        finally:
            mlx_module._FOREGROUND_OWNER_NAME = old_owner
            mlx_module._stamp_foreground_owner(old_owned_at)
            mlx_module._FOREGROUND_OWNER_STALE_AFTER = old_stale_after

        self.assertEqual(result, "ok")
        inner.assert_awaited_once()

    def test_force_clear_foreground_owner_respects_min_age(self):
        import core.brain.llm.mlx_client as mlx_module

        old_owner = mlx_module._FOREGROUND_OWNER_NAME
        old_owned_at = mlx_module._FOREGROUND_OWNER_ACQUIRED_AT
        try:
            mlx_module._FOREGROUND_OWNER_NAME = "chat_api:default"
            mlx_module._stamp_foreground_owner(time.time() - 10.0)
            young = mlx_module.force_clear_foreground_owner(
                reason="unit_test_young_owner",
                min_age_s=45.0,
            )
            self.assertFalse(young["cleared"])
            self.assertEqual(mlx_module._FOREGROUND_OWNER_NAME, "chat_api:default")

            mlx_module._stamp_foreground_owner(time.time() - 60.0)
            stale = mlx_module.force_clear_foreground_owner(
                reason="unit_test_stale_owner",
                min_age_s=45.0,
            )
            self.assertTrue(stale["cleared"])
            self.assertEqual(stale["holder"], "chat_api:default")
            self.assertIsNone(mlx_module._FOREGROUND_OWNER_NAME)
            self.assertEqual(mlx_module._FOREGROUND_OWNER_ACQUIRED_AT, 0.0)
        finally:
            mlx_module._FOREGROUND_OWNER_NAME = old_owner
            mlx_module._stamp_foreground_owner(old_owned_at)

    async def test_foreground_generate_reserves_owner_before_request_lock(self):
        import core.brain.llm.mlx_client as mlx_module

        client = MLXLocalClient(model_path=QWEN32_MODEL)
        old_owner = mlx_module._FOREGROUND_OWNER_NAME
        old_owned_at = mlx_module._FOREGROUND_OWNER_ACQUIRED_AT
        observed_owner = []

        async def _acquire(*_args, **_kwargs):
            observed_owner.append(mlx_module._FOREGROUND_OWNER_NAME)
            return True

        try:
            mlx_module._FOREGROUND_OWNER_NAME = None
            mlx_module._FOREGROUND_OWNER_ACQUIRED_AT = 0.0
            with ReplaceAttr(client, "_acquire_request_lock", AsyncCallProbe(side_effect=_acquire)):
                with ReplaceAttr(client, "_generate_inner", AsyncCallProbe(return_value="ok")):
                    result = await client.generate(
                        "hello",
                        foreground_request=True,
                        owner_label="live_user",
                        deadline=get_deadline(30.0),
                    )
        finally:
            mlx_module._FOREGROUND_OWNER_NAME = old_owner
            mlx_module._stamp_foreground_owner(old_owned_at)

        self.assertEqual(result, "ok")
        self.assertEqual(observed_owner, ["live_user"])

    async def test_reboot_worker_clears_matching_warmup_owner(self):
        import core.brain.llm.mlx_client as mlx_module

        client = MLXLocalClient(model_path=QWEN32_MODEL)
        old_owner = mlx_module._FOREGROUND_OWNER_NAME
        old_owned_at = mlx_module._FOREGROUND_OWNER_ACQUIRED_AT
        mlx_module._FOREGROUND_OWNER_NAME = "warmup:Qwen2.5-32B-Instruct-8bit"
        mlx_module._stamp_foreground_owner(time.time() - 20.0)

        try:
            await client.reboot_worker(reason="yield_to_solver", mark_failed=False)
            self.assertIsNone(mlx_module._FOREGROUND_OWNER_NAME)
            self.assertEqual(mlx_module._FOREGROUND_OWNER_ACQUIRED_AT, 0.0)
        finally:
            mlx_module._FOREGROUND_OWNER_NAME = old_owner
            mlx_module._stamp_foreground_owner(old_owned_at)


    async def test_primary_lane_generate_requires_explicit_foreground_request(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)

        inner = AsyncCallProbe(return_value="ok")
        with ReplaceAttr(client, "_generate_inner", inner):
            result = await client.generate("hello")

        self.assertEqual(result, "ok")
        self.assertFalse(inner.await_args.kwargs["foreground_request"])
        self.assertFalse(inner.await_args.kwargs["request_is_background"])

    async def test_generate_suppresses_stale_unlock_in_finally(self):
        client = MLXLocalClient(model_path=TEST_MODEL)
        lock_probe = LockProbe(release_error=RuntimeError("release unlocked lock"))
        client._request_lock = lock_probe

        with ReplaceAttr(client, "_generate_inner", AsyncCallProbe(return_value="ok")):
            result = await client.generate("hello")

        self.assertEqual(result, "ok")
        lock_probe.assert_released()

    async def test_generate_maps_timeout_kwarg_to_request_deadline(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)

        inner = AsyncCallProbe(return_value="ok")
        with ReplaceAttr(client, "_generate_inner", inner):
            result = await client.generate("hello", timeout=12.5)

        self.assertEqual(result, "ok")
        deadline = inner.await_args.kwargs["deadline"]
        self.assertAlmostEqual(deadline._timeout, 12.5, places=2)
        self.assertNotIn("timeout", inner.await_args.kwargs)

    async def test_generate_soft_times_out_init_budget_without_killing_worker(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        proc = ProcessProbe(alive=True)
        client._process = proc
        client._init_done = False
        client._set_lane_state("handshaking")
        client._init_future = asyncio.get_running_loop().create_future()

        result = await client._generate_inner(
            "hello",
            foreground_request=True,
            owner_label="test",
            deadline=get_deadline(0.5),
        )

        self.assertIsNone(result)
        proc.assert_not_killed()
        self.assertIs(client._process, proc)
        self.assertFalse(client._init_future.done())
        self.assertEqual(client._lane_state, "recovering")

    async def test_listener_routes_init_error_without_action_to_init_future(self):
        client = MLXLocalClient(model_path=TEST_MODEL)
        self._attach_local_ipc_queues(client)
        client._init_future = asyncio.get_running_loop().create_future()

        listener = asyncio.create_task(client._response_listener_loop())
        try:
            client._res_q.put({"status": "error", "message": "Init failed: boom"})
            result = await asyncio.wait_for(asyncio.shield(client._init_future), timeout=2.0)
        finally:
            listener.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await listener

        self.assertEqual(result["action"], "init")
        self.assertEqual(result["message"], "Init failed: boom")

    async def test_listener_attests_capture_identity_before_ready(self):
        client = MLXLocalClient(model_path=TEST_MODEL)
        self._attach_local_ipc_queues(client)
        process = ProcessProbe(alive=True)
        authority = build_worker_capture_launch_authority()
        identity = build_worker_capture_identity(
            worker_boot_id="a" * 32,
            worker_pid=process.pid,
            launch_challenge=authority.challenge,
        )
        client._process = process
        client._worker_capture_launch_authority = authority
        client._init_future = asyncio.get_running_loop().create_future()

        listener = asyncio.create_task(client._response_listener_loop())
        try:
            client._res_q.put(
                {
                    "status": "ok",
                    "action": "capture_identity_bootstrap",
                    "worker_action_capture_identity": identity.public_identity,
                }
            )
            for _ in range(100):
                if client._worker_capture_origin_binding:
                    break
                await asyncio.sleep(0.01)
        finally:
            listener.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await listener

        self.assertEqual(
            client._worker_capture_origin_binding["worker_identity"],
            identity.public_identity,
        )
        self.assertFalse(client._init_future.done())

    def test_handshake_age_is_anchored_to_process_birth(self):
        client = MLXLocalClient(model_path=TEST_MODEL)
        client._process_started_at = 100.0
        client._set_lane_state("handshaking")

        first_age = client._handshake_age_s(now=500.0)
        client._set_lane_state("handshaking")
        second_age = client._handshake_age_s(now=501.0)

        self.assertEqual(first_age, 400.0)
        self.assertEqual(second_age, 401.0)

    async def test_invalid_ready_receipt_retires_worker_without_replaying_frame(self):
        client = MLXLocalClient(model_path=TEST_MODEL)
        process = ProcessProbe(alive=True)
        client._process = process
        client._process_started_at = time.time()
        client._set_lane_state("handshaking")
        client._init_future = asyncio.get_running_loop().create_future()
        client._init_future.set_result({"status": "ok", "action": "init"})
        reboot = AsyncCallProbe(return_value=None)

        with ReplaceAttr(client, "reboot_worker", reboot):
            result = await client._ensure_worker_alive_inner(_init_retry=True)

        self.assertFalse(result)
        reboot.assert_awaited_once_with(
            reason="init_receipt_invalid",
            mark_failed=False,
        )

    async def test_listener_replacement_refuses_second_reader_on_same_queue(self):
        client = MLXLocalClient(model_path=TEST_MODEL)
        response_queue = queue.Queue()
        client._res_q = response_queue
        client._response_queue_generation = 7
        client._listener_response_queue = response_queue
        client._listener_queue_generation = 7
        release = asyncio.Event()

        async def cancellation_resistant_listener():
            while not release.is_set():
                try:
                    await asyncio.sleep(0)
                except asyncio.CancelledError:
                    continue

        listener = asyncio.create_task(cancellation_resistant_listener())
        client._listener_task = listener
        await asyncio.sleep(0)
        listener.cancel()
        await asyncio.sleep(0)

        async def timeout_immediately(_awaitable, **kwargs):
            self.assertEqual(kwargs["timeout"], 5.0)
            raise TimeoutError

        try:
            with replace_dotted(
                "core.brain.llm.mlx_client.asyncio.wait_for",
                timeout_immediately,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "response_listener_retirement_unconfirmed",
                ):
                    await client._ensure_listener_task()
            self.assertIs(client._listener_task, listener)
            self.assertIs(client._listener_response_queue, response_queue)
        finally:
            release.set()
            await asyncio.sleep(0)
            if not listener.done():
                listener.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await listener

    async def test_retired_listener_cannot_complete_replacement_generation(self):
        client = MLXLocalClient(model_path=TEST_MODEL)
        old_queue = queue.Queue()
        new_queue = queue.Queue()
        client._res_q = old_queue
        client._response_queue_generation = 3

        listener = asyncio.create_task(client._response_listener_loop(old_queue, 3))
        await asyncio.sleep(0.05)

        client._res_q = new_queue
        client._response_queue_generation = 4
        client._current_request_id = "replacement-request"
        client._current_gen_future = asyncio.get_running_loop().create_future()
        old_queue.put(
            {
                "status": "ok",
                "action": "generate",
                "id": "replacement-request",
                "text": "stale worker answer",
            }
        )

        await asyncio.wait_for(listener, timeout=2.0)
        self.assertFalse(client._current_gen_future.done())

    async def test_generation_waiter_flags_first_token_sla_breach(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        proc = ProcessProbe(alive=True)
        client._process = proc
        client._init_done = True
        client._set_lane_state("ready")
        req_id = "req-1"
        future = asyncio.get_running_loop().create_future()
        client._pending_generations[req_id] = future
        client._current_request_id = req_id
        client._current_request_started_at = 100.0
        client._last_generation_completed_at = 1.0
        client._current_request_prompt_chars = 0
        deadline = get_deadline(None)

        with replace_dotted(
            "core.brain.llm.mlx_client.asyncio.wait_for",
            SyncCallProbe(side_effect=TimeoutError()),
        ):
            with replace_dotted(
                "core.brain.llm.mlx_client.time.time",
                lambda: 100.0 + client._first_token_sla(foreground_request=True) + 1.0,
            ):
                result = await client._wait_for_generation_result(
                    req_id,
                    future,
                    deadline,
                    foreground_request=True,
                )

        self.assertIsNone(result)
        self.assertEqual(client._deferred_reboot_reason, "first_token_sla_exceeded")

    async def test_generation_waiter_aborts_when_process_rss_crosses_limit(self):
        class Snapshot:
            should_gc = False
            refuse_heavy_local_generation = True
            reason = "process_tree_rss:45.0GB/42.0GB (level=critical)"

        async def await_timeout(*_args, **_kwargs):
            await asyncio.sleep(0)
            raise TimeoutError

        client = MLXLocalClient(model_path=QWEN32_MODEL)
        client._process = ProcessProbe(alive=True)
        client._init_done = True
        client._set_lane_state("ready")
        req_id = "req-memory-pressure"
        future = asyncio.get_running_loop().create_future()
        client._pending_generations[req_id] = future
        client._current_gen_future = future
        client._current_request_id = req_id
        abort_calls = []

        def abort(reason):
            abort_calls.append(reason)
            return True

        client.force_abort_active_generation = abort

        with replace_dotted(
            "core.brain.llm.mlx_client._await_shared_future",
            await_timeout,
        ):
            with replace_dotted(
                "core.brain.llm.mlx_client.get_memory_pressure_snapshot",
                lambda: Snapshot(),
            ):
                result = await client._wait_for_generation_result(
                    req_id,
                    future,
                    get_deadline(30.0),
                    foreground_request=True,
                )

        self.assertIsNone(result)
        self.assertEqual(abort_calls, ["memory_pressure_during_generation"])
        self.assertTrue(future.cancelled())
        self.assertNotIn(req_id, client._pending_generations)

    async def test_long_prompt_extends_first_token_sla_for_heavy_lane(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        cold_sla = client._first_token_sla(foreground_request=True)

        client._last_generation_completed_at = 1.0
        client._current_request_prompt_chars = 24_740

        warm_long_prompt_sla = client._first_token_sla(foreground_request=True)

        self.assertGreater(warm_long_prompt_sla, 22.0)
        self.assertGreater(warm_long_prompt_sla, cold_sla)

    async def test_generation_waiter_caps_heartbeat_only_first_token_livelock(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        proc = ProcessProbe(alive=True)
        client._process = proc
        client._init_done = True
        client._set_lane_state("ready")
        req_id = "req-heartbeat-no-token"
        future = asyncio.get_running_loop().create_future()
        client._pending_generations[req_id] = future
        client._current_request_id = req_id
        client._current_request_started_at = 100.0
        client._last_generation_completed_at = 1.0
        client._current_request_prompt_chars = 24_740
        hard_ceiling = client._first_token_absolute_ceiling(foreground_request=True)
        now = 100.0 + hard_ceiling + 1.0
        client._last_heartbeat = now - 0.5
        client._last_progress_at = now - 0.5
        client._last_ready_at = now - 0.5

        self.assertLess(hard_ceiling, client._first_token_sla(foreground_request=True))

        with replace_dotted(
            "core.brain.llm.mlx_client.asyncio.wait_for",
            SyncCallProbe(side_effect=TimeoutError()),
        ):
            with replace_dotted("core.brain.llm.mlx_client.time.time", lambda: now):
                result = await client._wait_for_generation_result(
                    req_id,
                    future,
                    get_deadline(None),
                    foreground_request=True,
                )

        self.assertIsNone(result)
        self.assertEqual(client._deferred_reboot_reason, "recoverable_first_token_sla_exceeded")
        self.assertTrue(future.cancelled())
        self.assertNotIn(req_id, client._pending_generations)

    async def test_generation_waiter_flags_token_progress_stall(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        proc = ProcessProbe(alive=True)
        client._process = proc
        client._init_done = True
        client._set_lane_state("ready")
        req_id = "req-2"
        future = asyncio.get_running_loop().create_future()
        client._pending_generations[req_id] = future
        client._current_request_id = req_id
        client._current_request_started_at = 100.0
        client._current_first_token_at = 105.0
        client._last_token_progress_at = 105.0

        with replace_dotted(
            "core.brain.llm.mlx_client.asyncio.wait_for",
            SyncCallProbe(side_effect=TimeoutError()),
        ):
            with replace_dotted("core.brain.llm.mlx_client.time.time", lambda: 105.0 + client._token_stall_after() + 1.0):
                result = await client._wait_for_generation_result(
                    req_id,
                    future,
                    get_deadline(30.0),
                    foreground_request=True,
                )

        self.assertIsNone(result)
        self.assertEqual(client._deferred_reboot_reason, "token_progress_stalled")

    async def test_generation_waiter_recycles_fresh_heartbeat_token_stall(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        proc = ProcessProbe(alive=True)
        client._process = proc
        client._init_done = True
        client._set_lane_state("ready")
        req_id = "req-fresh-stall"
        future = asyncio.get_running_loop().create_future()
        client._pending_generations[req_id] = future
        client._current_request_id = req_id
        client._current_request_started_at = 100.0
        client._current_first_token_at = 105.0
        client._last_token_progress_at = 105.0
        now = 105.0 + client._token_stall_after(foreground_request=True) + 1.0
        client._last_heartbeat = now - 0.5

        with replace_dotted(
            "core.brain.llm.mlx_client.asyncio.wait_for",
            SyncCallProbe(side_effect=TimeoutError()),
        ):
            with replace_dotted("core.brain.llm.mlx_client.time.time", lambda: now):
                result = await client._wait_for_generation_result(
                    req_id,
                    future,
                    get_deadline(30.0),
                    foreground_request=True,
                )

        self.assertIsNone(result)
        self.assertEqual(client._deferred_reboot_reason, "recoverable_token_progress_stalled")
        self.assertTrue(future.cancelled())

    async def test_listener_drops_late_generation_for_previous_request(self):
        import core.brain.llm.mlx_client as mlx_module

        client = MLXLocalClient(model_path=QWEN32_MODEL)
        self._attach_local_ipc_queues(client)
        current_future = mlx_module._new_shared_future()
        client._current_gen_future = current_future
        client._current_request_id = "new-req"

        listener = asyncio.create_task(client._response_listener_loop())
        try:
            client._res_q.put({
                "status": "ok",
                "action": "generate",
                "id": "old-req",
                "text": "late stale answer",
            })
            await asyncio.sleep(0.25)
        finally:
            listener.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await listener

        self.assertFalse(current_future.done())

    async def test_warmup_precompile_requires_visible_readiness_after_empty_compile(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        client._warmup_in_flight = True
        client._process = ProcessProbe(alive=True)
        client._init_done = True
        client._recurrent_depth_status = {
            "active": True,
            "config": {"n_loops": 2},
            "expected_loops": 2,
            "required": True,
        }

        probe = AsyncCallProbe(side_effect=["", "ready"])
        with ReplaceAttr(client, "_generate_inner", probe):
            await client._run_warmup_precompile(
                request_is_background=False,
                foreground_request=True,
                owner_name="warmup:test",
                # Above _MIN_READINESS_PROBE_BUDGET_S. A warmup with less
                # than that left deliberately does NOT open a readiness
                # probe — it ends inside the budget it promised — so a 1.0s
                # timeout never reached the path this test is about and
                # failed on the budget refusal instead.
                warmup_timeout=30.0,
            )

        self.assertEqual(client.get_lane_status()["state"], "ready")
        self.assertTrue(client.get_lane_status()["conversation_ready"])
        self.assertEqual(len(probe.await_args_list), 2)
        precompile_kwargs = probe.await_args_list[0].kwargs
        self.assertTrue(precompile_kwargs["warmup_precompile"])
        self.assertEqual(precompile_kwargs["max_tokens"], 1)
        self.assertTrue(precompile_kwargs["foreground_request"])
        self.assertFalse(precompile_kwargs["request_is_background"])
        readiness_kwargs = probe.await_args_list[1].kwargs
        self.assertTrue(readiness_kwargs["health_probe"])
        self.assertTrue(readiness_kwargs["disable_prompt_cache"])
        self.assertTrue(readiness_kwargs["clear_prompt_cache"])
        self.assertEqual(readiness_kwargs["max_tokens"], 16)

    async def test_resident_primary_readiness_probe_bypasses_only_headroom_reservation(self):
        from core.brain.llm import mlx_client as mlx_client_module

        client = MLXLocalClient(model_path=QWEN32_MODEL)
        client._process = ProcessProbe(alive=True)
        client._init_done = True
        admitted_probe = AsyncCallProbe(return_value=False)

        with ReplaceAttr(mlx_client_module, "_FOREGROUND_OWNER_NAME", None):
            with ReplaceAttr(
                mlx_client_module,
                "_background_deferral_active",
                lambda _origin: "foreground_headroom_reserved",
            ):
                with ReplaceAttr(client, "_ensure_worker_alive", admitted_probe):
                    result = await client._generate_inner(
                        "Reply exactly: ready",
                        request_is_background=True,
                        foreground_request=False,
                        owner_label="warmup:test",
                        health_probe=True,
                        deadline=get_deadline(1.0),
                    )

        self.assertIsNone(result)
        self.assertEqual(len(admitted_probe.await_args_list), 1)

        blocked_probe = AsyncCallProbe(return_value=False)
        with ReplaceAttr(
            mlx_client_module,
            "_background_deferral_active",
            lambda _origin: "critical_memory_pressure",
        ):
            with ReplaceAttr(client, "_ensure_worker_alive", blocked_probe):
                result = await client._generate_inner(
                    "Reply exactly: ready",
                    request_is_background=True,
                    foreground_request=False,
                    owner_label="warmup:test",
                    health_probe=True,
                    deadline=get_deadline(1.0),
                )

        self.assertIsNone(result)
        self.assertEqual(blocked_probe.await_args_list, [])

    async def test_warmup_returns_false_when_worker_start_is_deferred(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)

        with ReplaceAttr(client, "_ensure_worker_alive", AsyncCallProbe(return_value=False)):
            result = await client.warmup(foreground_request=False)

        self.assertFalse(result)
        self.assertEqual(client.get_lane_status()["state"], "recovering")

    async def test_background_warmup_yields_before_precompile_when_foreground_owned(self):
        """The anti-thrash yield belongs to NON-PRIMARY background lanes.
        (Originally pinned with the 32B — that deadlocked the cortex live on
        2026-07-10: the owning foreground turn deferred the primary's own
        warmup forever. The primary exemption is pinned in
        test_mlx_runtime_contract.py.)"""
        from core.brain.llm import mlx_client as mlx_client_module

        client = MLXLocalClient(model_path="/models/Qwen2.5-7B-Instruct-4bit")
        precompile = AsyncCallProbe(return_value=None)

        with ReplaceAttr(client, "_ensure_worker_alive", AsyncCallProbe(return_value=True)):
            with ReplaceAttr(client, "_run_warmup_precompile", precompile):
                with ReplaceAttr(mlx_client_module, "_FOREGROUND_OWNER_NAME", "warmup:foreground"):
                    result = await client.warmup(foreground_request=False)

        self.assertFalse(result)
        self.assertEqual(precompile.await_args_list, [])

    async def test_warmup_returns_true_only_after_visible_readiness(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        client._process = ProcessProbe(alive=True)
        client._init_done = True
        client._recurrent_depth_status = {
            "active": True,
            "config": {"n_loops": 2},
            "expected_loops": 2,
            "required": True,
        }

        async def _mark_ready(**_kwargs):
            now = time.time()
            client._set_lane_state("ready")
            client._last_ready_at = now
            client._last_progress_at = now
            client._last_visible_readiness_at = now

        with ReplaceAttr(client, "_ensure_worker_alive", AsyncCallProbe(return_value=True)):
            with ReplaceAttr(client, "_run_warmup_precompile", AsyncCallProbe(side_effect=_mark_ready)):
                result = await client.warmup(foreground_request=False)

        self.assertTrue(result)
        self.assertTrue(client.get_lane_status()["conversation_ready"])

    async def test_warmup_precompile_rejects_empty_readiness_probe(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        client._warmup_in_flight = True
        client._process = ProcessProbe(alive=True)
        client._init_done = True
        recorded: list[tuple[tuple, dict]] = []

        def _record(*args, **kwargs):
            recorded.append((args, kwargs))

        with replace_dotted("core.brain.llm.mlx_client._record_mlx_degradation", _record):
            # The contract under test is "the readiness probe never returns
            # text", not a fixed number of calls: a finite side_effect list
            # made this test exhaust (StopIteration) whenever the precompile
            # path changed its call count, which is what happened when the
            # visible probe became unconditional (CP126 cdd743de).
            with ReplaceAttr(client, "_generate_inner", AsyncCallProbe(return_value="")):
                with ReplaceAttr(client, "reboot_worker", AsyncCallProbe()):
                    with self.assertRaises(RuntimeError):
                        await client._run_warmup_precompile(
                            request_is_background=False,
                            foreground_request=True,
                            owner_name="warmup:test",
                            # See above: below _MIN_READINESS_PROBE_BUDGET_S
                            # no probe opens at all, so the empty-probe
                            # rejection this test names was unreachable.
                            warmup_timeout=30.0,
                        )

        lane = client.get_lane_status()
        self.assertEqual(lane["state"], "recovering")
        self.assertFalse(lane["conversation_ready"])
        self.assertEqual(lane["last_error"], "warmup_readiness_no_text")
        self.assertEqual(recorded, [])

    async def test_warmup_records_exhausted_precompile_once_at_outer_boundary(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        client._process = ProcessProbe(alive=True)
        client._init_done = True
        recorded: list[tuple[tuple, dict]] = []

        def _record(*args, **kwargs):
            recorded.append((args, kwargs))

        with replace_dotted("core.brain.llm.mlx_client._record_mlx_degradation", _record):
            with ReplaceAttr(client, "_ensure_worker_alive", AsyncCallProbe(return_value=True)):
                with ReplaceAttr(
                    client,
                    "_run_warmup_precompile",
                    AsyncCallProbe(side_effect=RuntimeError("warmup_failed")),
                ):
                    result = await client.warmup(foreground_request=False)

        self.assertFalse(result)
        self.assertEqual(len(recorded), 1)

    async def test_foreground_empty_generation_marks_recoverable_reboot(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        client._process = ProcessProbe(alive=True)
        client._init_done = True
        self._attach_local_ipc_queues(client)
        client._set_lane_state("ready")

        with ReplaceAttr(client, "_ensure_worker_alive", AsyncCallProbe(return_value=True)):
            with ReplaceAttr(
                client,
                "_wait_for_generation_result",
                AsyncCallProbe(return_value={"status": "ok", "text": ""}),
            ):
                result = await client._generate_inner(
                    "hello",
                    _retry=False,
                    foreground_request=True,
                    owner_label="test",
                    deadline=get_deadline(30.0),
                )

        self.assertIsNone(result)
        self.assertEqual(client._deferred_reboot_reason, "recoverable_empty_generation")
        self.assertEqual(client._last_generation_completed_at, 0.0)
        self.assertEqual(client._last_user_facing_completed_at, 0.0)
        self.assertEqual(client._last_visible_readiness_at, 0.0)
        lane = client.get_lane_status()
        self.assertFalse(lane["conversation_ready"])
        self.assertIn("visible_conversation_probe_missing", lane["readiness_blockers"])

    async def test_foreground_empty_generation_retry_is_noncritical_when_recovered(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        client._process = ProcessProbe(alive=True)
        client._init_done = True
        self._attach_local_ipc_queues(client)
        client._set_lane_state("ready")
        client._recurrent_depth_status = {
            "active": True,
            "config": {"n_loops": 2},
            "expected_loops": 2,
            "required": True,
        }
        recorded = SyncCallProbe()

        with ReplaceAttr(client, "_record_degraded_event", recorded):
            with ReplaceAttr(
                client,
                "_wait_for_generation_result",
                AsyncCallProbe(
                    side_effect=[
                        {"status": "ok", "text": ""},
                        {"status": "ok", "text": "Recovered visible reply."},
                    ]
                ),
            ):
                result = await client._generate_inner(
                    "hello",
                    _retry=True,
                    foreground_request=True,
                    owner_label="test",
                    deadline=get_deadline(30.0),
                )

        self.assertEqual(result, "Recovered visible reply.")
        self.assertEqual(len(recorded.call_args_list), 1)
        event = recorded.call_args_list[0]
        self.assertEqual(event.args, ("empty_generation_retry",))
        self.assertEqual(event.kwargs["severity"], "info")
        self.assertEqual(event.kwargs["classification"], "non_critical_fallback")
        self.assertIsNone(client._deferred_reboot_reason)
        self.assertEqual(client._consecutive_empty, 0)

    async def test_surface_quality_rejection_preserves_worker_without_empty_retry(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        client._process = ProcessProbe(alive=True)
        client._init_done = True
        self._attach_local_ipc_queues(client)
        client._set_lane_state("ready")
        client._recurrent_depth_status = {
            "active": True,
            "config": {"n_loops": 2},
            "expected_loops": 2,
            "required": True,
        }
        recorded = SyncCallProbe()
        wait_probe = AsyncCallProbe(
            return_value={
                "status": "ok",
                "text": "",
                "tokens_used": 9,
                "surface_control_receipt": {
                    "surface_quality_gate_enabled": True,
                    "surface_quality_gate_passed": False,
                    "surface_quality_gate_attempts": 3,
                    "surface_quality_gate_reasons": [
                        "missing_requested_word_count",
                        "missing_current_topic_anchor",
                    ],
                },
            }
        )

        with ReplaceAttr(client, "_record_degraded_event", recorded):
            with ReplaceAttr(client, "_wait_for_generation_result", wait_probe):
                result = await client._generate_inner(
                    "hello",
                    _retry=True,
                    foreground_request=True,
                    owner_label="test",
                    deadline=get_deadline(30.0),
                )

        self.assertIsNone(result)
        self.assertEqual(len(wait_probe.await_args_list), 1)
        self.assertEqual(len(recorded.call_args_list), 0)
        self.assertIsNone(client._deferred_reboot_reason)
        self.assertEqual(client._consecutive_empty, 0)
        self.assertEqual(client.get_lane_status()["state"], "ready")
        self.assertEqual(
            client.get_last_surface_control_receipt()[
                "surface_quality_gate_reasons"
            ],
            ["missing_requested_word_count", "missing_current_topic_anchor"],
        )

    async def test_foreground_empty_generation_exhaustion_records_terminal_incident(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        client._process = ProcessProbe(alive=True)
        client._init_done = True
        self._attach_local_ipc_queues(client)
        client._set_lane_state("ready")
        client._recurrent_depth_status = {
            "active": True,
            "config": {"n_loops": 2},
            "expected_loops": 2,
            "required": True,
        }
        recorded = SyncCallProbe()

        with ReplaceAttr(client, "_record_degraded_event", recorded):
            with ReplaceAttr(
                client,
                "_wait_for_generation_result",
                AsyncCallProbe(
                    side_effect=[
                        {"status": "ok", "text": ""},
                        {"status": "ok", "text": ""},
                    ]
                ),
            ):
                result = await client._generate_inner(
                    "hello",
                    _retry=True,
                    foreground_request=True,
                    owner_label="test",
                    deadline=get_deadline(30.0),
                )

        self.assertIsNone(result)
        self.assertEqual(
            [call.args[0] for call in recorded.call_args_list],
            ["empty_generation_retry", "empty_generation_exhausted"],
        )
        terminal = recorded.call_args_list[-1]
        self.assertEqual(terminal.kwargs["severity"], "error")
        self.assertTrue(terminal.kwargs["foreground_request"])
        self.assertIn("no_visible_text", terminal.kwargs["detail"])
        self.assertEqual(client._deferred_reboot_reason, "recoverable_empty_generation")

    async def test_generate_reboots_recoverable_empty_generation_without_failed_lane(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)

        async def _empty_then_request_reboot(*args, **kwargs):
            client._deferred_reboot_reason = "recoverable_empty_generation"
            return None

        reboot_probe = AsyncCallProbe()
        with ReplaceAttr(client, "_generate_inner", AsyncCallProbe(side_effect=_empty_then_request_reboot)):
            with ReplaceAttr(client, "reboot_worker", reboot_probe):
                result = await client.generate("hello", foreground_request=True, owner_label="test")

        self.assertIsNone(result)
        reboot_probe.assert_awaited_once_with(reason="empty_generation", mark_failed=False)

    async def test_supervision_status_reports_recycle_candidate(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        proc = ProcessProbe(alive=True)
        client._process = proc
        client._init_done = True
        client._process_started_at = 100.0
        client._last_generation_completed_at = 600.0

        with replace_dotted("core.brain.llm.mlx_client.time.time", lambda: 2000.0):
            status = client.get_supervision_status()
            recyclable = client.should_recycle_for_fragmentation(
                max_uptime_s=900.0,
                min_idle_s=300.0,
            )

        self.assertTrue(status["alive"])
        self.assertAlmostEqual(status["process_uptime_s"], 1900.0, places=3)
        self.assertAlmostEqual(status["idle_for_s"], 1400.0, places=3)
        self.assertTrue(recyclable)

    async def test_heavy_model_swap_respects_cooldown_window(self):
        import core.brain.llm.mlx_client as mlx_module

        primary_path = "/models/32B"
        deep_path = "/models/72B"
        solver = MLXLocalClient(model_path=deep_path)

        solver_proc = ProcessProbe(alive=True)

        async def _spawn_solver():
            solver._init_future.set_result(
                ready_init_receipt(deep_path, client=solver, process=solver_proc)
            )
            return solver_proc

        old_last_heavy = mlx_module._GLOBAL_LAST_HEAVY_MODEL
        old_last_swap = mlx_module._GLOBAL_LAST_SWAP_TIME
        mlx_module._GLOBAL_LAST_HEAVY_MODEL = primary_path
        mlx_module._GLOBAL_LAST_SWAP_TIME = 100.0

        try:
            sleep_probe = AsyncCallProbe()
            with replace_dotted("core.brain.llm.model_registry.ACTIVE_MODEL", "Qwen2.5-32B-Instruct-8bit"), \
                 replace_dotted("core.brain.llm.model_registry.get_deep_model_path", lambda: deep_path), \
                 replace_dotted("core.brain.llm.model_registry.get_model_path", lambda name=None: primary_path if "32B" in str(name) or name is None else deep_path), \
                 replace_dotted("core.brain.llm.mlx_client.os.path.realpath", lambda path, *_args, **_kwargs: path), \
                 replace_dotted("core.brain.llm.mlx_client._declared_mlx_worker_footprint_gb", lambda _path: 40.0), \
                 replace_dotted("core.brain.llm.mlx_client.time.time", lambda: 105.0), \
                 replace_dotted("core.brain.llm.mlx_client.asyncio.sleep", sleep_probe), \
                 ReplaceAttr(solver, "_spawn_worker", AsyncCallProbe(side_effect=_spawn_solver)):
                await solver._ensure_worker_alive()
        finally:
            mlx_module._GLOBAL_LAST_HEAVY_MODEL = old_last_heavy
            mlx_module._GLOBAL_LAST_SWAP_TIME = old_last_swap

        # CP126 1effd581: the cooldown moved OUT of the model-load admission
        # context and the global spawn gate (holding a process-wide semaphore
        # while counting to twelve blocked every other lane from spawning),
        # and it is now sliced so a shutdown does not have to outlast it. The
        # contract this test guards is the TOTAL wait, which is unchanged.
        total_slept = sum(
            float(call.args[0]) for call in sleep_probe.await_args_list if call.args
        )
        self.assertAlmostEqual(total_slept, 7.0, places=6)
        self.assertTrue(solver._init_done)

    async def test_heavy_model_swap_can_bypass_cooldown_for_fast_restore(self):
        import core.brain.llm.mlx_client as mlx_module

        primary_path = "/models/32B"
        deep_path = "/models/72B"
        primary = MLXLocalClient(model_path=primary_path)

        primary_proc = ProcessProbe(alive=True)

        async def _spawn_primary():
            primary._init_future.set_result(
                ready_init_receipt(primary_path, client=primary, process=primary_proc)
            )
            return primary_proc

        old_last_heavy = mlx_module._GLOBAL_LAST_HEAVY_MODEL
        old_last_swap = mlx_module._GLOBAL_LAST_SWAP_TIME
        mlx_module._GLOBAL_LAST_HEAVY_MODEL = deep_path
        mlx_module._GLOBAL_LAST_SWAP_TIME = 100.0

        try:
            sleep_probe = AsyncCallProbe()
            with replace_dotted("core.brain.llm.model_registry.ACTIVE_MODEL", "Qwen2.5-32B-Instruct-8bit"), \
                 replace_dotted("core.brain.llm.model_registry.get_deep_model_path", lambda: deep_path), \
                 replace_dotted("core.brain.llm.model_registry.get_model_path", lambda name=None: primary_path if "32B" in str(name) or name is None else deep_path), \
                 replace_dotted("core.brain.llm.mlx_client.os.path.realpath", lambda path, *_args, **_kwargs: path), \
                 replace_dotted("core.brain.llm.mlx_client.time.time", lambda: 105.0), \
                 replace_dotted("core.brain.llm.mlx_client.asyncio.sleep", sleep_probe), \
                 ReplaceAttr(primary, "_spawn_worker", AsyncCallProbe(side_effect=_spawn_primary)):
                await primary._ensure_worker_alive(skip_swap_cooldown=True)
        finally:
            mlx_module._GLOBAL_LAST_HEAVY_MODEL = old_last_heavy
            mlx_module._GLOBAL_LAST_SWAP_TIME = old_last_swap

        sleep_probe.assert_not_awaited()
        self.assertTrue(primary._init_done)


class TestIPCWriterThread(unittest.TestCase):
    def test_essential_messages_displace_telemetry_when_buffer_full(self):
        """Ladder rung 1: shed one telemetry item, keep ordering through the
        local queue, no synchronous (blocking) parent-queue write from the
        producer path."""
        mp_queue = SimpleNamespace(put=SyncCallProbe())
        writer = IPCWriterThread(mp_queue)
        writer.local_queue = queue.Queue(maxsize=1)
        writer.local_queue.put({"status": "heartbeat"})

        item = {"status": "ok", "action": "generate", "text": "hello"}
        writer.put(item)

        assert writer.local_queue.get_nowait() == item, (
            "essential message must displace buffered telemetry"
        )
        assert mp_queue.put.call_args_list == [], (
            "no synchronous parent-queue write when shedding succeeds"
        )

    def test_essential_messages_bypass_buffer_full_of_essentials(self):
        """Ladder rung 2: nothing sheddable -> blocking bypass to the parent
        queue so init/generation/error messages are never lost."""
        mp_queue = SimpleNamespace(put=SyncCallProbe())
        writer = IPCWriterThread(mp_queue)
        writer.local_queue = queue.Queue(maxsize=1)
        writer.local_queue.put({"status": "error", "message": "earlier failure"})

        item = {"status": "ok", "action": "generate", "text": "hello"}
        writer.put(item)

        mp_queue.put.assert_called_once_with(item, block=True, timeout=5.0)

    def test_terminal_message_displaces_progress_when_buffer_full(self):
        mp_queue = SimpleNamespace(put=SyncCallProbe())
        writer = IPCWriterThread(mp_queue)
        writer.local_queue = queue.Queue(maxsize=1)
        writer.local_queue.put({"status": "progress", "tokens_generated": 8})

        terminal = {"status": "ok", "action": "generate", "text": "done"}
        writer.put(terminal)

        assert writer.local_queue.get_nowait() == terminal
        mp_queue.put.assert_not_called()

    def test_progress_cannot_displace_terminal_message(self):
        mp_queue = SimpleNamespace(put=SyncCallProbe())
        writer = IPCWriterThread(mp_queue)
        writer.local_queue = queue.Queue(maxsize=1)
        terminal = {"status": "ok", "action": "generate", "text": "done"}
        writer.local_queue.put(terminal)

        writer.put({"status": "progress", "tokens_generated": 8})

        assert writer.local_queue.get_nowait() == terminal
        mp_queue.put.assert_not_called()

    def test_undeliverable_terminal_marks_response_pipe_broken(self):
        mp_queue = SimpleNamespace(put=SyncCallProbe(side_effect=queue.Full()))
        writer = IPCWriterThread(mp_queue)
        writer.local_queue = queue.Queue(maxsize=1)
        writer.local_queue.put({"status": "error", "message": "earlier failure"})

        writer.put({"status": "ok", "action": "generate", "text": "done"})

        assert writer.broken.is_set()

    def test_heartbeat_is_dropped_when_buffer_full(self):
        mp_queue = SimpleNamespace(put=SyncCallProbe())
        writer = IPCWriterThread(mp_queue)
        writer.local_queue = queue.Queue(maxsize=1)
        writer.local_queue.put({"status": "heartbeat"})

        writer.put({"status": "heartbeat", "timestamp": 1.0})

        mp_queue.put.assert_not_called()


class TestMLXWorkerProgress(unittest.IsolatedAsyncioTestCase):
    def test_worker_stall_alarm_respects_active_32b_first_token_budget(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        client._current_request_id = "demo-request"
        client._current_first_token_at = 0.0
        client._current_first_token_hard_ceiling_s = 120.0

        stalled, budget = client._confirm_worker_reported_loop_stall(
            {
                "request_id": "demo-request",
                "job_age_s": 30.461,
                "loop_stalled": True,
            }
        )

        self.assertFalse(stalled)
        self.assertEqual(budget, 120.0)

        stalled, budget = client._confirm_worker_reported_loop_stall(
            {
                "request_id": "demo-request",
                "job_age_s": 120.1,
                "loop_stalled": True,
            }
        )

        self.assertTrue(stalled)
        self.assertEqual(budget, 120.0)

    async def test_response_listener_soft_cancels_confirmed_worker_loop_stall(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        client._req_q = queue.Queue()
        client._res_q = queue.Queue()
        client._current_request_id = "demo-request"
        client._current_request_started_at = 100.0
        client._current_request_seq = 7
        client._current_first_token_at = 0.0
        client._current_first_token_hard_ceiling_s = 120.0
        client._cancel_seq.value = 0

        listener = asyncio.create_task(client._response_listener_loop())
        try:
            client._res_q.put(
                {
                    "status": "heartbeat",
                    "request_id": "demo-request",
                    "job_age_s": 30.461,
                    "loop_stalled": True,
                }
            )
            await asyncio.sleep(0.25)
            self.assertEqual(client._cancel_seq.value, 0)
            self.assertIsNone(client._deferred_reboot_reason)

            client._res_q.put(
                {
                    "status": "heartbeat",
                    "request_id": "demo-request",
                    "job_age_s": 121.0,
                    "loop_stalled": True,
                }
            )
            await asyncio.sleep(0.25)
        finally:
            listener.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await listener

        self.assertEqual(client._cancel_seq.value, 7)
        self.assertEqual(client._deferred_reboot_reason, "recoverable_token_progress_stalled")

    def test_prompt_cache_budget_is_bounded_by_size_not_by_entry_count(self):
        """The 72B still retains nothing; the 32B is no longer starved.

        This asserted 2 entries for the 32B. While the entry count was the only
        memory bound it had to stay tiny — but 2 entries meant the internal
        lane held exactly one, so its several distinct prompt families evicted
        each other every tick, measured live and repeatedly as
        "trimmed hit — reused 3/792 tokens". Retained KV is now capped directly
        by _prompt_cache_total_token_budget_for_model, so entries can be
        generous and the eviction that matters is by total size.
        """
        self.assertEqual(_prompt_cache_entry_budget_for_model("/models/Qwen2.5-72B-Instruct-4bit"), 0)
        thirty_two = _prompt_cache_entry_budget_for_model("/models/Qwen2.5-32B-Instruct-8bit")
        self.assertGreaterEqual(
            thirty_two, 4, "two entries starved the internal lane to one"
        )
        # And the real ceiling is the token budget, which must exist for it.
        from core.brain.llm.mlx_worker import (
            _prompt_cache_total_token_budget_for_model,
        )

        self.assertGreater(
            _prompt_cache_total_token_budget_for_model("/models/Qwen2.5-32B-Instruct-8bit"),
            0,
            "a generous entry count is only safe because total KV is bounded",
        )

    def test_generation_progress_emits_on_first_token(self):
        self.assertTrue(
            _should_emit_generation_progress(
                1,
                last_emit_at=100.0,
                now=100.2,
            )
        )

    def test_generation_progress_emits_on_time_gap_before_token_modulus(self):
        self.assertTrue(
            _should_emit_generation_progress(
                3,
                last_emit_at=100.0,
                now=101.7,
            )
        )

    def test_generation_progress_stays_quiet_when_recent_and_off_cycle(self):
        self.assertFalse(
            _should_emit_generation_progress(
                3,
                last_emit_at=100.0,
                now=100.4,
            )
        )

    def test_heavy_model_prefill_is_chunked_below_the_stall_horizon(self):
        self.assertEqual(
            _prefill_step_size_for_model("/models/Qwen2.5-32B-Instruct-8bit"),
            128,
        )
        self.assertEqual(
            _prefill_step_size_for_model("/models/Qwen2.5-72B-Instruct-4bit"),
            64,
        )
        self.assertGreaterEqual(
            _prefill_step_size_for_model("/models/Qwen2.5-7B-Instruct-4bit"),
            256,
        )

    def test_32b_prefill_chunk_shrinks_with_measured_host_pressure(self):
        normal = SimpleNamespace(
            observation_available=True,
            available_gb=32.0,
            level="normal",
        )
        constrained = SimpleNamespace(
            observation_available=True,
            available_gb=18.0,
            level="warning",
        )
        critical = SimpleNamespace(
            observation_available=True,
            available_gb=8.0,
            level="critical",
        )

        self.assertEqual(
            _prefill_step_size_for_model(
                "/models/Qwen2.5-32B-Instruct-8bit",
                pressure_snapshot=normal,
            ),
            128,
        )
        self.assertEqual(
            _prefill_step_size_for_model(
                "/models/Qwen2.5-32B-Instruct-8bit",
                pressure_snapshot=constrained,
            ),
            64,
        )
        self.assertEqual(
            _prefill_step_size_for_model(
                "/models/Qwen2.5-32B-Instruct-8bit",
                pressure_snapshot=critical,
            ),
            32,
        )

    def test_prefill_pressure_can_never_increase_model_base_chunk(self):
        constrained = SimpleNamespace(
            observation_available=True,
            available_gb=5.0,
            level="emergency",
        )

        for model_path in (
            "/models/Qwen2.5-72B-Instruct-4bit",
            "/models/Qwen2.5-32B-Instruct-8bit",
            "/models/Qwen2.5-14B-Instruct-4bit",
            "/models/Qwen2.5-7B-Instruct-4bit",
        ):
            base = _prefill_step_size_for_model(model_path)
            pressured = _prefill_step_size_for_model(
                model_path,
                pressure_snapshot=constrained,
            )
            self.assertLessEqual(pressured, base)
            self.assertGreaterEqual(pressured, 32)

    def test_unavailable_prefill_pressure_sample_does_not_invent_pressure(self):
        unavailable = SimpleNamespace(
            observation_available=False,
            available_gb=0.0,
            level="emergency",
        )

        self.assertEqual(
            _prefill_step_size_for_model(
                "/models/Qwen2.5-32B-Instruct-8bit",
                pressure_snapshot=unavailable,
            ),
            128,
        )

    def test_prefill_progress_refreshes_watchdog_and_emits_correlated_phase(self):
        class WatchdogProbe:
            def __init__(self):
                self.calls = 0

            def activity(self):
                self.calls += 1

        class WriterProbe:
            def __init__(self):
                self.messages = []

            def put(self, message):
                self.messages.append(message)

        watchdog = WatchdogProbe()
        writer = WriterProbe()
        callback = _build_prefill_progress_callback(
            watchdog,
            writer,
            request_id="prefill-request",
            action="generate",
        )

        callback(0, 755)
        callback(128, 755)
        callback(256, 755)

        self.assertEqual(watchdog.calls, 3)
        self.assertEqual(
            [message["prompt_tokens_processed"] for message in writer.messages],
            [0, 128, 256],
        )
        self.assertTrue(all(message["phase"] == "prefill" for message in writer.messages))
        self.assertTrue(all(message["id"] == "prefill-request" for message in writer.messages))

    @pytest.mark.hardware
    def test_installed_mlx_lm_calls_prefill_hook_at_each_chunk_boundary(self):
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.models.qwen2 import Model, ModelArgs

        class WatchdogProbe:
            def __init__(self):
                self.calls = 0

            def activity(self):
                self.calls += 1

        class WriterProbe:
            def __init__(self):
                self.messages = []

            def put(self, message):
                self.messages.append(message)

        model = Model(
            ModelArgs(
                model_type="qwen2",
                hidden_size=16,
                num_hidden_layers=1,
                intermediate_size=32,
                num_attention_heads=2,
                rms_norm_eps=1e-6,
                vocab_size=32,
                num_key_value_heads=1,
                max_position_embeddings=64,
            )
        )
        watchdog = WatchdogProbe()
        writer = WriterProbe()
        callback = _build_prefill_progress_callback(
            watchdog,
            writer,
            request_id="real-prefill",
            action="generate",
        )
        generator = generate_step(
            mx.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            model,
            max_tokens=1,
            prefill_step_size=4,
            prompt_progress_callback=callback,
        )

        next(generator)

        coordinates = [
            message["prompt_tokens_processed"] for message in writer.messages
        ]
        self.assertEqual(coordinates[0], 0)
        self.assertIn(4, coordinates)
        self.assertIn(8, coordinates)
        self.assertEqual(coordinates[-1], 10)
        self.assertEqual(watchdog.calls, len(coordinates))

    async def test_prefill_progress_does_not_claim_a_first_token(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        client._req_q = queue.Queue()
        client._res_q = queue.Queue()
        client._mark_generation_started("prefill-request", prompt_chars=4096)

        listener = asyncio.create_task(client._response_listener_loop())
        try:
            client._res_q.put(
                {
                    "status": "progress",
                    "phase": "prefill",
                    "action": "generate",
                    "id": "prefill-request",
                    "prompt_tokens_processed": 128,
                    "prompt_tokens_total": 755,
                }
            )
            await asyncio.sleep(0.25)
        finally:
            listener.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await listener

        self.assertEqual(client._current_first_token_at, 0.0)
        self.assertEqual(client._last_token_progress_at, 0.0)
        self.assertEqual(client._current_prefill_tokens_processed, 128)
        self.assertEqual(client._current_prefill_tokens_total, 755)

    async def test_latent_stage_progress_does_not_claim_a_decoded_token(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        client._req_q = queue.Queue()
        client._res_q = queue.Queue()
        client._mark_generation_started(
            "latent-request",
            prompt_chars=4096,
            first_token_hard_ceiling_s=153.0,
        )
        baseline_progress_at = client._last_progress_at

        listener = asyncio.create_task(client._response_listener_loop())
        try:
            client._res_q.put(
                {
                    "status": "progress",
                    "action": "latent_reason",
                    "id": "latent-request",
                    "stage": "branch_select",
                    "elapsed_s": 112.0,
                    "spent_layer_apps": 160_000,
                }
            )
            await asyncio.sleep(0.25)
        finally:
            listener.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await listener

        self.assertEqual(client._current_first_token_at, 0.0)
        self.assertEqual(client._last_token_progress_at, 0.0)
        self.assertGreater(client._last_progress_at, baseline_progress_at)
        self.assertEqual(
            client._latent_progress_by_request["latent-request"]["stage"],
            "branch_select",
        )
        stalled, budget = client._confirm_worker_reported_loop_stall(
            {
                "request_id": "latent-request",
                "job_age_s": 46.7,
                "loop_stalled": True,
            }
        )
        self.assertFalse(stalled)
        self.assertEqual(budget, 153.0)

    async def test_textless_stream_progress_still_claims_a_decoded_token(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        client._req_q = queue.Queue()
        client._res_q = queue.Queue()
        client._mark_generation_started(
            "stream-request",
            prompt_chars=4096,
            first_token_hard_ceiling_s=153.0,
        )

        listener = asyncio.create_task(client._response_listener_loop())
        try:
            client._res_q.put(
                {
                    "status": "progress",
                    "action": "stream",
                    "id": "stream-request",
                    "tokens_generated": 1,
                }
            )
            await asyncio.sleep(0.25)
        finally:
            listener.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await listener

        self.assertGreater(client._current_first_token_at, 0.0)
        self.assertGreater(client._last_token_progress_at, 0.0)


class TestMLXRuntimeProbeFailure(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_probe_failure_marks_lane_failed_without_spawn_loop(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)

        spawn_probe = AsyncCallProbe(
            side_effect=RuntimeError("mlx_runtime_probe_failed:metal_device_enumeration_crash")
        )
        with ReplaceAttr(client, "_spawn_worker", spawn_probe):
            alive = await client._ensure_worker_alive()

        self.assertFalse(alive)
        spawn_probe.assert_awaited_once()
        self.assertEqual(client.get_lane_status()["state"], "failed")
        self.assertEqual(
            client.get_lane_status()["last_error"],
            "mlx_runtime_unavailable:metal_device_enumeration_crash",
        )

    async def test_runtime_probe_recovery_clears_failed_lane_and_backoff(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        client.note_lane_failed("mlx_runtime_unavailable:metal_device_enumeration_crash")
        client._spawn_backoff_until = time.time() + 120.0
        client._consecutive_spawn_failures = 3

        proc = ProcessProbe(alive=True)

        async def _spawn():
            client._init_future.set_result(
                ready_init_receipt(
                    QWEN32_MODEL,
                    client=client,
                    process=proc,
                    recurrent_depth={
                        "active": True,
                        "config": {"n_loops": 2},
                        "expected_loops": 2,
                        "required": True,
                        "loops": 2,
                    },
                )
            )
            return proc

        spawn_probe = AsyncCallProbe(side_effect=_spawn)
        with replace_dotted("core.brain.llm.mlx_client._probe_mlx_runtime", lambda force=False: (True, "mlx_runtime_ok")):
            with ReplaceAttr(client, "_spawn_worker", spawn_probe):
                alive = await client._ensure_worker_alive()

        self.assertTrue(alive)
        spawn_probe.assert_awaited_once()
        self.assertEqual(client.get_lane_status()["state"], "ready")
        self.assertEqual(client.get_lane_status()["last_error"], "")
        self.assertEqual(client._consecutive_spawn_failures, 0)
        self.assertEqual(client._spawn_backoff_until, 0.0)


def test_probe_reuses_fresh_positive_disk_cache(monkeypatch):
    import core.brain.llm.mlx_client as mlx_module

    monkeypatch.setattr(mlx_module.time, "time", lambda: 1000.0)
    monkeypatch.setattr(mlx_module, "_load_probe_cache_from_disk", lambda: (True, "mlx_runtime_ok", 950.0))
    monkeypatch.setattr(
        mlx_module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("probe should not run")),
    )
    monkeypatch.setattr(
        mlx_module,
        "_MLX_RUNTIME_PROBE",
        {"ok": None, "detail": "", "checked_at": 0.0},
    )

    ok, detail = mlx_module._probe_mlx_runtime(force=False)

    assert ok is True
    assert detail == "mlx_runtime_ok"


def test_probe_does_not_trust_stale_negative_disk_cache(monkeypatch):
    import core.brain.llm.mlx_client as mlx_module

    class _Completed:
        returncode = 0
        stdout = "mlx_runtime_ok\n"
        stderr = ""

    calls = []

    monkeypatch.setattr(mlx_module.time, "time", lambda: 1000.0)
    monkeypatch.setattr(
        mlx_module,
        "_load_probe_cache_from_disk",
        lambda: (False, "metal_device_enumeration_crash", 900.0),
    )
    monkeypatch.setattr(
        mlx_module.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)) or _Completed(),
    )
    monkeypatch.setattr(
        mlx_module,
        "_MLX_RUNTIME_PROBE",
        {"ok": None, "detail": "", "checked_at": 0.0},
    )
    monkeypatch.setattr(mlx_module, "_store_probe_cache_to_disk", lambda ok, detail: None)

    ok, detail = mlx_module._probe_mlx_runtime(force=False)

    assert ok is True
    assert detail == "mlx_runtime_ok"
    assert calls


class TestSpawnGateBoundedAcquire(unittest.IsolatedAsyncioTestCase):
    """The nightcap-wedge root: one wedged spawn holding the global gate must
    never freeze every other lane's warmup coroutine forever. Waiters time
    out, raise, and the caller defers with a receipt."""

    async def test_gate_timeout_raises_and_caller_defers(self):
        from core.brain.llm import mlx_client as mc

        self.assertTrue(mc._GLOBAL_SPAWN_GATE.acquire(blocking=False))
        original = mc._SPAWN_GATE_ACQUIRE_TIMEOUT_S
        mc._SPAWN_GATE_ACQUIRE_TIMEOUT_S = 0.1
        try:
            with self.assertRaises(TimeoutError):
                async with mc._spawn_gate_context(owner="blocked-test"):
                    pass  # pragma: no cover - never reached

            client = MLXLocalClient(model_path=TEST_MODEL)
            client._req_q = queue.Queue()
            try:
                alive = await client._ensure_worker_alive(foreground_request=True)
            finally:
                client.close()
            self.assertFalse(alive, "a gate timeout must defer, not hang or crash")
        finally:
            mc._SPAWN_GATE_ACQUIRE_TIMEOUT_S = original
            mc._GLOBAL_SPAWN_GATE.release()

    async def test_gate_releases_after_normal_use(self):
        from core.brain.llm import mlx_client as mc

        async with mc._spawn_gate_context(owner="normal-test"):
            snapshot = mc._spawn_gate_snapshot()
            self.assertTrue(snapshot["held"])
            self.assertEqual(snapshot["owner"], "normal-test")
        # gate must be free again
        self.assertFalse(mc._spawn_gate_snapshot()["held"])
        self.assertTrue(mc._GLOBAL_SPAWN_GATE.acquire(blocking=False))
        mc._GLOBAL_SPAWN_GATE.release()

    async def test_cancelled_waiter_cannot_acquire_and_leak_gate_later(self):
        """Regression for the live 330-second gate wedge.

        Cancelling ``asyncio.to_thread(Semaphore.acquire)`` left its worker
        thread alive. Once the real holder released, that abandoned thread
        acquired the semaphore forever because no context manager remained.
        """
        from core.brain.llm import mlx_client as mc

        self.assertTrue(mc._GLOBAL_SPAWN_GATE.acquire(blocking=False))
        original = mc._SPAWN_GATE_ACQUIRE_TIMEOUT_S
        mc._SPAWN_GATE_ACQUIRE_TIMEOUT_S = 1.0

        async def _waiter():
            async with mc._spawn_gate_context(owner="cancelled-test"):
                self.fail("cancelled waiter must never enter the gate")

        waiter = asyncio.create_task(_waiter())
        try:
            await asyncio.sleep(0.05)
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
        finally:
            mc._GLOBAL_SPAWN_GATE.release()
            mc._SPAWN_GATE_ACQUIRE_TIMEOUT_S = original

        await asyncio.sleep(0.1)
        self.assertFalse(mc._spawn_gate_snapshot()["held"])
        self.assertTrue(mc._GLOBAL_SPAWN_GATE.acquire(blocking=False))
        mc._GLOBAL_SPAWN_GATE.release()
