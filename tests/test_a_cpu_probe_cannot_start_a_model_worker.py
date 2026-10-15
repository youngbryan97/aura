"""A CPU probe cannot start a vision model through a side path."""
import builtins
from types import SimpleNamespace

import pytest

from core.runtime.model_worker_policy import (
    _model_worker_policy_invariant,
    require_model_workers_allowed,
)
from core.runtime.subprocess_gateway import (
    AcceleratorCapability,
    PythonProcessSpec,
    SubprocessGateway,
    _resolve_accelerator_claim,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def denied(monkeypatch):
    monkeypatch.setenv('AURA_MODEL_WORKERS_ALLOWED', '0')


def test_normal_policy_is_unchanged_and_explicit_denial_is_measured(monkeypatch):
    monkeypatch.setenv('AURA_MODEL_WORKERS_ALLOWED', '1')
    require_model_workers_allowed(source='test model')
    monkeypatch.setenv('AURA_MODEL_WORKERS_ALLOWED', '0')
    with pytest.raises(RuntimeError, match='prohibited'):
        require_model_workers_allowed(source='test model')
    assert _model_worker_policy_invariant() == ()


def test_model_worker_refusal_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'model_worker_policy_refuses_before_start')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())


@pytest.mark.asyncio
async def test_vision_client_refuses_before_a_lane_claim_or_spawn(denied, monkeypatch):
    from core.brain.llm import mlx_vision_client as vision

    def forbidden(*args, **kwargs):
        pytest.fail('model policy must be checked before model admission or spawn')

    client = vision.MLXVisionClient('not-loaded')
    monkeypatch.setattr(vision, 'prepare_model_lane_claim', forbidden)
    monkeypatch.setattr(client, '_spawn_worker_blocking', forbidden)
    assert await client.start_async() is False
    assert 'prohibited' in client._init_error
    assert client._process is None


def test_direct_worker_entry_refuses_before_mlx_import_or_heartbeat(denied, monkeypatch):
    from core.brain.llm import mlx_vision_worker as worker

    messages, original = [], builtins.__import__

    def imports(name, *args, **kwargs):
        if name.startswith('mlx'):
            pytest.fail('a refused worker imported accelerator code')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', imports)
    worker._mlx_vision_worker_loop('not-loaded', None, SimpleNamespace(put=messages.append))
    assert len(messages) == 1 and messages[0]['status'] == 'error' and messages[0]['action'] == 'init'


def test_generic_model_process_is_refused_before_process_construction(denied):
    from core.runtime.process_privilege import ProcessRole

    spec = PythonProcessSpec(target=print, args=(), source='proof_tooling:cpu-policy', name='forbidden-worker',
                             role=ProcessRole.MODEL_WORKER, requested_privileges=frozenset(),
                             accelerator_capability=AcceleratorCapability.MODEL)
    with pytest.raises(RuntimeError, match='prohibited'):
        SubprocessGateway().spawn_python_process(spec, context=object())


def test_inferred_model_claim_is_also_refused(denied, monkeypatch):
    from core.runtime import subprocess_gateway as gateway

    monkeypatch.setattr(gateway, '_inferred_model_lane_claim', lambda *args, **kwargs: object())
    with pytest.raises(RuntimeError, match='prohibited'):
        _resolve_accelerator_claim(['model-command'], source='proof_tooling:cpu-policy', timeout_s=1.,
                                   accelerator_capability=AcceleratorCapability.AUTO)


def test_cpu_cli_sets_one_envelope_and_refuses_a_model_endpoint(monkeypatch, tmp_path):
    from tools.measure_on_archived_games import _arguments

    for key in ('AURA_MODEL_WORKERS_ALLOWED', 'AURA_VISION_CPU_ONLY', 'AURA_PLAY_POINT_BACKEND', 'AURA_PLAY_POINT_DEVICE'):
        monkeypatch.setenv(key, '')
    args, _, _ = _arguments(['--games', '2', '--cpu-only', '--out', str(tmp_path / 'pass.json')])
    assert args.graphics == 'software' and args.vision_cpu and args.cpu_only
    import os

    assert os.environ['AURA_MODEL_WORKERS_ALLOWED'] == '0' and os.environ['AURA_PLAY_POINT_BACKEND'] == 'classical'
    assert os.environ['AURA_VISION_CPU_ONLY'] == '1' and os.environ['AURA_PLAY_POINT_DEVICE'] == 'cpu'
    with pytest.raises(SystemExit):
        _arguments(['--cpu-only', '--her-model', 'http://uncontacted', '--out', str(tmp_path / 'other.json')])
