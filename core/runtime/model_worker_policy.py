"""An operator can prohibit new model workers in this process and its children."""
from core.runtime.flags import FlagKind, declare
from core.verify.invariants import invariant

_ALLOWED = declare('AURA_MODEL_WORKERS_ALLOWED', kind=FlagKind.BOOL, default=True,
    description='Permit model-worker starts; false refuses them before process creation or model-lane admission',
    owner='core/runtime/model_worker_policy.py')


def model_workers_allowed() -> bool:
    return bool(_ALLOWED.value())


def require_model_workers_allowed(*, source: str) -> None:
    if not model_workers_allowed():
        raise RuntimeError(f'model worker start prohibited by process policy: {source}')


@invariant('runtime.model_worker_policy_refuses_before_start', scope='runtime',
           owner='core/runtime/model_worker_policy.py', observational=False)
def _model_worker_policy_invariant() -> tuple:
    assert isinstance(model_workers_allowed(), bool)
    if not model_workers_allowed():
        try:
            require_model_workers_allowed(source='contract probe')
        except RuntimeError:
            return ()
        raise AssertionError('a prohibited model worker was admitted')
    return ()
