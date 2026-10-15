"""Apply an explicit CPU policy before an Apple Vision request may execute."""
from __future__ import annotations

from typing import Any

from core.runtime.flags import FlagKind, declare
from core.verify import invariant

_CPU_ONLY = declare('AURA_VISION_CPU_ONLY', kind=FlagKind.BOOL, default=False,
    description='Require Apple Vision requests to acknowledge CPU-only execution before processing',
    owner='core/perception/vision_request_compute.py')


def configure_vision_request(request: Any, *, cpu_only: bool | None = None) -> bool:
    required = _CPU_ONLY.value() if cpu_only is None else cpu_only
    if not required:
        return False
    try:
        request.setUsesCPUOnly_(True)
        if not request.usesCPUOnly():
            raise RuntimeError('Vision did not acknowledge CPU-only execution')
    except AttributeError as error:
        raise RuntimeError('Vision cannot acknowledge CPU-only execution') from error
    return True


@invariant('perception.cpu_vision_requires_request_acknowledgement', scope='perception',
           owner='core/perception/vision_request_compute.py', observational=False)
def _cpu_vision_request_invariant() -> tuple:
    from types import SimpleNamespace

    chosen = []
    request = SimpleNamespace(setUsesCPUOnly_=chosen.append, usesCPUOnly=lambda: bool(chosen and chosen[-1]))
    assert not configure_vision_request(request, cpu_only=False) and not chosen
    assert configure_vision_request(request, cpu_only=True) and chosen == [True]
    return ()
