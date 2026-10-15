"""Select an operator's renderer without weakening a software-only request."""
from __future__ import annotations

from typing import Any

from core.verify.invariants import invariant


def graphics_arguments(backend: str, platform: str) -> list[str]:
    if backend == 'software':
        return ['--use-gl=angle', '--use-angle=swiftshader']
    if backend != 'host':
        raise ValueError('graphics backend must be host or software')
    return ['--enable-gpu', '--use-angle=metal'] if platform == 'darwin' else []


def software_graphics_verified(graphics: Any) -> bool:
    """Use the browser process's renderer report, never a page's claim."""
    renderer = graphics.get('renderer') if isinstance(graphics, dict) else None
    return isinstance(renderer, str) and 'swiftshader' in renderer.lower()


@invariant('capabilities.software_graphics_need_a_measured_cpu_renderer', scope='capabilities',
           owner='core/capabilities/browser_graphics_policy.py', observational=False)
def _software_graphics_invariant() -> tuple:
    args = graphics_arguments('software', 'darwin')
    assert args == graphics_arguments('software', 'linux')
    assert '--use-angle=metal' not in args and '--enable-unsafe-swiftshader' not in args
    assert software_graphics_verified({'renderer': 'ANGLE (SwiftShader Device (Subzero))'})
    assert not software_graphics_verified({'renderer': 'ANGLE Metal Renderer: Apple'})
    assert not software_graphics_verified(None)
    return ()
