"""Compatibility helpers for local TTS backends.

Coqui TTS 0.27.x imports ``transformers.pytorch_utils.isin_mps_friendly``.
The 5.x transformers line used by Aura's MLX lane removed that helper, while
MLX itself requires transformers >= 5.0.  This module restores the tiny helper
at runtime so TTS can coexist with the live inference stack.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def install_transformers_tts_compat() -> bool:
    """Install the Coqui/transformers compatibility shim if it is missing."""
    try:
        import torch
        from transformers import pytorch_utils
    # not a failure: the module is optional here, and its absence is the answer.
    except (ImportError, RuntimeError, AttributeError):
        return False

    if hasattr(pytorch_utils, "isin_mps_friendly"):
        return True

    def isin_mps_friendly(elements: Any, test_elements: Any) -> Any:
        """MPS-safe ``torch.isin`` replacement matching the removed helper."""
        if not isinstance(test_elements, torch.Tensor):
            if isinstance(test_elements, Iterable) and not isinstance(test_elements, (str, bytes)):
                test_elements = list(test_elements)
            test_elements = torch.as_tensor(test_elements, device=getattr(elements, "device", None))
        try:
            return torch.isin(elements, test_elements)
        except (RuntimeError, TypeError):
            cpu_result = torch.isin(elements.detach().cpu(), test_elements.detach().cpu())
            return cpu_result.to(elements.device)

    pytorch_utils.isin_mps_friendly = isin_mps_friendly
    return True
