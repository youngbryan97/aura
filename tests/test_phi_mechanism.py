from __future__ import annotations

import numpy as np


def test_phi_core_records_residual_stream_as_mechanism_complex():
    from core.consciousness.phi_core import PhiCore

    phi = PhiCore()
    base = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    for idx in range(80):
        signal = base * ((idx % 5) - 2)
        signal[0::3] += float(idx % 2)
        phi.record_residual_stream(signal, layer_idx=12)

    result = phi.compute_residual_phi()
    status = phi.get_status()

    assert result is not None
    assert result.tpm_n_samples >= 50
    assert status["residual_history_length"] >= 80
    assert status["residual_phi_s"] is not None
