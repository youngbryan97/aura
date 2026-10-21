"""Integrated information decomposed, as a diagnostic beside the battery's partition lines.

The battery decides irreducibility by intervention. Integrated information
decomposition (Mediano, Rosas, Carhart-Harris, Seth and Barrett 2019, 2021)
asks the same question of a recording: how much of what the whole carries
from one turn to the next is not carried by either part from its own past, once
the information both parts carry redundantly is counted once. Its revised
measure is

    Phi_R = I(X_t; X_t+1) - I(A_t; A_t+1) - I(B_t; B_t+1) + R

where R is the double-redundancy atom, here the minimum over the four mutual
informations between a part now and a part next (MMI redundancy). Phi_R below
zero says the parts' redundancy was all there was; above zero, the cut loses
information the parts alone do not carry. Read on the same components and the
same bias-corrected Gaussian estimator as the synergy line, on each cut of one
domain from the rest. Diagnostic: it counts toward no criterion.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from core.subject.recording import Recording
from core.subject.synergy import _components, _copula_normal, _gaussian_mi, without_clocks

__all__ = ["phi_r", "phi_r_by_domain"]


def phi_r(now_a: np.ndarray, now_b: np.ndarray, next_a: np.ndarray, next_b: np.ndarray) -> dict[str, float]:
    """Phi_R for one bipartition, from the parts' states now and next."""
    a, b, a_next, b_next = (_copula_normal(block) for block in (now_a, now_b, next_a, next_b))
    whole = _gaussian_mi(np.hstack([a, b]), np.hstack([a_next, b_next]))
    own_a = _gaussian_mi(a, a_next)
    own_b = _gaussian_mi(b, b_next)
    redundancy = min(own_a, own_b, _gaussian_mi(a, b_next), _gaussian_mi(b, a_next))
    return {
        "whole": round(float(whole), 6),
        "part_a": round(float(own_a), 6),
        "part_b": round(float(own_b), 6),
        "redundancy": round(float(redundancy), 6),
        "phi_r": round(float(whole - own_a - own_b + redundancy), 6),
    }


def phi_r_by_domain(recording: Recording) -> dict[str, Any]:
    """Phi_R of each one-domain cut against the rest, counters out, at the recording's grain."""
    recording = without_clocks(recording)
    out: dict[str, Any] = {}
    domains = list(recording.slices)
    for domain in domains:
        rest = [d for d in domains if d != domain]
        part = _components(recording.domain(domain))
        others = _components(np.hstack([recording.domain(d) for d in rest]))
        if part.shape[0] < 3 or others.shape[0] < 3:
            continue
        out[f"{domain}|rest"] = phi_r(part[:-1], others[:-1], part[1:], others[1:])
    return out
