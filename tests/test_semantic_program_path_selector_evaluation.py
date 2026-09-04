from __future__ import annotations

import pytest

from core.learning.semantic_program_path_selector_evaluation import (
    _one_sided_exact_p,
)


def test_selector_development_uses_the_exact_paired_tail() -> None:
    assert _one_sided_exact_p(treatment_only=5, control_only=0) == 0.03125
    assert _one_sided_exact_p(treatment_only=0, control_only=0) == 1.0
    assert _one_sided_exact_p(treatment_only=12, control_only=3) == pytest.approx(
        0.017578125
    )
