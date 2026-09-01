from __future__ import annotations

from core.learning.semantic_program_campaign import _one_sided_exact_p, _paired_control


def test_paired_control_counts_direction_and_exact_probability() -> None:
    treatment = [
        {"source_text_sha256": "a", "program_exact": True},
        {"source_text_sha256": "b", "program_exact": True},
        {"source_text_sha256": "c", "program_exact": False},
    ]
    control = [
        {"source_text_sha256": "a", "program_exact": False},
        {"source_text_sha256": "b", "program_exact": False},
        {"source_text_sha256": "c", "program_exact": True},
    ]

    result = _paired_control(treatment, control)

    assert result == {
        "treatment_only": 2,
        "control_only": 1,
        "discordant": 3,
        "one_sided_exact_p": 0.5,
    }


def test_exact_probability_handles_no_discordance_and_one_sided_win() -> None:
    assert _one_sided_exact_p(treatment_only=0, control_only=0) == 1.0
    assert _one_sided_exact_p(treatment_only=4, control_only=0) == 0.0625
