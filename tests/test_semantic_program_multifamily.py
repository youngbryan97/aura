from __future__ import annotations

import copy
from dataclasses import replace

from core.learning.semantic_program_multifamily import (
    SEMANTIC_PROGRAM_MULTIFAMILY_CAMPAIGN_SCHEMA,
    run_semantic_program_multifamily_campaign_from_examples,
)
from tests.test_semantic_program_basis import _basis, _manifest, _sha
from tests.test_semantic_program_transducer import _training


def test_one_shared_model_is_measured_separately_for_each_family() -> None:
    arithmetic_basis = _basis(boot="a" * 32, pid=111)
    sequence_basis = _basis(boot="b" * 32, pid=222)
    arithmetic = tuple(
        replace(
            copy.deepcopy(item),
            split=("train" if index < 48 else "validation" if index < 56 else "test"),
        )
        for index, item in enumerate(_training())
    )
    sequence = tuple(
        replace(
            item,
            ir=replace(
                item.ir,
                model_basis_receipt_sha256=_sha(sequence_basis),
            ),
        )
        for item in arithmetic
    )
    arithmetic = tuple(
        replace(
            item,
            ir=replace(
                item.ir,
                model_basis_receipt_sha256=_sha(arithmetic_basis),
            ),
        )
        for item in arithmetic
    )

    result = run_semantic_program_multifamily_campaign_from_examples(
        {"arithmetic": arithmetic, "sequence": sequence},
        manifests={
            "arithmetic": _manifest(arithmetic_basis, manifest_hash="1" * 64),
            "sequence": _manifest(sequence_basis, manifest_hash="2" * 64),
        },
    )

    assert result.report["schema"] == SEMANTIC_PROGRAM_MULTIFAMILY_CAMPAIGN_SCHEMA
    assert result.report["shared_model_count"] == 1
    assert result.report["family_router_present"] is False
    assert set(result.report["families"]) == {"arithmetic", "sequence"}
    assert all(
        report["held_out_total"] > 0
        for report in result.report["families"].values()
    )
    assert result.report["shared_coefficient_sha256"] == (
        result.model.training_receipt["coefficient_sha256"]
    )
