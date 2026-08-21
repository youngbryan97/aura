from __future__ import annotations

import logging

from core.brain.latent_cortex_service import (
    _report_qualified_recurrent_serving_status,
)


def test_active_qualified_package_is_named_at_boot(caplog) -> None:
    caplog.set_level(logging.INFO, logger="Aura.LatentCortexService")

    _report_qualified_recurrent_serving_status(
        {
            "active": True,
            "reason": "semantic_neural_serving_active",
            "receipt": {
                "package_id": "cp568-resident-semantic-neural-active-r1",
                "mode": "qualified_exact_semantic_v1",
                "promotion_mode": "active",
                "allowed_families": ["one", "two", "three", "four"],
                "activation_sha256": "a" * 64,
            },
        }
    )

    message = caplog.messages[-1]
    assert "Qualified semantic-neural serving ACTIVE" in message
    assert "cp568-resident-semantic-neural-active-r1" in message
    assert "families=4" in message


def test_explicit_kill_switch_is_not_reported_as_package_failure(caplog) -> None:
    caplog.set_level(logging.INFO, logger="Aura.LatentCortexService")

    _report_qualified_recurrent_serving_status(
        {
            "active": False,
            "reason": "semantic_neural_serving_disabled",
        }
    )

    assert caplog.messages[-1] == (
        "Qualified semantic-neural serving disabled by explicit kill switch"
    )
