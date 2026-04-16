import logging

import pytest

from core.cybernetics.ice_layer import ICELayer


def test_classify_anomaly_exposes_legacy_and_canonical_description_keys():
    ice = ICELayer()

    anomaly = ice.classify_anomaly("identity drift detected")

    assert anomaly["type"] == "SEMANTIC_DRIFT"
    assert anomaly["desc"] == "Loss of identity coherence."
    assert anomaly["description"] == anomaly["desc"]
    assert anomaly["containment"] == "RELOAD_CORE_NARRATIVE"


@pytest.mark.asyncio
async def test_executive_violation_uses_compatible_description_schema(caplog):
    ice = ICELayer()

    with caplog.at_level(logging.WARNING, logger="Aura.Cybernetics.ICE"):
        await ice._on_executive_violation({"label": "identity drift"})

    assert ice.get_status()["threat_level"] == pytest.approx(0.25)
    assert any(
        "Loss of identity coherence." in record.message
        for record in caplog.records
    )
