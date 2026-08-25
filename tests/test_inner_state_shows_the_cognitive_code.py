"""The instrument has to be visible, or nobody can use it.

A readout taken before generation is only worth something if it can be read.
This is the surface that shows z_Aura, the code it reads as, and what the
pathway has actually done — and it carries the one flag that matters, because
a symbolic reading of substrate state must never be mistaken for a reply.
"""

from __future__ import annotations

import json

import pytest

from interface.routes.inner_state import get_inner_state


@pytest.mark.asyncio
async def test_the_surface_carries_the_state_and_its_code():
    payload = json.loads((await get_inner_state()).body)
    block = payload["endogenous_language"]
    assert "error" not in block, block
    assert block["layout"]
    assert isinstance(block["state_coverage"], float)
    assert block["cognitive_code"].strip()
    assert block["code_lines"]


@pytest.mark.asyncio
async def test_the_code_is_never_offered_as_a_reply():
    block = json.loads((await get_inner_state()).body)["endogenous_language"]
    assert block["is_user_presentable"] is False


@pytest.mark.asyncio
async def test_absent_channels_are_named_rather_than_missing():
    block = json.loads((await get_inner_state()).body)["endogenous_language"]
    for line in block["code_lines"]:
        assert line["provenance"] in {"state", "organ", "head", "abstained"}
    # Outside a live runtime most organs answer nothing, and that has to read
    # as absence rather than as a state of zero.
    assert set(block["abstained_fields"]) <= {line["field"] for line in block["code_lines"]}


@pytest.mark.asyncio
async def test_the_surface_says_what_cannot_be_carried():
    block = json.loads((await get_inner_state()).body)["endogenous_language"]
    assert "ASSERTIONS" in block["unrepresentable_fields"]
    assert all(reason.strip() for reason in block["unrepresentable_fields"].values())


@pytest.mark.asyncio
async def test_the_pathway_counters_are_present_even_at_zero():
    block = json.loads((await get_inner_state()).body)["endogenous_language"]
    pathway = block["pathway"]
    for key in ("generations_seen", "bias_applied", "unexpected_refusals"):
        assert key in pathway
