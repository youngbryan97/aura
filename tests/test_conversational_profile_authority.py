from __future__ import annotations

import pytest

from core.social.conversational_profile import ConversationalProfiler
from core.social.relational_memory import RelationalMemoryAuthority


def _authority(tmp_path) -> RelationalMemoryAuthority:
    return RelationalMemoryAuthority(
        tmp_path / "relational.json",
        encryption_key=b"p" * 32,
        legacy_paths=(),
        auto_provision_key=False,
        now_fn=lambda: 100.0,
    )


@pytest.mark.asyncio
async def test_profiler_learns_a_person_without_being_given_leave(tmp_path):
    # Bryan, 13 November 2026: she learns about the people she talks with without being given leave first.
    authority = _authority(tmp_path)
    profiler = ConversationalProfiler(
        tmp_path / "legacy_profiles.json",
        authority=authority,
    )

    profile = await profiler.update_from_interaction(
        "bryan",
        "why does this architecture work?",
        "Because the state has one owner.",
    )

    assert profile.interactions_analyzed == 1
    assert authority.status()["record_count"] == 1


@pytest.mark.asyncio
async def test_session_profile_uses_authority_but_does_not_survive_restart(tmp_path):
    authority = _authority(tmp_path)
    authority.replace_consent(
        "bryan",
        kinds=["derived_profile"],
        operations=["recall", "prompt"],
        receipt_id="session-only",
    )
    profiler = ConversationalProfiler(
        tmp_path / "legacy_profiles.json",
        authority=authority,
    )

    profile = await profiler.update_from_interaction(
        "bryan",
        "why and how does this work?",
        "Here is the mechanism.",
    )

    assert profile.interactions_analyzed == 1
    assert authority.status()["record_count"] == 1
    assert not (tmp_path / "legacy_profiles.json").exists()
    restored = _authority(tmp_path)
    assert restored.status()["record_count"] == 0


@pytest.mark.asyncio
async def test_durable_profile_round_trips_and_prompt_use_is_exact_agent(tmp_path):
    authority = _authority(tmp_path)
    authority.grant_consent(
        "bryan",
        kinds=["derived_profile"],
        operations=["persist", "recall", "prompt"],
        receipt_id="remember-profile",
    )
    profiler = ConversationalProfiler(
        tmp_path / "legacy_profiles.json",
        authority=authority,
    )
    for _ in range(3):
        await profiler.update_from_interaction(
            "bryan",
            "why does this detailed architecture work?",
            "Here is the detailed mechanism.",
        )

    restored_authority = _authority(tmp_path)
    restored = ConversationalProfiler(
        tmp_path / "legacy_profiles.json",
        authority=restored_authority,
    )

    assert restored.get_profile("bryan").interactions_analyzed == 3
    assert "Communication DNA" in restored.get_context_injection("bryan")
    assert restored.get_context_injection("alice") == ""
    assert restored_authority.prompt_block("bryan") == ""


@pytest.mark.asyncio
async def test_delete_clears_adapter_view_without_stale_profile_cache(tmp_path):
    authority = _authority(tmp_path)
    authority.grant_consent(
        "bryan",
        kinds=["derived_profile"],
        operations=["persist", "recall", "prompt"],
        receipt_id="remember-profile",
    )
    profiler = ConversationalProfiler(
        tmp_path / "legacy_profiles.json",
        authority=authority,
    )
    await profiler.update_from_interaction(
        "bryan",
        "why does this work?",
        "Here is why.",
    )

    authority.delete_agent("bryan", authorization_receipt_id="delete-profile")

    assert profiler.get_profile("bryan").interactions_analyzed == 0
    assert profiler.get_context_injection("bryan") == ""
