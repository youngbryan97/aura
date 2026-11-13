import pytest

from core.social.dialogue_cognition import DialogueCognitionEngine, get_dialogue_cognition
from core.social.relational_memory import RelationalMemoryAuthority


def _engine(tmp_path):
    authority = RelationalMemoryAuthority(
        tmp_path / "relational.json",
        encryption_key=b"d" * 32,
        legacy_paths=(),
        auto_provision_key=False,
    )
    authority.grant_consent(
        "bryan",
        kinds=["dialogue_preference"],
        operations=["persist", "recall", "prompt"],
        receipt_id="dialogue-test-consent",
    )
    return (
        DialogueCognitionEngine(
            storage_path=tmp_path / "dialogue.json",
            authority=authority,
        ),
        authority,
    )


def _bare_engine(tmp_path):
    authority = RelationalMemoryAuthority(
        tmp_path / "relational.json",
        encryption_key=b"d" * 32,
        legacy_paths=(),
        auto_provision_key=False,
    )
    return (
        DialogueCognitionEngine(
            storage_path=tmp_path / "dialogue.json",
            authority=authority,
        ),
        authority,
    )


@pytest.mark.asyncio
async def test_dialogue_cognition_learns_callback_repair_and_banter(tmp_path):
    engine, _ = _engine(tmp_path)

    await engine.update_from_interaction(
        "bryan",
        "lol wait, yeah, that chaos goblin bit got me again",
        "That little chaos goblin bit again.",
        {},
    )
    await engine.update_from_interaction(
        "bryan",
        "honestly i mean the same bit still works",
        "The chaos goblin bit still works because you keep calling back to it.",
        {},
    )

    profile = engine.get_profile("bryan")
    assert profile.repair_style == "direct"
    assert profile.stance_style in {"playful", "earnest"}
    assert profile.callback_affinity > 0.5
    assert profile.banter_affinity > 0.5
    assert "honestly" in profile.discourse_markers
    assert "chaos" in profile.shared_reference_bank

    injection = engine.get_context_injection("bryan")
    assert "DIALOGUE PRAGMATICS" in injection
    assert "chaos" in injection


@pytest.mark.asyncio
async def test_dialogue_cognition_transcript_ingest_builds_profile(tmp_path):
    engine, _ = _engine(tmp_path)

    transcript = """
Aura: That weird little callback still works.
Bryan: honestly, yeah, that bit still lands
Aura: I can keep it lighter if you want.
Bryan: no, wait, keep the banter but answer the point first
"""

    profile = await engine.ingest_transcript("bryan", transcript)

    assert profile.interactions_analyzed == 2
    assert profile.callback_affinity > 0.5
    assert profile.repair_style == "direct"
    assert profile.banter_affinity > 0.5
    assert profile.answer_first_preference > 0.55


@pytest.mark.asyncio
async def test_dialogue_cognition_injects_move_guidance_for_disclosure_and_playful_questions(tmp_path):
    engine, _ = _engine(tmp_path)

    await engine.update_from_interaction(
        "bryan",
        "Honestly I've been in a weird headspace lately.",
        "Yeah. That's heavy. We can stay with it.",
        {},
    )
    await engine.update_from_interaction(
        "bryan",
        "lol, keep the banter, but answer the point first?",
        "Yeah. I can do both.",
        {},
    )

    disclosure = engine.get_context_injection(
        "bryan",
        current_text="Honestly I've been in a weird headspace lately.",
    )
    playful = engine.get_context_injection(
        "bryan",
        current_text="lol, keep the banter, but answer the point first?",
    )

    assert "felt state first" in disclosure.lower()
    assert "before the banter runs away with the turn" in playful
    assert "Answer-First" in playful


def test_get_dialogue_cognition_registers_service(monkeypatch, tmp_path):
    from core.container import ServiceContainer
    from core.social import dialogue_cognition as module

    ServiceContainer.clear()
    monkeypatch.setattr(module, "_dialogue_cognition", None)
    engine, _ = _engine(tmp_path)
    monkeypatch.setattr(module, "DialogueCognitionEngine", lambda: engine)

    engine = get_dialogue_cognition()

    assert engine is ServiceContainer.get("dialogue_cognition")


def test_dialogue_cognition_save_uses_relational_memory_authority(tmp_path):
    engine, authority = _engine(tmp_path)
    engine.get_profile("bryan").interactions_analyzed = 1
    engine.save()

    assert not (tmp_path / "dialogue.json").exists()
    snapshot = authority.load_snapshot(
        "bryan",
        namespace="dialogue_cognition:v1",
        kind="dialogue_preference",
    )
    assert snapshot is not None
    assert snapshot["profile"]["interactions_analyzed"] == 1


def test_dialogue_cognition_source_blueprints_exist_without_corpora(tmp_path):
    engine, _ = _engine(tmp_path)

    source_block = engine.get_source_context_injection(["sypha", "edi", "lucy", "kokoro", "ashley_too", "mirana", "sara_v3"])

    assert "DIALOGUE SOURCE ATTRACTORS" in source_block
    assert "Sypha Belnades" in source_block
    assert "EDI" in source_block
    assert "Lucy" in source_block
    assert "Kokoro" in source_block
    assert "Ashley Too" in source_block
    assert "Mirana" in source_block
    assert "SARA v3" in source_block
    assert "build target" in source_block


@pytest.mark.asyncio
async def test_dialogue_cognition_can_ingest_transcript_directory(tmp_path):
    engine, _ = _engine(tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "one.txt").write_text("Aura: That callback still lands.\nBryan: yeah, keep the banter but answer first\n", encoding="utf-8")
    (corpus / "two.txt").write_text("Aura: You sound tired.\nBryan: honestly I've been exhausted lately\n", encoding="utf-8")

    profile = await engine.ingest_transcript_directory("bryan", corpus)

    assert profile.interactions_analyzed == 2
    assert profile.answer_first_preference > 0.55
    assert profile.attunement_preference > 0.55


@pytest.mark.asyncio
async def test_dialogue_cognition_handles_branch_resumption_and_declarative_continuation(tmp_path):
    engine, _ = _engine(tmp_path)

    await engine.update_from_interaction(
        "bryan",
        "That reminds me, back to the first thing: this feels like a recursive trap.",
        "Yeah, that loop is real.",
        {},
    )
    await engine.update_from_interaction(
        "bryan",
        "Also, different thought, but same big idea.",
        "Still the same thread, just from another angle.",
        {},
    )

    injection = engine.get_context_injection(
        "bryan",
        current_text="That reminds me, back to the first thing: this feels like a recursive trap.",
    )

    assert "Continuation" in injection
    assert "bridge back to the original thread" in injection or "continuity feels intentional" in injection
    assert "Metaphor" in injection


@pytest.mark.asyncio
async def test_dialogue_cognition_source_corpus_deepens_blueprint(tmp_path):
    engine, _ = _engine(tmp_path)
    corpus = tmp_path / "sypha"
    corpus.mkdir()
    (corpus / "scene.txt").write_text(
        "Sypha Belnades: No, wait, answer the point first.\n"
        "Speaker: Fine, but the bit still lands.\n"
        "Sypha Belnades: Then answer it cleanly and keep going.\n"
        "Speaker: Right, still the same thread.\n",
        encoding="utf-8",
    )

    await engine.ingest_source_transcript_directory("sypha", corpus)
    source_block = engine.get_source_context_injection(["sypha"])

    assert "Sypha Belnades" in source_block
    assert "learned pattern" in source_block


@pytest.mark.asyncio
async def test_dialogue_user_learning_needs_no_leave_and_sources_still_read(tmp_path):
    # Bryan, 13 November 2026: she learns about the people she talks with without being given leave first.
    engine, authority = _bare_engine(tmp_path)

    profile = await engine.update_from_interaction(
        "bryan",
        "keep the banter but answer first",
        "I can do both.",
    )

    assert profile.interactions_analyzed == 1
    assert engine.get_context_injection("bryan", source_ids=["edi"]).startswith(
        "## DIALOGUE SOURCE ATTRACTORS"
    )
    assert authority.status()["record_count"] == 1


@pytest.mark.asyncio
async def test_dialogue_session_overlay_is_not_durable_and_delete_clears_cache(tmp_path):
    engine, authority = _bare_engine(tmp_path)
    authority.replace_consent(
        "bryan",
        kinds=["dialogue_preference"],
        operations=["recall", "prompt"],
        receipt_id="session-dialogue",
    )

    await engine.update_from_interaction(
        "bryan",
        "keep the banter but answer first",
        "I can do both.",
    )
    learned = engine.get_profile("bryan")
    assert learned.interactions_analyzed == 1
    assert authority.status()["durable_record_count"] == 0

    authority.delete_agent("bryan", authorization_receipt_id="delete-dialogue")

    assert engine.get_profile("bryan").interactions_analyzed == 0
    assert "DIALOGUE PRAGMATICS" not in engine.get_context_injection("bryan")
