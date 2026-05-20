import asyncio
import json

from core.context.context_compression import (
    CompressionLevel,
    ContextCompressionService,
)


def test_context_compression_quarantines_corrupt_state(tmp_path):
    state_file = tmp_path / "compression_state.json"
    state_file.write_text("{bad json", encoding="utf-8")

    service = ContextCompressionService(state_file=state_file)

    assert not state_file.exists()
    assert list(tmp_path.glob("compression_state.corrupt.*.json"))
    assert service._state.current_turn == 0


def test_route_failure_resets_stale_exclusion_to_full(tmp_path):
    service = ContextCompressionService(state_file=tmp_path / "state.json")
    service.register_file_access("core/a.py", "print('old')\n")
    service.advance_turn()
    service.advance_turn()
    service.advance_turn()
    service._state.files["core/a.py"].compression_level = CompressionLevel.EXCLUDED

    class BadBrain:
        async def generate(self, *_args, **_kwargs):
            return {"response": "not json"}

    result = asyncio.run(
        service.route_files(
            {"core/a.py": "print('old')\n"},
            "work on core/a.py",
            brain=BadBrain(),
        )
    )

    assert result["core/a.py"] is CompressionLevel.FULL
    assert service._state.files["core/a.py"].compression_level is CompressionLevel.FULL


def test_route_without_brain_resets_stale_exclusion_to_full(tmp_path):
    service = ContextCompressionService(state_file=tmp_path / "state.json")
    service.register_file_access("core/a.py", "print('old')\n")
    service.advance_turn()
    service.advance_turn()
    service.advance_turn()
    service._state.files["core/a.py"].compression_level = CompressionLevel.EXCLUDED

    result = asyncio.run(
        service.route_files(
            {"core/a.py": "print('old')\n"},
            "work on core/a.py",
            brain=None,
        )
    )

    assert result["core/a.py"] is CompressionLevel.FULL
    assert service._state.files["core/a.py"].compression_level is CompressionLevel.FULL


def test_route_supports_sync_brain_and_string_levels(tmp_path):
    service = ContextCompressionService(state_file=tmp_path / "state.json")
    service.register_file_access("core/a.py", "a = 1\n")
    service.advance_turn()
    service.advance_turn()
    service.advance_turn()

    class SyncBrain:
        def generate(self, *_args, **_kwargs):
            return {
                "response": json.dumps(
                    {"files": [{"path": "core/a.py", "level": "SUMMARY"}]}
                )
            }

    result = asyncio.run(
        service.route_files(
            {"core/a.py": "a = 1\n"},
            "summarize background files",
            brain=SyncBrain(),
        )
    )

    assert result["core/a.py"] is CompressionLevel.SUMMARY


def test_route_extracts_first_valid_json_object(tmp_path):
    service = ContextCompressionService(state_file=tmp_path / "state.json")
    service.register_file_access("core/a.py", "a = 1\n")
    service.advance_turn()
    service.advance_turn()
    service.advance_turn()

    class ChattyBrain:
        def generate(self, *_args, **_kwargs):
            return {
                "response": (
                    "Here is the routing decision:\n"
                    "{\"files\": [{\"path\": \"core/a.py\", \"level\": \"PARTIAL\"}]}\n"
                    "Done."
                )
            }

    result = asyncio.run(
        service.route_files(
            {"core/a.py": "a = 1\n"},
            "inspect the file",
            brain=ChattyBrain(),
        )
    )

    assert result["core/a.py"] is CompressionLevel.PARTIAL


def test_generate_summary_uses_deterministic_fallback_and_caches(tmp_path):
    service = ContextCompressionService(state_file=tmp_path / "state.json")

    class EmptyBrain:
        async def generate(self, *_args, **_kwargs):
            return {"response": ""}

    summary = asyncio.run(
        service.generate_summary(
            "core/example.py",
            "def run():\n    return 42\n",
            brain=EmptyBrain(),
        )
    )

    assert "core/example.py" in summary
    assert "2 lines" in summary
    assert service.apply_compression(
        "core/example.py",
        "def run():\n    return 42\n",
        CompressionLevel.SUMMARY,
    ).startswith("[Summary of core/example.py]")


def test_stale_summary_is_invalidated_on_content_change(tmp_path):
    service = ContextCompressionService(state_file=tmp_path / "state.json")
    service.register_file_access("core/example.py", "old\n")
    service._state.files["core/example.py"].summary = "old summary"

    compressed = service.apply_compression(
        "core/example.py",
        "new\n",
        CompressionLevel.SUMMARY,
    )

    assert "old summary" not in compressed
    assert service._state.files["core/example.py"].summary == ""
