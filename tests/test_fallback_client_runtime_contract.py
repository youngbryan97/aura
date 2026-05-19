from pathlib import Path

import pytest


def test_fallback_client_degradation_audit_is_clean():
    from tools.audit_degradation import analyze_file

    assert analyze_file(Path("core/brain/llm/fallback_client.py")) == []


class UnhealthyProvider:
    name = "cold_lane"

    def check_health(self):
        return False


class BrokenTextProvider:
    name = "broken_text_lane"

    def check_health(self):
        return True

    def generate_text(self, prompt, system_prompt=None, model=None):
        reason = f"{prompt}:{model}:text failure"
        raise RuntimeError(reason)


class WorkingTextProvider:
    name = "working_text_lane"

    def check_health(self):
        return True

    def generate_text(self, prompt, system_prompt=None, model=None):
        return f"recovered:{prompt}"


def test_fallback_text_records_failed_lane_and_uses_next_provider():
    from core.brain.llm.fallback_client import FallbackLLMClient

    client = FallbackLLMClient([UnhealthyProvider(), BrokenTextProvider(), WorkingTextProvider()])

    text = client.generate_text("hello", model="fast")

    attempts = client.get_status()["last_attempts"]
    assert text == "recovered:hello"
    assert [attempt["status"] for attempt in attempts] == ["skipped", "failed", "succeeded"]
    assert attempts[1]["provider"] == "broken_text_lane"
    assert client.get_status()["provider_failures"]["broken_text_lane"] == 1
    assert client.get_status()["provider_successes"]["working_text_lane"] == 1


class InvalidJsonProvider:
    name = "invalid_json_lane"

    def check_health(self):
        return True

    def generate_json(self, prompt, schema, system_prompt=None, model=None):
        return ["not", "a", "dict"]


class WorkingJsonProvider:
    name = "working_json_lane"

    def check_health(self):
        return True

    def generate_json(self, prompt, schema, system_prompt=None, model=None):
        return {"ok": True, "prompt": prompt, "schema_keys": sorted(schema)}


def test_fallback_json_rejects_invalid_payload_before_next_lane():
    from core.brain.llm.fallback_client import FallbackLLMClient

    client = FallbackLLMClient([InvalidJsonProvider(), WorkingJsonProvider()])

    payload = client.generate_json("shape", {"ok": bool})

    attempts = client.get_status()["last_attempts"]
    assert payload == {"ok": True, "prompt": "shape", "schema_keys": ["ok"]}
    assert [attempt["status"] for attempt in attempts] == ["failed", "succeeded"]
    assert "expected dict" in attempts[0]["error"]


class BrokenStreamProvider:
    name = "broken_stream_lane"

    def check_health(self):
        return True

    async def generate_stream(self, prompt, system_prompt=None, model=None, **kwargs):
        yield "partial-that-must-not-escape"
        reason = f"{prompt}:{model}:stream failure"
        raise RuntimeError(reason)


class WorkingStreamProvider:
    name = "working_stream_lane"

    def check_health(self):
        return True

    async def generate_stream(self, prompt, system_prompt=None, model=None, **kwargs):
        yield "clean"
        yield f"stream:{prompt}"


@pytest.mark.asyncio
async def test_fallback_stream_buffers_failed_provider_before_emitting():
    from core.brain.llm.fallback_client import FallbackLLMClient

    client = FallbackLLMClient([BrokenStreamProvider(), WorkingStreamProvider()])

    chunks = [chunk async for chunk in client.generate_stream("hello", model="stream")]

    attempts = client.get_status()["last_attempts"]
    assert chunks == ["clean", "stream:hello"]
    assert [attempt["status"] for attempt in attempts] == ["failed", "succeeded"]
    assert "partial-that-must-not-escape" not in chunks


def test_fallback_exhaustion_raises_model_unavailable_when_no_provider_is_healthy():
    from core.brain.llm.fallback_client import FallbackLLMClient
    from core.runtime.errors import ModelUnavailable

    client = FallbackLLMClient([UnhealthyProvider()])

    with pytest.raises(ModelUnavailable):
        client.generate_text("hello")

    assert client.get_status()["last_attempts"][0]["status"] == "skipped"
