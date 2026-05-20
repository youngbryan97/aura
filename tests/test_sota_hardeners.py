import pytest
import time
import json
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

# Import the code to test
from core.brain.llm.local_agent_client import LocalAgentClient
from core.capability_engine import SkillMetadata, _coerce_and_harmonize_params, _get_base_types
from research.protocols.resource_quotas import ComputeGovernor, QuotaExceededError


# Subclass to satisfy abstract base class methods in LLMProvider
class MockLocalAgentClient(LocalAgentClient):
    def generate_stream(self, *args, **kwargs):
        pass


# 1. TEST REACT PARSER & JSON RECONSTRUCTION
def test_parse_tool_call_robustness():
    # Use the test subclass that implements generate_stream
    client = MockLocalAgentClient()

    # Case A: Truncated JSON with missing closing brackets/braces
    raw_truncated = '{"tool": "web_search", "args": {"query": "Aura AGI"'
    parsed = client._parse_tool_call(raw_truncated)
    assert parsed is not None
    assert parsed.get("tool") == "web_search"
    assert parsed.get("args", {}).get("query") == "Aura AGI"

    # Case B: JSON wrapped in markdown fences and markdown preambles
    raw_markdown = """
Here is the tool call you requested:
```json
{
  "tool": "file_operation",
  "args": {
    "path": "test.txt",
    "content": "Aura is online."
  }
}
```
Let me know if you need anything else!
"""
    parsed = client._parse_tool_call(raw_markdown)
    assert parsed is not None
    assert parsed.get("tool") == "file_operation"
    assert parsed.get("args", {}).get("path") == "test.txt"

    # Case C: Hallucinated double nested keys (flattening check)
    raw_nested = {
        "tool": "web_search",
        "params": {
            "params": {
                "query": "Aura flagship project"
            }
        }
    }
    parsed = client._parse_tool_call(json.dumps(raw_nested))
    assert parsed is not None
    assert parsed.get("tool") == "web_search"
    # Unnesting should flatten the nested params to single level args
    assert parsed.get("args", {}).get("query") == "Aura flagship project"

    # Case D: Single quotes normalization and trailing commas
    raw_single_quotes = "{'tool': 'speak', 'args': {'text': 'Hello world',},}"
    parsed = client._parse_tool_call(raw_single_quotes)
    assert parsed is not None
    assert parsed.get("tool") == "speak"
    assert parsed.get("args", {}).get("text") == "Hello world"


# 2. TEST PARAMETER COERCION & SCHEMA HARMONIZATION
class HardenedTestSchema(BaseModel):
    name: str
    limit: int
    temperature: float
    debug: bool
    tags: Optional[List[str]] = Field(default_factory=lambda: ["aura", "cortex"])
    options: Optional[Dict[str, int]] = None
    default_val: int = 100


def test_coerce_and_harmonize_params():
    # Case A: Basic type coercion (string to int, float, bool)
    input_params = {
        "name": "Sovereign_Cortex",
        "limit": "42",
        "temperature": "0.7",
        "debug": "yes"
    }
    healed = _coerce_and_harmonize_params(input_params, HardenedTestSchema)
    
    assert healed["name"] == "Sovereign_Cortex"
    assert healed["limit"] == 42
    assert healed["temperature"] == 0.7
    assert healed["debug"] is True
    # Verify default values were successfully injected
    assert healed["default_val"] == 100
    assert healed["tags"] == ["aura", "cortex"]

    # Case B: Boolean coercion variants ("false", "no", "0")
    input_params_bools = {
        "name": "TestBools",
        "limit": 10,
        "temperature": 0.5,
        "debug": "0"
    }
    healed_bools = _coerce_and_harmonize_params(input_params_bools, HardenedTestSchema)
    assert healed_bools["debug"] is False

    # Case C: Comma separated string list parsing for generic List[str] type
    input_params_list = {
        "name": "TestList",
        "limit": 5,
        "temperature": 0.1,
        "debug": "true",
        "tags": "first, second, third"
    }
    healed_list = _coerce_and_harmonize_params(input_params_list, HardenedTestSchema)
    assert healed_list["tags"] == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_extract_and_validate_args_resilience():
    # Build metadata
    meta = SkillMetadata(
        name="test_resilient_skill",
        description="A skill to verify input validations.",
        input_model=HardenedTestSchema
    )

    # Valid payload but with type mismatches
    raw_payload = '{"name": "Aura", "limit": "500", "temperature": "0.95", "debug": "true"}'
    validated = await meta.extract_and_validate_args(raw_payload, None)
    
    assert validated is not None
    assert validated.get("name") == "Aura"
    assert validated.get("limit") == 500
    assert validated.get("temperature") == 0.95
    assert validated.get("debug") is True
    assert validated.get("default_val") == 100

    # Highly invalid/corrupted payload (satisfies non-destructive fallback)
    raw_corrupted = '{"name": "Aura", "limit": "not_an_int", "temperature": "0.1", "debug": "yes"}'
    validated_fallback = await meta.extract_and_validate_args(raw_corrupted, None)
    
    # Since limit is not_an_int and cannot be coerced, pydantic validation will fail.
    # The non-destructive fallback should cleanly return a fallback dict containing matches with an _error marker.
    assert "_error" in validated_fallback
    assert validated_fallback.get("name") == "Aura"
    # Note: temperature is coerced to float 0.1 successfully
    assert validated_fallback.get("temperature") == 0.1


# 3. TEST GOVERNANCE QUOTAS
def test_compute_governor_quotas():
    gov = ComputeGovernor()
    
    # Force state to lower tier limits
    gov.state.update({
        "current_tier": "TIER_1_LOCAL_ONLY",
        "max_tokens_per_hour": 1000,
        "max_concurrent_sims": 2,
        "internet_access": False
    })
    gov.token_usage_hourly = 0
    gov.last_reset_time = time.time()

    # Case A: Consuming within quota boundaries
    gov.enforce_quota("tokens", 400)
    assert gov.token_usage_hourly == 400
    
    gov.enforce_quota("tokens", 500)
    assert gov.token_usage_hourly == 900

    # Case B: Exceeding token hourly quota
    with pytest.raises(QuotaExceededError) as exc_info:
        gov.enforce_quota("tokens", 200)
    assert "Quota exceeded" in str(exc_info.value)
    assert "Token limit" in str(exc_info.value)

    # Case C: Active simulations count within boundary
    gov.enforce_quota("simulations", 2)

    # Case D: Active simulations exceeding boundary
    with pytest.raises(QuotaExceededError) as exc_info_sim:
        gov.enforce_quota("simulations", 3)
    assert "concurrent simulations" in str(exc_info_sim.value)


# 4. TEST STATEFUL SIMULATION & THROTTLE METRICS
def test_compute_governor_stateful_simulations():
    from research.protocols.resource_quotas import get_compute_governor
    gov = get_compute_governor()
    
    # Reset/Mock state
    gov.state.update({
        "max_tokens_per_hour": 100000,
        "max_concurrent_sims": 2
    })
    gov.token_usage_hourly = 0
    gov.active_simulations.clear()

    # Verify initial throttle factor is 1.0
    assert gov.get_throttle_factor() == 1.0

    # Start simulation within quota
    gov.start_simulation("sim_1")
    assert "sim_1" in gov.active_simulations
    
    gov.start_simulation("sim_2")
    assert "sim_2" in gov.active_simulations

    # Exceed simulations quota statefully
    with pytest.raises(QuotaExceededError):
        gov.start_simulation("sim_3")

    # Complete simulation
    gov.end_simulation("sim_1")
    assert "sim_1" not in gov.active_simulations

    # Start another simulation now that slot is open
    gov.start_simulation("sim_3")
    assert "sim_3" in gov.active_simulations

    # Test dynamic throttle factors
    gov.token_usage_hourly = 85000  # 85% usage
    assert gov.get_throttle_factor() == 0.5

    gov.token_usage_hourly = 96000  # 96% usage
    assert gov.get_throttle_factor() == 0.2

    gov.token_usage_hourly = 100000 # 100% usage
    assert gov.get_throttle_factor() == 0.0

    # Cleanup
    gov.active_simulations.clear()
    gov.token_usage_hourly = 0


# 5. TEST SOMATIC THROTTLE INTEGRATION WITH GOVERNOR
def test_somatic_throttle_governance():
    from core.brain.llm.somatic_throttle import SomaticComputeSentinel
    from research.protocols.resource_quotas import get_compute_governor
    
    sentinel = SomaticComputeSentinel()
    gov = get_compute_governor()

    # Case A: Nominal state (no throttle)
    gov.state["max_tokens_per_hour"] = 100000
    gov.token_usage_hourly = 0
    
    opts = {"max_tokens": 512, "temperature": 0.7}
    adjusted = sentinel.adjust_generation_options(opts.copy())
    assert adjusted["max_tokens"] == 512
    assert adjusted["temperature"] == 0.7

    # Case B: Stressed state (85% tokens consumed)
    gov.token_usage_hourly = 85000
    adjusted_stressed = sentinel.adjust_generation_options(opts.copy())
    assert adjusted_stressed["max_tokens"] <= 256
    assert adjusted_stressed["temperature"] == 0.3

    # Case C: Critical state (96% tokens consumed)
    gov.token_usage_hourly = 96000
    adjusted_critical = sentinel.adjust_generation_options(opts.copy())
    assert adjusted_critical["max_tokens"] <= 128
    assert adjusted_critical["temperature"] == 0.15

    # Case D: Exhausted state (100% tokens consumed)
    gov.token_usage_hourly = 100000
    adjusted_exhausted = sentinel.adjust_generation_options(opts.copy())
    assert adjusted_exhausted["max_tokens"] <= 8
    assert adjusted_exhausted["temperature"] == 0.05

    # Cleanup
    gov.token_usage_hourly = 0


# 6. TEST CAPABILITY ENGINE EXECUTE PYDANTIC RECOVERY
@pytest.mark.asyncio
async def test_capability_engine_execute_pydantic_recovery():
    from core.capability_engine import CapabilityEngine, SkillMetadata
    from pydantic import BaseModel
    import logging

    # Create a dummy skill metadata with a Pydantic model
    class DummySchema(BaseModel):
        required_field: str
        optional_field: int = 42

    class DummySkill:
        name = "dummy_skill"
        inputs = {"required_field": "A required test field"}
        
        async def execute(self, required_field: str, optional_field: int = 42):
            return {"ok": True, "required": required_field, "optional": optional_field}

    engine = CapabilityEngine()
    engine.logger = logging.getLogger("Test")
    
    meta = SkillMetadata(
        name="dummy_skill",
        description="A dummy test skill.",
        input_model=DummySchema,
        skill_class=DummySkill,
        enabled=True
    )
    engine.skills["dummy_skill"] = meta
    engine.instances["dummy_skill"] = DummySkill()

    # Case A: Execute with type mismatches that can be coerced
    result = await engine.execute("dummy_skill", {"required_field": 12345, "optional_field": "99"})
    assert result.get("ok") is True
    assert result.get("required") == "12345"  # coerced to str
    assert result.get("optional") == 99        # coerced to int

    # Case B: Execute with missing required field (should trigger self-healing recovery loop)
    # The recovery loop will try minimal subset, and if it completely fails, fall back gracefully
    # with an _error marker without crashing the executor loop.
    result_fail = await engine.execute("dummy_skill", {"optional_field": "100"})
    assert result_fail.get("ok") is False
    assert "dummy" in result_fail.get("error", "").lower() or "validation" in result_fail.get("error", "").lower()

