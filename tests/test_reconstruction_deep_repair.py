import pytest

from core.brain.llm.code_generator import LLMCodeGenerator, extract_python_code
from core.container import ServiceContainer
from core.runtime.self_healing import SelfHealing
from core.self_improvement.interface_contract import LabResult, PromotionVerdict
from core.self_modification.code_repair import AutonomousCodeRepair


def test_extract_python_code_prefers_fenced_source():
    text = "Here is the module:\n```python\n\ndef answer():\n    return 42\n```\nDone."
    assert extract_python_code(text) == "def answer():\n    return 42"


@pytest.mark.asyncio
async def test_llm_code_generator_uses_router_and_validates_python():
    class Router:
        def __init__(self):
            self.kwargs = {}

        async def think(self, prompt, **kwargs):
            self.kwargs = kwargs
            assert "Target module: core/example.py" in prompt
            return "```python\ndef run():\n    return 'ok'\n```"

    router = Router()
    generator = LLMCodeGenerator(router=router)

    code = await generator.generate_async(
        "# Module Reimplementation Task",
        {"module_path": "core/example.py", "attempt": 2, "stub_code": "def run(): ..."},
    )

    assert "def run" in code
    assert router.kwargs["origin"] == "reimplementation_lab"
    assert router.kwargs["is_background"] is True
    assert router.kwargs["system_prompt"]


@pytest.mark.asyncio
async def test_self_healing_request_deep_repair_uses_registered_lab(tmp_path):
    class Lab:
        async def run_reconstruction(self, module_path, max_attempts=None, metadata=None):
            assert module_path == "core/example.py"
            assert max_attempts == 1
            assert metadata["reason"] == "patch_repair_failed"
            return LabResult(
                success=False,
                module_path=module_path,
                verdict=PromotionVerdict.REJECT,
                attempts=1,
            )

    ServiceContainer.clear()
    ServiceContainer.register_instance("reimplementation_lab", Lab(), required=False)
    healer = SelfHealing()

    record = await healer.request_deep_repair(
        "core/example.py",
        reason="patch_repair_failed",
        metadata={"stage": "validation"},
        max_attempts=1,
    )

    assert record["result"] == "deep_repair_rejected"
    assert record["lab_result"]["module_path"] == "core/example.py"


@pytest.mark.asyncio
async def test_code_repair_fallback_calls_self_healing_deep_repair():
    class Lab:
        async def run_reconstruction(self, module_path, max_attempts=None, metadata=None):
            assert metadata["trigger"] == "patch_repair_failed"
            assert metadata["stage"] == "fix_generation"
            return LabResult(
                success=True,
                module_path=module_path,
                verdict=PromotionVerdict.PROMOTE,
                attempts=1,
            )

    ServiceContainer.clear()
    ServiceContainer.register_instance("reimplementation_lab", Lab(), required=False)
    repair = object.__new__(AutonomousCodeRepair)

    record = await repair._deep_repair_after_patch_failure(
        "core/example.py",
        10,
        {"summary": "patch failed"},
        stage="fix_generation",
    )

    assert record["result"] == "deep_repair_succeeded"
    assert record["lab_result"]["success"] is True
