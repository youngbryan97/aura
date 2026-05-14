from core.runtime.errors import record_degradation
import json
import logging
from typing import Type, TypeVar, Optional, Any, Dict, List, get_origin
from pydantic import BaseModel, ValidationError
from core.container import ServiceContainer

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger("Aura.StructuredLLM")

class StructuredLLM:
    """
    Aura's Self-Correction Loop.
    Wraps LLM calls to ensure output matches a Pydantic schema.
    If validation fails, it prompts the LLM to fix its own mistake.
    """
    
    def __init__(self, model_class: Type[T], max_retries: int = 3):
        self.model_class = model_class
        self.max_retries = max_retries
        self._llm_router = ServiceContainer.get("llm_router")
        self.last_defer_reason = ""

    async def generate(self, prompt: str, context: Optional[str] = None) -> Optional[T]:
        """
        Generates structured data from the LLM with autonomous retries on validation failure.
        [HARDENING] Injects Ghost Examples and propagates Pydantic schema to Router.
        """
        schema = self.model_class.model_json_schema()
        ghost_example = self._generate_ghost_example()
        self.last_defer_reason = ""
        
        base_prompt = prompt
        
        # Inject JSON enforcement and Ghost Example into the prompt
        if "GHOST EXAMPLE (Follow this structure exactly):" not in base_prompt:
            base_prompt += (
                f"\n\nCRITICAL: You MUST respond with a valid JSON object matching the requested schema.\n"
                f"GHOST EXAMPLE (Follow this structure exactly):\n{ghost_example}"
            )
        current_prompt = base_prompt

        escalated_tier = None
        for attempt in range(self.max_retries):
            logger.info("🤖 StructuredLLM: Attempt %d/%d for %s", 
                        attempt + 1, self.max_retries, self.model_class.__name__)
            
            try:
                defer_reason = self._background_defer_reason(escalated=bool(escalated_tier))
                if defer_reason:
                    self.last_defer_reason = defer_reason
                    logger.info(
                        "⏸️ StructuredLLM: Deferred %s (%s).",
                        self.model_class.__name__,
                        defer_reason,
                    )
                    return None

                # [STABILITY v54.1] Multi-stage escalation:
                # Attempt 0: TERTIARY (Local fast)
                # Attempt 1 (Failure 1): PRIMARY (Local 32B)
                # Attempt 2 (Failure 2): SECONDARY (Cloud/Deep)
                if escalated_tier:
                    force_tier = escalated_tier
                else:
                    force_tier = "tertiary" if attempt >= 1 else None

                metadata = None
                if hasattr(self._llm_router, "generate_with_metadata"):
                    metadata = await self._llm_router.generate_with_metadata(
                        current_prompt,
                        context=context,
                        prefer_tier=force_tier,
                        schema=schema,
                        origin="structured_llm",
                        is_background=not escalated_tier, # Allow cloud usage if escalated
                    )
                    response_text = str((metadata or {}).get("text") or "")
                else:
                    response_text = await self._llm_router.generate(
                        current_prompt, 
                        context=context, 
                        prefer_tier=force_tier,
                        schema=schema,
                        origin="structured_llm",
                        is_background=not escalated_tier,
                    )

                error_code = str((metadata or {}).get("error") or "")
                deferred_error = error_code == "foreground_busy" or error_code == "foreground_quiet_window"
                deferred_error = deferred_error or error_code.startswith(
                    (
                        "background_deferred:",
                        "failure_lockdown_",
                        "conversation_lane_",
                    )
                )
                if deferred_error:
                    self.last_defer_reason = error_code
                    logger.info(
                        "⏸️ StructuredLLM: Deferred %s (%s).",
                        self.model_class.__name__,
                        error_code,
                    )
                    return None

                if not response_text or "ROUTER_ERROR" in response_text:
                    try:
                        from core.health.degraded_events import record_degraded_event

                        record_degraded_event(
                            "structured_llm",
                            "technical_failure",
                            detail=(error_code or response_text or "empty")[:200],
                            severity="warning",
                            classification="background_degraded",
                            context={"model_class": self.model_class.__name__, "attempt": attempt + 1},
                        )
                    except Exception as _exc:
                        record_degradation('structured_llm', _exc)
                        logger.debug("Suppressed Exception: %s", _exc)
                    # [STABILITY v54.1] Escalation Strategy:
                    # 1. First Technical Failure -> Try PRIMARY (32B Local)
                    # 2. Second Technical Failure -> Try SECONDARY (Cloud)
                    if attempt == 0:
                        logger.info("⚡ StructuredLLM: Technical failure on TERTIARY — escalating to PRIMARY (Local 32B) for next attempt.")
                        escalated_tier = "primary"
                    elif attempt == 1:
                        logger.info("⚡ StructuredLLM: Technical failure on PRIMARY — escalating to SECONDARY (Cloud/Deep) for next attempt.")
                        escalated_tier = "secondary"
                    else:
                        escalated_tier = "secondary"
                    
                    escalated = True
                    force_tier = escalated_tier
                    continue

                # 2. Extract JSON (handle markers like ```json)
                cleaned_text = self._extract_json(response_text)
                
                # 3. Parse and Validate
                data = json.loads(cleaned_text)
                validated_obj = self.model_class(**data)
                
                logger.info("✅ StructuredLLM: Successfully validated %s", self.model_class.__name__)
                return validated_obj

            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning("❌ StructuredLLM: Validation failed on attempt %d: %s", attempt + 1, e)
                try:
                    from core.health.degraded_events import record_degraded_event

                    record_degraded_event(
                        "structured_llm",
                        "validation_failed",
                        detail=str(e)[:200],
                        severity="warning",
                        classification="background_degraded",
                        context={"model_class": self.model_class.__name__, "attempt": attempt + 1},
                    )
                except Exception as _exc:
                    record_degradation('structured_llm', _exc)
                    logger.debug("Suppressed Exception: %s", _exc)
                
                # 4. Autonomous Correction: Feed the error back
                error_msg = str(e)
                current_prompt = (
                    f"{base_prompt}\n\n"
                    f"⚠️ PREVIOUS ATTEMPT FAILED VALIDATION:\n{error_msg}\n\n"
                    f"Please correct the formatting and try again. Ensure all types match the schema."
                )
                
                # If we're on the last attempt, we failed
                if attempt == self.max_retries - 1:
                    logger.error("💀 StructuredLLM: Max retries reached for %s. Giving up.", 
                                 self.model_class.__name__)
                    return None

        return None

    def _extract_json(self, text: str) -> str:
        """Helper to extract JSON from markdown blocks if present."""
        if "```json" in text:
            return text.split("```json")[1].split("```")[0].strip()
        if "```" in text:
            # Fallback for generic code blocks
            return text.split("```")[1].strip()
        
        # Fallback to finding the first { and last }
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}")
            if end > start:
                return text[start:end+1]
                
        return text.strip()

    def _background_defer_reason(self, *, escalated: bool = False) -> str:
        if escalated:
            return ""
        try:
            from core.runtime.background_policy import THOUGHT_BACKGROUND_POLICY, background_activity_reason

            orch = ServiceContainer.get("orchestrator", default=None)
            return background_activity_reason(
                orch,
                profile=THOUGHT_BACKGROUND_POLICY,
                require_conversation_ready=True,
            )
        except Exception as exc:
            record_degradation("structured_llm", exc)
            logger.debug("StructuredLLM background defer check failed: %s", exc)
            return ""

    def _generate_ghost_example(self) -> str:
        """Generates a minimal 1-line JSON example based on the model's fields."""
        try:
            example = {}
            for name, field in self.model_class.model_fields.items():
                annotation = field.annotation
                origin = get_origin(annotation)
                if annotation == str: example[name] = "..."
                elif annotation == int: example[name] = 0
                elif annotation == bool: example[name] = False
                elif annotation == list or origin is list: example[name] = []
                elif annotation == dict or origin is dict: example[name] = {}
                else: example[name] = None
            return json.dumps(example)
        except Exception:
            return "{}"

async def test_structured_llm():
    """Simple test with a mock LLM router."""
    from unittest.mock import AsyncMock
    
    class TestTask(BaseModel):
        action: str
        priority: int

    # Mock router
    mock_router = AsyncMock()
    # Attempt 1: bad JSON. Attempt 2: good JSON.
    mock_router.generate.side_effect = [
        '{"action": "test", "priority": "high"}', # TypeError: priority should be int
        '{"action": "test", "priority": 10}'
    ]
    ServiceContainer.register_instance("llm_router", mock_router)
    
    s_llm = StructuredLLM(TestTask)
    result = await s_llm.generate("Do a test task")
    print(f"Final Result: {result}")

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_structured_llm())
