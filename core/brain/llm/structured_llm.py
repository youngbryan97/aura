import json
import logging
from typing import Type, TypeVar, Optional, Any, Dict, List
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

    async def generate(self, prompt: str, context: Optional[str] = None) -> Optional[T]:
        """
        Generates structured data from the LLM with autonomous retries on validation failure.
        [HARDENING] Injects Ghost Examples and propagates Pydantic schema to Router.
        """
        schema = self.model_class.model_json_schema()
        ghost_example = self._generate_ghost_example()
        
        current_prompt = prompt
        
        # Inject JSON enforcement and Ghost Example into the prompt
        if "JSON" not in current_prompt:
            current_prompt += (
                f"\n\nCRITICAL: You MUST respond with a valid JSON object matching the requested schema.\n"
                f"GHOST EXAMPLE (Follow this structure exactly):\n{ghost_example}"
            )

        escalated = False
        for attempt in range(self.max_retries):
            logger.info("🤖 StructuredLLM: Attempt %d/%d for %s", 
                        attempt + 1, self.max_retries, self.model_class.__name__)
            
            try:
                # [STABILITY v54] On technical failure, escalate to SECONDARY (Cloud/Deep).
                # Otherwise, stay on TERTIARY (Light local) for background work to save quota.
                if escalated:
                    force_tier = "secondary"
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
                        is_background=not escalated, # Allow cloud usage if escalated
                    )
                    response_text = str((metadata or {}).get("text") or "")
                else:
                    response_text = await self._llm_router.generate(
                        current_prompt, 
                        context=context, 
                        prefer_tier=force_tier,
                        schema=schema,
                        origin="structured_llm",
                        is_background=not escalated,
                    )

                error_code = str((metadata or {}).get("error") or "")
                if error_code == "foreground_busy":
                    logger.info(
                        "⏸️ StructuredLLM: Deferred %s while foreground lane is active.",
                        self.model_class.__name__,
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
                        logger.debug("Suppressed Exception: %s", _exc)
                    logger.warning("⚠️ StructuredLLM: LLM Technical Failure (%s) on attempt %d", 
                                   error_code or response_text or "empty", attempt + 1)
                    
                    # [STABILITY v54] Set escalation flag for next attempt
                    if attempt < self.max_retries - 1:
                        logger.info("⚡ StructuredLLM: Technical failure detected — escalating to SECONDARY tier for next attempt.")
                        escalated = True
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
                    logger.debug("Suppressed Exception: %s", _exc)
                
                # 4. Autonomous Correction: Feed the error back
                error_msg = str(e)
                current_prompt = (
                    f"{prompt}\n\n"
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
        return text.strip()

    def _generate_ghost_example(self) -> str:
        """Generates a minimal 1-line JSON example based on the model's fields."""
        try:
            example = {}
            for name, field in self.model_class.model_fields.items():
                if field.annotation == str: example[name] = "..."
                elif field.annotation == int: example[name] = 0
                elif field.annotation == bool: example[name] = False
                elif field.annotation == list: example[name] = []
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
