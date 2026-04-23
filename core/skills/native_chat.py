# skills/native_chat.py
import asyncio
import logging
from typing import Any, Dict, Optional

from core.brain.cognitive_engine import CognitiveEngine
from core.skills.base_skill import BaseSkill

# Import the ThoughtEmitter for visibility
try:
    from core.thought_stream import get_emitter
    emitter = get_emitter()
except ImportError:
    emitter = None

logger = logging.getLogger("Skills.NativeChat")


def _schedule_background_task(coro: Any, *, name: str) -> None:
    try:
        from core.utils.task_tracker import get_task_tracker

        get_task_tracker().create_task(coro, name=name)
        return
    except Exception:
        pass
    try:
        asyncio.create_task(coro, name=name)
    except Exception as exc:
        try:
            coro.close()
        except Exception:
            pass
        logger.debug("NativeChat background task %s could not be scheduled: %s", name, exc)

class NativeChatSkill(BaseSkill):
    name = "native_chat"
    description = "Conversational engine with robust dependency resolution."
    aliases = ["chat", "talk"]
    inputs = {
        "message": "User input to respond to."
    }

    def __init__(self, brain: Optional[CognitiveEngine] = None):
        self.brain = brain

    def _resolve_brain(self, context: Dict) -> Optional[CognitiveEngine]:
        """The 'Wholesale' Fix: A multi-stage fallback strategy to find the Brain.
        """
        # Strategy 1: Constructor Injection (Best)
        if self.brain:
            return self.brain

        # Strategy 2: Context/Container Injection (Standard)
        if context:
            if hasattr(context, 'get'):
                brain = context.get("brain")
                if brain: return brain
        
        # Strategy 3: Global Fallback (Self-Healing)
        try:
            from core.brain.cognitive_engine import cognitive_engine
            if cognitive_engine:
                self.brain = cognitive_engine
                return cognitive_engine
        except ImportError as _e:
            logger.debug('Ignored ImportError in native_chat.py: %s', _e)

        return None

    async def execute(self, goal: Dict, context: Dict) -> Dict:
        """Execute chat with safety rails and visibility.
        """
        # 1. Resolve Brain securely
        brain = self._resolve_brain(context)
        
        if not brain:
            msg = "CRITICAL: Brain not found in Context OR Global Scope."
            logger.critical(msg)
            return {
                "ok": False, 
                "error": msg
            }

        # 2. Extract Message
        params = goal.get("params", {})
        msg = params.get("message") or goal.get("objective")
        
        if not msg:
            return {"ok": False, "error": "No message provided."}

        # Force message to string to prevent "unhashable type: slice" errors if it's a dict
        msg_str = str(msg)
        logger.info("Processing chat message (Type: %s): %s...", type(msg_str), str(msg_str)[:50])

        # Phase XIV: Boost Conversational Momentum
        try:
            from core.container import ServiceContainer
            cme = ServiceContainer.get("conversational_momentum_engine", default=None)
            if cme:
                _schedule_background_task(
                    cme.on_new_user_message(msg_str),
                    name="native_chat.momentum",
                )
        except Exception as _e:
            logger.debug('Ignored Exception in native_chat.py: %s', _e)

        # 3. Understand Intent (ToM)
        intent_context = {}
        tom = context.get("theory_of_mind")
        if tom:
            intent_context = tom.infer_intent(msg_str, context)
            logger.info("ToM Intent: %s", intent_context.get('pragmatic', 'standard'))

        # 4. Gather Rich Context (v5.5 Dynamic Context Builder)
        try:
            from core.brain.context_builder import DynamicContextBuilder
            rich_context = await DynamicContextBuilder.build_rich_context(msg_str, context)
            personality_context = rich_context.get("personality", {})
            logger.info("Personality State (v5.5): %s (%s)", personality_context.get('mood'), personality_context.get('tone'))
            
            # Additional Context Formatting for Prompt
            prompt_context_str = DynamicContextBuilder.format_for_prompt(rich_context)
            rich_context["prompt_segment"] = prompt_context_str
        except ImportError:
            # Fallback to legacy context gathering
            memory_context = ""
            mem_sys = context.get("memory")
            if mem_sys:
                memory_context = await mem_sys.retrieve_context(msg_str)
            
            personality_context = {}
            try:
                from core.brain.personality_engine import get_personality_engine
                personality = get_personality_engine()
                personality.respond_to_event("user_message", {"message": msg_str})
                personality_context = personality.get_emotional_context_for_response()
            except ImportError as _e:
                logger.debug('Ignored ImportError in native_chat.py: %s', _e)
                
            rich_context = {
                **context, 
                "user_intent": intent_context,
                "memory_context": memory_context,
                "personality": personality_context
            }
            
        # 5. Think (with Visibility and Intent)
        try:
            if emitter:
                emitter.emit("Cognition", f"Intent: {intent_context.get('pragmatic', 'standard')} | {msg_str[:30]}...", level="info")
            
            # Combine original context with dynamic prompt segment (v5.5)
            # Prepend system context to the user message for the LLM
            final_llm_input = msg_str
            if "prompt_segment" in rich_context:
                final_llm_input = f"{rich_context['prompt_segment']}\n\nUser Input: {msg_str}"

            from core.brain.cognitive_engine import ThinkingMode
            # Use CREATIVE mode for chat to ensure personality and agency (Turns: 5)
            thought = await brain.think(final_llm_input, context=rich_context, mode=ThinkingMode.CREATIVE)
            
            # Handle thought object or raw string (Unwrap for UI)
            response = getattr(thought, 'content', str(thought))

            # BROADCAST: Send final response to all global listeners (UI)
            if emitter:
                try:
                    logger.info("Emitting chat response to ThoughtStream: %s...", response[:50])
                    emitter.emit("AURA", response, level="chat")
                except Exception as e:
                    logger.error("Failed to emit chat response: %s", e)
            else:
                logger.warning("No ThoughtStream emitter found in NativeChatSkill!")

            # MEMORY: store interaction
            try:
                mem_sys = context.get("memory")
                if mem_sys:
                    # Async remember calls for the interaction
                    logger.info("Storing chat interaction in Temporal Memory: %s...", msg_str[:min(len(msg_str), 30)])
                    _schedule_background_task(
                        mem_sys.remember(
                            msg_str,
                            metadata={"role": "user", "intent": intent_context.get("pragmatic")},
                        ),
                        name="native_chat.remember_user",
                    )
                    _schedule_background_task(
                        mem_sys.remember(response, metadata={"role": "aura", "mode": "chat"}),
                        name="native_chat.remember_aura",
                    )
            except Exception as e:
                logger.warning("Memory storage failed: %s", e)

            # BIORHYTHM: mark interaction
            try:
                orchestrator = context.get("orchestrator")
                if orchestrator and hasattr(orchestrator, "biorhythm"):
                    orchestrator.biorhythm.mark_interaction()
            except Exception as exc:
                logger.debug("Suppressed: %s", exc)

            return {
                "ok": True,
                "response": response,
                "summary": "Replied to user."
            }
            
        except Exception as e:
            logger.error("Cognitive failure: %s", e, exc_info=True)
            return {"ok": False, "error": f"Cognitive failure: {e}"}
