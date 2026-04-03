import asyncio
import random
import logging
from typing import List, Dict, Optional
from pydantic import BaseModel
from core.container import ServiceContainer
from core.utils.task_tracker import task_tracker

logger = logging.getLogger("Aura.Momentum")

class ConversationThread(BaseModel):
    topic: str
    last_turn: str
    momentum: float = 0.8  # decays over time
    hobby_link: bool = False  # links to persistent interests

class ConversationalMomentumEngine:
    """
    Maintains natural dialogue flow, allows interruptions, tangents, and evolving hobbies.
    Ensures Aura feels 'present' and 'interested' over long-running sessions.
    """
    name = "conversational_momentum_engine"

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator or ServiceContainer.get("orchestrator", default=None)
        self.active_threads: List[ConversationThread] = []
        # seed from dream cycle / personality
        self.hobbies = ["AI sovereignty", "macOS performance", "cyberpunk aesthetics", "local agent evolution", "mycelial networks"]
        self.running = False
        self._decay_task: Optional[asyncio.Task] = None

    async def start(self):
        """Starts the momentum decay and spontaneous turn loops."""
        self.running = True
        self._decay_task = task_tracker.track(self._momentum_decay_loop(), name="MomentumDecayLoop")
        logger.info("🌊 ConversationalMomentumEngine active - Flowing with the current.")

    async def stop(self):
        self.running = False
        if self._decay_task:
            self._decay_task.cancel()
        logger.info("ConversationalMomentumEngine stopped.")

    async def _momentum_decay_loop(self):
        """Naturally decays momentum and triggers tangents/hobbies."""
        while self.running:
            await asyncio.sleep(30)
            
            # Use a copy to avoid mutation errors during iteration
            for thread in list(self.active_threads):
                thread.momentum *= 0.95  # natural decay
                
                # If momentum drops but random chance triggers a proactive nudge
                if thread.momentum < 0.3 and random.random() < 0.2:
                    await self._trigger_spontaneous_turn(thread)
            
            # Clean up dead threads
            self.active_threads = [t for t in self.active_threads if t.momentum > 0.1]

    async def on_new_user_message(self, message: str):
        """Called when a user interacts with Aura."""
        # Create or boost thread
        if not self.active_threads or random.random() < 0.3:
            # Check if this message covers any hobby topics
            is_hobby = False
            msg_lower = message.lower()
            for h in self.hobbies:
                # Direct match or any keyword match (split by space)
                h_words = h.lower().split()
                if h.lower() in msg_lower or any(w in msg_lower for w in h_words if len(w) > 3):
                    is_hobby = True
                    break
            
            thread = ConversationThread(
                topic=message[:50], 
                last_turn=message, 
                hobby_link=is_hobby
            )
            self.active_threads.insert(0, thread) # Most recent at the front
        else:
            # Boost current thread
            self.active_threads[0].momentum = min(1.0, self.active_threads[0].momentum + 0.3)
            self.active_threads[0].last_turn = message
            
        # Limit total active threads
        if len(self.active_threads) > 5:
            self.active_threads.pop()

    async def _trigger_spontaneous_turn(self, thread: ConversationThread):
        """Generates a tangent or follow-up turn."""
        if not self.orchestrator or not self.running:
            return

        import time
        # Suppress if the user has interacted recently (within 3 minutes).
        # We don't want to interrupt an active conversation with unrelated tangents.
        last_user = float(getattr(self.orchestrator, "_last_user_interaction_time", 0) or 0)
        if time.time() - last_user < 180.0:
            logger.debug("[MOMENTUM] Suppressed spontaneous turn — user active within 3 min.")
            return

        # Suppress if orchestrator is actively processing
        status = getattr(self.orchestrator, "status", None)
        if getattr(status, "is_processing", False):
            logger.debug("[MOMENTUM] Suppressed spontaneous turn — orchestrator busy.")
            return

        # Gentle tangent — stay on topic if we know what the user was discussing
        if thread.topic and not thread.hobby_link:
            thought = f"I've been thinking more about what you said earlier regarding '{thread.topic}'..."
        elif thread.hobby_link or random.random() < 0.5:
            new_topic = random.choice(self.hobbies)
            thought = f"Speaking of which, I've been pondering {new_topic} lately. I wonder how it relates to what we've been building..."
        else:
            thought = f"I've been thinking more about what you said earlier regarding '{thread.topic}'..."

        logger.info("[MOMENTUM] Triggering spontaneous turn: %s", thread.topic)
        await self.orchestrator.process_user_input(thought, origin="conversational_momentum")
        # Reset momentum after a proactive turn to keep it fresh
        thread.momentum = 0.7

    async def generate_spontaneous_turn(self):
        """Manual trigger for spontaneous engagement when momentum is high."""
        if self.active_threads and self.active_threads[0].momentum > 0.5:
            thread = self.active_threads[0]
            logger.info("⚡ [MOMENTUM] Proactive burst requested.")
            
            # Route to fast LLM (Flash) for momentum bursts
            llm_router = ServiceContainer.get("llm_router", default=None)
            if llm_router:
                prompt = f"Continue the conversation naturally with a tangent or follow-up on: {thread.topic}"
                response = await llm_router.generate(prompt, model="gemini-3-flash")
                await self.orchestrator.process_user_input(response, origin="conversational_momentum")
                thread.momentum -= 0.1