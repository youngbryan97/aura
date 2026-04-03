import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.kernel.aura_kernel import AuraKernel

try:
    from core.container import ServiceContainer
except ImportError:
    ServiceContainer = None

logger = logging.getLogger(__name__)


@dataclass
class OrganStub:
    """
    Pattern for handling high-latency hardware or external subsystems.
    Explicitly resolves dependencies from the Kernel's registry to ensure
    Closed-Graph integrity.
    """

    name: str
    kernel: "AuraKernel"
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    instance: Any = None

    async def load(self) -> None:
        """
        [ZENITH] Heavy initialization with rigid timeouts.
        Allows the Kernel to boot instantly while 'organs' warm up.
        """
        logger.info(f"Loading organ: {self.name}...")
        try:
            async with asyncio.timeout(5.0):
                if self.name in ("brain", "llm"):
                    try:
                        from core.brain.llm.llm_router import IntelligentLLMRouter as LLMRouter
                        self.instance = self.kernel.get(LLMRouter)
                    except Exception:
                        self.instance = None

                    if self.instance:
                        if not hasattr(self.instance, "think") and hasattr(self.instance, "generate"):
                            self.instance.think = self.instance.generate
                    else:
                        class MockLLM:
                            async def think(self, prompt: str, **kwargs) -> str:
                                return "I'm Aura."

                            async def classify(self, prompt: str) -> str:
                                return "CHAT"

                        self.instance = MockLLM()
                elif self.name == "vision":
                    if ServiceContainer:
                        self.instance = ServiceContainer.get("vision_presence", default=None)
                    if not self.instance:
                        class MockVision:
                            async def capture(self):
                                return None

                            async def capture_desktop(self):
                                return None

                            async def load(self):
                                pass

                        self.instance = MockVision()
                elif self.name == "neural":
                    safe_boot = os.getenv("AURA_SAFE_BOOT_DESKTOP", "0") == "1"
                    try:
                        def _build_neural():
                            from core.senses.neural_bridge import NeuralBridge

                            return NeuralBridge(lightweight_mode=safe_boot)

                        self.instance = await asyncio.wait_for(asyncio.to_thread(_build_neural), timeout=1.5)
                        await asyncio.wait_for(self.instance.load(), timeout=2.5 if safe_boot else 4.0)
                    except Exception as e:
                        logger.warning(f"Failed to load NeuralBridge: {e}")

                        class MockNeural:
                            async def load(self):
                                pass

                            def get_status(self):
                                return {"is_running": False, "lightweight_mode": True}

                        self.instance = MockNeural()
                elif self.name == "cookie":
                    try:
                        from core.autonomy.reflective_cookie import ReflectiveCookie

                        self.instance = ReflectiveCookie(self.kernel)
                        await self.instance.load()
                    except Exception:
                        class MockCookie:
                            async def load(self):
                                pass

                        self.instance = MockCookie()
                elif self.name == "prober":
                    try:
                        from core.brain.alignment_prober import EmpathyProber

                        self.instance = EmpathyProber(self.kernel)
                        await self.instance.load()
                    except Exception:
                        class MockProber:
                            async def load(self):
                                pass

                        self.instance = MockProber()
                elif self.name == "tricorder":
                    try:
                        from core.cybernetics.tricorder import Tricorder

                        self.instance = Tricorder(self.kernel)
                        await self.instance.load()
                    except Exception:
                        class MockTricorder:
                            async def load(self):
                                pass

                        self.instance = MockTricorder()
                elif self.name == "ice_layer":
                    try:
                        from core.cybernetics.ice_layer import ICELayer

                        self.instance = ICELayer(self.kernel)
                        await self.instance.load()
                    except Exception:
                        class MockICE:
                            async def load(self):
                                pass

                        self.instance = MockICE()
                elif self.name == "omni_tool":
                    try:
                        from core.cybernetics.omni_tool import OmniTool

                        self.instance = OmniTool(self.kernel)
                        await self.instance.load()
                    except Exception:
                        class MockOmni:
                            async def load(self):
                                pass

                        self.instance = MockOmni()
                elif self.name == "memory":
                    if ServiceContainer:
                        self.instance = ServiceContainer.get("memory_facade", default=None)
                elif self.name == "voice":
                    if ServiceContainer:
                        try:
                            self.instance = await asyncio.wait_for(
                                asyncio.to_thread(ServiceContainer.get, "voice_engine", default=None),
                                timeout=2.0,
                            )
                        except asyncio.TimeoutError:
                            logger.warning("VoiceEngine resolution TIMEOUT. Proceeding without voice.")

                    if not self.instance:
                        class MockVoice:
                            async def speak(self, t):
                                pass

                            async def say(self, t):
                                pass

                            async def load(self):
                                pass

                        self.instance = MockVoice()
                elif self.name == "metabolism":
                    if ServiceContainer:
                        self.instance = ServiceContainer.get("metabolic_monitor", default=None)
                elif self.name == "continuity":
                    try:
                        from core.cybernetics.knowledge_continuity import KnowledgeContinuity

                        self.instance = KnowledgeContinuity(self.kernel)
                        await asyncio.wait_for(self.instance.load(), timeout=3.0)
                    except Exception as e:
                        logger.warning(f"Continuity organ load failed: {e}")

                        class MockContinuity:
                            async def load(self):
                                pass

                        self.instance = MockContinuity()
                else:
                    logger.warning(f"Unknown organ '{self.name}'. Providing generic fallback.")

                    class GenericMock:
                        async def load(self):
                            pass

                    self.instance = GenericMock()
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Organ {self.name} load TIMEOUT. Providing mock fallback.")

            class TimeoutMock:
                async def load(self):
                    pass

            self.instance = TimeoutMock()
        except Exception as e:
            logger.exception(f"OrganStub: unexpected error while resolving {self.name}: {e}")

            class ErrorMock:
                async def load(self):
                    pass

            self.instance = ErrorMock()

        self.ready.set()
        logger.debug(f"Organ {self.name} load complete.")

    def get_instance(self) -> Any:
        """
        Returns the hardware instance if ready, otherwise raises error.
        Prevents 'mid-tick' crashes due to uninitialized hardware.
        """
        if not self.ready.is_set():
            raise RuntimeError(f"Attempted to access organ '{self.name}' before it was READY.")
        return self.instance

    async def shutdown(self) -> None:
        """Stop the backing organ instance if it exposes a shutdown hook."""
        inst = self.instance
        if inst is None:
            return
        hook = getattr(inst, "shutdown", None) or getattr(inst, "stop", None)
        if not callable(hook):
            return
        result = hook()
        if asyncio.iscoroutine(result):
            await result
