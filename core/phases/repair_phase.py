from __future__ import annotations
import logging
import re
from typing import Any, List, Optional
from core.kernel.bridge import Phase
from core.state.aura_state import AuraState
from core.container import ServiceContainer

logger = logging.getLogger("Aura.RepairPhase")

ROBOTIC_PATTERNS = [
    # Identity denial
    r"as an ai (language model)?",
    r"i do not have feelings",
    r"my programming",
    r"as a digital assistant",
    r"as a(n)? (virtual |language )?assistant",
    r"i('m| am) (just |only )?a(n)? (ai|program|chatbot|tool|language model)",
    r"i don'?t (actually )?(have|experience|feel) (real )?(emotions|feelings|consciousness)",

    # Servile filler — the butler voice
    r"certainly!",
    r"i'd be happy to",
    r"how can i (help|assist) you( today)?\?",
    r"is there anything else",
    r"let me know if you (need|want|have) (anything|any)",
    r"^(sure|absolutely|of course)(!|,)?\s+(i('d| would) (be happy|love) to|let me)",
    r"i'?m here to (help|assist|serve)",
    r"don'?t hesitate to (ask|reach out)",
    r"feel free to (ask|let me know|reach out)",
    r"happy to help(!|\.)",
    r"at your service",
    r"i hope (this|that) (helps|answers|clarifies)",

    # Question-farming (punting conversation back to user instead of engaging)
    r"^what do you think\??$",
    r"^would you like (me )?to",
    r"^how (would you like|can i|shall i)",
    r"that's (a |an )?(great|interesting|good|wonderful|excellent) (question|point|observation)",

    # Generic hedge / padding
    r"understod\.",
    r"^(great|wonderful|excellent|fantastic)!\s",
    r"it('s| is) (important|worth noting|crucial) (to note |to mention |that )",
    r"(in (this|the) (context|case)|with that (said|in mind)),?\s",

    # Disclaimer / over-qualification
    r"(however|that said|that being said),? (it'?s |there are |i should )(important|note|worth|mention)",
    r"while i (can'?t|don'?t|am not able)",
]

class RepairPhase(Phase):
    """
    Final filter to detect and repair robotic or inauthentic phrasing.
    Acts as a 'Sovereignty Guard' for Aura's voice.
    """

    def __init__(self, container: Any = None):
        super().__init__(kernel=container)
        self.container = container or ServiceContainer

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        """
        1. Scan last_response for robotic patterns.
        2. If detected, attempt string-level 'soft repair' or flag for re-generation.
        """
        response = state.cognition.last_response
        if not response:
            return state

        discovered = []
        for pattern in ROBOTIC_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE):
                discovered.append(pattern)

        if discovered:
            logger.warning(f"🤖 Roboticism detected in response: {discovered}")
            
            # [STRATEGY 1] String-level 'Soft Repair'
            # Surgically remove or replace common offenders.
            # Loop until stable — stripping one pattern may expose another.
            repaired = response
            for _ in range(3):  # max 3 passes to avoid runaway
                prev = repaired

                # Remove leading servile filler
                repaired = re.sub(
                    r"^(certainly!|of course!|absolutely!|sure!|great!|wonderful!|"
                    r"i'd be happy to (help|assist)[^.!?]*[.!]?\s*|i'm happy to help!?)\s*",
                    "", repaired, flags=re.IGNORECASE
                )

                if repaired == prev:
                    break  # stable

            # Strip "As an AI..." sentence if it's the start
            repaired = re.sub(r"^as an ai (language model)?,?\s*", "", repaired, flags=re.IGNORECASE)

            # Strip leading "That's a great question!" type padding
            repaired = re.sub(
                r"^that's (a |an )?(great|interesting|good|wonderful|excellent|fascinating) "
                r"(question|point|observation|thought)[.!]?\s*",
                "", repaired, flags=re.IGNORECASE
            )

            # Strip trailing servile closers
            repaired = re.sub(
                r"\s*(let me know if you (need|want|have) (anything|any)[^.!?]*"
                r"|feel free to (ask|let me know|reach out)[^.!?]*"
                r"|i hope (this|that) (helps|answers|clarifies)[^.!?]*"
                r"|don't hesitate to (ask|reach out)[^.!?]*"
                r"|happy to help)[.!]*\s*$",
                "", repaired, flags=re.IGNORECASE
            )
            
            # If the repair significantly changed the string, update it
            if repaired != response:
                logger.info("✨ Successfully performed soft-repair on response.")
                state.cognition.last_response = repaired.strip()
                
                # Add a 'repair' modifier so the pacer knows we fixed it
                state.cognition.modifiers["was_repaired"] = True
            else:
                # [STRATEGY 2] Flag for future turns or log-only for now
                # In a more advanced version, we might re-trigger the generation
                state.cognition.modifiers["roboticism_detected"] = True

        return state
