"""Turn-taking + silence policy for conversational and movie modes.

Audit-driven contract: Aura needs a single source of truth for *when*
she may speak, vs. when she stays quiet, in conversation, movie, focus,
and collaborative modes. The state is updated by perception/audio
inputs and consumed by output paths before they emit.
"""
from __future__ import annotations


import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ConversationMode(str, Enum):
    CONVERSATION = "conversation"
    MOVIE = "movie"
    FOCUS = "focus"
    COLLABORATIVE = "collaborative"


@dataclass
class TurnTakingState:
    mode: ConversationMode = ConversationMode.CONVERSATION
    user_speaking: bool = False
    last_user_speech_at: float = 0.0
    last_aura_speech_at: float = 0.0
    pending_repair: bool = False
    scene_energy: float = 0.0


class TurnTakingEngine:
    def __init__(
        self,
        *,
        conversation_min_silence_s: float = 1.5,
        movie_min_silence_s: float = 12.0,
        focus_min_silence_s: float = 60.0,
        collaborative_min_silence_s: float = 0.5,
        backchannel_cooldown_s: float = 6.0,
        clock=time.time,
    ):
        self.state = TurnTakingState()
        self._cooldown_table = {
            ConversationMode.CONVERSATION: conversation_min_silence_s,
            ConversationMode.MOVIE: movie_min_silence_s,
            ConversationMode.FOCUS: focus_min_silence_s,
            ConversationMode.COLLABORATIVE: collaborative_min_silence_s,
        }
        self._backchannel_cooldown_s = backchannel_cooldown_s
        self._clock = clock

    # --- inputs ----------------------------------------------------------

    def set_mode(self, mode: ConversationMode) -> None:
        self.state.mode = mode

    def user_started_speaking(self) -> None:
        self.state.user_speaking = True
        self.state.last_user_speech_at = self._clock()

    def user_stopped_speaking(self) -> None:
        self.state.user_speaking = False
        self.state.last_user_speech_at = self._clock()

    def aura_emitted(self) -> None:
        self.state.last_aura_speech_at = self._clock()

    def update_scene_energy(self, energy: float) -> None:
        self.state.scene_energy = max(0.0, min(1.0, energy))

    def request_repair(self) -> None:
        self.state.pending_repair = True

    # --- decision --------------------------------------------------------

    def can_aura_speak(self) -> bool:
        if self.state.user_speaking:
            return False
        now = self._clock()
        min_silence = self._cooldown_table[self.state.mode]
        if now - self.state.last_user_speech_at < min_silence:
            return False
        if now - self.state.last_aura_speech_at < self._backchannel_cooldown_s:
            return False
        if self.state.mode == ConversationMode.MOVIE and self.state.scene_energy >= 0.6:
            return False
        if self.state.mode == ConversationMode.FOCUS and not self.state.pending_repair:
            return False
        return True

    def consume_repair(self) -> bool:
        if self.state.pending_repair:
            self.state.pending_repair = False
            return True
        return False
