"""CPU speech from a locally prepared Pocket TTS model and voice state."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from core.runtime.lockdep import LockRank, checked_lock

logger = logging.getLogger("Aura.Voice.Pocket")


class PocketSpeech:
    """Keep the model and its conditioning resident between spoken clauses."""

    name = "pocket"

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._model: Any = None
        self._voices: dict[str, Any] = {}
        self._lock = checked_lock("voice.tts.pocket", rank=LockRank.LEAF)

    def load(self) -> bool:
        config = self._directory / "config.yaml"
        states = sorted(self._directory.glob("*.safetensors"))
        if not config.is_file() or not states:
            return False
        try:
            import yaml
            from pocket_tts import TTSModel

            # Model downloads belong to preparation, never a voice call.
            settings = yaml.safe_load(config.read_text())
            weights = settings.get("weights_path")
            tokenizer = settings["flow_lm"]["lookup_table"]["tokenizer_path"]
            if not all(isinstance(p, str) and Path(p).is_absolute() and Path(p).is_file()
                       for p in (weights, tokenizer)):
                raise ValueError("Pocket speech requires local weights and tokenizer")
            self._model = TTSModel.load_model(config=config)
            self._voices = {p.stem: self._model.get_state_for_audio_prompt(p) for p in states}
            # Compile the speed adjustment before voice.ready, not on the
            # first turn whose delivery asks for a different pace.
            import librosa

            librosa.effects.time_stretch(np.zeros(4096, dtype=np.float32), rate=1.01)
            return True
        except (ImportError, OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Local Pocket speech unavailable; retaining Kokoro: %s", exc)
            self._model = None
            self._voices = {}
            return False

    @property
    def voices(self) -> frozenset[str]:
        return frozenset(self._voices)

    def synthesize(self, text: str, spec: Any) -> tuple[np.ndarray, int]:
        voice = spec.voice if spec.voice in self._voices else (
            "alba" if "alba" in self._voices else sorted(self._voices)[0])
        # The generator mutates its working state. Serialise whole clauses;
        # copy_state preserves the prepared speaker across turns.
        with self._lock:
            audio = self._model.generate_audio(self._voices[voice], text, copy_state=True)
            samples = audio.detach().cpu().numpy().astype(np.float32, copy=False)
        speed = float(spec.speed)
        if abs(speed - 1.0) > 0.01:
            import librosa

            samples = librosa.effects.time_stretch(samples, rate=max(.5, min(2., speed)))
        return samples, int(self._model.sample_rate)
