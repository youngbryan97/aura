from core.runtime.errors import record_degradation
import asyncio
import logging
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("Aura.NeuralBridge")


def _logging_streams_available() -> bool:
    for candidate in (logger, logging.getLogger()):
        for handler in getattr(candidate, "handlers", []):
            stream = getattr(handler, "stream", None)
            if stream is not None and getattr(stream, "closed", False):
                return False
    return True


NUM_CHANNELS = 8
SAMPLING_RATE = 250
WINDOW_SEC = 2.0
NUM_SAMPLES = int(SAMPLING_RATE * WINDOW_SEC)
COMMANDS = ["INTUITION", "LOGIC", "SYNCHRONICITY", "RECURSION"]
NUM_CLASSES = len(COMMANDS)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    denom = np.sum(exp)
    if denom <= 0:
        return np.ones_like(values) / max(len(values), 1)
    return exp / denom


class BCIClassifier:
    """Lightweight synthetic classifier for demo EEG streams.

    The previous implementation imported torch/scipy at module import time,
    which could stall desktop boot and inflate baseline RAM even though this
    bridge only simulates local thought telemetry.
    """

    def __init__(self):
        self._templates: Dict[int, np.ndarray] = {}
        self._template_norms: Dict[int, float] = {}

    def calibrate(self, sample_builder) -> None:
        templates: Dict[int, np.ndarray] = {}
        norms: Dict[int, float] = {}
        for class_idx in range(NUM_CLASSES):
            samples = [sample_builder(class_idx) for _ in range(6)]
            template = np.mean(samples, axis=0)
            templates[class_idx] = template
            norms[class_idx] = max(float(np.linalg.norm(template.ravel())), 1e-6)
        self._templates = templates
        self._template_norms = norms

    def eval(self) -> None:
        return None

    def predict(self, eeg_data: np.ndarray) -> tuple[int, float]:
        if not self._templates:
            return 0, 0.0

        flattened = eeg_data.ravel()
        sample_norm = max(float(np.linalg.norm(flattened)), 1e-6)
        scores = []
        for class_idx in range(NUM_CLASSES):
            template = self._templates[class_idx].ravel()
            score = float(np.dot(flattened, template) / (sample_norm * self._template_norms[class_idx]))
            scores.append(score)

        probabilities = _softmax(np.asarray(scores, dtype=np.float64))
        winner = int(np.argmax(probabilities))
        return winner, float(probabilities[winner])


class NeuralBridge:
    """Synthetic BCI neural bridge for ambient thought telemetry."""

    def __init__(self, *, lightweight_mode: bool = False):
        self.model = BCIClassifier()
        self.is_trained = False
        self._is_running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._event_bus = None
        self._rng = np.random.default_rng()
        self._lightweight_mode = lightweight_mode
        self._poll_interval_range = (8.0, 18.0) if lightweight_mode else (5.0, 15.0)

        self._last_command: Optional[str] = None
        self._confidence: float = 0.0

    async def load(self):
        logger.info("🧠 [NEURAL] Initializing BCI Neural Bridge...")

        try:
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._main_loop = None

        await asyncio.to_thread(self._calibrate)
        self.is_trained = True
        logger.info("✅ [NEURAL] BCI Calibration complete. 32B-Neural-Net ONLINE.")
        self.start()

    def _calibrate(self):
        start_time = time.time()
        self.model.calibrate(self._generate_synthetic_eeg)
        logger.debug(
            "Calibration finished in %.2fs (template-matched synthetic bridge)",
            time.time() - start_time,
        )

    def _generate_synthetic_eeg(self, class_label: int) -> np.ndarray:
        t = np.linspace(0, WINDOW_SEC, NUM_SAMPLES)
        data = np.zeros((NUM_CHANNELS, NUM_SAMPLES))
        for ch in range(NUM_CHANNELS):
            noise = self._rng.normal(0, 0.5, NUM_SAMPLES)
            if class_label == 0:
                sig = 1.5 * np.sin(2 * np.pi * 18 * t) + 0.8 * np.sin(2 * np.pi * 10 * t)
            elif class_label == 1:
                sig = 1.5 * np.sin(2 * np.pi * 22 * t) + 0.8 * np.sin(2 * np.pi * 12 * t)
            elif class_label == 2:
                sig = np.exp(-((t - 0.3) ** 2) / 0.05) * 3
            else:
                sig = 2.0 * np.sin(2 * np.pi * 10 * t)
            data[ch] = sig + noise + self._rng.normal(0, 0.2, NUM_SAMPLES)

        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float64)
        for ch in range(NUM_CHANNELS):
            data[ch] = np.convolve(data[ch], kernel, mode="same")
        data = data - np.mean(data, axis=1, keepdims=True)
        return data

    def start(self):
        if self._is_running:
            return
        self._stop_event.clear()
        self._is_running = True
        self._worker_thread = threading.Thread(
            target=self._run_inference_loop,
            daemon=True,
            name="AuraNeuralWorker",
        )
        self._worker_thread.start()
        logger.info("🌊 [NEURAL] Continuous telemetry loop started.")

    def stop(self):
        self._is_running = False
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)

    def _run_inference_loop(self):
        try:
            from core.event_bus import get_event_bus

            self._event_bus = get_event_bus()
        except ImportError:
            self._event_bus = None

        self.model.eval()
        while self._is_running:
            try:
                if self._stop_event.wait(timeout=float(self._rng.uniform(*self._poll_interval_range))):
                    break

                target_cls = int(self._rng.integers(0, NUM_CLASSES))
                eeg_data = self._generate_synthetic_eeg(target_cls)
                cmd_idx, confidence = self.model.predict(eeg_data)

                self._last_command = COMMANDS[cmd_idx]
                self._confidence = confidence
                if not self._is_running or self._stop_event.is_set():
                    break

                if _logging_streams_available():
                    logger.info(
                        "🧠 [NEURAL] Thought Decoded: %s (Conf: %.2f)",
                        self._last_command,
                        self._confidence,
                    )

                if self._event_bus:
                    try:
                        loop = getattr(self, "_main_loop", None)
                        if loop and not loop.is_closed():
                            asyncio.run_coroutine_threadsafe(
                                self._event_bus.publish(
                                    "core/senses/bci_event",
                                    {
                                        "command": self._last_command,
                                        "confidence": self._confidence,
                                        "timestamp": time.time(),
                                        "type": "SIMULATED_NEURAL_DECODE",
                                    },
                                ),
                                loop,
                            )
                        else:
                            logger.debug("⚠️ [NEURAL] Main loop unavailable. Skipping broadcast.")
                    except Exception as loop_error:
                        record_degradation('neural_bridge', loop_error)
                        logger.debug("Neural loop broadcast failure: %s", loop_error)
            except Exception as exc:
                record_degradation('neural_bridge', exc)
                logger.error("Neural loop error: %s", exc)
                if self._stop_event.wait(timeout=1.0):
                    break

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_running": self._is_running,
            "last_thought": self._last_command,
            "confidence": self._confidence,
            "channel_count": NUM_CHANNELS,
            "lightweight_mode": self._lightweight_mode,
        }
