"""How likely each candidate next turn is after a bare transcript, on her fast model.

core/together/what_was_proposed.py reads what an utterance asks of her from
which answers her own model expects. That needs likelihoods, and a generation
gives one sampled string. This holds her fast model in this process and runs
one forward pass over the transcript with every candidate appended: no
sampling, no decoding, no text written, and nothing told to the model. The
transcript is the one utterance in the bare chat markup with no system
message; the fast model's own template would write a default one.

The model is small (her reflex lane, about a gigabyte) and is loaded the first
time it is needed, on a thread of its own so the event loop never waits on
it, and only when the host has room: free memory of at least twice the
weights, once for the weights and once for the pass. `AURA_EAR=off` leaves it
unloaded, and then nothing is read.

It does not load in the desktop's parent process, which keeps MLX on the CPU
and leaves model work to the worker that owns the lane. On the live desktop
the same pass is a job of that worker (`as_a_job_of_the_worker`), asked for
through the client the turn already holds (`asked_of_the_worker`), so what
is read there is her cortex's expectation. A tool or a test in a process of
its own runs the fast model as written.
"""
from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.flags import FlagKind, declare
from core.runtime.lockdep import checked_lock

_FLAG_EAR = declare("AURA_EAR", kind=FlagKind.STRING, default='',
                    description="off turns off what she expects to hear next", owner="inference")

logger = logging.getLogger("Aura.Brain.LLM.Expects")

__all__ = ["ExpectsNext", "as_a_job_of_the_worker", "asked_of_the_worker", "get_expects_next", "score_next_turns",
           "surprise_of", "transcript_of"]

#: The most candidates one pass scores. The readers that use this score a few
#: second parts for each of a handful of kinds.
MOST_CANDIDATES = 64
#: The most tokens of what came before a stretch of heard words that the stretch is read against, and of the stretch.
#: Surprise read over the next several words tracks what people call surprising, and over only the next word does not
#: (Zhang 2025), so a stretch is read whole, up to this.
MOST_BEFORE_TOKENS, MOST_THEN_TOKENS = 192, 48
#: What a reading of surprise holds.
SURPRISE_READS = ("surprisal", "shift", "open", "tokens")


def transcript_of(said: str) -> str:
    """One utterance and her turn opened after it, with no system message."""
    return f"<|im_start|>user\n{' '.join(str(said or '').split())}<|im_end|>\n<|im_start|>assistant\n"


def score_next_turns(model: Any, tokenizer: Any, said: str, candidates: Sequence[str]) -> list[float]:
    """The log-likelihood of each candidate as the next turn after `said`, in one forward pass."""
    import mlx.core as mx

    prefix = list(tokenizer.encode(transcript_of(said), add_special_tokens=False))
    rows = [list(tokenizer.encode(str(c), add_special_tokens=False)) for c in candidates[:MOST_CANDIDATES]]
    if not prefix or not rows or any(not row for row in rows):
        return []
    longest = max(len(row) for row in rows)
    # Padded on the right: attention runs left to right, so what follows a
    # candidate cannot reach back into it.
    padded = [row + [row[-1]] * (longest - len(row)) for row in rows]
    logits = model(mx.array([prefix + row for row in padded]))
    # Only the positions that predict a candidate's tokens are read, and the
    # whole answer comes back as one small array: one evaluation a pass,
    # where reading it a number at a time was a hundred.
    start = len(prefix) - 1
    at = logits[:, start:start + longest, :]
    logprobs = at - mx.logsumexp(at, axis=-1, keepdims=True)
    chosen = mx.take_along_axis(logprobs, mx.array(padded)[..., None], axis=-1)[..., 0]
    real = mx.array([[1.0] * len(row) + [0.0] * (longest - len(row)) for row in rows])
    return [float(total) for total in mx.sum(chosen * real, axis=-1).tolist()]


def surprise_of(model: Any, tokenizer: Any, before: str, then: str) -> dict[str, float]:
    """How what was heard (``then``) landed on what she expected after what came before it, in one forward pass over
    the plain words (no chat markup: it is what was heard, not a turn of anybody's).

    - ``surprisal``: how unexpected its words were, in nats a token.
    - ``shift``: how much each of its words moved her expectation of the next, the mean divergence (KL) of each word's
      prediction of what follows from the one before it. People mark the boundaries of what they watch where this
      rises, and not where words are merely unlikely (Kumar, Zacks, Norman and Hasson); it is the surprise that draws
      the eye (Itti and Baldi 2005).
    - ``open``: how open what comes next is once it has been heard (entropy, nats): what suspense is made of (Ely,
      Frankel and Kamenica 2015).

    Nothing when either side is too short to read."""
    import mlx.core as mx

    context = list(tokenizer.encode(" ".join(str(before or "").split()), add_special_tokens=False))[-MOST_BEFORE_TOKENS:]
    heard = list(tokenizer.encode(" " + " ".join(str(then or "").split()), add_special_tokens=False))[:MOST_THEN_TOKENS]
    if len(context) < 2 or len(heard) < 2:
        return {}
    logits = model(mx.array([context + heard]))[0]
    # From the last word before it, which predicts its first, to its last, which predicts what follows it.
    rows = logits[len(context) - 1:].astype(mx.float32)
    logprobs = rows - mx.logsumexp(rows, axis=-1, keepdims=True)
    picked = mx.take_along_axis(logprobs[:-1], mx.array(heard)[:, None], axis=-1)[:, 0]
    probs = mx.exp(logprobs)
    shifts = mx.sum(probs[1:] * (logprobs[1:] - logprobs[:-1]), axis=-1)
    surprisal, shift, entropy = -mx.mean(picked), mx.mean(shifts), -mx.sum(probs[-1] * logprobs[-1])
    mx.eval(surprisal, shift, entropy)
    return {"surprisal": float(surprisal.item()), "shift": float(shift.item()), "open": float(entropy.item()),
            "tokens": float(len(heard))}


def as_a_job_of_the_worker(job: Any, model: Any, tokenizer: Any, engine: Any, metal: Any) -> dict[str, Any]:
    """The same pass as a job of the worker that owns the model lane.

    The job's `expect_next` is what was said followed by the candidates; its
    `surprise_of` is what came before and what was then heard (`surprise_of`).
    The pass runs with her steering stood down, since what is read is what the
    model expects and not what she feels, and with nothing sampled. The
    answer is the scores as JSON in the place a generation's text goes.
    """
    asked = [str(item) for item in (job.get("expect_next") or ())]
    surprise = [str(item) for item in (job.get("surprise_of") or ())]
    response: dict[str, Any] = {
        "id": job.get("id"), "action": "generate", "status": "ok", "text": "[]", "tokens_used": 0,
        "prompt_tokenization": {"chars": 0, "tokens": 0, "generated_tokens": 0}, "generation_performance": {},
    }
    if len(asked) < 2 and len(surprise) != 2:
        return response
    try:
        with contextlib.ExitStack() as held:
            held.enter_context(metal)
            if engine is not None and hasattr(engine, "controlled_measurement"):
                held.enter_context(engine.controlled_measurement())
                engine.set_alpha(0.0)
            if len(surprise) == 2:
                # As a list in SURPRISE_READS order, the shape the other pass answers in: a reply's text that is an
                # object can be read as a call to something.
                read = surprise_of(model, tokenizer, surprise[0], surprise[1])
                response["text"] = json.dumps([read[key] for key in SURPRISE_READS] if read else [])
            else:
                response["text"] = json.dumps(score_next_turns(model, tokenizer, asked[0], asked[1:]))
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as exc:
        response.update({"status": "error", "message": f"{type(exc).__name__}: {exc}"})
    return response


class _AsksWhatSheExpectsNext:
    """The client's side of the job: a part of MLXLocalClient, kept here with the pass it asks for."""

    async def expect_next(self, said: str, candidates: Sequence[str]) -> list[float]:
        """The worker's scores for each candidate as her next turn after `said`, or nothing."""
        try:
            raw = await self.generate_text_async(  # type: ignore[attr-defined]
                said, expect_next=[said, *candidates], max_tokens=1, temperature=0.0,
                origin="what_it_asks_of_her", internal_inference=True, foreground_request=True,
            )
            scores = json.loads(str(raw or "[]"))
        except (RuntimeError, TypeError, ValueError, OSError, TimeoutError) as exc:
            record_degradation(
                "expects_next", exc, severity="warning",
                action="one utterance was not read for what it asks of her; the worker gave no scores",
            )
            return []
        if not isinstance(scores, list) or len(scores) != len(candidates):
            return []
        return [float(score) for score in scores]

    async def surprised_by(self, before: str, then: str) -> dict[str, float]:
        """The worker's reading of how what was heard (``then``) landed after what came before it, or nothing."""
        try:
            raw = await self.generate_text_async(  # type: ignore[attr-defined]
                then, surprise_of=[before, then], max_tokens=1, temperature=0.0,
                origin="what_moves_her", internal_inference=True, foreground_request=True,
            )
            read = json.loads(str(raw or "[]"))
        except (RuntimeError, TypeError, ValueError, OSError, TimeoutError) as exc:
            record_degradation(
                "expects_next", exc, severity="warning",
                action="a stretch of what she heard was not read for how it landed; the worker gave no reading",
            )
            return {}
        if not isinstance(read, list) or len(read) != len(SURPRISE_READS) or not all(
                isinstance(value, (int, float)) for value in read):
            return {}
        return {key: float(value) for key, value in zip(SURPRISE_READS, read, strict=True)}


async def asked_of_the_worker(client: Any, said: str, candidates: Sequence[str]) -> list[float]:
    """The scores from the worker a turn already holds, or nothing from a client that cannot ask."""
    ask = getattr(client, "expect_next", None)
    return list(await ask(said, candidates)) if callable(ask) else []


class ExpectsNext:
    """Her fast model, held for scoring and nothing else."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._path = Path(model_path) if model_path else None
        self._thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aura-expects-next")
        self._model: Any = None
        self._tokenizer: Any = None
        self._refused = ""
        self._lease: Any = None

    def _where(self) -> Path:
        if self._path is None:
            from core.brain.llm.model_registry import FALLBACK_ENDPOINT, get_lane_runtime_model_path

            self._path = Path(get_lane_runtime_model_path(FALLBACK_ENDPOINT))
        return self._path

    def _has_room(self, path: Path) -> bool:
        import psutil

        weights = sum(item.stat().st_size for item in path.glob("*.safetensors"))
        return weights > 0 and psutil.virtual_memory().available >= 2 * weights

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if self._refused:
            return False
        if _FLAG_EAR.value().strip().lower() == "off":
            self._refused = "switched off"
            return False
        try:
            path = self._where()
            if not self._has_room(path):
                self._refused = "the host has no room for it"
                logger.info("👂 Her fast model is not loaded for scoring: %s.", self._refused)
                return False
            import mlx.core as mx

            if mx.default_device() != mx.gpu:
                # The desktop's parent process keeps MLX on the CPU, and model
                # work there belongs to the worker that owns the lane
                # (core/runtime/desktop_boot_safety.py). Live, 13 October, this
                # loaded there anyway and ground one core for minutes.
                self._refused = "this process keeps MLX off the GPU"
                logger.info("👂 Her fast model is not loaded for scoring: %s.", self._refused)
                return False
            from mlx_lm import load

            from core.runtime.model_lane_control import acquire_synchronous_in_process_model_lane

            # Held through the lane like every model this host loads, so the lane controller counts its memory
            # and can take it back; it is loaded again the next time it is needed.
            self._lease = acquire_synchronous_in_process_model_lane(
                owner_id=f"expects-next:{id(self)}",
                model_path=str(path),
                purpose="serve",
                priority=30,
                preemptible=True,
                evict=self._evict,
                metadata={"engine": "expects_next", "model_role": "reflex_scoring"},
            )
            self._model, self._tokenizer = load(str(path))
            logger.info("👂 Her fast model is loaded for scoring next turns (%s).", path.name)
            return True
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self._refused = f"{type(exc).__name__}: {exc}"
            record_degradation(
                "expects_next", exc, severity="warning",
                action="what an utterance asks of her is not read; her fast model did not load",
            )
            return False

    async def _evict(self, _owner: Any, reason: str) -> bool:
        """Give the model back to the lane; it loads again when it is next needed."""
        import asyncio

        self._model = self._tokenizer = None
        lease, self._lease = self._lease, None
        release = getattr(lease, "release", None)
        if callable(release):
            await asyncio.to_thread(release, reason=f"expects_next_eviction:{reason}")
        logger.info("👂 Her fast model for scoring was given back to the lane (%s).", reason)
        return True

    def _score(self, said: str, candidates: Sequence[str]) -> list[float]:
        if not self._load():
            return []
        return score_next_turns(self._model, self._tokenizer, said, candidates)

    def score(self, said: str, candidates: Sequence[str]) -> list[float]:
        """The log-likelihood of each candidate as her next turn after `said`, in nats.

        Empty when the model is not loaded. Waits on the model's own thread,
        so call it from a worker thread, never from the event loop.
        """
        try:
            return self._thread.submit(self._score, said, list(candidates)).result()
        except (RuntimeError, TypeError, ValueError, IndexError) as exc:
            record_degradation(
                "expects_next", exc, severity="warning",
                action="one utterance was not read for what it asks of her",
            )
            return []

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self._model is not None,
            "refused": self._refused,
            "model": self._path.name if self._path else "",
        }


_EXPECTS: ExpectsNext | None = None
_EXPECTS_LOCK = checked_lock("brain.llm.expects_next.singleton")


def get_expects_next() -> ExpectsNext:
    global _EXPECTS
    with _EXPECTS_LOCK:
        if _EXPECTS is None:
            _EXPECTS = ExpectsNext()
        return _EXPECTS
