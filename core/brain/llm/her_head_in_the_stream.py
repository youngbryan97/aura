"""Her own next-next-token head, drafting inside her worker's stream: two tokens a pass with everything she has on.

Bryan, 20 October 2026: one person, one Aura; as fast as is physically possible while keeping everything. Decoding
reads all 14 GB of her 4-bit weights for each pass, and a pass that reads two tokens costs about the same as one, so
the way past her model's 17 tokens a second on this machine is fewer passes a token. Her architecture carries a
multi-token prediction head (core/brain/llm/two_tokens_a_step.py ``MTPHead``): from her hidden state at a position
and the next token's embedding it predicts the token after. MEASURED that day on her persona 27B, its drafts were
kept 94 to 98 per cent of the time (``tools/measure_two_tokens_a_step.py``).

mlx_lm's speculative step cannot run her: it needs caches it can trim, and her gated delta net layers keep a running
state that cannot be trimmed. core/brain/llm/several_tokens_a_pass.py reads several tokens keeping each linear
layer's state after every one, and puts back the state at the last kept token. This is that pass in the shape her
worker's stream takes (mlx_lm ``stream_generate`` and its ``GenerationResponse``), with what her generations carry:

* her logits processors, applied to each checked position in order, with the token history mlx_lm gives them (the
  prompt's last token and every token since), so her private channel's bound counts as it does one token at a time;
* her hidden-state tap: before a position's processors run, the tap holds that position's hidden state, so her
  one-shot memory recalls from the token it is choosing after;
* the worker's prompt cache, read on from where it stands;
* her sampler: a draft is kept when the token sampled at its position is the draft, or, where her sampler says what it draws
  from, as often as her distribution and her head's overlap (core/brain/llm/her_sampler.py). Either keeps her distribution
  exactly; greedy is the case of an argmax sampler.

Two reads back from the GPU a pass (the tokens chosen, then the next draft), not one a token.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import mlx.core as mx

from core.brain.llm.her_sampler import kept_or_instead
from core.runtime.state_ownership import state_root

logger = logging.getLogger("LLM.HerHead")

__all__ = ["HER_HEAD", "head_generate_step", "her_head_may_check", "her_own_head", "stream_generate_with_her_head", "timed"]

#: Where her head's weights are: the original release's multi-token prediction layer, installed under her state root
#: (~/.aura unless a run names its own).
HER_HEAD = state_root() / "models" / "qwen3.8-27b-mtp-source" / "mtp_head.safetensors"


#: A draft from her own text: the least and the most tokens it is found by, and how many it is of to begin with and at most.
#: MEASURED 24 November 2026 on her 27B with her app closed, 700 tokens of a mend of a game after a prompt of 5,305 that holds
#: it: 24.8 tokens a second by her head alone, 36.1 with 3, 6, 6 and 16, 41.0 with these, and 39.6 to 41.6 with longer drafts.
TEXT_LEAST, TEXT_LONGEST, TEXT_DEPTH, TEXT_DEEPEST = 4, 8, 8, 24


#: The most tokens her head drafts a pass, and how often a depth beside the one that has been fastest is tried.
HEAD_DEEPEST, TRIED_EVERY, PASSES_HELD = 4, 12, 64


class HowDeep:
    """How many tokens her head drafts a pass: as many as has been fastest, in seconds a token, measured as she writes.

    Her head drafts the token after next from her hidden state and the next token. Its own output stands for the hidden state
    after that, so it drafts again from it, with less to go on each time. A pass that reads one more costs little, and gains a
    token where the draft holds; where it does not, what was read past it is undone. Which wins turns on what she is writing
    and on what else the machine is doing, so it is measured: each depth's seconds a token over its last passes, the fastest
    taken, and one beside it tried now and then."""

    def __init__(self, deepest: int = HEAD_DEEPEST) -> None:
        self.deepest = max(1, deepest)
        self._took: dict[int, tuple[float, float]] = {}
        self._passes = 0

    def a_token(self, depth: int) -> float | None:
        seconds, tokens = self._took.get(depth, (0.0, 0.0))
        return seconds / tokens if tokens else None

    @property
    def fastest(self) -> int:
        measured = {depth: rate for depth in self._took if (rate := self.a_token(depth)) is not None}
        return min(measured, key=measured.__getitem__) if measured else 1

    def next(self) -> int:
        """The depth of the next pass."""
        self._passes += 1
        best = self.fastest
        if self._passes % TRIED_EVERY:
            return best
        beside = [depth for depth in (best + 1, best - 1) if 1 <= depth <= self.deepest]
        # The one measured least lately is tried: a depth never measured before any other.
        return min(beside, key=lambda depth: self._took.get(depth, (0.0, 0.0))[1], default=best)

    def came(self, depth: int, tokens: int, seconds: float) -> None:
        """A pass of this depth gave so many tokens in so long. What came before it counts for a little less."""
        held_seconds, held_tokens = self._took.get(depth, (0.0, 0.0))
        keeps = 1.0 - 1.0 / PASSES_HELD
        self._took[depth] = (held_seconds * keeps + max(0.0, seconds), held_tokens * keeps + max(0, tokens))

    def says(self) -> str:
        return ", ".join(f"{depth}: {1.0 / rate:.1f}/s" for depth in sorted(self._took) if (rate := self.a_token(depth)))


#: What has been fastest, kept from one generation to the next.
_HOW_DEEP = HowDeep()

#: The token of a generation whose pass is looked into, where that is asked for.
LOOKED_INTO_AT = 60


def _looks_into_it() -> bool:
    from core.runtime.flags import env_str

    return env_str("AURA_LOOK_INTO_DECODE", default="", owner=__name__, description="Log what building one pass of each generation calls").strip().lower() in {"1", "true", "yes", "on"}


#: One position in this many is timed step by step, with the GPU waited for after each, and no other waits for it.
TIMED_EVERY = 40


class WhereItGoes:
    """Where a generation's time goes: reading a pass, choosing at each position, drafting; and at a choice, each of what is done
    to her logits. LIVE 25 November 2026 a pass of hers that read one token more took 26 ms more, and nothing said of what."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.times: dict[str, int] = {}
        #: How many drafts were looked at in each place of a pass, and how many of those were kept.
        self.drafts: dict[int, list[int]] = {}
        self._looked = False

    def spent(self, on: str, seconds: float) -> None:
        self.seconds[on] = self.seconds.get(on, 0.0) + seconds
        self.times[on] = self.times.get(on, 0) + 1

    @contextmanager
    def looked_into(self, made: int) -> Iterator[None]:
        """What building a pass calls, and how long each takes, for one pass of a generation where AURA_LOOK_INTO_DECODE asks."""
        if made < LOOKED_INTO_AT or self._looked or not _looks_into_it():
            yield
            return
        self._looked = True
        import cProfile
        import io
        import pstats

        profile, said = cProfile.Profile(), io.StringIO()
        profile.enable()
        try:
            yield
        finally:
            profile.disable()
            pstats.Stats(profile, stream=said).sort_stats("tottime").print_stats(18)
            logger.info("Building a pass: %s", " | ".join(line.strip() for line in said.getvalue().splitlines() if line.strip())[:4000])

    def draft(self, place: int, kept: bool) -> None:
        held = self.drafts.setdefault(place, [0, 0])
        held[0], held[1] = held[0] + 1, held[1] + kept

    def says(self) -> str:
        kept = "; drafts kept " + ", ".join(f"{100 * kept / of:.0f}% of {of}" for _place, (of, kept) in sorted(self.drafts.items())) if self.drafts else ""
        return kept.removeprefix("; ") + ("; " if kept else "") + ", ".join(f"{on} {1000 * self.seconds[on] / self.times[on]:.1f} ms x{self.times[on]}" for on in sorted(self.seconds, key=lambda on: -self.seconds[on]))


def _switched_on() -> bool:
    """AURA_HER_HEAD_DRAFTS, on unless set to 0. MEASURED 20 October 2026 on her 27B with every channel on
    (tools/her_decode_with_each_channel.py --with-her-head): 15.2 tokens a second without it, 21.5 with it, and her
    tokens the same up to the 59th, where her first and second choices stood 0.125 nats apart."""
    from core.runtime.flags import env_str

    drafts = env_str("AURA_HER_HEAD_DRAFTS", default="1", owner=__name__,
                     description="Her own head drafts tokens in the stream (0/false/no/off to turn it off)")
    return drafts.strip().lower() not in {"0", "false", "no", "off"}


def her_own_head(model: Any) -> Any:
    """Her multi-token prediction head joined to her model, where she has one and it is switched on; else None."""
    if model is None or not _switched_on():
        return None
    text = getattr(model, "language_model", model)
    if getattr(getattr(text, "args", None), "model_type", "") != "qwen3_5_text" or not HER_HEAD.is_file():
        return None
    try:
        from core.brain.llm.two_tokens_a_step import load_mtp_head

        head = load_mtp_head(model, HER_HEAD)
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
        from core.runtime.errors import record_degradation

        record_degradation("her_head_in_the_stream", exc, severity="warning", action="decoded without her head's drafts")
        return None
    logger.info("Speculative decoding: her own head drafts (%s).", HER_HEAD.name)
    return head


def of_what_was_said(processor: Any) -> bool:
    """Whether what a logits processor gives is of what was said and the logits it is given, and of nothing it keeps."""
    return str(getattr(processor, "__module__", "")).startswith("mlx_lm.") or bool(getattr(processor, "of_what_was_said", False))


def her_head_may_check(draft_model: Any, generation_kwargs: dict) -> bool | None:
    """Whether her head may draft for these generation arguments; None when the draft is not her head. Her pass keeps
    her logits processors, her tap and the prompt cache, which another draft model's could not."""
    from core.brain.llm.two_tokens_a_step import MTPHead

    if not isinstance(draft_model, MTPHead):
        return None
    return generation_kwargs.get("kv_bits") is None


def head_generate_step(prompt: mx.array, model: Any, head: Any, *, max_tokens: int = 256,
                       sampler: Callable[[mx.array], mx.array] | None = None,
                       logits_processors: Sequence[Callable[[Any, Any], Any]] | None = None,
                       prompt_cache: list | None = None, prefill_step_size: int = 512,
                       prompt_progress_callback: Callable[[int, int], None] | None = None,
                       tap: Any = None, drafts_from: Sequence[int] | None = None, how_deep: HowDeep | None = None,
                       goes: WhereItGoes | None = None, **_ignored: Any) -> Generator[tuple[int, mx.array, bool], None, None]:
    """Her tokens for ``prompt``, a pass reading the last kept token and her head's draft after it; and, where
    ``drafts_from`` is given (the whole prompt), what followed the same tokens before in it or in what she has written,
    wherever that begins with her head's own draft. A program mended is mostly the program again.

    Yields ``(token, logprobs, from_draft)`` as mlx_lm's speculative step does. A position's processors run only
    once the positions before it are kept: they keep state (her private channel's count, her memory's last entry)."""
    from mlx_lm.models.cache import KVCache, make_prompt_cache

    from core.brain.llm.several_tokens_a_pass import _logits, _read, _text, _TextIndex, _unread
    from core.consciousness.spoken_positions import speaking_several

    text = _text(model)
    how_deep = how_deep if how_deep is not None else _HOW_DEEP
    # Her worker's hidden-state tap, where it is in: its proxy around her backbone holds it (nonparametric_worker).
    tap = tap if tap is not None else getattr(text.model, "_tap", None)
    embed = text.model.embed_tokens
    cache = prompt_cache if prompt_cache is not None else make_prompt_cache(model)
    head_cache = KVCache()
    processors = list(logits_processors or [])
    sampler = sampler or (lambda logprobs: mx.argmax(logprobs, axis=-1))
    # Where her sampler says what it draws from, her head's drafts are drawn as hers are and kept by how the two overlap.
    of = sampler.probabilities if getattr(sampler, "draws", False) else None
    y = prompt.astype(mx.uint32)
    total, done = int(y.size), 0
    if prompt_progress_callback is not None:
        prompt_progress_callback(0, total)
    # The prompt but its last token, in steps, as mlx_lm prefills; her head reads each step beside her model, the
    # hidden state at each position with the token after it, so its own attention has the prompt.
    while total - done > 1:
        n = min(int(prefill_step_size or 512), total - done - 1)
        read = text.model(y[done:done + n][None], cache)
        head(read, embed(y[done + 1:done + n + 1][None]), head_cache)
        mx.eval([c.state for c in cache] + [c.state for c in [head_cache]])
        done += n
        if prompt_progress_callback is not None:
            prompt_progress_callback(done, total)
    history = y[done:]  # what mlx_lm hands its processors first: the prompt's tail
    hiddens: Any = None

    goes, positions = goes if goes is not None else WhereItGoes(), 0

    def choose(row: mx.array, at: int, draft: int | None = None, drawn_from: mx.array | None = None) -> tuple[int, mx.array]:
        """Her token at a position: her processors on its logits (shape [1, vocab]), then her sampler; where the ``draft`` there
        was drawn from a distribution, that draft or what is said instead of it."""
        nonlocal positions
        if tap is not None and hiddens is not None:
            tap.last_hidden, tap.of_a_pass = hiddens[at], (hiddens, at)
        positions += 1
        timed_here = positions % TIMED_EVERY == 0
        if timed_here:
            mx.eval(row)
        began = time.perf_counter()
        for processor in processors:
            row = processor(history, row)
            if timed_here:
                mx.eval(row)
                goes.spent(f"one {type(processor).__name__}", time.perf_counter() - began)
                began = time.perf_counter()
        logprobs = row - mx.logsumexp(row, axis=-1, keepdims=True)
        if of is not None and draft is not None and drawn_from is not None:
            chosen, _kept = kept_or_instead(of(logprobs.astype(mx.float32)), drawn_from, mx.array([draft]))
        else:
            chosen = sampler(logprobs)
        mx.eval(chosen)
        token = int(chosen.reshape(-1)[0].item())
        goes.spent("one sampling" if timed_here else "choosing", time.perf_counter() - began)
        return token, logprobs.squeeze(0)

    def kept(token: int) -> None:
        nonlocal history
        history = mx.concatenate([history, mx.array([token], dtype=history.dtype)])

    # What is done to her logits that is of what was said and of them and nothing else is done to her head's as well: where she
    # is kept from saying a thing again and her head is not, the two overlap least on the words that come most. The rest keeps
    # count of what it is given, and is given what she says.
    to_both = [processor for processor in processors if of_what_was_said(processor)]

    def drafted_after(hidden: mx.array, following: list[int], many: int = 1) -> tuple[list[int], list[mx.array]]:
        """Her head's drafts after what was kept, and what each was drawn from where it was drawn: one from her hidden state, and
        each after it from her head's own."""
        drawn_from: list[mx.array] = []
        said = mx.concatenate([history, mx.array([following[-1]], dtype=history.dtype)])

        def drafted(logits: mx.array) -> mx.array:
            nonlocal said
            if of is None:
                return mx.argmax(logits, axis=-1)
            logits = logits.astype(mx.float32)
            for processor in to_both:
                logits = processor(said, logits)
            drawn_from.append(of(logits))
            draft = mx.random.categorical(mx.log(drawn_from[-1]), axis=-1)
            said = mx.concatenate([said, draft.astype(said.dtype)])
            return draft

        out = head(hidden, embed(mx.array([following])), head_cache)
        drafts = [drafted(_logits(text, out[:, -1, :]))]
        for _ in range(many - 1):
            out = head(out[:, -1:], embed(drafts[-1][:, None]), head_cache)
            drafts.append(drafted(_logits(text, out[:, -1, :])))
        told = mx.concatenate(drafts).tolist()
        # What it drafted from its own output is not what she says: her head reads that from her hidden state when it is said.
        if many > 1:
            head_cache.trim(many - 1)
        return [int(one) for one in told], drawn_from

    # First pass: the prompt's last token gives her first token, and her head its first draft.
    hidden = text.model(history[None], cache)
    hiddens = hidden[0] if tap is not None else None
    token, logprobs = choose(_logits(text, hidden[:, -1, :]), -1)
    deep, began = how_deep.next(), time.perf_counter()
    drafts, drawn = drafted_after(hidden[:, -1:], [token], deep)
    drafting, from_draft, made, depth = time.perf_counter() - began, False, 0, TEXT_DEPTH
    index = None
    if drafts_from is not None:
        index = _TextIndex(TEXT_LONGEST)
        index.extend(drafts_from)
    while True:
        yield token, logprobs, from_draft
        made += 1
        if 0 <= max_tokens <= made:
            return
        kept(token)
        # A pass is timed from drafting it to its last token, less what is done with each token as it is given.
        began, elsewhere, proposal = time.perf_counter(), 0.0, drafts
        if index is not None:
            index.extend([token])
            again = index.draft(depth, least=TEXT_LEAST)
            if again and again[0] == drafts[0]:
                proposal = again
        by_her_head = proposal is drafts
        if max_tokens > 0:
            proposal = proposal[:max_tokens - made]
        building = time.perf_counter()
        with speaking_several(), goes.looked_into(made):  # all are tokens she speaks: steered, and sampled for Φ, as one at a time
            read, saved = _read(text, mx.array([[token, *proposal]]), cache)
        hiddens = read[0] if tap is not None else None
        logits = _logits(text, read)
        reading = time.perf_counter()
        goes.spent("building", reading - building)
        mx.eval(logits)
        goes.spent(f"reading {1 + len(proposal)}", time.perf_counter() - reading)
        took, whole = 0, 1 + len(proposal)
        try:
            for at, guess in enumerate(proposal):
                checked, checked_logprobs = choose(logits[:, at, :], at, guess, drawn[at] if by_her_head and at < len(drawn) else None)
                if by_her_head:
                    goes.draft(at, checked == guess)
                if checked != guess:
                    token, logprobs = checked, checked_logprobs
                    break
                took += 1
                given = time.perf_counter()
                yield guess, checked_logprobs, True
                elsewhere += time.perf_counter() - given
                made += 1
                if 0 <= max_tokens <= made:
                    return
                kept(guess)
                if index is not None:
                    index.extend([guess])
            else:
                # Every one kept: her token after them comes from the same pass.
                token, logprobs = choose(logits[:, took, :], took)
        finally:
            # What was read past the last token kept is as if it had never been, however this ends.
            undoing = time.perf_counter()
            _unread(cache, saved, whole, 1 + took)
            goes.spent("undoing", time.perf_counter() - undoing)
        from_draft = False
        if by_her_head:
            how_deep.came(deep, 1 + took, drafting + time.perf_counter() - began - elsewhere)
        elif len(proposal) > 1:
            depth = min(TEXT_DEEPEST, depth * 2) if took == len(proposal) else max(2, took + 1)
        deep, began = how_deep.next(), time.perf_counter()
        drafts, drawn = drafted_after(read[:, :1 + took], [*proposal[:took], token], deep)
        drafting = time.perf_counter() - began
        goes.spent(f"drafting {deep}", drafting)


def stream_generate_with_her_head(model: Any, tokenizer: Any, prompt: Any, max_tokens: int = 256,
                                  draft_model: Any = None, **kwargs: Any) -> Generator[Any, None, None]:
    """mlx_lm's ``stream_generate`` where ``draft_model`` is her head; mlx_lm's own for anything else."""
    from mlx_lm import stream_generate
    from mlx_lm.generate import GenerationResponse, generation_stream, wired_limit
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    from core.brain.llm.two_tokens_a_step import MTPHead

    if not isinstance(draft_model, MTPHead):
        kwargs.pop("drafts_from", None)
        yield from stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, draft_model=draft_model, **kwargs)
        return
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            special = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
            prompt = tokenizer.encode(prompt, add_special_tokens=special)
        prompt = mx.array(prompt)
    for unused in ("num_draft_tokens", "max_kv_size"):
        kwargs.pop(unused, None)
    detokenizer = tokenizer.detokenizer
    tic = time.perf_counter()
    prompt_tps, n, token, logprobs, from_draft, goes = 0.0, -1, 0, None, False, WhereItGoes()
    # Her weights held in memory for the generation, as mlx_lm's own stream holds them: unwired, they could be paged
    # out between passes while her app's other processes ran (MEASURED 20 October 2026: 9.9 tokens a second in her
    # worker, 21.5 by itself; prompts read at 145 a second against 390).
    with wired_limit(model, [generation_stream]):
        for n, (token, logprobs, from_draft) in enumerate(head_generate_step(prompt, model, draft_model,
                                                                              max_tokens=max_tokens, goes=goes, **kwargs)):
            if n == 0:
                prompt_tps = prompt.size / max(1e-9, time.perf_counter() - tic)
                tic = time.perf_counter()
            if token in tokenizer.eos_token_ids:
                break
            detokenizer.add_token(token)
            if (n + 1) == max_tokens:
                break
            yield GenerationResponse(text=detokenizer.last_segment, token=token, logprobs=logprobs, from_draft=from_draft,
                                     prompt_tokens=prompt.size, prompt_tps=prompt_tps, generation_tokens=n + 1,
                                     generation_tps=(n + 1) / max(1e-9, time.perf_counter() - tic),
                                     peak_memory=mx.get_peak_memory() / 1e9, finish_reason=None)
    detokenizer.finalize()
    if n > 1:
        logger.info("Her head drafts %d a pass; tokens a second by how many: %s", _HOW_DEEP.fastest, _HOW_DEEP.says())
        logger.info("Where her writing went: %s", goes.says())
    yield GenerationResponse(text=detokenizer.last_segment, token=token, logprobs=logprobs, from_draft=from_draft,
                             prompt_tokens=prompt.size, prompt_tps=prompt_tps, generation_tokens=n + 1,
                             generation_tps=(n + 1) / max(1e-9, time.perf_counter() - tic),
                             peak_memory=mx.get_peak_memory() / 1e9,
                             finish_reason="stop" if token in tokenizer.eos_token_ids else "length")


def timed(responses: Any, each: Callable[[], None]) -> Generator[Any, None, None]:
    """``responses`` passed on with ``each`` called as each comes, and at the end how long the generation waited on the
    model and how long the token loop around it took. LIVE 20 October 2026 her worker decoded at 8 tokens a second
    where her model, every channel and her head's drafts decode at 21.5 by themselves."""
    waited = looped = 0.0
    count = 0
    asked = time.perf_counter()
    try:
        for response in responses:
            came = time.perf_counter()
            waited, count = waited + came - asked, count + 1
            each()
            yield response
            asked = time.perf_counter()
            looped += asked - came
    finally:
        if count > 1:
            logger.info("[WORKER] %d tokens: %.2fs waiting on the model, %.2fs in the token loop around it",
                        count, waited, looped)
