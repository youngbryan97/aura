#!/usr/bin/env python3
"""Where her decoding speed goes: her model decoded plainly, then with each thing her worker adds to every token.

LIVE 20 October 2026: her 27B decodes about 16 to 17 tokens a second when it is run plainly (L09 attempt 3's step 1,
her app down), about 12 in her app alone, and 8 to 10 while she watches something. Her worker adds to every token:
logits processors (her endogenous vocabulary bias, her one-shot memory's recall, a tap that reads each token's hidden
state), and her steering hooks on the residual stream. This decodes one fixed prompt, greedily, the same number of
tokens each time, plainly and then with each of those built the worker's own way, and reports tokens a second.

Run with her app down: two copies of her model do not fit beside each other.

    her_decode_with_each_channel.py --output FILE [--tokens 256] [--model DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROMPT = "Tell me how the bicycle changed the lives of ordinary people in the 1890s, in four paragraphs."
LONG_PROMPT = ("Earlier today we talked about old machines, how people first rode them, what the roads were like, "
               "who could afford one, what doctors and newspapers said about them, and why women's clothing changed. "
               "We also talked about the factories that made them and the clubs that formed around them.")


def _decode(model: Any, tokenizer: Any, ids: list[int], tokens: int, processors: list[Any], *, head: Any = None,
            spoken: list[int] | None = None, margins: list[float] | None = None) -> dict[str, Any]:
    """Decode ``tokens`` tokens greedily; tokens a second from the first token to the last (prefill left out). With
    ``head``, her own head drafts (core/brain/llm/her_head_in_the_stream.py); the tokens go into ``spoken``."""
    import mlx.core as mx

    from core.brain.llm.her_head_in_the_stream import stream_generate_with_her_head

    times: list[float] = []
    drafted = 0
    began = time.perf_counter()
    for response in stream_generate_with_her_head(model, tokenizer, ids, max_tokens=tokens, draft_model=head,
                                                  logits_processors=processors or None,
                                                  sampler=lambda logits: mx.argmax(logits, axis=-1)):
        times.append(time.perf_counter())
        drafted += int(bool(response.from_draft))
        if spoken is not None:
            spoken.append(int(response.token))
        if margins is not None:  # how far her chosen token led the next: a near tie is where rounding can part two runs
            top = mx.sort(response.logprobs.reshape(-1))[-2:]
            margins.append(round(float(top[1] - top[0]), 4))
    decoded = len(times) - 1
    # The prompt read, from the start to the first token: its tokens a second.
    read_s = times[0] - began if times else 0.0
    return {"tokens": len(times), "tokens_per_second": round(decoded / (times[-1] - times[0]), 2) if decoded > 0 else 0.0,
            "prompt_tokens": len(ids), "prompt_read_per_second": round(len(ids) / read_s, 1) if read_s > 0 else 0.0,
            **({"from_her_head": drafted} if head is not None else {})}


def _alike(without: list[int], with_head: list[int], margins: list[float] | None = None) -> dict[str, Any]:
    """Whether her head's tokens are those decoded without it, where they first part when not, and by how much her
    chosen token led the next there, decoded one at a time (in nats)."""
    parted = next((i for i, (a, b) in enumerate(zip(without, with_head, strict=False)) if a != b), None)
    lead = margins[parted] if margins and parted is not None and parted < len(margins) else None
    return {"same_tokens": parted is None and len(without) == len(with_head), "first_parted_at": parted,
            "lead_where_parted": lead, "smallest_lead_before": min(margins[:parted]) if margins and parted else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--with-her-head", action="store_true", help="also decode with her own head drafting")
    parser.add_argument("--model", type=Path,
                        default=Path("~/.aura/models/Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15"))
    args = parser.parse_args()

    from mlx_lm import load

    from core.brain.llm.chat_format import reasoning_effort_for_generation, render_chat_template
    from core.runtime.model_lane_control import standalone_model_lane

    model_path = args.model.expanduser().resolve(strict=True)
    rows: list[dict[str, Any]] = []

    def record(name: str, outcome: dict[str, Any]) -> None:
        rows.append({"with": name, **outcome})
        print(json.dumps(rows[-1]), flush=True)

    with standalone_model_lane(owner_id="her-decode-with-each-channel", model_path=str(model_path), purpose="evaluation",
                               preemptible=False, require_exclusive=True, allow_owner_eviction=True,
                               metadata={"tool": Path(__file__).name}):
        model, tokenizer = load(str(model_path))
        text = render_chat_template(tokenizer, [{"role": "user", "content": PROMPT}], add_generation_prompt=True,
                                    enable_thinking=False, reasoning_effort=reasoning_effort_for_generation(thinking=False))
        ids = [int(t) for t in tokenizer.encode(text, add_special_tokens=False)]
        # A prompt the length of one of her turns (about 1,500 tokens) for the reading, as well as the short one.
        long_text = render_chat_template(tokenizer, [{"role": "user", "content": (LONG_PROMPT + "\n\n") * 12 + PROMPT}],
                                         add_generation_prompt=True, enable_thinking=False,
                                         reasoning_effort=reasoning_effort_for_generation(thinking=False))
        long_ids = [int(t) for t in tokenizer.encode(long_text, add_special_tokens=False)]
        _decode(model, tokenizer, ids, 16, [])  # warm: the first decode compiles what the rest reuse
        record("nothing added", _decode(model, tokenizer, ids, args.tokens, []))
        record("a processor that does nothing", _decode(model, tokenizer, ids, args.tokens, [lambda _tokens, logits: logits]))
        record("nothing added, a turn-length prompt", _decode(model, tokenizer, long_ids, 64, []))

        job = {"principal": "bryan", "origin": "desktop_quick_user"}
        endogenous: list[Any] = []
        built: Any = None
        try:
            import numpy as np

            from core.brain.llm.endogenous_decode import JOB_STATE_KEY, install_endogenous_processor
            from core.brain.llm.endogenous_state import STATE_DIM, EndogenousState

            # Its cost does not depend on what the state says: every dimension present, at the middle.
            state = EndogenousState(values=np.full(STATE_DIM, 0.5, dtype=np.float32), present=np.ones(STATE_DIM, dtype=bool))
            processors: list[Any] = []
            receipt, fault = install_endogenous_processor(tokenizer, {**job, JOB_STATE_KEY: state.to_payload()}, processors)
            endogenous = processors
            record(f"her endogenous bias ({receipt.get('reason') or 'installed'})",
                   _decode(model, tokenizer, ids, args.tokens, processors) if processors else {"skipped": fault or True})
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as why:
            record("her endogenous bias", {"skipped": str(why)})

        try:
            from core.brain.nonparametric_worker import last_recall_outcome, maybe_build_foreground

            built = maybe_build_foreground(model, job=job)
            if built is None:
                record("her one-shot memory", {"skipped": last_recall_outcome()})
            else:
                tap, processor = built
                with tap:
                    record("her one-shot memory (tap and recall)", _decode(model, tokenizer, ids, args.tokens, [processor]))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as why:
            record("her one-shot memory", {"skipped": str(why)})

        try:
            import multiprocessing

            from core.brain.llm import mlx_worker
            from core.consciousness import phi_residual_channel, steering_channel

            # The channels the live worker is given: her substrate's state in, the residual ring out to PhiCore.
            spawn = multiprocessing.get_context("spawn")
            substrate, ring = steering_channel.create_channel(spawn), phi_residual_channel.create_channel(spawn)
            steering_channel.publish(substrate)
            attachment = mlx_worker._attach_affective_steering(model, tokenizer, substrate, ring, spawn.Value("b", False, lock=False),
                                                               model_path=str(model_path))
            engine = attachment.engine
            if engine is None or not engine.active_hooks():
                record("her steering hooks", {"skipped": str(attachment.disposition)})
            else:
                record(f"her steering hooks and the residual ring ({len(engine.active_hooks())}, alpha {float(engine._alpha):.2f})",
                       _decode(model, tokenizer, ids, args.tokens, []))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError) as why:
            record("her steering hooks", {"skipped": f"{type(why).__name__}: {why}"})

        # All of them at once, as her worker decodes a turn.
        head = None
        if args.with_her_head:
            from core.brain.llm.her_head_in_the_stream import HER_HEAD
            from core.brain.llm.two_tokens_a_step import load_mtp_head

            head = load_mtp_head(model, HER_HEAD)
        if built is not None:
            tap, processor = built
            with tap:
                without: list[int] = []
                leads: list[float] = []
                record("all of them together", _decode(model, tokenizer, ids, args.tokens, [*endogenous, processor], spoken=without,
                                                       margins=leads))
                record("all of them together, a turn-length prompt", _decode(model, tokenizer, long_ids, 64, [*endogenous, processor]))
                if head is not None:
                    with_head: list[int] = []
                    outcome = _decode(model, tokenizer, ids, args.tokens, [*endogenous, processor], head=head, spoken=with_head)
                    record("all of them together, her head drafting", {**outcome, **_alike(without, with_head, leads)})
                    record("all of them together, her head drafting, a turn-length prompt",
                           _decode(model, tokenizer, long_ids, 64, [*endogenous, processor], head=head))
        else:
            record("all of them together (no one-shot memory)", _decode(model, tokenizer, ids, args.tokens, endogenous))
            record("all of them together, a turn-length prompt", _decode(model, tokenizer, long_ids, 64, endogenous))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"model": str(model_path), "prompt": PROMPT, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
