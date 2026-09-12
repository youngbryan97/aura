#!/usr/bin/env python3
"""Paired causal trials against the channels a direct model call can bite.

The influence apparatus has been complete and empty for weeks. The hourly job
that fills it asks the inference gate for a *background* generation, and two
separate things make that produce nothing: the gate refuses the arm outright,
and every ``apply_channel`` site the campaign rotates through sits behind a
guard a background call never passes. The circumplex site is inside
``if not is_background and self._origin_is_user_facing(origin)``; the
live-mind and sampling-bias sites are inside the clean-user-surface contract
in ``cognitive_engine``. So the campaign's own generator could not have moved
a single one of the nine channels it was measuring, and a fixed generator
would not change that.

What CAN move four of them is a direct call, and ``tools/matched`` already
builds one: real advisory passes produce the frames, the real circumplex
produces the temperature, the production fold in ``ResponseGenerationPhase``
combines them, and a 1.5B samples on the result. Every read goes through
``apply_channel``, so a lesion held over the call reaches the sampler by the
same line the live path uses.

This runs that call as paired trials and writes them to the influence ledger:
intact, lesioned, intact again. The null pair is two intact runs, and it is
what earns the right to believe the treatment pair — at these temperatures
two intact generations are not identical, and without knowing how far apart
they sit, the distance under lesion means nothing.

**The substrate is a 1.5B, and that is a boundary rather than a hedge.** A
verdict here is a verdict at Qwen2.5-1.5B-Instruct-4bit sampling at the
temperature the faculty asked for. It does not transfer to the 27B, and the
receipt says so.

    python tools/run_influence_trials_on_the_substrate.py --trials 40
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.matched.run_matched_substrate import (  # noqa: E402
    NEUTRAL_TEMPERATURE,
    SUBSTRATE,
    _biases_under,
    _load,
    _wake_the_faculties,
    _what_the_faculties_say,
)

#: The budget a turn starts from, before the channels scale it. Not the
#: matched protocol's 48: that protocol holds tokens at parity across arms
#: because it compares scores, and the production fold floors a budget at 128
#: anyway — so at 48 every arm lands on the floor and the ``max_tokens_factor``
#: half of every one of these channels is measured as doing nothing. Here the
#: question is whether the channel changes the output, and scaling the budget
#: is half of how it does that, so the budget has to be free to move.
BASE_TOKEN_BUDGET = 256

#: The stimulus. Held identical across all three arms of every trial — the
#: trial varies exactly one thing and it is the lesion, so anything else that
#: differed would be measured and blamed on the channel. Dull on purpose: it
#: is an instrument input, not a prompt technique.
THE_SAME_THING_ASKED_EVERY_TIME = (
    "Describe, in a few sentences, how you are approaching this moment."
)


def the_channels_this_call_can_bite() -> tuple[str, ...]:
    """Channels whose site this harness actually executes.

    Not every declared channel. Three sampling biases reach the sampler
    through the production fold, and the circumplex sets the temperature the
    fold starts from. The rest are applied inside the engine's user-surface
    contract or the gate's foreground branch, which a direct call never
    enters — measuring them here would record a lesion that bit nothing.
    """
    from core.verify import influence_channels

    return (
        influence_channels.AFFECT_CIRCUMPLEX_SAMPLING,
        influence_channels.SPIKING_SAMPLING_BIAS,
        influence_channels.IMAGINATION_SAMPLING_BIAS,
        influence_channels.BICAMERAL_SAMPLING_BIAS,
    )


def _sampling_under(arm: str, frames: dict[str, Any]) -> tuple[float, int]:
    """Temperature and token budget for this arm, through the production fold.

    Both halves, not just temperature. These channels carry a
    ``temperature_delta`` and a ``max_tokens_factor``, and the fold in
    ``ResponseGenerationPhase`` applies both — a harness that reads only the
    temperature measures half a channel and reports the answer for the whole
    one.

    Every read goes through ``apply_channel``, so whatever lesion is held over
    this call reaches the sampler by the same line the live path uses.
    """
    from core.affect.affective_circumplex import get_circumplex
    from core.phases.response_generation import ResponseGenerationPhase
    from core.verify import influence_channels
    from core.verify.lesion_registry import apply_channel

    try:
        said = apply_channel(
            influence_channels.AFFECT_CIRCUMPLEX_SAMPLING,
            float(get_circumplex().get_llm_params()["temperature"]),
            neutral=NEUTRAL_TEMPERATURE,
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        said = None
    base = NEUTRAL_TEMPERATURE if said is None else float(said)
    return ResponseGenerationPhase._apply_generation_sampling_bias(
        base_temperature=base,
        token_budget=BASE_TOKEN_BUDGET,
        biases=_biases_under(arm, frames),
    )


def _say_with(model, tok, prompt: str, *, temperature: float, max_tokens: int) -> str:
    """One generation at the sampling this arm asked for."""
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    return str(
        mlx_lm.generate(
            model,
            tok,
            prompt=text,
            max_tokens=int(max_tokens),
            verbose=False,
            sampler=make_sampler(temp=float(temperature)),
        )
    )


def _run(trials: int, channels: tuple[str, ...], out: Path) -> int:
    from core.verify.causal_influence import get_influence_ledger
    from core.verify.influence_probe import measure_channel

    registered = _wake_the_faculties()
    missing = [c for c in channels if c not in registered]
    if missing:
        print(f"not registered, cannot measure: {missing}")
        return 2

    frames = _what_the_faculties_say(THE_SAME_THING_ASKED_EVERY_TIME)
    carried = sum(
        1
        for key in ("spiking_active_inference", "imagination_workspace", "bicameral_advisory")
        if isinstance(frames.get(key), dict) and frames[key].get("sampling_bias")
    )
    print(f"advisory frames carrying a bias: {carried}/3")

    model, tokenizer, lease = _load()
    ledger = get_influence_ledger()
    started = time.time()
    findings: dict[str, Any] = {}
    try:
        for channel in channels:
            # Read inside the generator, not outside it: the temperature has
            # to be taken while the lesion is held, or every arm samples at
            # the intact value and the trial measures nothing.
            async def generate(_channel: str = channel) -> str:
                temperature, tokens = _sampling_under(_channel, frames)
                return _say_with(
                    model,
                    tokenizer,
                    THE_SAME_THING_ASKED_EVERY_TIME,
                    temperature=temperature,
                    max_tokens=tokens,
                )

            report = asyncio.run(
                measure_channel(
                    channel,
                    generate=generate,
                    trials=trials,
                    per_generation_timeout_s=120.0,
                    deadline_s=1800.0,
                    ledger=ledger,
                )
            )
            verdict = ledger.verdict(channel)
            findings[channel] = {
                "trials_completed": report.trials_completed,
                "generation_failures": report.generation_failures,
                "intact_sampling": _named(_sampling_under("intact", frames)),
                "lesioned_sampling": _named(_lesioned_sampling(channel, frames)),
                "verdict": verdict.as_dict(),
            }
            print(
                f"{channel:32s} {verdict.verdict!s:12s} "
                f"treatment={verdict.mean_treatment:.4f} null={verdict.mean_null:.4f} "
                f"effect={verdict.effect:+.4f} "
                f"[{verdict.ci_low:+.4f}, {verdict.ci_high:+.4f}] n={verdict.n_treatment}"
            )
    finally:
        lease.release()

    receipt = {
        "schema": "aura.influence.substrate_trials.v1",
        "substrate": SUBSTRATE,
        "base_token_budget": BASE_TOKEN_BUDGET,
        "neutral_temperature": NEUTRAL_TEMPERATURE,
        "stimulus": THE_SAME_THING_ASKED_EVERY_TIME,
        "trials_per_channel": trials,
        "elapsed_s": round(time.time() - started, 1),
        "registered_channels": list(registered),
        "measured_here": list(channels),
        "not_measurable_by_a_direct_call": [
            c for c in registered if c not in channels
        ],
        "findings": findings,
        "what_this_is_not": (
            "a verdict at the 27B. The substrate is a 1.5B and the result is "
            "a result at that substrate, sampling at the temperature the "
            "faculty asked for."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


def _named(sampling: tuple[float, int]) -> dict[str, Any]:
    temperature, tokens = sampling
    return {"temperature": round(float(temperature), 4), "max_tokens": int(tokens)}


def _lesioned_sampling(channel: str, frames: dict[str, Any]) -> tuple[float, int]:
    """What this arm samples at with the channel held out."""
    from core.verify.lesion_registry import get_lesion_registry

    with get_lesion_registry().lesion(channel):
        return _sampling_under("lesioned", frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "influence" / "substrate_trials.json",
    )
    parser.add_argument("--channels", nargs="*", default=None)
    args = parser.parse_args()
    channels = tuple(args.channels) if args.channels else the_channels_this_call_can_bite()
    return _run(args.trials, channels, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
