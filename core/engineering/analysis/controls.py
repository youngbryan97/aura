"""Feedback loops: how fast, how steady, and whether it oscillates.

A control loop has three failure modes and they are all visible in two
numbers. Too slow, and it never catches up. Too fast, and it overshoots and
rings. Unstable, and it runs away. Damping ratio and natural frequency say
which one a design is heading for before anything is switched on.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from core.engineering.analysis import Finding, register
from core.engineering.units import Q


@register(
    "loop_response",
    "Control loop response",
    "How quickly does it settle, and does it overshoot?",
    domains=("signal",),
    discipline="controls",
)
def loop_response(design) -> Iterable[Finding]:
    for part in design.parts:
        if "controller" not in part.tags and "loop" not in part.tags:
            continue
        frequency = part.ratings.get("natural_frequency")
        damping = part.ratings.get("damping_ratio")
        if frequency is None or damping is None:
            continue
        wn = float(frequency.as_("rad/s").value) if frequency.unit != "Hz" else (
            2.0 * math.pi * float(frequency.value)
        )
        zeta = float(damping.value)
        if wn <= 0 or zeta < 0:
            continue
        if zeta < 1.0:
            overshoot = math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta * zeta))
            behaviour = (
                f"it overshoots by {overshoot * 100:.0f}% and rings before settling"
            )
        elif zeta == 1.0:
            overshoot = 0.0
            behaviour = "it arrives without overshooting, as fast as it can without ringing"
        else:
            overshoot = 0.0
            behaviour = "it creeps up to the target without overshooting, and takes its time"
        settling = 4.0 / (zeta * wn) if zeta * wn > 0 else float("inf")
        yield Finding(
            id=f"controls.response.{part.id}",
            name=f"Step response of {part.name}",
            value=Q(settling, "s"),
            formula="t_settle = 4 / (zeta omega_n), overshoot = exp(-pi zeta / sqrt(1 - zeta^2))",
            inputs={"omega_n": Q(wn, "rad/s"), "zeta": damping},
            method="Second-order underdamped step response, 2% settling band",
            plain=(
                f"Asked to move to a new target, the "
                f"{part.lay_name or part.name.lower()} settles in "
                f"{Q(settling, 's').text()} and {behaviour}."
            ),
            subject=part.id,
            verdict="pass" if 0.4 <= zeta <= 1.0 else "watch",
            advice=(
                ""
                if 0.4 <= zeta <= 1.0
                else "Add damping, or slow the loop down."
                if zeta < 0.4
                else "Raise the gain; the loop is sluggish for no benefit."
            ),
        )


@register(
    "sensor_resolution",
    "Sensor resolution",
    "How small a change can it actually see?",
    domains=("signal",),
    discipline="controls",
)
def sensor_resolution(design) -> Iterable[Finding]:
    for part in design.parts:
        if "sensor" not in part.tags:
            continue
        span = part.ratings.get("range") or part.ratings.get("full_scale")
        bits = part.ratings.get("bits") or part.ratings.get("resolution_bits")
        if span is None or bits is None:
            continue
        steps = 2 ** int(float(bits.value))
        step = float(span.value) / steps
        noise = part.ratings.get("noise")
        detail = ""
        if noise is not None:
            usable = math.log2(float(span.value) / float(noise.value)) if float(noise.value) > 0 else 0
            detail = (
                f" Noise of {noise.text()} swamps the bottom bits, leaving about "
                f"{usable:.1f} bits that mean anything."
            )
        yield Finding(
            id=f"controls.resolution.{part.id}",
            name=f"Resolution of {part.name}",
            value=Q(step, span.unit or ""),
            formula="step = range / 2^bits",
            inputs={"range": span, "bits": bits},
            method="Ideal quantiser step over the declared full scale",
            plain=(
                f"Across a {span.text()} range with {int(float(bits.value))} bits, the "
                f"smallest change it can report is {Q(step, span.unit or '').text()}.{detail}"
            ),
            subject=part.id,
        )


@register(
    "sample_rate",
    "Sampling",
    "Is it looking often enough to keep up?",
    domains=("signal", "data"),
    discipline="controls",
)
def sample_rate(design) -> Iterable[Finding]:
    for part in design.parts:
        rate = part.ratings.get("sample_rate")
        bandwidth = part.ratings.get("signal_bandwidth")
        if rate is None or bandwidth is None:
            continue
        sampling = float(rate.as_("Hz").value)
        signal = float(bandwidth.as_("Hz").value)
        ratio = sampling / (2.0 * signal) if signal > 0 else float("inf")
        yield Finding(
            id=f"controls.sampling.{part.id}",
            name=f"Sampling margin, {part.name}",
            value=Q(ratio, "count"),
            formula="margin = f_sample / (2 f_signal)",
            inputs={"f_sample": rate.as_("Hz"), "f_signal": bandwidth.as_("Hz")},
            method="Nyquist criterion",
            plain=(
                f"Sampling at {rate.as_('Hz').text()} against a "
                f"{bandwidth.as_('Hz').text()} signal is {ratio:.1f} times the minimum. "
                + (
                    "Comfortable; a factor of five or more is normal practice for control."
                    if ratio >= 5
                    else "Above the theoretical minimum but tight. Anything faster than it "
                    "expects will be read as a slow signal that is not there."
                    if ratio > 1
                    else "Below the minimum. Fast changes will appear as slow ones that do "
                    "not exist, and no filtering afterwards can undo that."
                )
            ),
            subject=part.id,
            verdict="pass" if ratio >= 5 else ("watch" if ratio > 1 else "fail"),
            margin=ratio - 1.0,
            advice=""
            if ratio >= 5
            else "Sample faster, or put an anti-alias filter ahead of the converter.",
        )


@register(
    "latency_budget",
    "Reaction time",
    "How long between something happening and the response?",
    domains=("signal", "data"),
    discipline="controls",
)
def latency_budget(design) -> Iterable[Finding]:
    delays = []
    for part in design.parts:
        delay = part.ratings.get("latency") or part.ratings.get("delay")
        if delay is not None:
            delays.append((part, delay))
    if len(delays) < 2:
        return
    total = sum(float(value.as_("s").value) for _p, value in delays)
    worst = max(delays, key=lambda pair: float(pair[1].as_("s").value))
    yield Finding(
        id="controls.latency",
        name="End-to-end delay",
        value=Q(total, "s"),
        formula="t_total = sum of every stage's delay",
        inputs={part.name: value.as_("s") for part, value in delays},
        method="Serial latency accumulation over the signal path",
        plain=(
            f"From the event to the response takes {Q(total, 's').text()}. The "
            f"{worst[0].lay_name or worst[0].name.lower()} is the slowest stage at "
            f"{worst[1].as_('s').text()}, so that is the one worth fixing."
        ),
        subject="controls",
    )
