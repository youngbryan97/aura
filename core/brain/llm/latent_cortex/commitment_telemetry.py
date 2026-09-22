"""Declared channels for what an episode committed and what it covered.

The number that matters is not "how many passes did we spend" — it is "how
many DISTINCT candidates did we examine". Under i.i.d. branch sampling from a
peaked model those diverge badly: eight passes, two distinct answers, and
nothing in the operator view saying so. That gap is the entire reason more
branches and more depth have bought so little, and it has been invisible.

Every channel here is declared with an id, a unit and limits, per the
telemetry contract. The limits are the interesting part:

  duplicate_passes goes YELLOW at 2 and RED at 4 — not because duplicates
  are a fault, but because a run spending half its budget re-deriving one
  answer is a run whose best-of-N is best-of-2, and an operator should see
  that without reading a receipt.

  measured_narrowing has no red band. A low value can mean the constraints
  are weak OR that the candidates were already tight; it is diagnostic, not
  a fault, and giving it a red limit would manufacture alarms.
"""

from __future__ import annotations

from typing import Any

from core.runtime.errors import record_degradation

CHANNEL_COMMITMENTS = "rlc.commitments"
CHANNEL_DISTINCT = "rlc.distinct_candidates"
CHANNEL_DUPLICATE_PASSES = "rlc.duplicate_passes"
CHANNEL_NARROWING = "rlc.measured_narrowing"
CHANNEL_REFUSALS = "rlc.commit_refusals"

_declared = False


def declare() -> bool:
    """Declare the channels once. Safe to call repeatedly."""
    global _declared
    if _declared:
        return True
    try:
        from core.fsw.telemetry_dictionary import ChannelType, channel

        channel(
            identifier=0x1201, name=CHANNEL_COMMITMENTS, type=ChannelType.INT,
            unit="count", owner="latent_cortex.commitment_ratchet",
            description="Constraints irreversibly committed in one episode.",
            group="rlc",
        )
        channel(
            identifier=0x1202, name=CHANNEL_DISTINCT, type=ChannelType.INT,
            unit="count", owner="latent_cortex.commitment_ratchet",
            description=(
                "Distinct candidate answers examined. The quantity best-of-N "
                "actually depends on, as opposed to passes spent."
            ),
            group="rlc",
        )
        channel(
            identifier=0x1203, name=CHANNEL_DUPLICATE_PASSES, type=ChannelType.INT,
            unit="count", owner="latent_cortex.commitment_ratchet",
            description=(
                "Passes that re-derived an answer already examined. This is "
                "the wasted budget that makes best-of-8 behave like best-of-2."
            ),
            yellow_high=2, red_high=4,
            group="rlc",
        )
        channel(
            identifier=0x1204, name=CHANNEL_NARROWING, unit="fraction",
            owner="latent_cortex.commitment_ratchet",
            description=(
                "Fraction of the candidate pool eliminated by measured "
                "commitments. Diagnostic, deliberately without a red band: a "
                "low value can mean weak constraints or tight candidates."
            ),
            group="rlc",
        )
        channel(
            identifier=0x1205, name=CHANNEL_REFUSALS, type=ChannelType.INT,
            unit="count", owner="latent_cortex.commitment_ratchet",
            description=(
                "Commits refused — contradictory, duplicate, or narrowing "
                "nothing. A refusal means the step had no new information, "
                "and the pass it would have conditioned is not worth spending."
            ),
            group="rlc",
        )
        _declared = True
        return True
    except (ImportError, ValueError, TypeError, KeyError) as exc:
        record_degradation(
            "commitment_telemetry", exc, severity="debug",
            action="commitment channels not declared; episode coverage is "
                   "visible only in receipts this run",
        )
        return False


def sample(receipt: dict[str, Any], *, passes: int | None = None) -> None:
    """Write one episode's ratchet receipt to the declared channels."""
    if not isinstance(receipt, dict) or not receipt:
        return
    if not declare():
        return
    try:
        from core.fsw.telemetry_dictionary import write
    # not a failure: the module is optional here, and its absence is the answer.
    except ImportError:
        return

    def _put(name: str, value: Any) -> None:
        if value is None:
            return
        try:
            write(name, value)
        except (ValueError, TypeError, KeyError) as exc:
            record_degradation(
                "commitment_telemetry", exc, severity="debug",
                action=f"channel {name} not written",
            )

    turns = int(receipt.get("turns") or 0)
    pool_initial = int(receipt.get("pool_initial") or 0)
    pool_remaining = int(receipt.get("pool_remaining") or 0)
    _put(CHANNEL_COMMITMENTS, turns)
    _put(CHANNEL_REFUSALS, len(receipt.get("refusals") or ()))
    # Only report narrowing that was actually measured. Writing 0.0 for an
    # episode that had no pool would put "no narrowing" on a chart that reads
    # as "narrowed nothing", which is a different and unearned claim.
    if receipt.get("narrowing_is_measured"):
        _put(CHANNEL_NARROWING, float(receipt.get("measured_narrowing") or 0.0))
    if pool_initial:
        _put(CHANNEL_DISTINCT, pool_initial)
        if passes is not None:
            _put(CHANNEL_DUPLICATE_PASSES, max(0, int(passes) - pool_initial))
    _ = pool_remaining


__all__ = [
    "CHANNEL_COMMITMENTS",
    "CHANNEL_DISTINCT",
    "CHANNEL_DUPLICATE_PASSES",
    "CHANNEL_NARROWING",
    "CHANNEL_REFUSALS",
    "declare",
    "sample",
]
