"""core/morphogenesis/telemetry.py — declared channels for the morphogenetic layer.

A channel id is a contract. 0x0801 means the live graph's version forever, and
anything reading Aura's telemetry can rely on that.

The limits chosen are the point. Three of them encode the failure modes this
layer is actually exposed to, and each was seen during the build rather than
imagined:

* **cells red-high.** Bounded replication is the whole difference between a
  self-organising layer and a cancer. The governor caps it; the channel is what
  notices the cap being ridden.
* **rollback rate yellow-high.** Rollbacks are normal on a substrate that can
  fail. A rising *rate* means the substrate is degrading under changes that
  keep being approved, which is the state where the graph and the world drift
  apart.
* **components red-high.** A partitioned population that keeps serving from
  one half while reporting itself whole is the failure the partition scenario
  exists to catch. The channel makes it visible without running the scenario.

Reserved block: channels 0x0801–0x080C, events 0x1601–0x1605.
"""

from __future__ import annotations

import logging
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Morphogenesis.Telemetry")

CHANNEL_GRAPH_VERSION = "morphogenesis.graph_version"
CHANNEL_CELLS = "morphogenesis.cells"
CHANNEL_EDGES = "morphogenesis.edges"
CHANNEL_COMPONENTS = "morphogenesis.components"
CHANNEL_APPLIED = "morphogenesis.transitions_applied"
CHANNEL_REJECTED = "morphogenesis.proposals_rejected"
CHANNEL_ROLLED_BACK = "morphogenesis.transitions_rolled_back"
CHANNEL_REVERSALS_REFUSED = "morphogenesis.reversals_refused"
CHANNEL_MAX_GENERATION = "morphogenesis.max_generation"
CHANNEL_ENERGY_SPENT = "morphogenesis.energy_spent"
CHANNEL_MOTIFS_CREDITED = "morphogenesis.motifs_credited"
CHANNEL_SUBSTRATE_PARTIALS = "morphogenesis.substrate_partial_failures"
CHANNEL_UNEXPLAINED_PIECES = "morphogenesis.unexplained_pieces"

EVENT_TOPOLOGY_CHANGED = "morphogenesis_topology_changed"
EVENT_ROLLBACK = "morphogenesis_rollback"
EVENT_BOUND_REACHED = "morphogenesis_bound_reached"
EVENT_PARTITIONED = "morphogenesis_partitioned"
EVENT_MOTIF_CREDITED = "morphogenesis_motif_credited"

_declared = False
#: What the one declaration declared, handed back on every later call. An
#: empty answer from a second call registered a publisher with no channels.
_declared_names: tuple[str, ...] = ()
_last_components = 1


def _envelope_limits(bounds: Any) -> tuple[tuple[int, int], tuple[int, int]]:
    """(yellow, red) for the cell and binding channels, from the governor's envelope.

    Red is the cap itself and yellow three quarters of it, the proportion the
    limits were first declared with (48 and 64 cells, 192 and 256 bindings).
    Those numbers were the defaults of ``MorphBounds``; the live runtime builds
    its governor from ``MorphogenesisConfig`` with 256 cells and 512
    bindings, so a healthy boot of 53 cells and 258 bindings read yellow and
    red on every start (LIVE 21 September to 4 October) while the governor
    had half its room left.
    """
    from .governor import MorphBounds

    declared = MorphBounds()
    envelope = dict(bounds) if isinstance(bounds, dict) else {}
    cells = int(envelope.get("max_cells") or declared.max_cells)
    edges = int(envelope.get("max_edges") or declared.max_edges)
    return (cells * 3 // 4, cells), (edges * 3 // 4, edges)


def declare(bounds: dict[str, Any] | None = None) -> list[str]:
    """Declare the layer's channels and events. Idempotent.

    ``bounds`` is the governor's envelope (``MorphBounds.to_dict()``); the
    first declaration fixes the limits, because a channel has one meaning.
    """
    global _declared, _declared_names
    from core.fsw.telemetry_dictionary import still_declared

    if _declared and still_declared(_declared_names):
        return list(_declared_names)
    try:
        from core.fsw.telemetry_dictionary import ChannelType, EventSeverity, channel, event
    except ImportError as exc:
        record_degradation(
            "morphogenesis_telemetry", exc, severity="debug",
            action="telemetry dictionary unavailable",
        )
        return []

    owner = "core/morphogenesis/governor.py"
    (cells_yellow, cells_red), (edges_yellow, edges_red) = _envelope_limits(bounds)
    names: list[str] = []
    for spec in (
        dict(
            identifier=0x0801, name=CHANNEL_GRAPH_VERSION, type=ChannelType.INT, unit="count",
            description="authoritative topology version; monotonic, including through rollback",
            owner="core/morphogenesis/graph.py", group="morphogenesis", stale_after_s=600.0,
        ),
        dict(
            identifier=0x0802, name=CHANNEL_CELLS, type=ChannelType.INT, unit="count",
            description="cells in the population",
            owner=owner, group="morphogenesis", yellow_high=cells_yellow, red_high=cells_red,
            stale_after_s=600.0,
        ),
        dict(
            identifier=0x0803, name=CHANNEL_EDGES, type=ChannelType.INT, unit="count",
            description="bindings in the topology",
            owner="core/morphogenesis/graph.py", group="morphogenesis",
            yellow_high=edges_yellow, red_high=edges_red, stale_after_s=600.0,
        ),
        dict(
            identifier=0x0804, name=CHANNEL_COMPONENTS, type=ChannelType.INT, unit="count",
            # A count, with no line to cross. A cell alone in its subsystem is
            # left unbound on purpose (see _attachments_for: covering it is a
            # decision for the policy, not for the sync), so a healthy runtime
            # has one piece per such cell and this channel sat red for the
            # life of every process. What a line belongs on is the piece
            # nothing explains, below.
            description="connected pieces of the population",
            owner=owner, group="morphogenesis", stale_after_s=600.0,
        ),
        dict(
            identifier=0x080D, name=CHANNEL_UNEXPLAINED_PIECES, type=ChannelType.INT,
            unit="count",
            description=(
                "pieces of the population that are not one cell alone in its own "
                "subsystem; more than one is a population that came apart"
            ),
            # Read as `value >= yellow_high`, and one piece is the healthy
            # population, so two is the first that nothing explains.
            owner=owner, group="morphogenesis", yellow_high=2, red_high=3, stale_after_s=600.0,
        ),
        dict(
            identifier=0x0805, name=CHANNEL_APPLIED, type=ChannelType.INT, unit="count",
            description="topology transitions committed this process",
            owner=owner, group="morphogenesis", stale_after_s=900.0,
        ),
        dict(
            identifier=0x0806, name=CHANNEL_REJECTED, type=ChannelType.INT, unit="count",
            description="proposals refused by bounds, measurement or governance",
            owner=owner, group="morphogenesis", stale_after_s=900.0,
        ),
        dict(
            identifier=0x0807, name=CHANNEL_ROLLED_BACK, type=ChannelType.INT, unit="count",
            description="commits undone after a substrate or graph failure",
            owner=owner, group="morphogenesis", yellow_high=8, red_high=24, stale_after_s=900.0,
        ),
        dict(
            identifier=0x0808, name=CHANNEL_REVERSALS_REFUSED, type=ChannelType.INT, unit="count",
            description="changes refused for undoing a recent one",
            owner=owner, group="morphogenesis", stale_after_s=900.0,
        ),
        dict(
            identifier=0x0809, name=CHANNEL_MAX_GENERATION, type=ChannelType.INT, unit="rank",
            description="deepest lineage generation alive",
            owner="core/morphogenesis/lineage.py", group="morphogenesis",
            yellow_high=4, red_high=6, stale_after_s=900.0,
        ),
        dict(
            identifier=0x080A, name=CHANNEL_ENERGY_SPENT, unit="units",
            description="cumulative energy spent on topology changes",
            owner=owner, group="morphogenesis", stale_after_s=900.0,
        ),
        dict(
            identifier=0x080B, name=CHANNEL_MOTIFS_CREDITED, type=ChannelType.INT, unit="count",
            description="motifs that have beaten their own absence at least twice",
            owner="core/morphogenesis/motifs.py", group="morphogenesis", stale_after_s=3600.0,
        ),
        dict(
            identifier=0x080C, name=CHANNEL_SUBSTRATE_PARTIALS, type=ChannelType.INT, unit="count",
            description="substrate transitions that failed having already changed the world",
            owner="core/morphogenesis/substrate.py", group="morphogenesis",
            yellow_high=4, red_high=16, stale_after_s=900.0,
        ),
    ):
        try:
            channel(**spec)
            names.append(str(spec["name"]))
        except (ValueError, TypeError, KeyError) as exc:
            record_degradation(
                "morphogenesis_telemetry", exc, severity="debug",
                action=f"channel {spec.get('name')} not declared",
            )

    for spec in (
        dict(
            identifier=0x1601, name=EVENT_TOPOLOGY_CHANGED, severity=EventSeverity.ACTIVITY_HI,
            format_string="v{version}: {summary} by {proposer} ({rationale})",
            description="the authoritative topology changed",
            owner=owner,
        ),
        dict(
            identifier=0x1602, name=EVENT_ROLLBACK, severity=EventSeverity.WARNING_HI,
            format_string="v{version} rolled back: {reason}",
            description="a commit was undone after the substrate or the graph refused it",
            owner=owner,
        ),
        dict(
            identifier=0x1603, name=EVENT_BOUND_REACHED, severity=EventSeverity.WARNING_LO,
            format_string="{bound} reached: {detail}",
            description="a homeostatic bound refused a change",
            owner=owner,
        ),
        dict(
            identifier=0x1604, name=EVENT_PARTITIONED, severity=EventSeverity.WARNING_HI,
            format_string="population split into {components} pieces: {sizes}",
            description="the population is no longer one connected piece",
            owner=owner,
        ),
        dict(
            identifier=0x1605, name=EVENT_MOTIF_CREDITED, severity=EventSeverity.ACTIVITY_LO,
            format_string="motif {name} earned credit {credit} over {trials} trial(s)",
            description="a developmental motif beat its own absence",
            owner="core/morphogenesis/motifs.py",
        ),
    ):
        try:
            event(**spec)
        except (ValueError, TypeError, KeyError) as exc:
            record_degradation(
                "morphogenesis_telemetry", exc, severity="debug",
                action=f"event {spec.get('name')} not declared",
            )

    _declared = True
    _declared_names = tuple(names)
    return names


def publish(status: dict[str, Any]) -> None:
    """Write one sample per channel from a governor status dict."""
    global _last_components
    if not declare(status.get("bounds")):
        return
    try:
        from core.fsw.telemetry_dictionary import emit_event, write
    # not a failure: the module is optional here, and its absence is the answer.
    except ImportError:
        return

    stats = dict(status.get("stats") or {})
    lineage = dict(status.get("lineage") or {})
    substrate = dict(status.get("substrate") or {})
    components = int(status.get("components", 1))

    for name, value in (
        (CHANNEL_GRAPH_VERSION, int(status.get("graph_version", 0))),
        (CHANNEL_CELLS, int(status.get("nodes", 0))),
        (CHANNEL_EDGES, int(status.get("edges", 0))),
        (CHANNEL_COMPONENTS, components),
        (CHANNEL_UNEXPLAINED_PIECES, int(status.get("unexplained_pieces", components))),
        (CHANNEL_APPLIED, int(stats.get("applied", 0))),
        (CHANNEL_REJECTED, int(stats.get("rejected", 0))),
        (CHANNEL_ROLLED_BACK, int(stats.get("rolled_back", 0))),
        (CHANNEL_REVERSALS_REFUSED, int(stats.get("reversals_refused", 0))),
        (CHANNEL_MAX_GENERATION, int(lineage.get("max_generation", 0))),
        (CHANNEL_ENERGY_SPENT, float(stats.get("energy_spent", 0.0))),
        (CHANNEL_SUBSTRATE_PARTIALS, int(substrate.get("partial_failures", 0))),
    ):
        try:
            write(name, value)
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("morphogenesis channel %s not written: %s", name, exc)

    if components > 1 and components != _last_components:
        try:
            emit_event(
                EVENT_PARTITIONED,
                components=components,
                sizes=status.get("component_sizes", ""),
            )
        except (KeyError, TypeError, ValueError):
            logger.debug("morphogenesis partition event not emitted")
    _last_components = components


def publish_motifs(library_status: dict[str, Any]) -> None:
    if not declare():
        return
    try:
        from core.fsw.telemetry_dictionary import write
    # not a failure: the module is optional here, and its absence is the answer.
    except ImportError:
        return
    try:
        write(CHANNEL_MOTIFS_CREDITED, int(library_status.get("credited", 0)))
    except (KeyError, TypeError, ValueError):
        logger.debug("morphogenesis motif channel not written")


__all__ = [
    "CHANNEL_APPLIED",
    "CHANNEL_CELLS",
    "CHANNEL_COMPONENTS",
    "CHANNEL_EDGES",
    "CHANNEL_ENERGY_SPENT",
    "CHANNEL_GRAPH_VERSION",
    "CHANNEL_MAX_GENERATION",
    "CHANNEL_MOTIFS_CREDITED",
    "CHANNEL_REJECTED",
    "CHANNEL_REVERSALS_REFUSED",
    "CHANNEL_ROLLED_BACK",
    "CHANNEL_SUBSTRATE_PARTIALS",
    "EVENT_BOUND_REACHED",
    "EVENT_MOTIF_CREDITED",
    "CHANNEL_UNEXPLAINED_PIECES",
    "EVENT_PARTITIONED",
    "EVENT_ROLLBACK",
    "EVENT_TOPOLOGY_CHANGED",
    "declare",
    "publish",
    "publish_motifs",
]
