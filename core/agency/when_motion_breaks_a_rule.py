"""Observed departures through a boundary the task requires to hold.

The contract names edges and its provenance. Open edges have no implied wall.
Only visible motion supplies evidence; an extrapolated missing track is a
hypothesis. Application variables and terminal messages cannot prove motion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.verify import invariant


def departure_edge(event: dict[str, Any], shape: tuple[int, int]) -> str:
    """The first edge crossed from the last visible position, before a missing track is extrapolated."""
    tall, wide = shape
    x, y = event.get("x"), event.get("y")
    vx, vy = event.get("vx", 0), event.get("vy", 0)
    if not all(isinstance(n, (int, float)) for n in (x, y, vx, vy)):
        return ""
    visible = event.get("last_visible")
    if isinstance(visible, list) and len(visible) == 2:
        x0, y0 = visible
        halfw, halfh = event.get("width", 0)/2, event.get("height", 0)/2
        candidates = []
        for distance, speed, name in ((x0+halfw, -vx, "left"), (wide+halfw-x0, vx, "right"),
                                       (y0+halfh, -vy, "top"), (tall+halfh-y0, vy, "bottom")):
            if speed >= 8:
                crossing = distance/speed
                elapsed = event.get("at", 0)-event.get("last_seen_at", 0)
                if 0 <= crossing <= elapsed + 0.08:
                    candidates.append((crossing, name))
        return min(candidates)[1] if candidates else ""
    edge = min((x, "left", -vx), (wide-x, "right", vx), (y, "top", -vy), (tall-y, "bottom", vy))
    return edge[1] if edge[0] <= 8 and edge[2] >= 8 else ""


@dataclass
class MotionChecks:
    required_edges: frozenset[str] = frozenset()
    provenance: str = ""
    violations: list[dict[str, Any]] = field(default_factory=list)
    unconfirmed: list[dict[str, Any]] = field(default_factory=list)

    def see(self, moves: Any, happened: list[dict[str, Any]], at: float, controlled: int | None) -> list[dict[str, Any]]:
        """Require a visibly outside centre; a lost track alone cannot prove an escape."""
        if not self.provenance or any(e.get("what") == "new screen" for e in happened):
            return []
        found = []
        events = [e for e in happened if e.get("what") == "gone"]
        # Exact drawing bounds can show a partially clipped object before it
        # vanishes. Use the measured path, never the predicted control position.
        for thing in getattr(moves, "things", {}).values():
            if thing.seen == at and thing.path:
                events.append({"thing": thing.number, "x": thing.path[-1][1], "y": thing.path[-1][2],
                               "vx": thing.vx, "vy": thing.vy, "at": at,
                               "last_visible": list(thing.path[-1][1:]), "last_seen_at": thing.seen})
        for event in events:
            if event.get("thing") == controlled:
                continue
            x, y = event.get("x"), event.get("y")
            vx, vy = event.get("vx", 0), event.get("vy", 0)
            if not all(isinstance(n, (int, float)) for n in (x, y, vx, vy)):
                continue
            visible = event.get("last_visible", [x, y])
            if not isinstance(visible, list) or len(visible) != 2 or not all(isinstance(v, (int, float)) for v in visible):
                continue
            x0, y0 = visible
            tall, wide = moves.shape
            crossed = [("left", -x0, -vx), ("right", x0-wide, vx),
                       ("top", -y0, -vy), ("bottom", y0-tall, vy)]
            edge = next((name for name, beyond, speed in crossed if beyond > 0.5 and speed >= 8
                         and name in self.required_edges), "")
            if not edge:
                predicted = departure_edge(event, moves.shape)
                if event.get("what") == "gone" and predicted in self.required_edges and len(self.unconfirmed) < 16:
                    self.unconfirmed.append({"at": at, "edge": predicted, "thing": event.get("thing"),
                                             "last_visible": visible, "reason": "missing track; crossing was only extrapolated"})
                continue
            if edge not in self.required_edges:
                continue
            receipt = {"at": at, "edge": edge, "thing": event.get("thing"),
                       "position": visible, "velocity": [vx, vy], "provenance": self.provenance,
                       "last_seen_at": event.get("last_seen_at", at), "evidence": "visible centre outside required boundary",
                       "finding": f"a moving object left through the {edge}, which should turn it back"}
            self.violations.append(receipt)
            found.append(receipt)
        return found


def _open_edges_are_not_faults() -> bool:
    from types import SimpleNamespace

    moves = SimpleNamespace(shape=(100, 160))
    event = [{"what": "gone", "thing": 3, "x": 60, "y": -1, "vx": 20, "vy": -50}]
    closed = MotionChecks(frozenset({"top", "bottom"}), "source reflection contract")
    return not MotionChecks().see(moves, event, 1.0, 1) and bool(closed.see(moves, event, 1.0, 1))


@invariant("agency.motion_checks_require_boundary_contract", scope="agency",
           owner="core/agency/when_motion_breaks_a_rule.py", observational=False)
def _motion_checks_invariant() -> tuple:
    assert _open_edges_are_not_faults(), "a boundary verdict lacked a contract or missed an observed breach"
    return ()
