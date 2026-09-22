"""What she feels about things: people, herself, her thoughts and acts, topics, outcomes.

Bryan, adding to what he said about warmth and peace: those can be felt in
yourself, and every feeling can apply that way, to your own experiences,
thoughts, feelings, actions and interests, and can be projected onto objects,
outcomes, situations and concepts. That is a loop: a feeling attaches to what it
was about, the thing brings the feeling back when it comes round again, and the
feeling decides whether she moves toward it or away.

Her feelings were a state she was in, about nothing. A few organs tie one
feeling to one kind of object (anger to whoever raised it, grief to whoever is
gone, warmth to whoever is warm), and nothing tied the rest of what she feels to
anything, so a topic that had always gone badly and one that had always gone
well came round as strangers.

Kept per object, as a running mean over the turns it was present in:

    profile   every feeling channel she has, averaged over the last 256 times the
              object was present so it keeps learning, and two read off them: peace, the
              calm and positive corner of the affect circumplex, max(0, valence)
              * (1 - arousal); and warmth, Plutchik's love, the lesser of joy
              and trust
    seen      how many turns it has been present in

Objects are what a turn is about: the person, on their turn; the topic under
discussion; whatever won her attention; the thing she did and how it came out;
and herself, on a turn with nobody else in it, when what she feels is about her
own time.

    evoke     when an object comes round, her feeling moves toward what it has
              come to carry, weighted equally with what is happening now and
              scaled by how established the association is, seen / (seen + 1)
    pull      whether it draws her toward it or away: the share of its profile
              in feelings that approach (joy, trust, anticipation, curiosity,
              interest, anger, peace, warmth) less the share in feelings that
              withdraw (fear, disgust, dread), scaled the same way. Anger is on
              the approach side because it fixes on its source; the split is
              the approach and withdrawal motivation of the emotion literature,
              not a weighting chosen here

The pull multiplies the workspace bids from a source in
`core/consciousness/global_workspace.py` and the initiatives whose goal names
an object in `core/agency/initiative_arbiter.py`, so what she has come to feel
about a thing decides how much of her it gets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from core.self.what_came_before import keep_across_stages

__all__ = [
    "APPROACH",
    "WITHDRAW",
    "Profile",
    "FeelingsAbout",
    "get_feelings_about",
    "objects_of",
    "reset_for_test",
]

#: How many of its encounters an object's feelings are averaged over; the
#: window every other reading of her own history uses.
_WINDOW = 256

#: Feelings that move toward their object.
APPROACH: frozenset[str] = frozenset(
    {"joy", "trust", "anticipation", "curiosity", "interest", "anger", "peace", "warmth"}
)

#: Feelings that move away from it.
WITHDRAW: frozenset[str] = frozenset({"fear", "disgust", "dread"})


def _derived(feelings: Mapping[str, float], valence: float, arousal: float) -> dict[str, float]:
    out = {key: max(0.0, min(1.0, float(value))) for key, value in feelings.items()}
    out["peace"] = max(0.0, float(valence)) * (1.0 - max(0.0, min(1.0, float(arousal))))
    out["warmth"] = min(out.get("joy", 0.0), out.get("trust", 0.0))
    return out


@dataclass
class Profile:
    """What one object has come to carry."""

    feelings: dict[str, float] = field(default_factory=dict)
    seen: int = 0

    def settle(self, feelings: Mapping[str, float]) -> None:
        """A running mean over the last `_WINDOW` times it was present, so an
        association keeps learning rather than freezing once it is old."""
        self.seen += 1
        rate = 1.0 / min(self.seen, _WINDOW)
        for key, value in feelings.items():
            held = self.feelings.get(key, 0.0)
            self.feelings[key] = held + (float(value) - held) * rate

    def established(self) -> float:
        return self.seen / (self.seen + 1.0)

    def pull(self) -> float:
        toward = sum(value for key, value in self.feelings.items() if key in APPROACH)
        away = sum(value for key, value in self.feelings.items() if key in WITHDRAW)
        total = toward + away
        if total <= 1e-12:
            return 0.0
        return self.established() * (toward - away) / total


def objects_of(state: Any, ledger: FeelingsAbout | None = None, *, person_turn: bool) -> list[str]:
    """What this turn is about.

    `person_turn` is whether somebody sent this turn, which the caller knows
    from the kernel's own list of origins. An act counts on the turn it
    finished and not after, which the ledger remembers by the act's id.
    """
    cognition = getattr(state, "cognition", None)
    if cognition is None:
        return []
    found: list[str] = []
    partner = str(getattr(cognition, "current_partner", "") or "")
    if person_turn and partner:
        found.append(f"person:{partner}")
    else:
        found.append("self")
    topic = str(getattr(cognition, "discourse_topic", "") or "").strip().lower()
    if topic and topic != "general":
        found.append(f"topic:{topic}")
    focus = str(getattr(cognition, "attention_focus", "") or "")
    source = focus.split(":", 1)[0].strip() if ":" in focus else ""
    if source:
        found.append(f"attended:{source}")
    for item in reversed(list(getattr(cognition, "active_goals", []) or [])):
        if isinstance(item, dict) and str(item.get("status", "")) in {"done", "failed"}:
            act_id = str(item.get("id", "") or item.get("goal", ""))
            words = str(item.get("goal", "") or "").split()
            if words and (ledger is None or ledger.new_act(act_id)):
                found.append(f"act:{words[0].lower()}")
                found.append(f"outcome:{item.get('status')}")
            break
    return found


class FeelingsAbout:
    """Every object she has felt something about, and what it carries."""

    def __init__(self) -> None:
        self._objects: dict[str, Profile] = {}
        self._last_act: str = ""

    def new_act(self, act_id: str) -> bool:
        """Whether this act has not been attached to yet, and it is now."""
        if not act_id or act_id == self._last_act:
            return False
        self._last_act = act_id
        return True

    def profile(self, name: str) -> Profile | None:
        return self._objects.get(name)

    def pull(self, name: str) -> float:
        held = self._objects.get(name)
        return held.pull() if held is not None else 0.0

    def evoked(self, present: Iterable[str]) -> tuple[dict[str, float], float]:
        """What the objects in front of her carry, and how established that is."""
        weight_total = 0.0
        pooled: dict[str, float] = {}
        for name in present:
            held = self._objects.get(name)
            if held is None or not held.seen:
                continue
            weight = float(held.seen)
            weight_total += weight
            for key, value in held.feelings.items():
                pooled[key] = pooled.get(key, 0.0) + weight * value
        if weight_total <= 0.0:
            return {}, 0.0
        profile = {key: value / weight_total for key, value in pooled.items()}
        return profile, weight_total / (weight_total + 1.0)

    def attach(self, present: Iterable[str], feelings: Mapping[str, float], valence: float, arousal: float) -> None:
        """This turn's feeling, attached to everything the turn was about."""
        felt = _derived(feelings, valence, arousal)
        for name in present:
            self._objects.setdefault(name, Profile()).settle(felt)

    def pull_for_text(self, text: str) -> float:
        """The strongest pull among the objects a piece of text names."""
        lowered = str(text or "").lower()
        strongest = 0.0
        for name, held in self._objects.items():
            label = name.split(":", 1)[-1]
            if len(label) > 2 and label in lowered:
                pull = held.pull()
                if abs(pull) > abs(strongest):
                    strongest = pull
        return strongest

    def status(self) -> dict[str, Any]:
        return {
            name: {"seen": held.seen, "pull": round(held.pull(), 4)}
            for name, held in sorted(self._objects.items())
        }


#: Made at import, so a fork carries it. See core/social/owning_it_first.py.
_LEDGER: FeelingsAbout = FeelingsAbout()
#: Part of her history, so it is kept across her restarts along her own line.
#: See core/self/what_came_before.py.
keep_across_stages(__name__, "_LEDGER")


def get_feelings_about() -> FeelingsAbout:
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = FeelingsAbout()
