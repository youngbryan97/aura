"""Where her notes on a person live between restarts.

`PersonModel` is the shape and `InterpersonalObserver` is the writer. Neither
outlives the process, and a relationship model that resets on every restart is
not a relationship model — it is a session cache with opinions. Worse, it fails
in the shape this whole subsystem exists to refuse: what survives a restart
would be whatever the *prose* summary happened to keep, so the structured
record would lose to the lossy one precisely because the lossy one was durable.

Three positions, in the order they matter:

* **Restoring is not re-observing.** Reload goes through ``from_dict``, never by
  replaying ``observe``. Replaying would stamp every occurrence with the time of
  the restart — turning a year of evidence into a burst of sightings on boot day,
  which is the frequency bug in a different costume — and would drop anything he
  had corrected, because ``observe`` refuses a corrected claim. That refusal is
  right at the door and wrong on reload.
* **An unreadable file is an incident, not an empty model.** Starting fresh and
  carrying on is how "she forgot everything about me" becomes a thing nobody
  notices for a week. The file is moved aside, kept, and the loss is recorded.
* **Consent is checked here, not only at the callers.** The store is reachable
  from more than one seam, and a gate that lives at the seams is a gate that the
  next seam forgets. The kind is the same ``derived_profile`` the rest of the
  profile surface uses, deliberately: a new kind of its own would escape every
  revocation a person has already made.
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

from core.governance_context import local_internal_governed_scope
from core.memory.interpersonal_model import Observation, PersonModel
from core.memory.interpersonal_observer import Exchange, InterpersonalObserver
from core.memory.memory_blocks import DEFAULT_LIMIT, MemoryBlock, MemoryBlockSet
from core.runtime.atomic_writer import AtomicWriteError, read_json_envelope
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.InterpersonalStore")

__all__ = [
    "InterpersonalStore",
    "get_interpersonal_store",
    "BLOCK_LABEL",
    "BLOCK_SOURCE",
    "CONSENT_KIND",
    "SCHEMA_VERSION",
]

#: The consent kind this store answers to. Shared with the rest of the derived
#: profile surface on purpose — see the module docstring.
CONSENT_KIND = "derived_profile"

SCHEMA_VERSION = 1

#: The block this store owns in the context window, and the author name that
#: owns it. A block carrying this as ``derived_from`` will refuse a write from
#: anybody else, which is what keeps a background summariser off it.
BLOCK_LABEL = "person"
BLOCK_SOURCE = "interpersonal_memory"

_STORE_ERRORS = (OSError, RuntimeError, TypeError, ValueError, KeyError)


def _person_key(person: str) -> str:
    return " ".join(str(person or "").strip().split())[:160]


def _filename(person: str) -> str:
    """A digest, not the person's name.

    Names in filenames put someone's identity in every directory listing, log
    line and backup index that ever touches the state directory, and they make
    the path a function of untrusted text.
    """
    return hashlib.sha256(_person_key(person).encode("utf-8")).hexdigest()[:32] + ".json"


class InterpersonalStore:
    """Her person models, durable, consent-gated, one file each."""

    def __init__(self, *, root: Path | None = None, authority: object | None = None) -> None:
        self._root = Path(root) if root is not None else state_root() / "data" / "interpersonal"
        self._authority = authority
        self._models: dict[str, PersonModel] = {}
        self._observers: dict[str, InterpersonalObserver] = {}
        self._load_failures = 0
        self._save_failures = 0
        self._consent_skips = 0
        self._written = 0

    # -- consent -----------------------------------------------------------

    def _resolve_authority(self) -> object | None:
        if self._authority is not None:
            return self._authority
        try:
            from core.social.relational_memory import get_relational_memory_authority

            self._authority = get_relational_memory_authority()
        except _STORE_ERRORS + (ImportError,) as exc:
            record_degradation(
                "memory.interpersonal_store.consent",
                exc,
                action="treated interpersonal memory as not consented",
            )
            return None
        return self._authority

    def allows(self, person: str, operation: str) -> bool:
        """Whether this person has consented to `operation` on derived profile data.

        Fail-closed: no authority, or an authority that raises, means no.
        """
        key = _person_key(person)
        if not key:
            return False
        authority = self._resolve_authority()
        if authority is None or not hasattr(authority, "allows"):
            return False
        try:
            return bool(authority.allows(key, CONSENT_KIND, operation))
        except _STORE_ERRORS as exc:
            record_degradation(
                "memory.interpersonal_store.consent",
                exc,
                action=f"denied {operation} on interpersonal memory after consent lookup failed",
            )
            return False

    # -- models ------------------------------------------------------------

    def path_for(self, person: str) -> Path:
        return self._root / _filename(person)

    def model_for(self, person: str) -> PersonModel:
        """The model for this person, loaded from disk on first use."""
        key = _person_key(person)
        if not key:
            raise ValueError("a person model needs someone to be about")
        existing = self._models.get(key)
        if existing is not None:
            return existing
        model = self._load(key)
        self._models[key] = model
        return model

    def observer_for(self, person: str) -> InterpersonalObserver:
        key = _person_key(person)
        observer = self._observers.get(key)
        if observer is None:
            observer = InterpersonalObserver(self.model_for(key))
            self._observers[key] = observer
        return observer

    # -- the write side ----------------------------------------------------

    async def observe_turn(
        self,
        person: str,
        *,
        episode_id: str,
        user_text: str = "",
        assistant_text: str = "",
        at: float | None = None,
        superseded_episode_ids: tuple[str, ...] = (),
    ) -> list[Observation]:
        """Notice what one turn says about someone, and record it durably.

        Returns what was written, which is usually nothing: the detectors are
        deliberately shallow and most turns say nothing explicit about anyone.
        """
        key = _person_key(person)
        if not key or not episode_id:
            return []
        if not self.allows(key, "recall"):
            self._consent_skips += 1
            return []

        model = self.model_for(key)
        removed = sum(
            model.remove_episode(episode_id)
            for episode_id in dict.fromkeys(superseded_episode_ids)
        )
        observer = self.observer_for(key)
        exchange = Exchange(
            episode_id=episode_id,
            user_text=user_text or "",
            assistant_text=assistant_text or "",
            at=time.time() if at is None else at,
        )
        written = observer.observe_exchange(exchange)
        if written:
            self._written += len(written)
        if written or removed:
            await self.save(key)
        return written

    # -- the read side -----------------------------------------------------

    def render(self, person: str, *, per_facet: int = 6) -> str:
        """The prompt block for this person, or "" if there is nothing to say.

        Synchronous and pure: this is called while a turn is being assembled,
        and it reads memory that is already resident.
        """
        key = _person_key(person)
        if not key or not self.allows(key, "prompt"):
            return ""
        try:
            model = self.model_for(key)
        except _STORE_ERRORS as exc:
            record_degradation(
                "memory.interpersonal_store.render",
                exc,
                action="omitted the interpersonal block from context assembly",
            )
            return ""
        if len(model) == 0:
            return ""
        return model.render(per_facet=per_facet)

    # -- as a memory block -------------------------------------------------

    def as_block(
        self,
        person: str,
        *,
        label: str = BLOCK_LABEL,
        limit: int = DEFAULT_LIMIT,
        per_facet: int = 6,
    ) -> MemoryBlock | None:
        """This person's notes as a memory block, owned by this store.

        ``derived_from`` is what keeps sleep-time consolidation away from it.
        Without it the block is prose like any other, and a background pass
        would eventually summarise the one thing in the context window that
        must not be summarised — which would make the entire typed store
        pointless, since what reaches the model is the block.

        Budget is met by dropping whole records, never by trimming text. Fewer
        entries per facet is the same policy eviction already uses — the
        weakest evidence goes first — and it leaves every surviving claim with
        its conditions, its count and its counter-examples intact. Trimming
        characters would take the qualifiers, because they sit at the end of
        the line.
        """
        key = _person_key(person)
        if not key or not self.allows(key, "prompt"):
            return None
        model = self.model_for(key)
        if len(model) == 0:
            return None

        # Records before readings. The dynamics are derived from the very lines
        # printed above them, so they can be recomputed from what survives while
        # the evidence cannot be recomputed from them.
        for budget in range(per_facet, 0, -1):
            for include_dynamics in (True, False):
                text = model.render(per_facet=budget, include_dynamics=include_dynamics)
                if len(text) <= limit:
                    return MemoryBlock(
                        label=label,
                        value=text,
                        limit=limit,
                        description=(
                            "What I know about this person, rendered from evidence. "
                            "Each line carries how I know it and how often."
                        ),
                        derived_from=BLOCK_SOURCE,
                    )

        # Refused rather than truncated, for the same reason the block system
        # refuses an overflowing write: a silent trim drops exactly the part
        # somebody decided was worth keeping.
        record_degradation(
            "memory.interpersonal_store.block",
            ValueError(
                f"a single record about this person exceeds the {limit}-character "
                f"block budget; the block was omitted rather than trimmed"
            ),
            action="omitted the interpersonal block instead of truncating it",
        )
        return None

    def refresh_block(
        self,
        blocks: MemoryBlockSet,
        person: str,
        *,
        label: str = BLOCK_LABEL,
    ) -> MemoryBlock | None:
        """Re-render this person's block in an existing set.

        The write is attributed to this store because the store is the only
        author the block will accept — see ``MemoryBlock.derived_from``.
        """
        block = self.as_block(person, label=label)
        if block is None:
            return None
        if label not in blocks:
            return blocks.attach(block)
        return blocks.rewrite(
            label,
            block.value,
            author=BLOCK_SOURCE,
            reason="re-rendered from the interpersonal record",
        )

    # -- durability --------------------------------------------------------

    def _load(self, person: str) -> PersonModel:
        path = self.path_for(person)
        if not path.exists():
            return PersonModel(person)
        try:
            envelope = read_json_envelope(path)
            model = PersonModel.from_dict(envelope["payload"])
        except _STORE_ERRORS + (UnicodeDecodeError, AtomicWriteError) as exc:
            self._load_failures += 1
            self._quarantine(path, exc)
            return PersonModel(person)
        if model.person != person:
            # The digest resolved to a file about somebody else. Conflating two
            # people is the one merge this subsystem refuses outright.
            self._load_failures += 1
            self._quarantine(
                path,
                ValueError(
                    f"file at {path.name} holds notes about a different person"
                ),
            )
            return PersonModel(person)
        return model

    def _quarantine(self, path: Path, exc: BaseException) -> None:
        """Move an unreadable file aside rather than overwrite it.

        Kept rather than deleted: it is the only remaining evidence of what she
        knew, and a loss with the payload destroyed is a loss nobody can undo.
        """
        aside = path.with_suffix(f".unreadable.{int(time.time())}.json")
        try:
            with local_internal_governed_scope(
                "memory.interpersonal_store.quarantine",
                domain="memory_write",
            ):
                get_file_write_gateway().move_path(
                    path, aside, source="memory.interpersonal_store.quarantine"
                )
        except _STORE_ERRORS as move_exc:
            record_degradation(
                "memory.interpersonal_store",
                move_exc,
                severity="critical",
                action="could not quarantine unreadable person notes; started empty",
                enforce_failure_policy=False,
            )
            aside = None  # type: ignore[assignment]
        record_degradation(
            "memory.interpersonal_store",
            exc,
            severity="critical",
            action=(
                "started with no notes about this person; the previous file was "
                f"kept at {aside.name if aside else '(not moved)'}"
            ),
            enforce_failure_policy=False,
        )

    async def save(self, person: str) -> bool:
        """Persist one person's model. Async because an on-loop fsync is a stall."""
        key = _person_key(person)
        model = self._models.get(key)
        if model is None:
            return False
        try:
            gateway = get_file_write_gateway()
            with local_internal_governed_scope(
                "memory.interpersonal_store.save",
                domain="memory_write",
            ):
                await gateway.ensure_directory_async(
                    self._root, source="memory.interpersonal_store.save"
                )
                await gateway.write_json_async(
                    self.path_for(key),
                    model.to_dict(),
                    schema_version=SCHEMA_VERSION,
                    schema_name="interpersonal_person_model",
                    source="memory.interpersonal_store.save",
                )
        except _STORE_ERRORS as exc:
            self._save_failures += 1
            record_degradation(
                "memory.interpersonal_store.save",
                exc,
                action="kept this person's notes in memory only; they will not survive a restart",
            )
            return False
        return True

    async def save_all(self) -> int:
        saved = 0
        for person in list(self._models):
            saved += int(await self.save(person))
        return saved

    def preferences_for_prompt(self, person: str, *, limit: int = 12) -> dict[str, str]:
        """What he has said he likes and values, as the prompt block reads it.

        `world.user_preferences` is described in `core/being/individual_preferences.py`
        as "durable, persisted, and injected into every prompt: learned from
        conversation". Nothing wrote it. The context assembler has read it since
        it was added and found nothing there every time, and the column reading
        it stayed flat through a six-hour recording.

        Keys are his own words, because the record's whole discipline is that
        person-knowledge is never reworded. Values say how she knows it.
        """
        key = _person_key(person)
        if not key or key not in self._models:
            return {}
        rows: list[tuple[float, str, str]] = []
        for observation in self._models[key]:
            if str(observation.facet) not in {"preference", "value"}:
                continue
            claim = str(observation.claim or "").strip()
            if not claim:
                continue
            last = observation.last_seen() or 0.0
            rows.append((float(last), claim, observation.evidence_phrase or "she saw it"))
        rows.sort(key=lambda row: -row[0])
        return {claim: phrase for _, claim, phrase in rows[: max(0, int(limit))]}

    def models(self) -> dict[str, PersonModel]:
        """Every person model resident, keyed the way the store keys them.

        A reader that wants to ask how one record differs from the others needs
        the others. See core/social/particular.py.
        """
        return dict(self._models)

    def get_status(self) -> dict[str, object]:
        return {
            "people": len(self._models),
            "observations": sum(len(model) for model in self._models.values()),
            "written": self._written,
            "load_failures": self._load_failures,
            "save_failures": self._save_failures,
            "consent_skips": self._consent_skips,
            "consent_kind": CONSENT_KIND,
            "root": str(self._root),
        }


_store: InterpersonalStore | None = None


def get_interpersonal_store() -> InterpersonalStore:
    global _store
    if _store is None:
        _store = InterpersonalStore()
    return _store
