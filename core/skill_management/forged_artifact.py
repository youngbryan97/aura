"""The on-disk form of a forged skill, and what binds it to the evidence.

Verification proves something about a specific string of Python. The file that
ends up in ``skills/`` is not that string — it also carries an adapter so the
catalog can discover it. If nothing connects the two, "this skill was verified"
degrades into "some ancestor of this file once was", and an edit, a partial
write or a hand-patch would leave the claim standing over code nobody checked.

So the layout is fixed and the join is a content hash.

Layout
------
Two regions separated by :data:`VERIFIED_REGION_MARKER`::

    <verified region>          model-authored, sandbox-executed, hashed
    # ---- marker ----
    <adapter>                  Aura-authored, fixed by ADAPTER_TEMPLATE

The verified region is a complete module on its own: a ``run(params) -> dict``
with imports restricted to the pure-computation allowlist. It is exactly what
:func:`~core.skill_management.skill_verification.verify_draft` executed, byte
for byte, so the evidence describes this file and not a relative of it.

The adapter below the marker is a ``BaseSkill`` subclass that calls ``run``.
It sits at module level, unguarded, because :mod:`core.skills.discovery`
collects classes from ``tree.body`` and nothing else. Its import scan is more
forgiving — ``_top_level_import_statements`` descends into ``try`` and ``if`` —
but the class scan is not, so a subclass tucked inside
``if BaseSkill is not None:`` is invisible to the catalog and the skill is
silently excluded.

That single constraint is what rules out the tidier arrangement where one
module is both executable under the sandbox and discoverable by the catalog:
the adapter needs a top-level class, a top-level class needs ``BaseSkill``
resolved at import, and ``BaseSkill`` cannot be imported inside the sandbox
because the repository is not readable there. It cannot be both files, so it is
one file with two regions and an explicit seam.

Why the hash is checked at execution and not only at write
----------------------------------------------------------
A verification receipt written at forge time proves what was true at forge time.
Between then and the call, the file is an ordinary file on an ordinary disk.
:func:`load_verified_region` recomputes the digest and refuses when it has
moved, so a forged skill whose source changed after verification fails closed
instead of running. That is the difference between a system that was verified
and a system that stays verified.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from core.runtime.lockdep import checked_lock

__all__ = [
    "VERIFIED_REGION_MARKER",
    "ADAPTER_TEMPLATE",
    "ArtifactError",
    "ForgedArtifact",
    "LedgerEntry",
    "ForgeLedger",
    "get_forge_ledger",
    "assemble",
    "split",
    "digest_of",
    "class_name_for",
    "load_verified_region",
    "next_version_path",
]

#: Separates the two regions. Long and unmistakable on purpose: a marker that
#: could plausibly occur in generated code would let a draft declare where its
#: own verified region ends.
VERIFIED_REGION_MARKER = (
    "# ==== AURA FORGE: end of verified region — everything below is Aura's adapter ===="
)

#: The adapter, in full. Aura writes this; a model never contributes to it.
#: ``execute`` is async because ``BaseSkill`` declares it async, and it does no
#: awaiting of its own — ``run`` is a pure synchronous function by contract, so
#: there is nothing here to await and nothing that can block the loop beyond the
#: computation the caller asked for.
ADAPTER_TEMPLATE = '''from typing import Any

from core.skills.base_skill import BaseSkill


class {class_name}(BaseSkill):
    """{description}"""

    name = "{skill_name}"
    description = "{description}"
    #: Model-authored code that runs under a kernel boundary. Not
    #: ``pure_compute``: that maps to the ``observe`` authority class, which is
    #: what a skill Aura's own authors wrote and reviewed gets. This one is
    #: bounded compute, and the boundary is load-bearing rather than incidental.
    effect_scope = "sandboxed_compute"

    async def execute(
        self, params: dict[str, Any], context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        result = {entrypoint}(dict(params or {{}}))
        if not isinstance(result, dict):
            return {{
                "ok": False,
                "error": (
                    "forged skill returned "
                    f"{{type(result).__name__}}, and the contract is a dict"
                ),
            }}
        return result
'''

_IDENTIFIER = re.compile(r"[^0-9a-zA-Z_]+")


class ArtifactError(ValueError):
    """The file is not a well-formed forged artifact."""


def class_name_for(skill_name: str) -> str:
    """A valid, stable CamelCase class name for a skill name.

    Digits and stray punctuation are the interesting cases. ``3d_render``
    becomes ``Skill3dRender`` rather than ``3dRender``, which is not an
    identifier and would have failed at import — after the file was written and
    the catalog reloaded.
    """
    cleaned = _IDENTIFIER.sub("_", str(skill_name or "").strip())
    parts = [p for p in cleaned.split("_") if p]
    if not parts:
        raise ArtifactError("skill name has no usable characters")
    camel = "".join(p[:1].upper() + p[1:] for p in parts)
    if not camel.endswith("Skill"):
        camel = f"{camel}Skill"
    # A leading digit is the case that matters: ``3d_render`` yields
    # ``3dRenderSkill``, which is not an identifier and would have failed at
    # import — after the file was written and the catalog reloaded.
    return camel if camel[0].isalpha() else f"Forged{camel}"


def digest_of(verified_source: str) -> str:
    """Content address of a verified region."""
    return hashlib.blake2b(verified_source.encode("utf-8"), digest_size=16).hexdigest()


@dataclass(frozen=True)
class ForgedArtifact:
    """A deployable file and the evidence it is bound to."""

    skill_name: str
    class_name: str
    verified_source: str
    text: str
    digest: str

    @property
    def adapter(self) -> str:
        return split(self.text)[1]


def assemble(
    verified_source: str,
    *,
    skill_name: str,
    description: str,
    entrypoint: str = "run",
) -> ForgedArtifact:
    """Build the deployable file from a verified region.

    The verified region is copied in unchanged. Nothing here reformats,
    reindents or "cleans up" model-authored source, because any edit would
    invalidate the digest the evidence is filed under — silently, since the
    result would still be valid Python.
    """
    if VERIFIED_REGION_MARKER in verified_source:
        raise ArtifactError("the verified region contains the region marker")
    if not verified_source.strip():
        raise ArtifactError("the verified region is empty")

    class_name = class_name_for(skill_name)
    safe_description = str(description or skill_name).replace('"', "'").replace("\\", "/")
    safe_description = " ".join(safe_description.split())[:200]
    adapter = ADAPTER_TEMPLATE.format(
        class_name=class_name,
        skill_name=str(skill_name),
        description=safe_description,
        entrypoint=entrypoint,
    )
    body = verified_source if verified_source.endswith("\n") else verified_source + "\n"
    text = f"{body}\n{VERIFIED_REGION_MARKER}\n\n{adapter}"
    return ForgedArtifact(
        skill_name=str(skill_name),
        class_name=class_name,
        verified_source=verified_source,
        text=text,
        digest=digest_of(verified_source),
    )


def split(text: str) -> tuple[str, str]:
    """Return ``(verified_region, adapter)``. Raises when the file is malformed.

    An absent marker and a repeated one are both refusals rather than a
    best-effort parse. A file with two markers has an ambiguous verified region,
    and guessing which one is authoritative is how a second region gets
    smuggled past the hash.
    """
    occurrences = text.count(VERIFIED_REGION_MARKER)
    if occurrences == 0:
        raise ArtifactError("file carries no verified-region marker")
    if occurrences > 1:
        raise ArtifactError(f"file carries {occurrences} verified-region markers")
    head, _, tail = text.partition(VERIFIED_REGION_MARKER)
    return head[:-1] if head.endswith("\n") else head, tail.lstrip("\n")


def load_verified_region(path: Path | str, *, expected_digest: str) -> str:
    """Read the verified region and refuse if it no longer hashes to the evidence.

    ``expected_digest`` comes from the verification record, so this is the check
    that keeps "verified" true over time rather than only at the moment of
    writing.
    """
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArtifactError(f"cannot read forged skill at {target}: {exc}") from exc
    region, _ = split(text)
    actual = digest_of(region)
    if actual != expected_digest:
        raise ArtifactError(
            f"{target.name} no longer matches its verification evidence "
            f"(recorded {expected_digest[:12]}, found {actual[:12]})"
        )
    return region


@dataclass(frozen=True)
class LedgerEntry:
    """One forged skill, its evidence, and how it has fared since."""

    skill_name: str
    digest: str
    path: str
    verified_at: float
    boundary: str
    probes_executed: int
    probes_precommitted: int
    summary: str
    version: int = 1
    successes: int = 0
    failures: int = 0

    @property
    def reliability(self) -> float | None:
        """Measured success rate, or None when it has never been called.

        None rather than a default, because a skill nobody has run has no
        reliability and any number here would be read as one. Callers that need
        to sort can decide what to do with the absence; they cannot decide that
        if it arrives as ``0.5``.
        """
        total = self.successes + self.failures
        return self.successes / total if total else None

    def to_dict(self) -> dict[str, object]:
        return {
            "skill_name": self.skill_name,
            "digest": self.digest,
            "path": self.path,
            "verified_at": self.verified_at,
            "boundary": self.boundary,
            "probes_executed": self.probes_executed,
            "probes_precommitted": self.probes_precommitted,
            "summary": self.summary,
            "version": self.version,
            "successes": self.successes,
            "failures": self.failures,
        }

    @staticmethod
    def from_dict(raw: object) -> LedgerEntry:
        if not isinstance(raw, dict):
            raise ArtifactError("ledger entry is not an object")
        name = str(raw.get("skill_name") or "").strip()
        digest = str(raw.get("digest") or "").strip()
        if not name or not digest:
            raise ArtifactError("ledger entry is missing its name or digest")
        return LedgerEntry(
            skill_name=name,
            digest=digest,
            path=str(raw.get("path") or ""),
            verified_at=float(raw.get("verified_at") or 0.0),
            boundary=str(raw.get("boundary") or "none"),
            probes_executed=int(raw.get("probes_executed") or 0),
            probes_precommitted=int(raw.get("probes_precommitted") or 0),
            summary=str(raw.get("summary") or ""),
            version=int(raw.get("version") or 1),
            successes=int(raw.get("successes") or 0),
            failures=int(raw.get("failures") or 0),
        )


def next_version_path(path: Path | str) -> Path:
    """Where the current file should be archived before it is replaced.

    Voyager renames a superseded skill to ``V2``, ``V3`` and keeps it. The
    reason is worth restating: a regenerated skill is not necessarily better
    than the one it replaces, and overwriting is the one operation that makes
    finding that out impossible. The archive lands beside the skill under a
    dot-directory so the catalog's walk skips it.
    """
    target = Path(path)
    archive = target.parent / ".forge_versions"
    n = 2
    while (archive / f"{target.stem}.v{n}.py").exists():
        n += 1
    return archive / f"{target.stem}.v{n}.py"


class ForgeLedger:
    """Which forged skills exist, what proved them, and how they have fared.

    Held apart from the files themselves so that a skill deleted from disk stops
    being a capability immediately, while its evidence survives for the audit.
    The reverse — a file with no ledger entry — is the case that matters at
    runtime: :meth:`entry_for` returns nothing, the digest check has nothing to
    check against, and the execution path refuses. Code that appeared in
    ``skills/`` without going through the forge therefore does not run, which is
    the property that makes the directory safe to leave writable.
    """

    def __init__(self, path: Path | None = None) -> None:

        if path is None:
            from core.config import config

            path = Path(config.paths.data_dir) / "forge" / "verified_skills.json"
        self._path = Path(path)
        self._lock = checked_lock("core.skill_management.forged_artifact")
        self._entries: dict[str, LedgerEntry] = {}
        self._loaded = False

    @property
    def path(self) -> Path:
        return self._path

    def _load_locked(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            from core.runtime.errors import record_degradation

            record_degradation(
                "forge_ledger",
                exc,
                action="started with an empty forge ledger after the file failed to load",
                severity="warning",
            )
            return
        for item in (raw or {}).get("skills", []):
            try:
                entry = LedgerEntry.from_dict(item)
            except ArtifactError:
                continue
            self._entries[entry.skill_name] = entry

    def entries(self) -> list[LedgerEntry]:
        with self._lock:
            self._load_locked()
            return sorted(self._entries.values(), key=lambda e: e.skill_name)

    def entry_for(self, skill_name: str) -> LedgerEntry | None:
        with self._lock:
            self._load_locked()
            return self._entries.get(str(skill_name))

    def _payload_locked(self) -> str:
        return json.dumps(
            {"skills": [e.to_dict() for e in sorted(self._entries.values(), key=lambda x: x.skill_name)]},
            indent=2,
            sort_keys=True,
        )

    async def record_async(self, entry: LedgerEntry) -> None:
        """Add or replace an entry and persist it through the write gateway.

        The version and the reliability counts both belong to a *digest*, not to
        a name, and the two cases are opposites.

        Re-recording the same digest — a re-verification of unchanged code —
        keeps both. Without that, verifying a skill again resets its version to
        one and erases every success and failure it has accumulated, so the
        record would say the skill is new and untried each time anyone checked
        it.

        A new digest is different code. Its version increments and the counts
        start at zero, because the successes belonged to the implementation that
        was just replaced and carrying them over would credit new code with the
        old code's record.
        """
        with self._lock:
            self._load_locked()
            previous = self._entries.get(entry.skill_name)
            if previous is not None:
                if previous.digest == entry.digest:
                    entry = LedgerEntry(
                        **{
                            **entry.to_dict(),  # type: ignore[arg-type]
                            "version": previous.version,
                            "successes": previous.successes,
                            "failures": previous.failures,
                        }
                    )
                else:
                    entry = LedgerEntry(
                        **{
                            **entry.to_dict(),  # type: ignore[arg-type]
                            "version": previous.version + 1,
                            "successes": 0,
                            "failures": 0,
                        }
                    )
            self._entries[entry.skill_name] = entry
            payload = self._payload_locked()
        await self._write_async(payload)

    async def record_outcome_async(self, skill_name: str, *, succeeded: bool) -> None:
        """Fold a real call into the entry's counts.

        This is the half Voyager leaves out: its library records that a skill
        once worked, never that it has since stopped. A skill whose failures
        accumulate is visible here without anyone having to re-verify it.
        """
        with self._lock:
            self._load_locked()
            entry = self._entries.get(str(skill_name))
            if entry is None:
                return
            self._entries[entry.skill_name] = LedgerEntry(
                **{
                    **entry.to_dict(),  # type: ignore[arg-type]
                    "successes": entry.successes + (1 if succeeded else 0),
                    "failures": entry.failures + (0 if succeeded else 1),
                }
            )
            payload = self._payload_locked()
        await self._write_async(payload)

    async def _write_async(self, payload: str) -> None:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.errors import record_degradation
        from core.runtime.file_write_gateway import get_file_write_gateway

        gateway = get_file_write_gateway()
        try:
            with local_internal_governed_scope("skill_management.forge_ledger"):
                await gateway.ensure_directory_async(
                    self._path.parent, source="skill_management.forge_ledger"
                )
                await gateway.write_text_async(
                    self._path, payload, source="skill_management.forge_ledger"
                )
        except (OSError, RuntimeError, ValueError) as exc:
            record_degradation(
                "forge_ledger",
                exc,
                action="kept the in-memory forge ledger but failed to persist it",
                severity="warning",
            )


_ledger: ForgeLedger | None = None


def get_forge_ledger() -> ForgeLedger:
    global _ledger
    if _ledger is None:
        _ledger = ForgeLedger()
    return _ledger
