"""The heads this organ carries, written down and read back.

A head is a small trained thing with a revision. Saving one is cheap; losing
the revision is not, because a rehydrated track record that cannot say which
head produced it is a record of nothing. Every path here carries the revision
alongside the weights.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation only; that module imports this one
    from .service import ControlPoint

import json
import sqlite3
from pathlib import Path
from typing import Any

from core.ontogeny.calibration import (
    OPERATIONAL_SHADOW,
    CalibrationObservation,
)
from core.ontogeny.experience import (
    OutcomeKind,
)
from core.runtime.errors import record_degradation

#: Episodes whose bucket is remembered so a later resolution can find its
#: tally. Bounded: an outcome that lands after this many decisions have gone
#: by is folded in by the next rehydration instead.
_BUCKET_MEMORY = 20_000


class _KeepsItsHeadsOnDisk:
    """Lifted whole from OntogenyCore; see service.py."""

    def rehydrate_track_records(self, limit: int = 6000) -> dict[str, int]:
        """Rebuild the tallies from the corpus. Slow, so it runs on maintenance."""
        rebuilt: dict[str, int] = {}
        with self._lock:
            names = list(self._control_points)
        for name in names:
            try:
                episodes = self._spine.episodes(name, limit=limit)
            except (RuntimeError, OSError, ValueError) as exc:
                record_degradation("ontogeny", exc, severity="debug",
                                   action=f"track-record rehydration skipped for {name}")
                continue
            rebuilt[name] = self._track.hydrate(name, episodes)
        return rebuilt

    def rehydrate_operational_calibration(self, limit: int = _BUCKET_MEMORY) -> dict[str, int]:
        """Rebuild operational cohorts from immutable decision-time shadows.

        The episode reader intentionally exposes a narrow projection and older
        versions omitted ``context_json`` from that projection. Read only that
        provenance column here so restart recovery remains source-bound without
        rewriting or deleting historical incidents.
        """
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.

        rebuilt: dict[str, int] = {}
        with self._lock:
            control_points = list(self._control_points.values())
        for cp in control_points:
            try:
                episodes = self._spine.episodes(cp.name, evidence_only=True, limit=limit)
                contexts = self._episode_contexts([episode.episode_id for episode in episodes])
            except (RuntimeError, OSError, ValueError, sqlite3.Error) as exc:
                record_degradation(
                    "ontogeny", exc, severity="debug",
                    action=f"operational calibration rehydration skipped for {cp.name}",
                )
                continue
            observations: list[CalibrationObservation] = []
            for episode in episodes:
                if episode.outcome is None or not episode.outcome.kind.is_evidence:
                    continue
                probability = (episode.shadow or {}).get(episode.decision)
                if probability is None or episode.shadow_version is None:
                    continue
                probability = min(1.0, max(0.0, float(probability)))
                context = contexts.get(episode.episode_id, episode.context or {})
                revision = str(context.get("runtime_revision") or "legacy-unbound")
                predicted_success = probability >= 0.5
                observations.append(CalibrationObservation(
                    episode_id=episode.episode_id,
                    control_point=episode.control_point,
                    confidence=max(probability, 1.0 - probability),
                    correct=predicted_success == (episode.outcome.kind is OutcomeKind.SUCCESS),
                    decided_at=episode.decided_at,
                    observed_at=episode.outcome.resolved_at,
                    runtime_revision=revision,
                    head_version=int(episode.shadow_version),
                    action=episode.decision,
                    provenance=OPERATIONAL_SHADOW,
                ))
            rebuilt[cp.name] = self._operational_calibration.replace_observations(
                cp.name,
                observations,
                provenance=OPERATIONAL_SHADOW,
            )
            self._operational_calibration.activate(
                cp.name,
                runtime_revision=self._revision_for(cp.name),
                head_version=self._head_version(cp),
                provenance=OPERATIONAL_SHADOW,
            )
        return rebuilt

    def _heads_dir(self) -> Path:
        """Head checkpoints live beside the corpus that produced them.

        Deriving this from the spine rather than from config is the whole
        point. The provenance gate keeps test episodes out of the live corpus,
        but a head is *derived* from a corpus, and a head fitted on simulated
        episodes and written to the live directory is the same contamination
        one level up — it would be loaded by the real instance at next boot and
        would start scoring real decisions from things that never happened.
        Tying every artefact to the store's own root makes a sandbox total
        instead of partial.
        """
        return self._spine.db_path.parent / "heads"

    def _save_head(self, cp: ControlPoint) -> None:

        if not cp.heads:
            return
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            payload: dict[str, Any] = {
                "schema_id": cp.schema.schema_id,
                "actions": list(cp.actions),
                "moments": cp.moments.state_dict() if cp.moments else {},
                "heads": {action: head.state_dict() for action, head in cp.heads.items()},
            }
            gateway = get_file_write_gateway()
            target = self._heads_dir() / f"{cp.name.replace('.', '_')}.json"
            with local_internal_governed_scope(
                "ontogeny_head", domain="state_mutation", receipt_prefix="ontogeny-head"
            ):
                gateway.ensure_directory(target.parent, source="ontogeny_head")
                gateway.write_text(
                    target, json.dumps(payload, ensure_ascii=False), source="ontogeny_head"
                )
        except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
            record_degradation(
                "ontogeny", exc, severity="warning",
                action=f"head checkpoint for {cp.name} not written",
            )

    def _load_heads(self) -> None:
        from .service import (
            logger,
        )

        directory = self._heads_dir()
        if not directory.exists():
            return
        with self._lock:
            control_points = list(self._control_points.values())
        for cp in control_points:
            target = directory / f"{cp.name.replace('.', '_')}.json"
            if not target.exists():
                continue
            try:
                payload = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                record_degradation("ontogeny", exc, severity="warning",
                                   action=f"head checkpoint for {cp.name} unreadable")
                continue
            if payload.get("schema_id") != cp.schema.schema_id:
                logger.info(
                    "ontogeny: heads for %s were fitted on schema %s, now %s — discarding them",
                    cp.name, payload.get("schema_id"), cp.schema.schema_id,
                )
                continue
            heads = cp.ensure_heads(self._units)
            restored = 0
            for action, state in (payload.get("heads") or {}).items():
                head = heads.get(action)
                if head is not None and head.load_state(state):
                    restored += 1
            if restored and cp.moments is not None:
                cp.moments.load_state(payload.get("moments", {}))
            logger.info("ontogeny: restored %d/%d heads for %s", restored, len(heads), cp.name)

    def _activate_operational_cohorts(self) -> None:
        with self._lock:
            control_points = list(self._control_points.values())
        for cp in control_points:
            self._operational_calibration.activate(
                cp.name,
                runtime_revision=self._revision_for(cp.name),
                head_version=self._head_version(cp),
                provenance=OPERATIONAL_SHADOW,
            )
