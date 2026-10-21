"""Match source artwork beside ordinary play, then require current material.

One optional CPU operation owns a copied capture. New frames never queue
behind it. A delayed result must traverse the original continuing point
basis before publication; it cannot select a body, goal or collision rule.
"""
from __future__ import annotations

import contextvars
import math
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.perception.a_material_frame_kept_for_its_owner import keep_material_frame
from core.perception.a_source_point_on_current_material import SourcePointOnTrack, SourcePointSeen
from core.perception.a_source_point_on_visible_marks import SourcePointOnVisibleMarks, VisibleSourcePoint, attach_source_point_to_visible_marks
from core.perception.points_a_consumer_still_needs import MaterialRetention, MaterialRegion
from core.perception.a_visual_reference_in_the_picture import reference_appearances
from core.perception.source_reference_proposals import SourceReferenceProposal
from core.perception.named_material_in_a_consumed_frame import MaterialMarks
from core.perception.visual_references_for_current_sources import (
    RegisteredVisualReference, _belongs, acquire_visual_references, current_visual_sources, reference_variant_coverage,
)
from core.runtime.executors import submit_blocking_io
from core.runtime.lockdep import checked_thread_semaphore
from core.verify import invariant

_SLOTS = checked_thread_semaphore('perception.source_reference_operations', 1)


def _acquired_for_play(tables: tuple, layers: tuple, root: Any):
    catalog = acquire_visual_references(tables, root=root)
    if not layers:
        return catalog
    from core.perception.composed_source_artwork_for_pixel_search import with_source_layer_proposals
    from core.perception.visual_references_for_current_sources import ReferencesAcquired, interleaved_reference_variants

    references, refused = with_source_layer_proposals(catalog.references, layers)
    return ReferencesAcquired(interleaved_reference_variants(references), (*catalog.unavailable, *refused))


def retention_for_source_frames(observer: Any, at: float, frames: tuple) -> MaterialRetention | None:
    """Keep only currently observed source marks within the existing point budget."""
    marks = getattr(observer, 'material_marks', None)
    if (not isinstance(marks, MaterialMarks) or marks.pixels != getattr(observer, 'pixels_consumed', None)
            or type(at) not in (int, float) or not math.isfinite(at) or marks.pixels.at != at
            or not isinstance(frames, tuple) or len(frames) > 64):
        return None
    visible = {(p.number, p.born) for p in marks.points}
    retained, regions = set(), []
    for item in frames:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        reference, binding = item
        if (not isinstance(reference, RegisteredVisualReference) or not isinstance(binding, SourcePointOnVisibleMarks)
                or reference.source != binding.source or reference.coordinates.reference_sha256 != binding.reference_sha256):
            continue
        sample = binding.inspect(observer, at, source=reference.source).current
        if sample is None:
            continue
        anchors = tuple((a[0], a[1]) for a in binding.anchors if (a[0], a[1]) in visible)
        retained.update(anchors)
        if sample.sampling_region is not None:
            regions.append((sample.sampling_region, anchors))
    names = tuple(sorted(retained)[:64])
    if not names:
        return None
    proposed = []
    for box, anchors in regions:
        supported = tuple(key for key in anchors if key in names)
        if len(supported) >= 3:
            proposed.append(MaterialRegion(box, supported))
    return MaterialRetention(marks.epoch, at, at+.5, names, tuple(proposed[:4]))


@dataclass(frozen=True)
class SourceReferencePoint:
    """A current source origin, with provenance and no inferred task role."""

    source: str
    character_id: int
    exported_name: str
    reference_sha256: str
    point: SourcePointSeen
    hypotheses: int = 1


@dataclass(frozen=True)
class SourceReferenceFrame:
    """A current source coordinate frame with no object or controlled identity."""

    source: str
    character_id: int
    exported_name: str
    reference_sha256: str
    point: VisibleSourcePoint


def _coalesced(points: list[SourceReferencePoint]) -> tuple[tuple, tuple]:
    """Enclose agreeing pose estimates without counting them as independent evidence."""
    groups: dict = {}
    for item in points:
        key = (item.source, item.character_id, item.reference_sha256,
               item.point.number, item.point.born, item.point.at)
        groups.setdefault(key, []).append(item)
    kept, refused = [], []
    for group in groups.values():
        first = group[0]
        if len(group) == 1:
            kept.append(first)
            continue
        centre = tuple(math.fsum(item.point.position[axis] for item in group)/len(group) for axis in (0, 1))
        error = max(math.dist(centre, item.point.position)+item.point.error for item in group)
        if error > 1.:
            refused.append('reference_pose_disagreement')
            continue
        point = SourcePointSeen(first.point.number, first.point.born, first.point.at, centre, error)
        kept.append(SourceReferencePoint(first.source, first.character_id, first.exported_name,
                                        first.reference_sha256, point, len(group)))
    return tuple(kept), tuple(refused)


@dataclass(frozen=True)
class _Matched:
    reference: RegisteredVisualReference
    bindings: tuple[SourcePointOnTrack, ...]
    refused: tuple[str, ...]
    views: tuple[tuple[float, float], ...] = ()
    frames: tuple[SourcePointOnVisibleMarks, ...] = ()
    appearances: tuple = ()
    capture: Any = None


@dataclass
class _Work:
    raw: Future
    generation: int
    kind: str
    started: float


def _matched(reference: RegisteredVisualReference, frame: Any, *, angle: float = 0., view: tuple | None = None) -> _Matched:
    bindings, refused, frames = [], [], []
    # Each job covers the scale range at one orientation. Successive catalog
    # passes cover the turn; all poses still need original-pixel qualification.
    scales = tuple(sorted({1., *map(float, np.geomspace(.25, 4., 31))}))
    angles = (angle,)
    if view is not None:
        scale, orientation = view
        scales = tuple(float(v) for v in scale*np.geomspace(.8, 1.25, 7) if .05 <= v <= 8)
        angles = tuple((orientation+step+math.pi) % (2*math.pi)-math.pi
                       for step in (-math.pi/16, 0., math.pi/16))
    poses = reference_appearances(frame.picture, reference.pixels, scales=scales, angles=angles)
    for pose in poses:
        attached = reference.coordinates.on_track((0., 0.), pose, frame.picture, reference.pixels,
                                                  frame, frame.at, source=reference.source)
        if attached.binding is None:
            refused.append(attached.reason)
        else:
            bindings.append(attached.binding)
        # An origin attached to a fragment cannot suppress the independently
        # observed whole reference. These retain separate ownership.
        visible = attach_source_point_to_visible_marks(reference.coordinates, (0., 0.), pose,
            frame.picture, reference.pixels, frame, frame.at, source=reference.source)
        if visible.binding is not None:
            frames.append(visible.binding)
        else:
            refused.append(visible.reason)
    views = tuple((float(math.sqrt(np.linalg.det(pose.linear))),
                   float(math.atan2(pose.linear[1][0], pose.linear[0][0]))) for pose in poses)
    return _Matched(reference, tuple(bindings), tuple(refused) if poses else ('no_qualified_appearance',),
                    views, tuple(frames), poses, frame.pixels_consumed)


def _finished(raw: Future, slots: Any) -> None:
    try:
        if not raw.cancelled():
            raw.exception()
    finally:
        slots.release()


def _with_current_material_frames(retained: tuple, offered: tuple, observer: Any, at: float, tables: tuple) -> tuple:
    """Renew a measured source frame using its own current named material."""
    if not isinstance(offered, tuple) or len(offered) > 32:
        return retained, ('source_frame_offer_capacity_unavailable',)
    refused = []
    for offer in offered:
        if not isinstance(offer, tuple) or len(offer) != 2:
            refused.append('source_frame_offer_unreadable')
            continue
        reference, fresh = offer
        if (not isinstance(reference, RegisteredVisualReference) or not isinstance(fresh, SourcePointOnVisibleMarks)
                or reference.source != fresh.source or reference.key[2] != fresh.reference_sha256
                or not any(_belongs(reference, table) for table in current_visual_sources(tables))):
            refused.append('source_frame_offer_source_changed')
            continue
        sample = fresh.inspect(observer, at, source=reference.source)
        if sample.current is None:
            refused.append(sample.reason)
            continue
        names = {(a[0], a[1]) for a in fresh.anchors}
        retained = tuple((r, old) for r, old in retained if r.variant_key != reference.variant_key or old.epoch != fresh.epoch
                         or len(names & {(a[0], a[1]) for a in old.anchors}) < 3)
        if len(retained) < 64:
            retained = (*retained, (reference, fresh))
        else:
            refused.append('source_frame_capacity_unavailable')
    return retained, tuple(refused)


class SourceReferencesInPlay:
    """An event-loop-owned catalog and at most one worker-owned operation."""

    def __init__(self, *, root: Any = None, interval_s: float = 1.) -> None:
        if type(interval_s) not in (int, float) or not math.isfinite(interval_s) or interval_s < 0:
            raise ValueError('the reference cadence must be finite and nonnegative')
        self.root = root
        self.interval_s = interval_s
        self.tables: tuple = ()
        self.layers: tuple = ()
        self.catalog: tuple = ()
        self.variant_coverage: tuple = ()
        self.compositions: tuple = ()
        self.bindings: tuple = ()
        self.points: tuple = ()
        self.frames: tuple = ()
        self.source_frames: tuple = ()
        self.retention: MaterialRetention | None = None
        self.proposals: tuple = ()
        self._proposal_revision = 0
        self.unavailable: tuple = ()
        self.refused: tuple = ()
        self.views: dict = {}
        self.status = 'no_current_source'
        self._work: _Work | None = None
        self._generation = 0
        self._cursor = 0
        self._closed = False
        self._next = -math.inf
        self._loaded = False

    def _submit(self, kind: str, call: Any, now: float, *, capture: Any = None) -> None:
        slots = _SLOTS
        if self._work is not None or not slots.acquire(blocking=False):
            self.status = 'reference_capacity_unavailable'
            return
        context = contextvars.copy_context()
        try:
            if capture is not None:
                frame = capture()
                if frame is None:
                    slots.release()
                    self.status = 'capture_not_owned'
                    return
                measure = call
                call = lambda: measure(frame)
            raw = submit_blocking_io(context.run, call, label='source_reference_'+kind)
        except BaseException:
            slots.release()
            raise
        raw.add_done_callback(lambda done: _finished(done, slots))
        self._work = _Work(raw, self._generation, kind, now)
        self._next = now+self.interval_s
        self.status = kind+'_pending'

    def _consume(self, observer: Any = None, at: float | None = None) -> None:
        work = self._work
        if work is None or not work.raw.done():
            return
        self._work = None
        if work.generation != self._generation or work.raw.cancelled():
            return
        try:
            result = work.raw.result()
        except Exception as exc:
            from core.runtime.errors import record_degradation

            record_degradation('source_reference_work', exc, action='reference measurement unavailable')
            self.status = 'reference_work_failed'
            self.unavailable = (type(exc).__name__+': '+str(exc)[:180],)
            self._loaded = True
            return
        if work.kind == 'catalog':
            self.catalog, self.unavailable = result.references, result.unavailable
            self.variant_coverage = reference_variant_coverage(self.catalog)
            from core.perception.composed_source_artwork_for_pixel_search import composition_record

            self.compositions = tuple(record for r in self.catalog if (record := composition_record(r)) is not None)
            self._loaded = True
            # Metadata readiness must not spend the interval between pixel searches.
            self._next = work.started
            self.status = 'catalog_ready' if self.catalog else 'no_registered_reference'
        else:
            self.refused = result.refused
            if result.appearances and result.capture is not None:
                self._proposal_revision += 1
                proposal = SourceReferenceProposal(result.reference, result.capture,
                                                     result.appearances, self._proposal_revision)
                self.proposals = (*tuple(old for old in self.proposals
                                        if old.reference.variant_key != result.reference.variant_key), proposal)[-32:]
            if result.views:
                self.views[result.reference.variant_key] = result.views
            # A failed appearance search cannot erase continuing measured
            # material. Replace only an instance actually rebound by this job.
            delivered = []
            for binding in result.bindings:
                sample = binding.inspect(observer, at, source=result.reference.source) if observer is not None else None
                if sample is None or sample.current is not None:
                    delivered.append(binding)
                else:
                    self.refused = (*self.refused[-31:], sample.reason)
            replaced = {binding.query.key[:4] for binding in delivered}
            self.bindings = tuple((r, b) for r, b in self.bindings
                                  if r.variant_key != result.reference.variant_key or b.query.key[:4] not in replaced)
            self.bindings = (*self.bindings, *((result.reference, b) for b in delivered))[:256]
            for fresh in result.frames:
                sample = fresh.inspect(observer, at, source=result.reference.source) if observer is not None else None
                if sample is not None and sample.current is None:
                    self.refused = (*self.refused[-31:], sample.reason)
                    continue
                names = {(a[0], a[1]) for a in fresh.anchors}
                self.frames = tuple((r, old) for r, old in self.frames
                    if r.variant_key != result.reference.variant_key or old.epoch != fresh.epoch
                    or len(names & {(a[0], a[1]) for a in old.anchors}) < 3)
                if len(self.frames) < 64:
                    self.frames = (*self.frames, (result.reference, fresh))
                else:
                    self.unavailable = (*self.unavailable[-31:], 'source_frame_capacity_unavailable')
            self.status = 'reference_measured'

    def advance(self, observer: Any, picture: Any, at: float, tables: tuple, *, now: float | None = None) -> tuple:
        """Poll without waiting; publish only points measured in this current capture."""
        self.points = ()
        self.source_frames, self.retention = (), None
        if self._closed:
            return ()
        now = time.monotonic() if now is None else now
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (at, now)):
            self.status = 'capture_not_measured'
            return ()
        tables = current_visual_sources(tables)
        layers = source_layers_in_play(tables)
        if tables != self.tables or layers != self.layers:
            self.layers = layers
            self.tables, self.catalog, self.bindings = tables, (), ()
            self.variant_coverage = ()
            self.compositions = ()
            self.frames = ()
            self.proposals = ()
            self.unavailable, self.refused = (), ()
            self.views = {}
            self._loaded = False
            self._generation += 1
            self._cursor, self._next = 0, -math.inf
            if self._work is not None:
                self._work.raw.cancel()
        self._consume(observer, at)
        if not self.tables:
            self.status = 'no_current_source'
            return ()
        from core.perception.search_hints_for_source_frame_variants import with_source_variant_hints

        self.proposals = with_source_variant_hints(self.proposals, self.catalog)
        offered = getattr(observer, 'fresh_source_material_frames', ())
        if isinstance(offered, tuple) and len(offered) <= 32:
            registered = {r.variant_key: r for r in self.catalog}
            registered.update((p.reference.variant_key, p.reference) for p in self.proposals)
            delivered = []
            for item in offered:
                if not isinstance(item, tuple) or len(item) != 2:
                    continue
                key, binding = item
                if (isinstance(key, tuple) and len(key) == 4 and isinstance(key[0], str)
                        and type(key[1]) is int and isinstance(key[2], str)
                        and type(key[3]) in (int, str) and key in registered):
                    delivered.append((registered[key], binding))
            self.frames, refused = _with_current_material_frames(self.frames, tuple(delivered), observer, at, self.tables)
            self.refused = (*self.refused[-31:], *refused)
        kept, points = [], []
        for reference, binding in self.bindings:
            sample = binding.inspect(observer, at, source=reference.source)
            if sample.current is not None:
                kept.append((reference, binding))
                points.append(SourceReferencePoint(reference.source, reference.declaration.character_id,
                    reference.declaration.name, reference.coordinates.reference_sha256, sample.current))
            else:
                self.refused = (*self.refused[-31:], sample.reason)
        self.bindings = tuple(kept)
        self.points, conflicting = _coalesced(points)
        self.refused = (*self.refused[-31:], *conflicting)
        kept_frames, source_frames = [], []
        for reference, binding in self.frames:
            sample = binding.inspect(observer, at, source=reference.source)
            if sample.current is not None:
                kept_frames.append((reference, binding))
                source_frames.append(SourceReferenceFrame(reference.source, reference.declaration.character_id,
                    reference.declaration.name, reference.coordinates.reference_sha256, sample.current))
            else:
                self.refused = (*self.refused[-31:], sample.reason)
        self.frames, self.source_frames = tuple(kept_frames), tuple(source_frames)
        self.retention = retention_for_source_frames(observer, at, self.frames)
        if self._work is not None:
            self.status = self._work.kind+('_late' if now-self._work.started > 5. else '_pending')
            return self.points
        if now < self._next:
            return self.points
        if not self._loaded:
            captured_tables, captured_layers = self.tables, self.layers
            self._submit('catalog', lambda: _acquired_for_play(captured_tables, captured_layers, self.root), now)
        elif self.catalog:
            reference = self.catalog[self._cursor % len(self.catalog)]
            turn = self._cursor//len(self.catalog)
            learned = self.views.get(reference.variant_key, ())
            # Keep broad discovery alive even when a previous pose becomes
            # wrong. A learned view proposes pixels; it grants no identity.
            view = learned[(turn//3) % len(learned)] if learned and turn % 3 != 2 else None
            phase = (turn//3 if learned else turn) % 16
            angle = (phase*math.pi/8+math.pi) % (2*math.pi)-math.pi
            self._cursor += 1
            self._submit('match', lambda frame: _matched(reference, frame, angle=angle, view=view), now,
                         capture=lambda: keep_material_frame(observer, picture, at))
        return self.points

    def retire(self) -> None:
        """Close publication; a running operation keeps its permit until it ends."""
        self._closed = True
        self._generation += 1
        self.catalog, self.bindings, self.points = (), (), ()
        self.layers = ()
        self.compositions = ()
        self.frames, self.source_frames, self.retention = (), (), None
        self.proposals = ()
        self.views = {}
        if self._work is not None:
            self._work.raw.cancel()
            self._work = None
        self.status = 'closed'


def source_tables_in_play() -> tuple:
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    return current_visual_sources(getattr(THE_GUIDE.get(), 'program_visual_assets', ()))


def source_layers_in_play(tables: tuple) -> tuple:
    from core.cognition.a_guide_to_a_place import THE_GUIDE
    from core.perception.source_layers_with_copied_positions import SourceLayerProposal

    layers = getattr(THE_GUIDE.get(), 'program_visual_layers', ())
    sources = {table.source for table in tables}
    if not isinstance(layers, tuple) or len(layers) > 32:
        return ()
    return tuple(layer for layer in layers if isinstance(layer, SourceLayerProposal) and layer.source in sources)


@invariant('perception.retired_reference_workers_cannot_publish', scope='perception',
           owner='core/perception/source_references_while_play_advances.py', observational=False)
def _reference_worker_retirement_invariant() -> tuple:
    worker = SourceReferencesInPlay()
    worker.retire()
    assert worker.advance(None, None, 1., (), now=1.) == () and worker.status == 'closed'
    return ()


@invariant('perception.reference_metadata_does_not_spend_match_cadence', scope='perception',
           owner='core/perception/source_references_while_play_advances.py', observational=False)
def _reference_catalog_readiness_invariant() -> tuple:
    from types import SimpleNamespace

    worker = SourceReferencesInPlay(interval_s=1.)
    result = SimpleNamespace(references=(), unavailable=())
    finished = SimpleNamespace(done=lambda: True, cancelled=lambda: False, result=lambda: result)
    worker._work, worker._next = _Work(finished, worker._generation, 'catalog', 10.), 11.
    worker._consume()
    assert worker._loaded and worker._next == 10. and worker._work is None
    assert not worker.points and not worker.source_frames
    worker.retire()
    return ()
