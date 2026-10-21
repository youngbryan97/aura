"""Acquire registered artwork only for the program bytes read on this visit.

The local index points to external renderer receipts. Each acquisition checks
the source, declaration and image bytes again. A missing or damaged bundle is
unavailable evidence, and an exported name supplies no observed task role.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path

import numpy as np
from PIL import Image

from core.perception.a_visual_reference_in_the_picture import _digest
from core.perception.coordinates_in_a_visual_reference import (
    SourceImageCoordinates, calibrated_source_coordinates,
)
from core.perception.visual_assets_in_a_flash_file import (
    DeclaredVisualAsset, FlashVisualAssets, read_flash_visual_assets,
)
from core.verify import invariant

MAX_REFERENCES = 32


@dataclass(frozen=True)
class RegisteredVisualReference:
    source: str
    declaration: DeclaredVisualAsset
    coordinates: SourceImageCoordinates
    pixels: np.ndarray = field(repr=False, compare=False)
    frame_number: int = 1
    declared_frame_count: int | None = None

    def __post_init__(self) -> None:
        if (type(self.frame_number) is not int or not 1 <= self.frame_number <= 4096
                or self.declared_frame_count is not None and (type(self.declared_frame_count) is not int
                    or not self.frame_number <= self.declared_frame_count <= 4096)):
            raise ValueError('a visual reference needs a bounded source frame and optional declared count')

    @property
    def key(self) -> tuple:
        return self.source, self.declaration.character_id, self.coordinates.reference_sha256

    @property
    def variant_key(self) -> tuple:
        """Equal pixels in different source frames can have different origins."""
        return (*self.key, self.frame_number)


@dataclass(frozen=True)
class ReferencesAcquired:
    references: tuple[RegisteredVisualReference, ...] = ()
    unavailable: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferenceVariantCoverage:
    source: str
    character_id: int
    declared_frames: int | None
    available_frames: tuple[int, ...]
    missing_frames: tuple[int, ...] | None


def reference_variant_coverage(references: tuple[RegisteredVisualReference, ...]) -> tuple[ReferenceVariantCoverage, ...]:
    """Static reference coverage cannot establish an active frame or task outcome."""
    groups = {}
    for reference in references:
        groups.setdefault((reference.source, reference.declaration.character_id), []).append(reference)
    result = []
    for (source, character), variants in groups.items():
        counts = {r.declared_frame_count for r in variants}
        count = next(iter(counts)) if len(counts) == 1 else None
        available = tuple(sorted({r.frame_number for r in variants}))
        missing = tuple(frame for frame in range(1, count+1) if frame not in available) if count is not None else None
        result.append(ReferenceVariantCoverage(source, character, count, available, missing))
    return tuple(result)


def interleaved_reference_variants(references: tuple[RegisteredVisualReference, ...]) -> tuple[RegisteredVisualReference, ...]:
    """Give every asset family a visit before spending its next visual variant."""
    groups = {}
    for reference in references:
        groups.setdefault((reference.source, reference.declaration.character_id), []).append(reference)
    for variants in groups.values():
        variants.sort(key=lambda reference: reference.frame_number)
    return tuple(group[index] for index in range(max((len(g) for g in groups.values()), default=0))
                 for group in groups.values() if index < len(group))


def reference_index_root() -> Path:
    from core.runtime.state_ownership import state_root

    return state_root()/'state'/'visual-references'


def current_visual_sources(tables: object) -> tuple[FlashVisualAssets, ...]:
    """Only bounded complete declarations from a current read can select artwork."""
    if not isinstance(tables, tuple) or len(tables) > 6:
        return ()
    return tuple(table for table in tables if isinstance(table, FlashVisualAssets) and table.complete
                 and re.fullmatch(r'swf:[0-9a-f]{64}', table.source))


def _read(path: Path, most: int) -> bytes:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= most:
        raise ValueError('reference file is missing, linked or over budget')
    with path.open('rb') as stream:
        raw = stream.read(most+1)
    if len(raw) > most:
        raise ValueError('reference file grew beyond its byte budget')
    return raw


def _inside(root: Path, name: object) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError('a reference must name a contained relative file')
    path = root/name
    if any(part.is_symlink() for part in (path, *path.parents) if part.is_relative_to(root)):
        raise ValueError('a reference path crosses a link')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('a reference path leaves its bundle')
    return path


def _image(raw: bytes) -> np.ndarray:
    import io

    with Image.open(io.BytesIO(raw)) as image:
        if image.format != 'PNG' or min(image.size) < 16 or max(image.size) > 512:
            raise ValueError('a reference needs bounded PNG pixels')
        return np.array(image.convert('RGBA'))


def registered_reference(bundle: Path, *, receipt_sha256: str | None = None) -> RegisteredVisualReference:
    """Recheck an external static export and its independently measured coordinates."""
    receipt = _read(bundle/'receipt.json', 1_000_000)
    if receipt_sha256 is not None and hashlib.sha256(receipt).hexdigest() != receipt_sha256:
        raise ValueError('the published renderer receipt changed')
    record = json.loads(receipt)
    if (record.get('schema') != 'aura.program_visual_reference.v1' or record.get('status') != 'rendered'
            or not isinstance(record.get('images'), list) or len(record['images']) != 1):
        raise ValueError('one successfully rendered registered frame is required')
    raw_source = _read(bundle/'source.swf', 40*1024*1024)
    table = read_flash_visual_assets(raw_source)
    declared = record['declaration']
    asset = table.named(declared['name'])
    if asset is None or table.source != record['source'] or json.loads(json.dumps(asdict(asset))) != declared:
        raise ValueError('the rendered declaration no longer matches its source bytes')
    frame_number = record.get('frame_number', 1)
    if type(frame_number) is not int or frame_number < 1:
        raise ValueError('the reference frame must be a positive integer')
    from core.perception.visual_frames_in_a_source_declaration import read_declared_visual_frames

    timeline = read_declared_visual_frames(raw_source, asset)
    if 'frame_number' in record:
        if (not timeline.complete or frame_number > timeline.frame_count
                or type(record.get('declared_frame_count')) is not int
                or record.get('declared_frame_count') != timeline.frame_count):
            raise ValueError('the reference frame is not in the current complete source timeline')
        if asset.kind == 'sprite' and Path(record['images'][0]['path']).stem != str(frame_number):
            raise ValueError('the rendered image does not identify the selected source frame')
    image = record['images'][0]
    png = _read(_inside(bundle/'rendered', image['path']), 8_000_000)
    if hashlib.sha256(png).hexdigest() != image['file_sha256']:
        raise ValueError('reference image bytes changed')
    pixels = _image(png)
    if image['size'] != [pixels.shape[1], pixels.shape[0]]:
        raise ValueError('reference dimensions changed')
    registration = record['registration']
    svg = _read(_inside(bundle, registration['svg']), 2*1024*1024)
    raster_raw = _read(_inside(bundle, registration['raster']), 8_000_000)
    raster = _image(raster_raw)
    if (hashlib.sha256(raster_raw).hexdigest() != registration['raster_file_sha256']
            or hashlib.sha256(raster.tobytes()).hexdigest() != registration['raster_raw_sha256']):
        raise ValueError('independent coordinate raster bytes changed')
    coordinates = calibrated_source_coordinates(table.source, svg, raster, pixels)
    if coordinates is None or json.loads(json.dumps(asdict(coordinates))) != registration['coordinates']:
        raise ValueError('the stored source coordinates do not match measured raster alignment')
    pixels.setflags(write=False)
    return RegisteredVisualReference(table.source, asset, coordinates, pixels, frame_number,
                                     timeline.frame_count if timeline.complete else None)


def acquire_visual_references(tables: tuple, *, root: Path | None = None) -> ReferencesAcquired:
    """Load a bounded local catalog for current source identities, off the actor loop."""
    root = reference_index_root() if root is None else root
    found, unavailable, count, identities = [], [], 0, set()
    for table in current_visual_sources(tables):
        directory = root/table.source[4:]
        if not directory.exists():
            unavailable.append(table.source+': no published references')
            continue
        if directory.is_symlink():
            unavailable.append(table.source+': linked index refused')
            continue
        for path in sorted(islice(directory.iterdir(), MAX_REFERENCES+1)):
            count += 1
            if count > MAX_REFERENCES:
                return ReferencesAcquired(interleaved_reference_variants(tuple(found)), (*unavailable, 'reference index budget exceeded'))
            try:
                item = json.loads(_read(path, 8192))
                if item.get('schema') != 'aura.visual_reference_index.v1' or item.get('source') != table.source:
                    raise ValueError('the index does not name the current source')
                bundle = Path(item['bundle'])
                if not bundle.is_absolute() or not re.fullmatch('[0-9a-f]{64}', item['receipt_sha256']):
                    raise ValueError('the index has no complete local bundle receipt')
                reference = registered_reference(bundle, receipt_sha256=item['receipt_sha256'])
                if not _belongs(reference, table):
                    raise ValueError('the reference is not a declaration of the current source')
                if reference.variant_key not in identities:
                    identities.add(reference.variant_key)
                    found.append(reference)
            except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
                unavailable.append(f'{table.source}: {type(exc).__name__}: {str(exc)[:180]}')
    return ReferencesAcquired(interleaved_reference_variants(tuple(found)), tuple(unavailable))


def _belongs(reference: RegisteredVisualReference, table: FlashVisualAssets) -> bool:
    return (reference.source == table.source and reference.coordinates.source == table.source
            and reference.coordinates.reference_sha256 == _digest(reference.pixels)
            and table.named(reference.declaration.name) == reference.declaration)


@invariant('perception.reference_catalog_requires_the_current_source', scope='perception',
           owner='core/perception/visual_references_for_current_sources.py', observational=False)
def _current_reference_source_invariant() -> tuple:
    source = 'swf:'+'1'*64
    asset = DeclaredVisualAsset('arbitrary', 7, 'sprite', 0, 39, (10,))
    pixels = np.zeros((32, 32, 4), np.uint8)
    coordinates = SourceImageCoordinates(source, _digest(pixels), '0'*64, (32, 32),
                                         ((1., 0.), (0., 1.)), (0., 0.), 0.)
    reference = RegisteredVisualReference(source, asset, coordinates, pixels)
    table = FlashVisualAssets(source, (asset,))
    assert _belongs(reference, table)
    assert not _belongs(reference, FlashVisualAssets('swf:'+'2'*64, (asset,)))
    assert current_visual_sources((FlashVisualAssets(source, (asset,), ('incomplete',)),)) == ()
    return ()
