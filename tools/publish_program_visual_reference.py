"""Register a checked external artwork export for read-only runtime acquisition."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def publish_reference_bundle(bundle: Path, *, root: Path | None = None) -> Path:
    """Index a checked local export; runtime acquisition still rechecks its bytes.

    The bundle remains an external dependency. Moving or deleting it makes
    the reference unavailable. Runtime perception has no publication method.
    """
    from core.governance_context import local_internal_governed_scope
    from core.perception.visual_references_for_current_sources import (
        _read, reference_index_root, registered_reference,
    )
    from core.runtime.file_write_gateway import get_file_write_gateway

    bundle = bundle.expanduser().resolve()
    reference = registered_reference(bundle)
    receipt = _read(bundle/'receipt.json', 1_000_000)
    registered_reference(bundle, receipt_sha256=hashlib.sha256(receipt).hexdigest())
    directory = (reference_index_root() if root is None else root)/reference.source[4:]
    path = directory/f'{reference.declaration.character_id}-{reference.frame_number}-{reference.coordinates.reference_sha256}.json'
    record = dict(schema='aura.visual_reference_index.v1', source=reference.source,
                  bundle=str(bundle), receipt_sha256=hashlib.sha256(receipt).hexdigest())
    with local_internal_governed_scope('publish_visual_reference'):
        gateway = get_file_write_gateway()
        gateway.ensure_directory(directory, source='publish_visual_reference')
        gateway.write_text(path, json.dumps(record, sort_keys=True)+'\n', source='publish_visual_reference')
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(dict(index=str(publish_reference_bundle(args.bundle)),
                          task_passed=None, presentation_qualified=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
