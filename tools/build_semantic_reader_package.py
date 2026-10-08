#!/usr/bin/env python3
"""Package a qualified program reader for her runtime, and switch it on or back.

``build`` copies the reader and the five documents its qualification rests
on into ``artifacts/rlc/<package>/`` and writes the v2 activation
(core/learning/compositional_semantic_qualification.build_semantic_reader_activation),
which refuses unless G04's and G05's closure rules hold in their reports and
independent checks and every G10 predicate holds.

``activate`` makes a built package her operational one by writing its
activation to ``training/fused-model/compositional-semantic-active.json``.
``rollback`` puts back exactly what that file held before, or removes it when
it held nothing, so the runtime returns to the package it used before.

Usage:
    build_semantic_reader_package.py build --reader FILE --qualification FILE
        --transfer-report FILE --transfer-verification FILE
        --public-report FILE --public-verification FILE --claim TEXT
    build_semantic_reader_package.py activate --package DIR
    build_semantic_reader_package.py rollback --package DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ACTIVE = ROOT / "training/fused-model/compositional-semantic-active.json"


def build(args: argparse.Namespace) -> Path:
    from core.learning.compositional_semantic_qualification import (
        build_semantic_reader_activation,
        canonical_document_bytes,
        semantic_reader_package_id,
    )
    from core.learning.semantic_operation_peaks import semantic_reader_from_dict

    raw = args.reader.expanduser().read_bytes()
    package = ROOT / "artifacts/rlc" / semantic_reader_package_id(
        semantic_reader_from_dict(json.loads(raw)).receipt_sha256)
    evidence = package / "evidence"
    evidence.mkdir(parents=True, exist_ok=False)
    reader_path = package / "transducer.json"
    reader_path.write_bytes(raw)
    copies = {}
    for name in ("qualification", "transfer_report", "transfer_verification", "public_report",
                 "public_verification"):
        source = getattr(args, name).expanduser()
        target = evidence / f"{name}.json"
        shutil.copyfile(source, target)
        copies[name] = target
    activation = build_semantic_reader_activation(
        repo_root=ROOT, reader_path=reader_path, qualification_path=copies["qualification"],
        transfer_report_path=copies["transfer_report"],
        transfer_verification_path=copies["transfer_verification"],
        public_report_path=copies["public_report"], public_verification_path=copies["public_verification"],
        claim_boundary=args.claim)
    (package / "activation.json").write_bytes(canonical_document_bytes(activation))
    return package


def activate(package: Path) -> None:
    activation = (package / "activation.json").read_bytes()
    before = package / "previous_active.json"
    if before.exists():
        raise SystemExit(f"{before} exists; roll back before activating again")
    record = {"existed": ACTIVE.exists()}
    if ACTIVE.exists():
        record["bytes_sha256"] = hashlib.sha256(ACTIVE.read_bytes()).hexdigest()
        (package / "previous_active.bytes").write_bytes(ACTIVE.read_bytes())
    before.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    ACTIVE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE.write_bytes(activation)


def rollback(package: Path) -> None:
    before = package / "previous_active.json"
    record = json.loads(before.read_text(encoding="utf-8"))
    if record["existed"]:
        previous = (package / "previous_active.bytes").read_bytes()
        if hashlib.sha256(previous).hexdigest() != record["bytes_sha256"]:
            raise SystemExit("the kept copy of the previous activation changed")
        ACTIVE.write_bytes(previous)
        (package / "previous_active.bytes").unlink()
    else:
        ACTIVE.unlink(missing_ok=True)
    before.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    built = sub.add_parser("build")
    for name in ("reader", "qualification", "transfer_report", "transfer_verification", "public_report",
                 "public_verification"):
        built.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, required=True)
    built.add_argument("--claim", required=True)
    for name in ("activate", "rollback"):
        sub.add_parser(name).add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        print(build(args))
    elif args.command == "activate":
        activate(args.package)
    else:
        rollback(args.package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
