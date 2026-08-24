#!/usr/bin/env python3
"""Create the fresh 27B recovery package, or verify a certificate against it.

Two subcommands, both CPU-only:

    materialize   derive the package identity from the active descriptor
    verify        check a certificate describes THIS checkpoint and inherits
                  nothing from the 32B verdict it supersedes

Materializing before the campaign runs is the point: the identity exists, and
carries no verdict, so there is a thing for evidence to attach to that cannot
be confused with the package holding the old claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.recovery_package_identity import (  # noqa: E402
    RECOVERY_EVIDENCE_ROOT,
    DescriptorIdentity,
    RecoveryPackageError,
    build_package,
    certificate_errors,
    descriptor_from_manifest,
    load_manifest,
)
from tools.export_active_descriptor import installation_root  # noqa: E402


def _descriptor() -> DescriptorIdentity:
    return descriptor_from_manifest(load_manifest(installation_root()))


def _materialize(args: argparse.Namespace) -> int:
    descriptor = _descriptor()
    package = build_package(descriptor, campaign=args.campaign)
    out = args.out or (
        REPO_ROOT / RECOVERY_EVIDENCE_ROOT / "package.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(package, indent=1, sort_keys=True))
    print(f"wrote {out}")
    print(f"package_id     {package['package_id']}")
    print(f"fingerprint    {descriptor.fingerprint()}")
    print(f"verdict        {package['verdict']} (nothing measured yet)")
    print(f"evidence root  {package['evidence_root']}")
    print(f"legacy claim   {package['legacy_claim']['package_id']} -> authority none")
    return 0


def _verify(args: argparse.Namespace) -> int:
    descriptor = _descriptor()
    certificate = json.loads(args.certificate.read_text())
    errors = certificate_errors(certificate, descriptor)
    if errors:
        for error in errors:
            print(f"  {error}")
        print(f"\n{len(errors)} finding(s); the certificate is not admissible.")
        return 1
    print("OK: the certificate describes this checkpoint and inherits nothing.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("materialize")
    make.add_argument("--campaign", default="rlc-27b-recovery")
    make.add_argument("--out", type=Path, default=None)
    make.set_defaults(handler=_materialize)

    check = sub.add_parser("verify")
    check.add_argument("certificate", type=Path)
    check.set_defaults(handler=_verify)

    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except RecoveryPackageError as exc:
        print(f"refused: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
