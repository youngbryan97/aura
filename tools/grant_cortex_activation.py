#!/usr/bin/env python3
"""Write a standing authorization for cortex activation. An operator runs this.

`activate_upgrade` takes an operator authorization string at the moment of
the swap. This writes the other kind: one decision, made in advance, about a
named family of candidates, with an expiry and a ceiling.

It is a separate tool rather than a function inside the runtime for the
reason the whole design turns on — nothing on an autonomous path may write a
grant, and the clearest way to keep that true is for the only caller to be a
person at a terminal.

    python tools/grant_cortex_activation.py \
        --granted-by "bryan" \
        --model-prefix "/Users/bryan/.aura/models/Qwen3.9-" \
        --days 14 --activations 1 \
        --reason "planned Qwen3.9 generation upgrade window"

Read what is currently granted, and by whom:

    python tools/grant_cortex_activation.py --show
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.self_modification.standing_authorization import (  # noqa: E402
    LONGEST_GRANT_S,
    read_standing_grant,
    where_a_grant_is_kept,
    write_standing_grant,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--granted-by", default="")
    parser.add_argument(
        "--model-prefix",
        action="append",
        default=[],
        help="a model path prefix this grant covers; repeatable",
    )
    parser.add_argument(
        "--descriptor-digest",
        action="append",
        default=[],
        help="an exact artifact descriptor digest this grant covers; repeatable",
    )
    parser.add_argument("--days", type=float, default=0.0)
    parser.add_argument("--activations", type=int, default=1)
    parser.add_argument("--reason", default="")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--fused-model-dir", default="")
    args = parser.parse_args()

    if args.fused_model_dir:
        where = Path(args.fused_model_dir)
    else:
        from core.brain.llm.model_registry import get_fused_model_root

        where = Path(get_fused_model_root())

    if args.show:
        grant = read_standing_grant(where)
        if grant is None:
            print(f"no standing grant at {where_a_grant_is_kept(where)}")
            return 0
        left = (grant.expires_at - time.time()) / 86400
        print(json.dumps(grant.to_dict(), indent=2, sort_keys=True))
        print(
            f"\n{grant.used} of {grant.most_activations} activation(s) used; "
            f"{left:.1f} days left"
        )
        return 0

    if not args.granted_by:
        parser.error("--granted-by names the operator making this decision")
    if args.days <= 0:
        parser.error(
            f"--days is required and is at most {LONGEST_GRANT_S / 86400:.0f}"
        )

    grant = write_standing_grant(
        where,
        granted_by=args.granted_by,
        model_path_prefixes=tuple(args.model_prefix),
        descriptor_digests=tuple(args.descriptor_digest),
        valid_for_s=args.days * 86400,
        most_activations=args.activations,
        reason=args.reason,
    )
    print(f"wrote {where_a_grant_is_kept(where)}")
    print(json.dumps(grant.to_dict(), indent=2, sort_keys=True))
    print(
        "\nThis authorizes activation of a covered candidate that also passes "
        "its capability comparison and every critical gate. It does not "
        "authorize anything else, and it does not replace the evidence."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
