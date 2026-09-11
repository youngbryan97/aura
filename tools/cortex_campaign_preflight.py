#!/usr/bin/env python3
"""Whether a cortex-inclusive campaign could be authoritative, before it runs.

The offline organism answers with a deterministic stub, which is right for
paired causal measurement and removes every path that runs through real
language generation. So a whole-Aura claim needs a second campaign through the
real cortex, and that campaign has requirements a run cannot recover from once
it has started: a fallback model serving one arm, a sampler that was not fixed,
a chat template that changed between the arms, a configured pointer that is not
the resident weights.

This checks them without loading anything. It reads the active pointer, the
serving profile and the decoding configuration, and reports what a
cortex-inclusive run would be able to claim. Every answer is `ready`, `blocked`
with the reason, or `unknown` — and `unknown` is not `ready`.

    python tools/cortex_campaign_preflight.py
    python tools/cortex_campaign_preflight.py --json out.json

Nothing here loads the model or touches the live instance. A real Aura is
usually running on this machine and a second cortex-sized model beside it is
what this tool exists to plan, not to do.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: What a paired causal arm needs from the decoder. A sampler left free puts
#: uncontrolled variation between two arms that differ only in a displacement,
#: and the displacement is then the smaller of the two things being measured.
DETERMINISTIC: dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
}


def _check(name: str, ok: bool | None, detail: str, **extra: Any) -> dict[str, Any]:
    status = "unknown" if ok is None else ("ready" if ok else "blocked")
    return {"check": name, "status": status, "detail": detail, **extra}


def _pointer() -> tuple[Any, dict[str, Any]]:
    """The one active cortex pointer, and what it says about itself."""
    try:
        from core.brain.llm.model_registry import get_active_cortex_spec

        spec = get_active_cortex_spec()
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        return None, _check(
            "active pointer", None, f"the registry could not be read: {exc}"
        )
    if spec is None:
        return None, _check(
            "active pointer", False, "no active cortex pointer is configured"
        )
    return spec, _check(
        "active pointer",
        True,
        f"{getattr(spec, 'base_model', '?')} {getattr(spec, 'tag', '')}".strip(),
        model_path=str(getattr(spec, "model_path", "")),
        pointer_sha256=getattr(spec, "pointer_sha256", ""),
        descriptor_sha256=getattr(spec, "descriptor_sha256", ""),
        serving_profile_sha256=getattr(spec, "serving_profile_sha256", ""),
        repository_id=getattr(spec, "repository_id", ""),
        revision=getattr(spec, "revision", ""),
        exact_identity=bool(getattr(spec, "exact_identity", False)),
    )


def _weights_present(spec: Any) -> dict[str, Any]:
    if spec is None:
        return _check("weights on disk", None, "no pointer resolved, so nothing to look at")
    path = Path(getattr(spec, "model_path", "") or "")
    if not path:
        return _check("weights on disk", None, "the pointer names no model path")
    if not path.exists():
        return _check("weights on disk", False, f"{path} does not exist")
    index = path / "model.safetensors.index.json"
    config = path / "config.json"
    return _check(
        "weights on disk",
        config.is_file(),
        str(path),
        has_index=index.is_file(),
        has_config=config.is_file(),
    )


def _tokenizer(spec: Any) -> dict[str, Any]:
    if spec is None:
        return _check("tokenizer", None, "no pointer resolved, so nothing to look at")
    path = Path(getattr(spec, "model_path", "") or "")
    if not path.exists():
        return _check("tokenizer", None, "no model path to read a tokenizer from")
    names = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
    found = [name for name in names if (path / name).is_file()]
    return _check(
        "tokenizer",
        bool(found),
        ", ".join(found) or "no tokenizer file beside the weights",
        files=found,
    )


def _chat_template(spec: Any) -> dict[str, Any]:
    """The template is part of the prompt, so it is part of the measurement.

    Two places carry it and either will do. Newer exports put it in
    `chat_template.jinja` beside the weights; older ones inline it in
    `tokenizer_config.json`. Looking in only the second reported a blocker on a
    model that has one.
    """
    import hashlib

    if spec is None:
        return _check("chat template", None, "no pointer resolved, so nothing to look at")
    path = Path(getattr(spec, "model_path", "") or "")
    if not path.exists():
        return _check("chat template", None, "no model path to read a template from")

    jinja = path / "chat_template.jinja"
    if jinja.is_file():
        digest = hashlib.sha256(jinja.read_bytes()).hexdigest()
        return _check(
            "chat template", True, f"chat_template.jinja, sha256 {digest[:16]}",
            sha256=digest, source="chat_template.jinja",
        )

    config = path / "tokenizer_config.json"
    if not config.is_file():
        return _check("chat template", False, "no template file and no tokenizer config")
    try:
        blob = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return _check("chat template", None, f"tokenizer config unreadable: {exc}")
    template = blob.get("chat_template")
    if not template:
        return _check(
            "chat template", False,
            "no chat_template.jinja beside the weights and none in the tokenizer config",
        )
    digest = hashlib.sha256(str(template).encode("utf-8")).hexdigest()
    return _check(
        "chat template", True, f"tokenizer config, sha256 {digest[:16]}",
        sha256=digest, source="tokenizer_config.json",
    )


def _decoding(spec: Any = None) -> dict[str, Any]:
    """Greedy, or a sampler whose generator is fixed. Anything else is noise."""
    settings: dict[str, Any] = {}
    # What the weights themselves ship with, which is what serves unless
    # something overrides it.
    path = Path(getattr(spec, "model_path", "") or "") if spec is not None else Path()
    generation = path / "generation_config.json"
    if generation.is_file():
        try:
            blob = json.loads(generation.read_text(encoding="utf-8"))
            for key in DETERMINISTIC:
                if key in blob:
                    settings[key] = blob[key]
        except (OSError, ValueError):
            pass
    try:
        from core.config import get_config

        config = get_config()
        for key in DETERMINISTIC:
            value = getattr(config, key, None)
            if value is not None:
                settings[key] = value
    except (ImportError, AttributeError, RuntimeError):
        pass
    for key in DETERMINISTIC:
        env = os.environ.get(f"AURA_{key.upper()}")
        if env is not None:
            settings[key] = env
    if not any(value is not None for value in settings.values()):
        return _check(
            "deterministic decoding",
            None,
            "no temperature or sampler setting could be read; the campaign has "
            "to set them explicitly",
            wanted=DETERMINISTIC,
        )
    mismatched = {
        key: settings.get(key)
        for key, wanted in DETERMINISTIC.items()
        if settings.get(key) is not None and float(settings[key]) != float(wanted)
    }
    return _check(
        "deterministic decoding",
        not mismatched,
        "greedy" if not mismatched else f"sampler is free: {mismatched}",
        settings=settings,
        wanted=DETERMINISTIC,
    )


def _no_fallback() -> dict[str, Any]:
    """A fallback serving one arm and the real cortex the other is two systems."""
    try:
        from core.brain.llm.model_registry import external_llama_cortex_allowed

        external = external_llama_cortex_allowed()
    except (ImportError, RuntimeError):
        external = None
    if external is None:
        return _check("no fallback lane", None, "could not read the external-lane switch")
    return _check(
        "no fallback lane",
        not external,
        "external server cortex is retired" if not external else
        "an external llama lane is allowed, so two arms could be served by two systems",
    )


def _kv_restore() -> dict[str, Any]:
    """Whether the recurrent state can be snapshotted and put back."""
    try:
        from core.subject.driver import Snapshot

        fields = set(getattr(Snapshot, "__dataclass_fields__", {}))
    except (ImportError, AttributeError) as exc:
        return _check("recurrent state restore", None, f"driver snapshot unreadable: {exc}")
    wanted = {"singletons", "services", "phases"}
    missing = sorted(wanted - fields)
    return _check(
        "recurrent state restore",
        not missing,
        "the fork carries module singletons, built services and phase accumulators"
        if not missing
        else f"the fork does not carry: {missing}",
        carries=sorted(fields),
    )


def preflight() -> dict[str, Any]:
    spec, pointer = _pointer()
    # Every check emits a row, whether or not the pointer resolved. A missing
    # row is worse than an unknown one: a reader scanning the list would not
    # notice the absence, and "we did not look" has to read differently from
    # "we looked and it was fine".
    checks = [
        pointer,
        _weights_present(spec),
        _tokenizer(spec),
        _chat_template(spec),
        _decoding(spec),
        _no_fallback(),
        _kv_restore(),
    ]

    blocked = [row for row in checks if row["status"] == "blocked"]
    unknown = [row for row in checks if row["status"] == "unknown"]
    return {
        "checks": checks,
        "blocked": [row["check"] for row in blocked],
        "unknown": [row["check"] for row in unknown],
        # Unknown is not ready. A campaign that cannot say what served it
        # cannot claim what it served.
        "scope_a_run_could_claim": (
            "cortex_inclusive" if not blocked and not unknown else "substrate_only"
        ),
        "note": (
            "Nothing here loads the model. A real Aura is usually running on "
            "this machine, and a second cortex-sized model beside it is what "
            "this plans rather than what it does."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    parser.add_argument("--check", action="store_true", help="exit non-zero unless ready")
    args = parser.parse_args()

    report = preflight()
    for row in report["checks"]:
        mark = {"ready": "  ", "blocked": "✗ ", "unknown": "? "}[row["status"]]
        print(f"{mark}{row['check']:26} {row['detail']}")
    print()
    print(f"a run now could claim: {report['scope_a_run_could_claim']}")
    if report["blocked"]:
        print("blocked:  " + ", ".join(report["blocked"]))
    if report["unknown"]:
        print("unknown:  " + ", ".join(report["unknown"]))

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    if args.check:
        return 0 if report["scope_a_run_could_claim"] == "cortex_inclusive" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
