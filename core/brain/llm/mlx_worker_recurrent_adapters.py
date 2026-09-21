"""Loading and attaching the optional recurrent adapters: the certified adapter, the unified shadow package, the qualified activation, and the expert adapter's structural check.

Lifted whole out of `mlx_worker`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root


def _attach_certified_recurrent_adapter(
    model: Any,
    *,
    model_path: str,
    personality_adapter_path: str | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Attach the certified recurrent adapter configured for this live state.

    An absent default pointer means no adapter has earned promotion yet. Once
    an operator explicitly configures a pointer, every missing, stale,
    untrusted, identity-mismatched, or unloadable artifact is fatal to worker
    initialization; serving the base path while claiming trained recurrence
    would invalidate both runtime truth and the scientific certificate.
    """

    from core.brain.llm.latent_cortex.runtime_identity import (
        WORKER_ACTIVATION_SCHEMA,
        inactive_worker_recurrent_adapter_activation,
    )

    configured_pointer = os.environ.get(
        "AURA_RLC_ACTIVATION_POINTER",
        "",
    ).strip()
    activation_root = state_root() / "data/adapters/latent-cortex"
    pointer_path = (
        Path(configured_pointer).expanduser()
        if configured_pointer
        else activation_root / "active.json"
    )
    if not pointer_path.exists():
        if configured_pointer:
            raise RuntimeError(f"configured_recurrent_adapter_pointer_missing:{pointer_path}")
        return inactive_worker_recurrent_adapter_activation(), None

    configured_trust_root = os.environ.get(
        "AURA_RLC_ACTIVATION_TRUST_ROOT",
        "",
    ).strip()
    trust_root_path = (
        Path(configured_trust_root).expanduser()
        if configured_trust_root
        else activation_root / "trust-root.pem"
    )
    try:
        from core.brain.llm.latent_cortex.live_adapter_activation import (
            read_live_adapter_trust_root,
        )

        trusted_root = read_live_adapter_trust_root(trust_root_path)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"recurrent_adapter_trust_root_unreadable:{trust_root_path}") from exc

    approved_root = activation_root / "releases"
    from core.brain.llm.latent_cortex.live_adapter_activation import (
        attach_certified_live_adapter,
    )

    receipt = attach_certified_live_adapter(
        model,
        model_path=model_path,
        personality_adapter_path=personality_adapter_path,
        pointer_path=pointer_path,
        trusted_root_public_key_pem=trusted_root,
        approved_adapter_roots=(approved_root,),
        now_unix=int(time.time()),
    )
    identity = receipt.get("adapter_identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("recurrent_adapter_identity_receipt_missing")
    activation = {
        "schema": WORKER_ACTIVATION_SCHEMA,
        "configured": True,
        "active": True,
        "reason": "certified_gain_proven",
        "receipt_sha256": receipt["receipt_sha256"],
        "activation_sha256": receipt["activation_sha256"],
        "adapter_composite_identity_sha256": identity["composite_identity_sha256"],
        "campaign_name": receipt["campaign_name"],
        "claim_tier": receipt["claim_tier"],
        "verified_verdict": receipt["verified_verdict"],
        "loaded_projection_count": receipt["loaded_projection_count"],
    }
    return activation, receipt


def _load_unified_recurrent_shadow(
    model: Any,
    tokenizer: Any,
    *,
    model_path: str,
) -> tuple[Any | None, dict[str, Any]]:
    """Load separately certified recurrent tissue with zero serving authority."""

    from core.brain.llm.unified_recurrent_shadow_contract import (
        LOAD_SCHEMA,
        seal_shadow_load_receipt,
        shadow_load_receipt_errors,
    )

    from .mlx_worker import (
        _FLAG_UNIFIED_RECURRENT_SHADOW_PACKAGE,
    )

    configured = _FLAG_UNIFIED_RECURRENT_SHADOW_PACKAGE.value().strip()
    pointer_path: Path | None = None
    if configured:
        package = Path(configured).expanduser()
    else:
        from core.brain.llm.unified_recurrent_shadow_pointer import (
            default_shadow_activation_paths,
            resolve_shadow_pointer,
        )

        pointer_path, releases_root = default_shadow_activation_paths()
        if pointer_path.exists() or pointer_path.is_symlink():
            try:
                package = resolve_shadow_pointer(
                    pointer_path,
                    releases_root=releases_root,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"configured_unified_recurrent_shadow_pointer_invalid:{pointer_path}:{exc}"
                ) from exc
        else:
            package = None
    if package is None:
        return None, seal_shadow_load_receipt(
            {
                "schema": LOAD_SCHEMA,
                "configured": False,
                "loaded": False,
                "reason": "not_configured",
                "package_id": "",
                "manifest_sha256": "",
                "checkpoint_sha256": "",
                "controller_sha256": "",
                "families": [],
                "task_depths": [],
                "recurrence_depth": 0,
                "model_identity_strength": "none",
                "mode": "shadow_only",
                "serving_authority": False,
            }
        )

    try:
        from core.brain.llm.unified_recurrent_shadow import (
            load_unified_recurrent_shadow,
        )

        loaded = load_unified_recurrent_shadow(
            package,
            model=model,
            tokenizer=tokenizer,
            model_path=Path(model_path),
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        # A shadow that does not match this model is DECLINED, not fatal.
        #
        # LIVE, 2026-08-13: every boot logged
        #   MLXWorker CRITICAL: configured_unified_recurrent_shadow_invalid:
        #   .../releases/cp366-source-migrated-recurrent: unified shadow
        #   mechanics or resident model binding differs
        #   🔄 [MLX] Worker init failed ... Retrying spawn
        #
        # The integrity check is right — that package was built against a
        # different resident binding and must not be used. But it runs in
        # shadow_only mode with serving_authority False, so it is by
        # construction not load-bearing, and refusing it took the whole worker
        # down and put boot into a respawn loop. An optional component that
        # cannot serve must not be able to stop the thing it decorates.
        record_degradation(
            "mlx_worker",
            exc,
            action=(
                "declined an incompatible unified recurrent shadow and started "
                "the worker without it"
            ),
        )
        return None, seal_shadow_load_receipt(
            {
                "schema": LOAD_SCHEMA,
                "configured": True,
                "loaded": False,
                "reason": f"incompatible:{exc}"[:300],
                "package_id": str(package),
                "manifest_sha256": "",
                "checkpoint_sha256": "",
                "controller_sha256": "",
                "families": [],
                "task_depths": [],
                "recurrence_depth": 0,
                "model_identity_strength": "none",
                "mode": "shadow_only",
                "serving_authority": False,
            }
        )
    receipt = dict(loaded.receipt)
    errors = shadow_load_receipt_errors(receipt)
    if errors:
        raise RuntimeError(
            "configured_unified_recurrent_shadow_receipt_invalid:" + ",".join(errors)
        )
    return loaded, receipt


def _load_unified_recurrent_qualified_activation(
    loaded_shadow: Any | None,
    shadow_status: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Load typed serving authority separately from non-serving tissue."""

    from core.brain.llm.unified_recurrent_qualified_activation import (
        activation_matches_shadow_receipt,
        seal_qualified_activation_load_receipt,
    )
    from core.brain.llm.unified_recurrent_qualified_activation_store import (
        default_qualified_activation_path,
        read_qualified_activation,
    )

    activation_path = default_qualified_activation_path()
    if not (activation_path.exists() or activation_path.is_symlink()):
        return None, seal_qualified_activation_load_receipt(
            configured=False,
            loaded=False,
            reason="not_configured",
            activation=None,
        )
    try:
        activation = read_qualified_activation(activation_path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"configured_unified_recurrent_qualified_activation_invalid:{activation_path}:{exc}"
        ) from exc
    if loaded_shadow is None or shadow_status.get("loaded") is not True:
        # Authority fails closed, but the ordinary model worker does not.
        # This activation cannot authorize chat or arbitrary reasoning and
        # cannot execute without its exact shadow tissue.  Keeping it loaded
        # as *inactive evidence* makes that refusal explicit without letting
        # a stale optional package crash unrelated base inference.
        return None, seal_qualified_activation_load_receipt(
            configured=True,
            loaded=False,
            reason="qualified_activation_shadow_unavailable",
            activation=activation,
        )
    if not activation_matches_shadow_receipt(activation, shadow_status):
        raise RuntimeError("qualified_activation_shadow_identity_differs")
    if activation.get("mode") == "qualified_typed_pending":
        return None, seal_qualified_activation_load_receipt(
            configured=True,
            loaded=False,
            reason="qualified_activation_pending_canary",
            activation=activation,
        )
    receipt = seal_qualified_activation_load_receipt(
        configured=True,
        loaded=True,
        reason="qualified_activation_loaded",
        activation=activation,
    )
    return activation, receipt


def _handle_unified_recurrent_qualified_decode(
    job: dict[str, Any],
    *,
    loaded_shadow: Any | None,
    qualified_activation: Mapping[str, Any] | None,
    model: Any,
    contract_key: bytes | None,
    consumed_canary_nonces: set[str] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    activity: Callable[[], None] | None = None,
    reclaim: Callable[[], bool],
) -> dict[str, Any]:
    """Execute one typed decode only after exact activation admission."""

    from .mlx_worker import (
        logger,
    )

    if job.get("action") != "unified_recurrent_qualified_decode":
        raise ValueError("unified_recurrent_qualified_decode_action_differs")
    # At call time: the surface modules read names `mlx_worker` defines
    # while they load, and this module is imported before those exist.
    from .mlx_worker_surface_repair import _verify_contract_authority

    refusal = _verify_contract_authority(job, contract_key)
    if refusal:
        raise ValueError(refusal)
    if loaded_shadow is None:
        raise RuntimeError("unified_recurrent_shadow_not_loaded")
    canary_activation = job.get("unified_recurrent_qualified_canary_activation")
    canary_authority = job.get("unified_recurrent_qualified_canary_authority")
    if qualified_activation is None and not isinstance(canary_activation, Mapping):
        raise RuntimeError("unified_recurrent_qualified_activation_not_loaded")
    from core.brain.llm.unified_recurrent_qualified_activation import (
        activation_matches_shadow_receipt,
        qualified_activation_errors,
    )
    from core.brain.llm.unified_recurrent_qualified_decode import (
        authorize_qualified_decode_result,
        qualified_canary_authority_matches,
        run_qualified_decode,
    )

    if canary_activation is not None:
        if qualified_activation is not None:
            raise RuntimeError("qualified_canary_requires_inactive_durable_authority")
        if qualified_activation_errors(canary_activation):
            raise RuntimeError("qualified_canary_activation_invalid")
        if (
            not isinstance(canary_authority, Mapping)
            or not isinstance(getattr(loaded_shadow, "canary_battery", None), Mapping)
            or not qualified_canary_authority_matches(
                canary_authority,
                activation=canary_activation,
                request=job.get("unified_recurrent_qualified_decode_contract") or {},
                battery=loaded_shadow.canary_battery,
            )
        ):
            raise RuntimeError("qualified_canary_authority_invalid")
        nonce = str(canary_authority.get("nonce") or "")
        if consumed_canary_nonces is None or nonce in consumed_canary_nonces:
            raise RuntimeError("qualified_canary_authority_replayed")
        consumed_canary_nonces.add(nonce)
        effective_activation = canary_activation
    else:
        effective_activation = qualified_activation
    if not isinstance(effective_activation, Mapping) or not activation_matches_shadow_receipt(
        effective_activation,
        loaded_shadow.receipt,
    ):
        raise RuntimeError("qualified_activation_shadow_identity_differs")

    def report_progress(event: Mapping[str, int | str]) -> None:
        log_progress = logger.info if canary_activation is not None else logger.debug
        log_progress(
            "Unified recurrent qualified decode progress: stage=%s generated=%s/%s",
            event.get("stage"),
            event.get("generated_token_count"),
            event.get("maximum_token_count"),
        )

    try:
        result = run_qualified_decode(
            loaded_shadow,
            model,
            job.get("unified_recurrent_qualified_decode_contract"),
            cancel_check=cancel_check,
            activity=activity,
            progress=report_progress,
        )
        authorized = authorize_qualified_decode_result(
            result,
            effective_activation,
            canary_only=canary_activation is not None,
        )
    finally:
        reclaimed = reclaim()
    if reclaimed is not True:
        raise RuntimeError("unified_recurrent_qualified_decode_memory_not_reclaimed")
    return {
        "id": str(job.get("id") or ""),
        "action": "unified_recurrent_qualified_decode",
        "status": "ok",
        "receipt": authorized,
        "allocator_reclaimed": True,
    }


def _validate_expert_adapter_dir(adapter_dir: str) -> Path:
    """Structural + policy validation BEFORE any resident-weight mutation.

    set_expert_adapter previously accepted any path from any IPC caller and
    mutated resident weights first, validating nothing: no approved-root
    policy, no adapter structure check. Validation failures here leave the
    resident model completely untouched.
    """
    from .mlx_worker import (
        _RESTORABLE_FINE_TUNE_TYPES,
        _expert_adapter_approved_roots,
    )

    path = Path(str(adapter_dir or "")).expanduser()
    try:
        path = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FileNotFoundError(f"expert_adapter_dir_unresolvable:{adapter_dir}") from exc
    if not path.is_dir():
        raise NotADirectoryError(f"expert_adapter_not_a_directory:{path}")
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"expert_adapter_missing_config:{path}")
    try:
        adapter_config = json.loads(config_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"expert_adapter_config_unparseable:{path}") from exc
    # CP126 15f68852. The config was parsed and then ignored. mlx_lm's
    # load_adapters honours fine_tune_type: a "full" (or "dora"/unknown)
    # adapter does NOT produce restorable wrapper modules — it writes into
    # the resident weights. Detach can only unwrap .linear/.embedding
    # wrappers, so such a load permanently rewrites the personality model
    # this worker is serving, with no way back short of a reload.
    #
    # The type is therefore constrained BEFORE any weight mutation, rather
    # than discovered afterwards by a detach that silently restores nothing.
    if not isinstance(adapter_config, dict):
        raise ValueError(f"expert_adapter_config_not_an_object:{path}")
    fine_tune_type = (
        str(
            adapter_config.get("fine_tune_type")
            or adapter_config.get("fine_tune")
            # Absent means LoRA in mlx_lm's own default.
            or "lora"
        )
        .strip()
        .lower()
    )
    if fine_tune_type not in _RESTORABLE_FINE_TUNE_TYPES:
        raise ValueError(f"expert_adapter_fine_tune_type_not_restorable:{fine_tune_type}:{path}")
    has_weights = any(path.glob("*.safetensors")) or (path / "adapters.npz").is_file()
    if not has_weights:
        raise ValueError(f"expert_adapter_missing_weights:{path}")
    for root in _expert_adapter_approved_roots():
        try:
            resolved_root = root.expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        try:
            path.relative_to(resolved_root)
            return path
        except ValueError:
            continue
    raise PermissionError(f"expert_adapter_path_not_approved:{path}")
