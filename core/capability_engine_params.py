"""Parameters a caller handed in, shaped to the schema the skill declares.

Lifted whole out of `capability_engine`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import json
from typing import Any


def _safe_field_default(field_name: str, field_obj: Any) -> tuple[Any, bool]:
    from .capability_engine import (
        _FIELD_DEFAULT_FACTORY_ERRORS,
        _record_capability_degradation,
    )

    default_factory = getattr(field_obj, "default_factory", None)
    if default_factory is None:
        return None, False

    try:
        return default_factory(), True
    except _FIELD_DEFAULT_FACTORY_ERRORS as exc:
        _record_capability_degradation(
            exc,
            action=f"omitted invalid default factory value for parameter {field_name!r}",
            severity="warning",
        )
        return None, False


def _get_field_info(field_name: str, field_obj: Any) -> tuple[Any, Any, bool]:
    annotation = None
    default_val = None
    has_default = False

    if hasattr(field_obj, "annotation"):
        annotation = field_obj.annotation
        from pydantic_core import PydanticUndefined

        if field_obj.default is not PydanticUndefined:
            default_val = field_obj.default
            has_default = True
        else:
            default_val, has_default = _safe_field_default(field_name, field_obj)
    elif hasattr(field_obj, "type_"):
        annotation = field_obj.type_
        if field_obj.default is not None:
            default_val = field_obj.default
            has_default = True
        else:
            default_val, has_default = _safe_field_default(field_name, field_obj)

    return annotation, default_val, has_default


def _minimal_model_payload(fields: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    minimal: dict[str, Any] = {}
    for field_name, field_obj in fields.items():
        _, default_val, has_default = _get_field_info(field_name, field_obj)
        if has_default:
            minimal[field_name] = default_val
        elif field_name in params:
            minimal[field_name] = params[field_name]
    return minimal


def _get_base_types(annotation: Any) -> list[Any]:
    if annotation is None:
        return []
    import types
    from typing import Union, get_args, get_origin

    origin = get_origin(annotation)
    # Check if union type (typing.Union or PEP 604 | )
    if origin is Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        args = get_args(annotation)
        types_list = []
        for arg in args:
            types_list.extend(_get_base_types(arg))
        return types_list

    if annotation is type(None):
        return []

    if origin is not None:
        return [origin]

    return [annotation]


def _coerce_and_harmonize_params(params: dict[str, Any], input_model: Any) -> dict[str, Any]:
    """Coerces parameter types and injects defaults from a Pydantic model class."""
    from .capability_engine import (
        _PARAMETER_COERCION_ERRORS,
        _record_capability_degradation,
    )

    if not input_model or not isinstance(params, dict):
        return params

    # 1. Get fields map
    fields = {}
    if hasattr(input_model, "model_fields"):
        fields = input_model.model_fields
    elif hasattr(input_model, "__fields__"):
        fields = input_model.__fields__

    if not fields:
        return params

    healed = dict(params)

    # 2. Coerce existing params
    for name, val in list(healed.items()):
        if name not in fields:
            continue

        field_obj = fields[name]
        annotation, _, _ = _get_field_info(name, field_obj)
        if not annotation:
            continue

        target_types = _get_base_types(annotation)
        if not target_types:
            continue

        # If value already matches one of target types, keep it
        if any(isinstance(val, t) for t in target_types if isinstance(t, type)):
            continue

        # Otherwise, let's coerce!
        coerced = False
        coercion_error: BaseException | None = None
        for t in target_types:
            if coerced:
                break
            if not isinstance(t, type):
                continue

            try:
                if t is bool:
                    val_str = str(val).strip().lower()
                    if val_str in {"true", "yes", "1", "t", "y", "on"}:
                        healed[name] = True
                        coerced = True
                    elif val_str in {"false", "no", "0", "f", "n", "off", ""}:
                        healed[name] = False
                        coerced = True
                    elif isinstance(val, (int, float)):
                        healed[name] = bool(val)
                        coerced = True
                elif t is int:
                    if isinstance(val, float):
                        healed[name] = int(val)
                        coerced = True
                    elif isinstance(val, str):
                        cleaned = val.strip()
                        healed[name] = int(float(cleaned))
                        coerced = True
                elif t is float:
                    if isinstance(val, (int, str)):
                        healed[name] = float(val)
                        coerced = True
                elif t is str:
                    if isinstance(val, (dict, list)):
                        healed[name] = json.dumps(val)
                        coerced = True
                    else:
                        healed[name] = str(val)
                        coerced = True
                elif t is list:
                    if isinstance(val, str):
                        cleaned = val.strip()
                        if cleaned.startswith("[") and cleaned.endswith("]"):
                            healed[name] = json.loads(cleaned)
                            coerced = True
                        elif "," in cleaned:
                            healed[name] = [item.strip() for item in cleaned.split(",")]
                            coerced = True
                        else:
                            healed[name] = [cleaned]
                            coerced = True
                elif t is dict:
                    if isinstance(val, str):
                        cleaned = val.strip()
                        if cleaned.startswith("{") and cleaned.endswith("}"):
                            healed[name] = json.loads(cleaned)
                            coerced = True
            except _PARAMETER_COERCION_ERRORS as exc:
                coercion_error = exc
                continue

        if not coerced and coercion_error is not None:
            # A value that won't coerce (e.g. limit="not_an_int") is bad INPUT
            # gracefully handled by keeping the original — not a capability_engine
            # fault. Under a fail-closed policy + production governance this
            # warning would otherwise escalate to a CRITICAL service failure and
            # raise, spiking existential threat (the same July-2026 live pathology
            # already fixed at the sanitized-fallback site below).
            _record_capability_degradation(
                coercion_error,
                action=f"kept original value for parameter {name!r} after coercion failed",
                severity="warning",
                enforce_failure_policy=False,
            )

    # 3. Inject Defaults for missing keys
    for name, field_obj in fields.items():
        if name not in healed:
            _, default_val, has_default = _get_field_info(name, field_obj)
            if has_default:
                healed[name] = default_val

    return healed


def _a_model_from_a_schema(schema: dict[str, Any]) -> Any:
    """Wrap a plain JSON schema so it answers `model_json_schema` like a model.

    A skill that has a schema and not a model should not have to grow a
    dependency to say what it returns.
    """

    class _SaysItsSchema:
        def __init__(self, said: dict[str, Any]) -> None:
            self._said = dict(said)

        def model_json_schema(self) -> dict[str, Any]:
            return dict(self._said)

    return _SaysItsSchema(schema)


def _note_double_nested_params(log: Any, skill_name: str, params: Any, normalized_params: Any) -> None:
    """Say so when the model nested the params twice, before they are unpacked."""
    if (
        normalized_params != params
        and "params" in params
        and isinstance(params["params"], dict)
    ):
        log.warning(
            "[%s] Unpacking double-nested params from LLM hallucination.", skill_name
        )
