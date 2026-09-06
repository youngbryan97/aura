"""Child-process implementation for the skill catalog contract probe."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import io
import json
import os
import signal
import socket
import subprocess
import sys
import webbrowser
from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import Any


class _ProbeTimeoutError(TimeoutError):
    pass


class _ConstructorDependencyWitness:
    """Identity-bearing witness for validating declared constructor injection."""

    def __init__(self, name: str):
        self.name = name

    def __getattr__(self, name: str) -> Any:
        attempted_access = f"{self.name}.{name}"
        raise RuntimeError(
            "skill constructor used dependency behavior during catalog validation: "
            f"{attempted_access}"
        )


_DENIED_EFFECT_ATTEMPTS: list[str] = []


def _deny_effect(*_args: Any, **_kwargs: Any) -> Any:
    operation = str(_kwargs.pop("_probe_operation", "external effect"))
    _DENIED_EFFECT_ATTEMPTS.append(operation)
    raise RuntimeError(f"{operation} is disabled during the skill catalog probe")


def _effect_denier(operation: str) -> Callable[..., Any]:
    def deny(*args: Any, **kwargs: Any) -> Any:
        kwargs["_probe_operation"] = operation
        return _deny_effect(*args, **kwargs)

    return deny


class _ProbeSocket(socket.socket):
    def connect(self, *_args: Any, **_kwargs: Any) -> Any:
        return _deny_effect(_probe_operation="socket.connect")

    def connect_ex(self, *_args: Any, **_kwargs: Any) -> int:
        _deny_effect(_probe_operation="socket.connect_ex")
        return 1


def _install_filesystem_audit_guard() -> None:
    sandbox_root = os.path.realpath(os.environ["AURA_ROOT"])

    def _inside_sandbox(candidate: Any) -> bool:
        if isinstance(candidate, int):
            return True
        try:
            resolved = os.path.realpath(os.fspath(candidate))
            return os.path.commonpath((sandbox_root, resolved)) == sandbox_root
        except (OSError, TypeError, ValueError):
            return False

    def _audit(event: str, args: tuple[Any, ...]) -> None:
        if event == "open" and args:
            mode = str(args[1] or "") if len(args) > 1 else ""
            flags = int(args[2] or 0) if len(args) > 2 and isinstance(args[2], int) else 0
            mutating = any(marker in mode for marker in ("a", "w", "x", "+")) or bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC)
            )
            if mutating and not _inside_sandbox(args[0]):
                raise RuntimeError("skill catalog probe blocked a write outside its sandbox")
        elif event in {
            "os.chmod",
            "os.chown",
            "os.mkdir",
            "os.remove",
            "os.rename",
            "os.rmdir",
            "shutil.copyfile",
            "shutil.copymode",
            "shutil.copystat",
            "shutil.rmtree",
        }:
            paths = args[:2] if event in {"os.rename", "shutil.copyfile"} else args[:1]
            if any(not _inside_sandbox(path) for path in paths):
                raise RuntimeError("skill catalog probe blocked a mutation outside its sandbox")
        elif event in {"socket.bind", "socket.connect", "subprocess.Popen"}:
            raise RuntimeError("skill catalog probe blocked an external process or network effect")

    sys.addaudithook(_audit)


def _install_effect_guards() -> None:
    _install_filesystem_audit_guard()
    replacements: tuple[tuple[Any, str, Any], ...] = (
        (socket, "socket", _ProbeSocket),
        (socket, "create_connection", _effect_denier("socket.create_connection")),
        (subprocess, "Popen", _effect_denier("subprocess.Popen")),
        (subprocess, "run", _effect_denier("subprocess.run")),
        (subprocess, "call", _effect_denier("subprocess.call")),
        (subprocess, "check_call", _effect_denier("subprocess.check_call")),
        (subprocess, "check_output", _effect_denier("subprocess.check_output")),
        (asyncio, "create_subprocess_exec", _effect_denier("asyncio.create_subprocess_exec")),
        (asyncio, "create_subprocess_shell", _effect_denier("asyncio.create_subprocess_shell")),
        (os, "system", _effect_denier("os.system")),
        (os, "popen", _effect_denier("os.popen")),
        (webbrowser, "open", _effect_denier("webbrowser.open")),
        (webbrowser, "open_new", _effect_denier("webbrowser.open_new")),
        (webbrowser, "open_new_tab", _effect_denier("webbrowser.open_new_tab")),
    )
    for module, name, replacement in replacements:
        vars(module)[name] = replacement


def _timeout_handler(_signum: int, _frame: Any) -> None:
    timeout_message = "skill catalog probe stage timed out"
    raise _ProbeTimeoutError(timeout_message)


@contextlib.contextmanager
def _stage_timeout(seconds: float) -> Iterator[None]:
    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def _schema_for_class(skill_class: type[Any]) -> dict[str, Any]:
    input_model = getattr(skill_class, "input_model", None)
    if input_model is not None:
        schema_builder = getattr(input_model, "model_json_schema", None)
        if not callable(schema_builder):
            raise TypeError("input_model does not expose model_json_schema()")
        schema = schema_builder()
        if not isinstance(schema, dict) or schema.get("type") not in {None, "object"}:
            raise TypeError("input_model JSON schema must describe an object")
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        return schema

    legacy_inputs = getattr(skill_class, "inputs", None)
    if isinstance(legacy_inputs, dict) and legacy_inputs:
        properties = {
            str(name): {"description": str(description), "type": "string"}
            for name, description in legacy_inputs.items()
        }
        return {
            "additionalProperties": False,
            "properties": properties,
            "required": list(properties),
            "type": "object",
        }
    return {"additionalProperties": True, "properties": {}, "type": "object"}


def _constructor_arguments(
    skill_class: type[Any], declaration: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    signature = inspect.signature(skill_class)
    declared = tuple(str(item) for item in declaration.get("constructor_dependencies") or ())
    required = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    ]
    required_names = tuple(parameter.name for parameter in required)
    if set(required_names) != set(declared):
        raise TypeError(
            "required constructor dependencies do not match declaration: "
            f"required={required_names!r} declared={declared!r}"
        )
    positional: list[Any] = []
    keyword: dict[str, Any] = {}
    for parameter in required:
        probe = _ConstructorDependencyWitness(parameter.name)
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional.append(probe)
        else:
            keyword[parameter.name] = probe
    return positional, keyword


def _requirements_for_class(skill_class: type[Any]) -> tuple[dict[str, Any], bool, list[str]]:
    requirements = getattr(skill_class, "requirements", None)
    payload = {
        "commands": list(getattr(requirements, "commands", ()) or ()),
        "packages": list(getattr(requirements, "packages", ()) or ()),
        "supported_platforms": list(getattr(requirements, "supported_platforms", ()) or ()),
    }
    checker = getattr(requirements, "check", None)
    if not callable(checker):
        return payload, True, []
    ready, errors = checker()
    return payload, bool(ready), [str(error) for error in errors]


def _quarantine(declaration: dict[str, Any], stage: str, exc: BaseException) -> dict[str, Any]:
    return {
        "catalog_id": declaration.get("catalog_id"),
        "class_name": declaration.get("class_name"),
        "error": f"{type(exc).__name__}: {exc}"[:1200],
        "module_path": declaration.get("module_path"),
        "name": declaration.get("name"),
        "stage": stage,
        "status": "quarantined",
    }


def _validate_class(module: Any, declaration: dict[str, Any]) -> dict[str, Any]:
    stage = "class_lookup"
    try:
        skill_class = getattr(module, str(declaration["class_name"]))
        if not inspect.isclass(skill_class):
            raise TypeError("declared implementation is not a class")

        stage = "base_contract"
        from core.skills.base_skill import BaseSkill

        if not issubclass(skill_class, BaseSkill):
            raise TypeError("skill does not inherit the canonical governed BaseSkill")
        if inspect.isabstract(skill_class):
            raise TypeError("skill class is abstract")
        if str(getattr(skill_class, "name", "")) != str(declaration["name"]):
            raise ValueError("runtime skill name does not match the source declaration")
        if not callable(getattr(skill_class, "execute", None)):
            raise TypeError("skill has no executable execute() contract")
        if not callable(getattr(skill_class, "safe_execute", None)):
            raise TypeError("skill has no governed safe_execute() contract")
        runtime_scope = str(getattr(skill_class, "effect_scope", "") or "").strip().lower()
        if runtime_scope and runtime_scope != str(declaration["effect_scope"]):
            raise ValueError(
                f"runtime effect_scope {runtime_scope!r} differs from catalog "
                f"{declaration['effect_scope']!r}"
            )

        stage = "construction"
        positional, keyword = _constructor_arguments(skill_class, declaration)
        instance = skill_class(*positional, **keyword)
        if str(getattr(instance, "name", "")) != str(declaration["name"]):
            raise ValueError("constructed skill changed its declared name")

        stage = "schema"
        schema = _schema_for_class(skill_class)
        json.dumps(schema, sort_keys=True)

        stage = "requirements"
        requirements, dependency_ready, dependency_errors = _requirements_for_class(skill_class)
        execute = instance.execute
        route_class = "async" if inspect.iscoroutinefunction(execute) else "sync"
        return {
            "catalog_id": declaration["catalog_id"],
            "class_name": declaration["class_name"],
            "dependency_errors": dependency_errors,
            "dependency_ready": dependency_ready,
            "description": str(getattr(skill_class, "description", declaration["description"])),
            "execution_profile": str(getattr(skill_class, "execution_profile", "cpu")),
            "input_schema": schema,
            # What the skill hands back, where it says so machine-readably.
            # `output` on a skill class is prose; `result_schema` is a claim a
            # caller can check before it reads the result.
            "result_schema": (
                dict(getattr(skill_class, "result_schema", None) or {}) or None
            ),
            "contract_version": str(getattr(skill_class, "contract_version", "1")),
            "is_core_personality": bool(getattr(skill_class, "is_core_personality", False)),
            "memory_mb_estimate": int(getattr(skill_class, "memory_mb_estimate", 256)),
            "metabolic_cost": int(getattr(skill_class, "metabolic_cost", 1)),
            "module_path": declaration["module_path"],
            "name": declaration["name"],
            "requirements": requirements,
            "route_class": route_class,
            "stage": "complete",
            "status": "valid",
            "timeout_seconds": float(getattr(skill_class, "timeout_seconds", 30.0)),
        }
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
        return _quarantine(declaration, stage, exc)


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    declarations = list(payload.get("declarations") or [])
    by_module: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for declaration in declarations:
        by_module[str(declaration.get("module_path") or "")].append(declaration)

    validations: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    for module_path in sorted(by_module):
        module_declarations = sorted(
            by_module[module_path], key=lambda item: (str(item.get("name")), str(item.get("class_name")))
        )
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        try:
            with _stage_timeout(8.0), contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(
                captured_stderr
            ):
                module = importlib.import_module(module_path)
        except (
            ImportError,
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
            OSError,
            _ProbeTimeoutError,
        ) as exc:
            validations.extend(_quarantine(item, "import", exc) for item in module_declarations)
            module = None
        if module is not None:
            for declaration in module_declarations:
                with _stage_timeout(5.0), contextlib.redirect_stdout(
                    captured_stdout
                ), contextlib.redirect_stderr(captured_stderr):
                    validations.append(_validate_class(module, declaration))
        diagnostic = "\n".join(
            part.strip() for part in (captured_stdout.getvalue(), captured_stderr.getvalue()) if part.strip()
        )
        if diagnostic:
            diagnostics.append(f"{module_path}: {diagnostic[:600]}")

    validations.sort(key=lambda item: (str(item.get("name")), str(item.get("module_path"))))
    return {
        "catalog_digest": payload.get("catalog_digest"),
        "diagnostics": diagnostics[:20],
        "validations": validations,
    }


def main() -> int:
    if os.environ.get("AURA_SKILL_CATALOG_PROBE") != "1":
        sys.stderr.write("skill catalog probe must run through the bounded parent boundary\n")
        return 2
    try:
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, dict):
            raise TypeError("probe payload must be an object")
        _install_effect_guards()
        result = _run(payload)
        sys.stdout.write(json.dumps(result, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
        return 0
    except (RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
