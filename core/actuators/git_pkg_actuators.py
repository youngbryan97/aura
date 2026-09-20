"""core/actuators/git_pkg_actuators.py
===================================
Actuators for Git and package installation operations.

Hardening (CP126): git runs only inside a confined workspace root, refuses
option-injection in user positionals (end-of-options `--`), validates clone
URL scheme/host and destination containment, and records a pre-op HEAD for
mutating operations. Package installs are pinned by default, reject flag-like
specs, run non-interactively, and have their output bounded and credential-
redacted.
"""

import os
import re
import subprocess
import sys
from typing import Any
from urllib.parse import urlparse

from core.actuators.actuator_registry import ActuatorResult, BaseActuator
from core.actuators.authority import verify_actuator_authority
from core.runtime.subprocess_gateway import get_subprocess_gateway

# ── Shared constraints ───────────────────────────────────────────────────────

_MAX_OUTPUT_CHARS = 64 * 1024

_CLONE_ALLOWED_SCHEMES = {"https", "ssh", "git"}
_PRIVATE_HOST_MARKERS = (
    "localhost", "127.0.0.1", "0.0.0.0", "::1",
    "169.254.169.254",  # cloud metadata endpoint
    ".internal", ".local",
)
# Package spec: a name with optional extras and a version specifier. Must NOT
# start with a hyphen (which would be parsed as a pip option).
_PACKAGE_SPEC_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*(\[[A-Za-z0-9,._\-]+\])?([=<>!~]=?[A-Za-z0-9._\-*,]+)*$")
_REF_RE = re.compile(r"^[A-Za-z0-9._/@+\-]+$")


def _workspace_root() -> str:
    root = os.environ.get("AURA_GIT_WORKSPACE", "").strip() or os.getcwd()
    return os.path.realpath(os.path.abspath(root))


def _validate_git_cwd(requested: Any, root: str) -> tuple[str | None, str]:
    if requested is None:
        return root, ""
    if not isinstance(requested, str) or not requested.strip():
        return None, "cwd must be a non-empty string"
    resolved = os.path.realpath(os.path.abspath(requested))
    if resolved != root and not resolved.startswith(root + os.sep):
        return None, f"cwd escapes the git workspace root ({root})"
    if not os.path.isdir(resolved):
        return None, f"cwd does not exist: {resolved}"
    return resolved, ""


def _safe_positional(value: Any, *, kind: str) -> tuple[str | None, str]:
    """Reject option-injection and control characters in a user positional."""
    if not isinstance(value, str) or not value.strip():
        return None, f"{kind} must be a non-empty string"
    if value.startswith("-"):
        return None, f"{kind} may not start with '-' (option injection)"
    if any(c in value for c in "\x00\n\r"):
        return None, f"{kind} contains control characters"
    return value, ""


def _validate_ref(value: Any, *, kind: str) -> tuple[str | None, str]:
    ok, err = _safe_positional(value, kind=kind)
    if ok is None:
        return None, err
    if ".." in value or not _REF_RE.match(value):
        return None, f"{kind} is not a valid git ref"
    return value, ""


def _validate_clone_url(url: Any) -> tuple[str | None, str]:
    if not isinstance(url, str) or not url.strip() or url.startswith("-"):
        return None, "clone url is missing or malformed"
    if any(c in url for c in "\x00\n\r"):
        return None, "clone url contains control characters"
    # scp-like syntax: user@host:path
    scp = re.match(r"^[A-Za-z0-9._%+\-]+@([A-Za-z0-9._\-]+):", url)
    host = scp.group(1) if scp else (urlparse(url).hostname or "")
    scheme = "ssh" if scp else urlparse(url).scheme.lower()
    if scheme not in _CLONE_ALLOWED_SCHEMES:
        return None, f"clone scheme '{scheme}' is not allowed (use https/ssh/git)"
    if not host:
        return None, "clone url has no host"
    low = host.lower()
    if any(marker in low for marker in _PRIVATE_HOST_MARKERS):
        return None, f"clone host '{host}' is not permitted"
    return url, ""


def _validate_dest_path(path: Any, root: str) -> tuple[str | None, str]:
    if path is None:
        return None, ""  # optional
    if not isinstance(path, str) or not path.strip() or path.startswith("-"):
        return None, "clone destination is malformed"
    resolved = os.path.realpath(os.path.abspath(path))
    if resolved != root and not resolved.startswith(root + os.sep):
        return None, "clone destination escapes the workspace root"
    return resolved, ""


def _bound(text: Any) -> str:
    s = str(text or "")
    if len(s) > _MAX_OUTPUT_CHARS:
        return s[:_MAX_OUTPUT_CHARS] + f"\n...[truncated {len(s) - _MAX_OUTPUT_CHARS} chars]"
    return s


def _redact_urls(text: str) -> str:
    """Strip credentials embedded in URLs from tool output."""
    return re.sub(r"(\w+://)[^/\s:@]+:[^/\s@]+@", r"\1***:***@", text)


def _clean_output(text: Any) -> str:
    return _bound(_redact_urls(str(text or "")))


class GitActuator(BaseActuator):
    requires_authority = True

    @property
    def name(self) -> str:
        return "git_operation"

    @property
    def description(self) -> str:
        return "Perform git operations (status, diff, branch, commit, checkout, clone) within the codebase."

    def validate_params(self, params: dict[str, Any]) -> bool:
        if not isinstance(params, dict) or "action" not in params:
            return False
        action = params["action"]
        if action not in ("status", "diff", "branch", "commit", "checkout", "clone", "log"):
            return False
        if action == "clone" and "url" not in params:
            return False
        if action == "commit" and "message" not in params:
            return False
        if action == "checkout" and "branch_or_commit" not in params:
            return False  # required target validated up front, never a KeyError
        if action in {"branch", "commit", "checkout"} and not bool(params.get("allow_mutation")):
            return False
        if action == "clone" and not bool(params.get("allow_external_clone")):
            return False
        return True

    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        _authorized, _auth_reason = verify_actuator_authority(params, actuator=self.name)
        if not _authorized:
            return ActuatorResult(False, _auth_reason, {})
        if not self.validate_params(params):
            return ActuatorResult(False, "Invalid git action or missing parameters.", {})

        action = params["action"]
        root = _workspace_root()
        cwd, cwd_err = _validate_git_cwd(params.get("cwd"), root)
        if cwd is None:
            return ActuatorResult(False, f"Refused git op: {cwd_err}", {})

        cmd, build_err = self._build_command(action, params, root)
        if cmd is None:
            return ActuatorResult(False, f"Refused git op: {build_err}", {})

        # Precondition for mutating ops: capture the pre-op HEAD as a rollback
        # ref, and honor an optional expected_head guard.
        pre_head = ""
        if action in {"commit", "checkout", "branch"}:
            pre_head = self._current_head(cwd)
            expected = params.get("expected_head")
            if isinstance(expected, str) and expected and pre_head and not pre_head.startswith(expected):
                return ActuatorResult(
                    False,
                    f"Refused git {action}: expected HEAD {expected} but repository is at {pre_head[:12]}.",
                    {"pre_op_head": pre_head},
                )

        try:
            res = get_subprocess_gateway().run(cmd, cwd=cwd, timeout=30.0, source="git_actuator", accelerator_capability="auto")
            success = res.returncode == 0
            updates = {
                "exit_code": res.returncode,
                "stdout": _clean_output(res.stdout),
                "stderr": _clean_output(res.stderr),
            }
            if pre_head:
                updates["pre_op_head"] = pre_head
            return ActuatorResult(success, f"Git {action} executed with exit code {res.returncode}.", updates)
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            return ActuatorResult(False, f"Git execution failed: {e}", {})

    def _build_command(self, action: str, params: dict[str, Any], root: str) -> tuple[list[str] | None, str]:
        """Assemble a git argv with option-injection guards and `--` separators."""
        if action == "status":
            return ["git", "-c", "core.fsmonitor=false", "status"], ""
        if action == "log":
            try:
                n = max(1, min(1000, int(params.get("n", 5))))
            except (TypeError, ValueError):
                return None, "log count must be an integer"
            return ["git", "-c", "core.fsmonitor=false", "log", "-n", str(n)], ""
        if action == "diff":
            cmd = ["git", "-c", "core.fsmonitor=false", "diff"]
            if "target" in params:
                target, err = _safe_positional(params["target"], kind="diff target")
                if target is None:
                    return None, err
                cmd += ["--", target]
            return cmd, ""
        if action == "branch":
            cmd = ["git", "-c", "core.fsmonitor=false", "branch"]
            if "name" in params:
                name, err = _validate_ref(params["name"], kind="branch name")
                if name is None:
                    return None, err
                cmd += ["--", name]
            return cmd, ""
        if action == "commit":
            message = params["message"]
            if not isinstance(message, str) or not message.strip():
                return None, "commit message must be a non-empty string"
            return ["git", "commit", "-m", message], ""
        if action == "checkout":
            ref, err = _validate_ref(params["branch_or_commit"], kind="checkout target")
            if ref is None:
                return None, err
            return ["git", "checkout", "--", ref], ""
        if action == "clone":
            url, err = _validate_clone_url(params.get("url"))
            if url is None:
                return None, err
            cmd = ["git", "clone", "--", url]
            dest, derr = _validate_dest_path(params.get("path"), root)
            if params.get("path") is not None:
                if dest is None:
                    return None, derr
                cmd.append(dest)
            return cmd, ""
        return None, f"unsupported action: {action}"

    @staticmethod
    def _current_head(cwd: str) -> str:
        try:
            res = get_subprocess_gateway().run(
                ["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"], cwd=cwd, timeout=10.0, source="git_actuator_precondition",
                accelerator_capability="none",
            )
            if res.returncode == 0:
                return str(res.stdout or "").strip()
        except (subprocess.SubprocessError, OSError, ValueError):
            return ""
        return ""


class PackageInstallActuator(BaseActuator):
    requires_authority = True

    @property
    def name(self) -> str:
        return "package_install"

    @property
    def description(self) -> str:
        return "Install a Python package into the virtual environment using pip."

    def validate_params(self, params: dict[str, Any]) -> bool:
        if not isinstance(params, dict) or "package_name" not in params:
            return False
        pkg = params["package_name"]
        if not isinstance(pkg, str) or not pkg.strip():
            return False
        if not _PACKAGE_SPEC_RE.match(pkg.strip()):
            return False
        if not bool(params.get("allow_install")):
            return False
        return True

    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        _authorized, _auth_reason = verify_actuator_authority(params, actuator=self.name)
        if not _authorized:
            return ActuatorResult(False, _auth_reason, {})
        if not self.validate_params(params):
            return ActuatorResult(False, "Safety validation failed: package name or explicit install approval is invalid.", {})

        pkg = params["package_name"].strip()

        # Pinned by default: an unpinned spec can silently install a different
        # version (or a typo-squat) on every run. Opt out explicitly.
        allow_unpinned = str(os.environ.get("AURA_PKG_ALLOW_UNPINNED", "")).strip().lower() in {"1", "true", "yes", "on"}
        if "==" not in pkg and not allow_unpinned and not bool(params.get("allow_unpinned")):
            return ActuatorResult(
                False,
                f"Refused install: '{pkg}' is not version-pinned (use name==X.Y.Z or set allow_unpinned).",
                {},
            )

        # CP126 cc1d78b7: a pinned VERSION still executes installer code from
        # whatever index is configured. A digest is what actually binds the
        # artifact. Hashes are used when supplied, and their absence is
        # reported rather than passed off as a verified install.
        hashes = params.get("hashes")
        hash_args: list[str] = []
        if isinstance(hashes, (list, tuple)) and hashes:
            for digest in hashes:
                text = str(digest or "").strip()
                if not re.fullmatch(r"(sha256|sha384|sha512):[0-9a-f]{64,128}", text):
                    return ActuatorResult(
                        False, f"Refused install: malformed hash '{text[:40]}'.", {}
                    )
                hash_args += ["--hash", text]
        require_hashes = bool(hash_args)

        # CP126 3a5c4a39: installing into sys.executable mutates the RUNNING
        # interpreter with no isolated target, transaction or rollback, so a
        # bad resolution degrades Aura itself until a restart. A different
        # target may be named; using the live interpreter is an explicit,
        # acknowledged choice.
        target_python = str(params.get("python_executable") or "").strip() or sys.executable
        mutates_running_interpreter = os.path.realpath(target_python) == os.path.realpath(sys.executable)
        if mutates_running_interpreter and not bool(params.get("allow_mutating_running_env")):
            return ActuatorResult(
                False,
                "Refused install: this would mutate the RUNNING interpreter "
                f"({target_python}). Pass allow_mutating_running_env=True to "
                "accept that, or name an isolated python_executable.",
                {"target_python": target_python},
            )

        # CP126 6da6af1d: a dry run resolves dependencies and reports what
        # WOULD change before anything is written, which is the closest thing
        # to a preview this path can offer.
        dry_run = bool(params.get("dry_run"))

        # `--` terminates option parsing so a spec can never become a pip flag.
        cmd = [target_python, "-m", "pip", "install", "--no-input", "--disable-pip-version-check"]
        if require_hashes:
            cmd.append("--require-hashes")
        if dry_run:
            cmd.append("--dry-run")
        cmd += hash_args + ["--", pkg]

        # CP126 3a5c4a39: capture the pre-install state so a caller can tell
        # what changed, and reconcile an uncertain outcome afterwards.
        before = self._installed_version(target_python, pkg)

        try:
            timeout_s = float(params.get("timeout_s") or 120.0)
            timeout_s = max(10.0, min(900.0, timeout_s))
            res = get_subprocess_gateway().run(
                cmd, timeout=timeout_s, source="package_install_actuator",
                accelerator_capability="auto",
            )
            success = res.returncode == 0
            after = self._installed_version(target_python, pkg)
            return ActuatorResult(
                success,
                f"Package '{pkg}' installation finished with exit code {res.returncode}.",
                {
                    "exit_code": res.returncode,
                    "stdout": _clean_output(res.stdout),
                    "stderr": _clean_output(res.stderr),
                    "pinned": "==" in pkg,
                    # CP126 cc1d78b7: say what was actually verified.
                    "hash_verified": require_hashes,
                    "supply_chain_bound": (
                        "artifact digest verified" if require_hashes
                        else "version pin only — no artifact digest, signature or provenance"
                    ),
                    "dry_run": dry_run,
                    "target_python": target_python,
                    "mutated_running_interpreter": mutates_running_interpreter and not dry_run,
                    # CP126 6da6af1d: reconcile the outcome instead of leaving
                    # a partial install uncertain.
                    "version_before": before,
                    "version_after": after,
                    "changed": before != after,
                    "restart_required": bool(
                        mutates_running_interpreter and not dry_run and before != after
                    ),
                },
            )
        except subprocess.TimeoutExpired:
            after = self._installed_version(target_python, pkg)
            return ActuatorResult(
                False,
                f"Package '{pkg}' install timed out; outcome reconciled from the environment.",
                {
                    "exit_code": -1,
                    "timed_out": True,
                    "version_before": before,
                    "version_after": after,
                    "changed": before != after,
                    "partial_install_suspected": before != after,
                },
            )
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            return ActuatorResult(False, f"Package installation failed: {e}", {})

    @staticmethod
    def _installed_version(python_executable: str, spec: str) -> str:
        """The installed version of ``spec``'s distribution, or ''."""
        name = re.split(r"[\[=<>!~]", str(spec or ""), maxsplit=1)[0].strip()
        if not name:
            return ""
        probe = (
            "import importlib.metadata as m,sys\n"
            f"try: print(m.version({name!r}))\n"
            "except Exception: print('')\n"
        )
        try:
            res = get_subprocess_gateway().run(
                [python_executable, "-I", "-S", "-c", probe],
                timeout=20.0,
                source="package_install_actuator_probe",
                accelerator_capability="none",
            )
            return (res.stdout or "").strip()
        except (subprocess.SubprocessError, OSError, ValueError):
            return ""
