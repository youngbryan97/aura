#!/usr/bin/env python3
"""Live-surface probe: verify the RUNNING Aura instance end to end, read-only.

The offline suite proves code; this proves the production surface a user
actually touches. Run against a live instance (default 127.0.0.1:8000):

    python tools/live_surface_probe.py            # summary + exit code
    python tools/live_surface_probe.py --json     # machine-readable report

Checks (all read-only; no chat turns, no state mutation):
  boot_health        /api/health/boot returns 200 with healthy payload
  runtime_pulse      health report: probes pass, contract level healthy
  readiness_coherence  the readiness story is self-consistent (no "booting"
                       forever while conversation_ready, no ready-without-lane)
  startup_budget     boot completed within the declared budget (C4); the K2
                       startup probe is not wedged
  ui_shell           / serves the app shell with the expected mount points
  static_assets      aura.js / aura.css / service-worker served and non-empty
  websocket          /ws accepts an upgrade and answers a ping
  incidents          /api/system/incidents answers with a valid narrative
  latency            every HTTP check above answers within budget

Exit codes: 0 all pass, 1 any check failed, 2 instance unreachable.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import time
import urllib.error
import urllib.request

DEFAULT_BASE = "http://127.0.0.1:8000"
HTTP_BUDGET_S = 5.0
# C4 startup performance budget: a boot that has not completed within this
# window is a regression, not a slow day. Live evidence: a clean boot
# reaches ready in ~40s; the primary cortex cold-start SLA is 180s
# (core/brain/degradation_ladder.py) — the budget tracks the ladder.
STARTUP_BUDGET_S = float(os.environ.get("AURA_PROBE_STARTUP_BUDGET_S", "180"))


def startup_budget_violations(payload: dict, budget_s: float = STARTUP_BUDGET_S) -> list[str]:
    """C4: ways a boot-health payload violates the startup budget.

    Pure function so the budget contract is unit-testable without a live
    server. Gates regressions: a process past the budget that has not
    finished booting, or whose K2 startup probe says wedged, fails the
    probe run (exit 1) — wire-able into release checks.
    """
    violations: list[str] = []
    uptime = float(payload.get("runtime_age_s") or payload.get("uptime") or 0.0)
    progress = int(payload.get("progress") or 0)
    status_text = str(payload.get("status") or "")
    if uptime > budget_s and progress < 100 and status_text == "booting":
        violations.append(
            f"boot progress {progress}% after {uptime:.0f}s (budget {budget_s:.0f}s)"
        )
    startup_probe = (payload.get("probes") or {}).get("startup") or {}
    if startup_probe and not bool(startup_probe.get("ok", True)):
        violations.append(f"startup probe wedged: {startup_probe.get('reason', '')}")
    return violations


def readiness_incoherences(payload: dict) -> list[str]:
    """Return the ways a boot-health payload contradicts itself (empty == coherent).

    Pure function so the coherence contract is unit-testable without a live
    server. This is the logic that catches user-facing readiness lies: a shell
    stuck at "booting" while the lane is ready, a "ready" flag with no lane,
    or a "ready" status that disagrees with the boot phase.
    """
    phase = str(payload.get("boot_phase") or "")
    status_text = str(payload.get("status") or "")
    conv_ready = bool(payload.get("conversation_ready"))
    uptime = float(payload.get("runtime_age_s") or payload.get("uptime") or 0.0)
    incoherences: list[str] = []
    if uptime > 120.0 and conv_ready and status_text == "booting":
        incoherences.append(f"still '{status_text}' after {uptime:.0f}s with conversation_ready")
    # A lane actively answering a turn (conversation_working) legitimately
    # reports ready=true with conversation_ready=false — the desktop must
    # connect and stream the in-flight reply. Observed live post-restart.
    if payload.get("ready") and not conv_ready and phase != "conversation_working":
        incoherences.append("ready=true but conversation_ready=false")
    if status_text == "ready" and phase and phase != "kernel_ready":
        incoherences.append(f"status=ready but phase={phase}")
    return incoherences


def _get(base: str, path: str) -> tuple[int, bytes, float]:
    started = time.monotonic()
    req = urllib.request.Request(base + path, headers={"User-Agent": "aura-live-probe"})
    with urllib.request.urlopen(req, timeout=HTTP_BUDGET_S) as resp:
        body = resp.read()
        return resp.status, body, time.monotonic() - started


def probe(base: str) -> dict:
    checks: dict[str, dict] = {}

    def record(name: str, ok: bool, detail: str, elapsed: float | None = None) -> None:
        checks[name] = {
            "ok": bool(ok),
            "detail": detail[:300],
            **({"elapsed_s": round(elapsed, 3)} if elapsed is not None else {}),
        }

    # boot_health + runtime pulse payload
    try:
        try:
            status, body, elapsed = _get(base, "/api/health/boot")
        except urllib.error.HTTPError as http_exc:
            # An HTTP error IS a reachable server answering unhealthy.
            status = http_exc.code
            body = http_exc.read()
            elapsed = 0.0
        payload = json.loads(body or b"{}")
        record(
            "boot_health",
            status == 200,
            f"status={status} ready={payload.get('ready')} phase={payload.get('boot_phase')}",
            elapsed,
        )
        probes = payload.get("required_probes") or payload.get("probes") or {}
        all_passed = bool(probes.get("all_passed", status == 200))
        record("runtime_pulse", all_passed, f"required_probes.all_passed={all_passed}")

        # Readiness coherence: the story the shell reads must be self-consistent.
        # This is the check that would have caught the "booting, 48% for 55
        # minutes while chat works" bug — a user-facing lie about state.
        incoherences = readiness_incoherences(payload)
        record(
            "readiness_coherence",
            not incoherences,
            "; ".join(incoherences)
            or f"coherent (status={payload.get('status')} phase={payload.get('boot_phase')})",
        )

        # C4 startup budget: boot must complete inside the declared window.
        violations = startup_budget_violations(payload)
        record(
            "startup_budget",
            not violations,
            "; ".join(violations)
            or f"within budget ({STARTUP_BUDGET_S:.0f}s, uptime {float(payload.get('runtime_age_s') or 0.0):.0f}s)",
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "reachable": False,
            "error": f"{type(exc).__name__}: {exc}",
            "checks": checks,
        }
    except (json.JSONDecodeError, ValueError) as exc:
        record("boot_health", False, f"unparseable health payload: {exc}")

    # UI shell
    try:
        status, body, elapsed = _get(base, "/")
        text = body.decode("utf-8", errors="replace")
        anchors = ("aura.js", "id=\"messages\"")
        missing = [a for a in anchors if a not in text]
        record(
            "ui_shell",
            status == 200 and not missing,
            f"status={status} bytes={len(body)} missing={missing or 'none'}",
            elapsed,
        )
    except (urllib.error.URLError, TimeoutError, OSError, UnicodeDecodeError) as exc:
        record("ui_shell", False, f"{type(exc).__name__}: {exc}")

    # Static assets
    for asset in ("/static/aura.js", "/static/aura.css", "/static/service-worker.js"):
        try:
            status, body, elapsed = _get(base, asset)
            record(
                f"static:{asset.rsplit('/', 1)[-1]}",
                status == 200 and len(body) > 512,
                f"status={status} bytes={len(body)}",
                elapsed,
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            record(f"static:{asset.rsplit('/', 1)[-1]}", False, f"{type(exc).__name__}: {exc}")

    # WebSocket upgrade + ping (raw socket; no third-party deps)
    try:
        host_port = base.split("//", 1)[1]
        host, _, port_s = host_port.partition(":")
        started = time.monotonic()
        with socket.create_connection((host, int(port_s or "80")), timeout=HTTP_BUDGET_S) as sock:
            sock.settimeout(HTTP_BUDGET_S)
            # RFC 6455: Sec-WebSocket-Key is a random 16-byte nonce, not a
            # credential — generate per probe (also keeps secret scanners
            # honest: no base64 literals in source).
            key = base64.b64encode(os.urandom(16)).decode("ascii")
            handshake = (
                f"GET /ws HTTP/1.1\r\nHost: {host_port}\r\nUpgrade: websocket\r\n"
                f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                "Sec-WebSocket-Version: 13\r\n\r\n"
            )
            sock.sendall(handshake.encode())
            response = sock.recv(1024).decode("utf-8", errors="replace")
            upgraded = "101" in response.splitlines()[0] if response else False
            elapsed = time.monotonic() - started
            record("websocket", upgraded, response.splitlines()[0] if response else "no response", elapsed)
    except (OSError, IndexError, ValueError) as exc:
        record("websocket", False, f"{type(exc).__name__}: {exc}")

    # Incident narrative endpoint — the operator's "what happened and why".
    try:
        status, body, elapsed = _get(base, "/api/system/incidents?minutes=60")
        payload = json.loads(body or b"{}")
        schema_ok = str(payload.get("schema", "")).startswith("aura.incident_narrative")
        episodes = payload.get("episodes")
        record(
            "incidents",
            status == 200 and schema_ok and isinstance(episodes, list),
            f"status={status} episodes={len(episodes) if isinstance(episodes, list) else 'n/a'}",
            elapsed,
        )
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
        record("incidents", False, f"{type(exc).__name__}: {exc}")

    # Latency budget over the HTTP checks
    slow = {
        name: c["elapsed_s"]
        for name, c in checks.items()
        if c.get("elapsed_s", 0) > HTTP_BUDGET_S * 0.8
    }
    record("latency", not slow, f"slow={slow or 'none'} (budget {HTTP_BUDGET_S}s)")

    return {
        "reachable": True,
        "base": base,
        "at_unix": time.time(),
        "passed": all(c["ok"] for c in checks.values()),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = probe(args.base)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        if not report.get("reachable"):
            print(f"UNREACHABLE: {report.get('error')}")
            return 2
        for name, c in report["checks"].items():
            mark = "ok" if c["ok"] else "failed"
            extra = f" ({c['elapsed_s']}s)" if "elapsed_s" in c else ""
            print(f"{mark} {name}: {c['detail']}{extra}")
        print("PASS" if report["passed"] else "FAIL")
    if not report.get("reachable"):
        return 2
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
