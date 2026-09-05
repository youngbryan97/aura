from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import signal
import time
import urllib.parse
from typing import Any

from core.bus.local_pipe_bus import LocalPipeBus
from core.capabilities.phantom_browser import PhantomBrowser
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.network_gateway import get_network_gateway
from core.runtime.url_policy import describe_decision
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.SensoryGate")

#: Result envelope version so a caller can tell an observation from an error
#: dictionary that happens to have the same keys (CP126 c8c56e76).
SENSORY_RESULT_SCHEMA = "aura.sensory.observation.v1"

#: End-to-end budget for one browse request: dependency start, navigation,
#: human delay and extraction together (CP126 1fb6515c).
DEFAULT_BROWSE_DEADLINE_S = 45.0
DEFAULT_SEARCH_DEADLINE_S = 15.0
MAX_REQUEST_DEADLINE_S = 120.0

#: Bounds on what a single observation may return to the parent.
MAX_CONTENT_CHARS = 200_000

#: How long shutdown may spend closing the browser and draining tasks before
#: the actor stops waiting and lets the process exit.
SHUTDOWN_GRACE_S = 10.0

#: Heartbeats missed before the actor concludes the supervisor is gone.
MAX_HEARTBEAT_FAILURES = 5

SENSORY_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)


def _record_sensory_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, object] | None = None,
) -> None:
    record_degradation(
        "sensory_gate",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=False,
        extra=extra,
    )


class SensoryGateActor:
    """Isolated sensory actor for browser/search observation work."""

    def __init__(
        self,
        connection: Any,
        *,
        supervisor_pid: int | None = None,
        shutdown_token: str = "",
        authorized_principals: tuple[str, ...] = (),
    ):
        self.bus = LocalPipeBus(is_child=True, connection=connection)
        self.browser: PhantomBrowser | None = None
        self._is_active = True
        self._heartbeat_interval = 3.0
        self._heartbeat_failures = 0
        self._background_tasks: set[asyncio.Task] = set()
        self._shutdown_event: asyncio.Event | None = None
        # CP126 4dc9dc31: shutdown used to be an ordinary handler that any
        # message could trigger. It now requires the supervisor's token and a
        # nonce that cannot be replayed.
        self._shutdown_token = str(
            shutdown_token or os.environ.get("AURA_SENSORY_SHUTDOWN_TOKEN", "")
        )
        self._used_shutdown_nonces: set[str] = set()
        self._shutdown_reason = ""
        # CP126 b25cb82b: without a parent-death monitor a lost pipe leaves an
        # orphan holding a browser forever.
        self._supervisor_pid = int(
            supervisor_pid or os.environ.get("AURA_SENSORY_SUPERVISOR_PID", 0) or os.getppid()
        )
        self._authorized_principals = tuple(
            principal for principal in authorized_principals if principal
        ) or tuple(
            part.strip()
            for part in os.environ.get("AURA_SENSORY_PRINCIPALS", "").split(",")
            if part.strip()
        )
        self._last_observation_ts = 0.0
        self._last_heartbeat_ok_ts = time.time()

    def _track_task(self, coro: Any, *, name: str | None = None) -> asyncio.Task:
        task = get_task_tracker().create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def _request_shutdown(self, reason: str = "") -> None:
        if reason and not self._shutdown_reason:
            self._shutdown_reason = reason
            logger.info("👁️ SensoryGate shutdown requested: %s", reason)
        self._is_active = False
        if self._shutdown_event is not None:
            self._shutdown_event.set()

    async def _cancel_background_tasks(self, *, timeout: float = SHUTDOWN_GRACE_S) -> None:
        """Cancel and drain tracked tasks within a deadline.

        CP126 1fb6515c: the gather was unbounded, so a task that ignored
        cancellation could hold process teardown open indefinitely.
        """
        tasks = [task for task in self._background_tasks if not task.done()]
        if not tasks:
            self._background_tasks.clear()
            return
        for task in tasks:
            task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=timeout
            )
        except TimeoutError:
            stuck = [task.get_name() for task in tasks if not task.done()]
            _record_sensory_degradation(
                TimeoutError("background_tasks_did_not_cancel"),
                action="abandoned uncancellable sensory tasks so shutdown could finish",
                severity="warning",
                extra={"tasks": stuck},
            )
            logger.error("❌ SensoryGate tasks did not cancel in time: %s", stuck)
        self._background_tasks.clear()

    async def run(self) -> None:
        """Main actor loop."""
        logger.info("👁️ SensoryGate Actor starting...")
        self._shutdown_event = asyncio.Event()
        try:
            try:
                self.browser = PhantomBrowser(visible=False, principal="sensory_gate")
            except SENSORY_RECOVERABLE_ERRORS as exc:
                _record_sensory_degradation(
                    exc,
                    action="failed sensory gate startup before accepting browser/search work",
                    severity="critical",
                )
                raise

            self.bus.register_handler("browse", self._handle_browse)
            self.bus.register_handler("search", self._handle_search)
            self.bus.register_handler("ping", lambda payload, tid: "pong")
            self.bus.register_handler("shutdown", self._handle_shutdown)

            self.bus.start()
            if not self._is_active:
                self._request_shutdown("bus_inactive_at_start")
            self._track_task(self._heartbeat_loop(), name="sensory_gate.heartbeat")
            self._track_task(self._liveness_loop(), name="sensory_gate.liveness")

            logger.info("👁️ SensoryGate Actor ready.")
            await self._shutdown_event.wait()
        finally:
            self._request_shutdown("run_loop_exited")
            await self._cancel_background_tasks()
            try:
                await self.bus.stop()
            except SENSORY_RECOVERABLE_ERRORS as exc:
                _record_sensory_degradation(
                    exc,
                    action="continued sensory gate shutdown after bus stop failed",
                    severity="degraded",
                )
                logger.error("❌ SensoryGate bus shutdown failed: %s", exc)
            if self.browser is not None:
                try:
                    # CP126 1fb6515c: an unbounded close could hang teardown
                    # behind a wedged browser process.
                    await asyncio.wait_for(self.browser.close(), timeout=SHUTDOWN_GRACE_S)
                except TimeoutError as exc:
                    _record_sensory_degradation(
                        exc,
                        action="abandoned a browser close that exceeded the shutdown budget",
                        severity="degraded",
                    )
                    logger.error("❌ SensoryGate browser close exceeded its budget")
                except SENSORY_RECOVERABLE_ERRORS as exc:
                    _record_sensory_degradation(
                        exc,
                        action="continued sensory gate shutdown after browser close failed",
                        severity="degraded",
                    )
                    logger.error("❌ SensoryGate browser shutdown failed: %s", exc)
            logger.info("👁️ SensoryGate Actor stopped.")

    def _health_snapshot(self) -> dict[str, Any]:
        """What the actor can actually attest to right now.

        CP126 62c1e3f3: the heartbeat reported "healthy" whenever a local
        counter was zero, then reset that counter after an unchecked send — so
        a dead transport or a wedged browser kept claiming health.
        """
        browser_ready = False
        browser_detail = "absent"
        if self.browser is not None:
            try:
                status = self.browser.get_status()
                browser_ready = bool(
                    status.get("is_active", getattr(self.browser, "is_active", False))
                )
                browser_detail = "active" if browser_ready else "inactive"
            except SENSORY_RECOVERABLE_ERRORS as exc:
                browser_detail = f"probe_failed: {type(exc).__name__}"

        bus_ready = bool(getattr(self.bus, "is_running", True)) and self._is_active
        healthy = bus_ready and browser_ready and self._heartbeat_failures == 0
        return {
            "pid": os.getpid(),
            "ts": time.time(),
            "status": "healthy" if healthy else "degraded",
            "bus_ready": bus_ready,
            "browser": browser_detail,
            "consecutive_heartbeat_failures": self._heartbeat_failures,
            "last_observation_age_s": (
                round(time.time() - self._last_observation_ts, 3)
                if self._last_observation_ts
                else None
            ),
            "supervisor_pid": self._supervisor_pid,
        }

    async def _heartbeat_loop(self) -> None:
        """Send heartbeats to the supervisor and escalate when they stop landing."""
        while self._is_active:
            try:
                await self.bus.send("heartbeat", self._health_snapshot())
                self._heartbeat_failures = 0
                self._last_heartbeat_ok_ts = time.time()
            except SENSORY_RECOVERABLE_ERRORS as exc:
                self._heartbeat_failures += 1
                _record_sensory_degradation(
                    exc,
                    action="kept sensory actor alive and retried supervisor heartbeat on next interval",
                    severity="warning",
                    extra={"consecutive_heartbeat_failures": self._heartbeat_failures},
                )
                logger.error("❌ Heartbeat failed: %s", exc)
                if self._heartbeat_failures >= MAX_HEARTBEAT_FAILURES:
                    # CP126 b25cb82b: retrying forever is how an orphan is born.
                    self._request_shutdown(
                        f"heartbeat_unreachable_after_{self._heartbeat_failures}_attempts"
                    )

            shutdown_event = self._shutdown_event
            if shutdown_event is None:
                await asyncio.sleep(self._heartbeat_interval)
                continue
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=self._heartbeat_interval)
            except TimeoutError:
                continue

    async def _liveness_loop(self) -> None:
        """Stop the actor when the supervisor is gone.

        CP126 b25cb82b: the main loop waited only on its own shutdown event, so
        a supervisor that died without delivering the shutdown message left a
        child holding a browser against an unusable bus.
        """
        while self._is_active:
            if self._supervisor_pid and not self._parent_alive():
                self._request_shutdown(f"supervisor_pid_{self._supervisor_pid}_gone")
                return
            shutdown_event = self._shutdown_event
            if shutdown_event is None:
                await asyncio.sleep(self._heartbeat_interval)
                continue
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=self._heartbeat_interval)
                return
            except TimeoutError:
                continue

    def _parent_alive(self) -> bool:
        try:
            os.kill(self._supervisor_pid, 0)
        except ProcessLookupError:
            return False
        except (PermissionError, OSError):
            # Still there, we just may not signal it.
            return True
        # A reparented orphan reports init as its parent.
        return os.getppid() == self._supervisor_pid or self._supervisor_pid == os.getppid()

    @staticmethod
    def _deadline(payload: dict[str, Any], default: float) -> float:
        try:
            requested = float(payload.get("deadline_s", default))
        except (TypeError, ValueError):
            return default
        if requested <= 0 or requested != requested:  # non-positive or NaN
            return default
        return min(requested, MAX_REQUEST_DEADLINE_S)

    def _authorized(self, payload: dict[str, Any]) -> tuple[str, str]:
        """The calling principal, or ('', reason) when the call is unauthorized."""
        principal = str(payload.get("principal") or "").strip()
        if not self._authorized_principals:
            # No roster configured: accept, but the receipt says the request
            # was unattributed rather than pretending it was authenticated.
            return principal or "unattributed", ""
        if not principal:
            return "", "request carried no principal"
        if principal not in self._authorized_principals:
            return "", f"principal {principal!r} is not authorized for sensory work"
        return principal, ""

    def _refusal(self, kind: str, reason: str, trace_id: str, **extra: Any) -> dict[str, Any]:
        return {
            "schema": SENSORY_RESULT_SCHEMA,
            "kind": kind,
            "ok": False,
            "error": reason,
            "trace_id": trace_id,
            "observation_only": True,
            "requires_governance_for_effects": True,
            **extra,
        }

    async def _handle_browse(self, payload: dict[str, Any], trace_id: str) -> dict[str, Any]:
        """Observe a URL under the canonical outbound policy.

        CP126 3bba0f36: caller-controlled URL text used to reach PhantomBrowser
        directly — no principal, no scheme or host policy, no private-network
        or local-file denial, no port bound, no receipt. The gate now shares
        the same policy the web-fetch actuator enforces.
        """
        if not isinstance(payload, dict):
            return self._refusal("browse", "payload must be an object", trace_id)

        principal, auth_error = self._authorized(payload)
        if auth_error:
            _record_sensory_degradation(
                PermissionError(auth_error),
                action="refused an unauthorized browse request",
                severity="warning",
                extra={"trace_id": trace_id},
            )
            return self._refusal("browse", auth_error, trace_id)

        raw_url = str(payload.get("url") or "").strip()
        if not raw_url:
            return self._refusal("browse", "No URL provided", trace_id)

        decision = describe_decision(raw_url, principal=principal, stage="request")
        if not decision["allowed"]:
            _record_sensory_degradation(
                PermissionError(decision["reason"]),
                action="refused a browse request that failed outbound URL policy",
                severity="warning",
                extra={"trace_id": trace_id, "host": decision["host"]},
            )
            logger.warning(
                "🚫 [%s] Browse refused: %s", trace_id[:8], decision["reason"]
            )
            return self._refusal("browse", decision["reason"], trace_id, policy=decision)

        url = decision["url"]
        if self.browser is None:
            error = RuntimeError("browser_unavailable")
            _record_sensory_degradation(
                error,
                action="blocked browse request because browser was not initialized",
                severity="degraded",
                extra={"trace_id": trace_id},
            )
            return self._refusal("browse", "browser_unavailable", trace_id, policy=decision)

        deadline_s = self._deadline(payload, DEFAULT_BROWSE_DEADLINE_S)
        logger.info("🌐 [%s] Browsing: %s", trace_id[:8], url)
        started = time.monotonic()
        try:
            # CP126 1fb6515c: dependency start, navigation, delay and
            # extraction are ONE operation with ONE budget, so a slow site
            # cannot delay later serial control handling indefinitely.
            return await asyncio.wait_for(
                self._observe(url, principal, decision, trace_id, started),
                timeout=deadline_s,
            )
        except TimeoutError as exc:
            _record_sensory_degradation(
                exc,
                action="returned a browse timeout receipt within the request deadline",
                severity="warning",
                extra={"trace_id": trace_id, "url": url, "deadline_s": deadline_s},
            )
            return self._refusal(
                "browse",
                f"browse exceeded its {deadline_s:.0f}s deadline",
                trace_id,
                url=url,
                policy=decision,
                elapsed_s=round(time.monotonic() - started, 3),
            )
        except SENSORY_RECOVERABLE_ERRORS as exc:
            _record_sensory_degradation(
                exc,
                action="returned browse error result without crashing sensory actor",
                severity="warning",
                extra={"trace_id": trace_id, "url": url},
            )
            logger.error("❌ [%s] Browse failed: %s", trace_id[:8], exc)
            return self._refusal("browse", str(exc), trace_id, url=url, policy=decision)

    async def _observe(
        self,
        url: str,
        principal: str,
        decision: dict[str, Any],
        trace_id: str,
        started: float,
    ) -> dict[str, Any]:
        """Navigate, then actually extract and describe what was observed.

        CP126 c8c56e76: PhantomBrowser.browse() returns a *bool*, and that bool
        was stored under "content" — so a caller could not tell a navigation
        result from page text, and got no status, final URL, content digest or
        completeness signal.
        """
        navigated = await self.browser.browse(url)
        if not navigated:
            return self._refusal(
                "browse", "navigation_failed", trace_id, url=url, policy=decision,
                elapsed_s=round(time.monotonic() - started, 3),
            )

        final_url = url
        page = getattr(self.browser, "page", None)
        if page is not None:
            final_url = str(getattr(page, "url", url) or url)
        landing = describe_decision(final_url, principal=principal, stage="landing")
        if not landing["allowed"] and final_url != url:
            # A redirect that leaves policy is a policy failure, not a result.
            _record_sensory_degradation(
                PermissionError(landing["reason"]),
                action="discarded an observation whose redirect target failed URL policy",
                severity="warning",
                extra={"trace_id": trace_id, "final_url": final_url},
            )
            return self._refusal(
                "browse", f"redirect left policy: {landing['reason']}", trace_id,
                url=url, final_url=final_url, policy=decision, landing_policy=landing,
            )

        content = ""
        extraction_error = ""
        try:
            content = str(await self.browser.read_content() or "")
        except SENSORY_RECOVERABLE_ERRORS as exc:
            extraction_error = f"{type(exc).__name__}: {exc}"
            _record_sensory_degradation(
                exc,
                action="returned a navigation-only observation after extraction failed",
                severity="warning",
                extra={"trace_id": trace_id, "url": url},
            )

        truncated = len(content) > MAX_CONTENT_CHARS
        if truncated:
            content = content[:MAX_CONTENT_CHARS]
        self._last_observation_ts = time.time()
        return {
            "schema": SENSORY_RESULT_SCHEMA,
            "kind": "browse",
            "ok": bool(content) and not extraction_error,
            "trace_id": trace_id,
            "principal": principal,
            "url": url,
            "final_url": final_url,
            "redirected": final_url != url,
            "navigated": True,
            "content": content,
            "content_chars": len(content),
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "complete": not truncated and not extraction_error,
            "truncated": truncated,
            "extraction_error": extraction_error,
            "observed_at": self._last_observation_ts,
            "elapsed_s": round(time.monotonic() - started, 3),
            "policy": decision,
            "landing_policy": landing,
            "observation_only": True,
            "requires_governance_for_effects": True,
        }

    async def _handle_search(self, payload: dict[str, Any], trace_id: str) -> dict[str, Any]:
        """Handle search request via Wikipedia OpenSearch API."""
        if not isinstance(payload, dict):
            return self._refusal("search", "payload must be an object", trace_id)
        query = str(payload.get("query") or "").strip()
        if not query:
            return self._refusal("search", "No query provided", trace_id)

        deadline_s = self._deadline(payload if isinstance(payload, dict) else {}, DEFAULT_SEARCH_DEADLINE_S)
        started = time.monotonic()
        logger.info("🔍 [%s] Wikipedia search: %s", trace_id[:8], query)
        try:
            url = (
                "https://en.wikipedia.org/w/api.php?action=opensearch"
                f"&search={urllib.parse.quote(query)}&limit=3&namespace=0&format=json"
            )

            def fetch() -> Any:
                response = get_network_gateway().request(
                    "GET",
                    url,
                    headers={"User-Agent": "Aura/1.0"},
                    timeout=5,
                    source="sensory_gate.wikipedia_search",
                    read_only=True,
                )
                if not response.get("ok"):
                    raise OSError(str(response.get("error") or "Wikipedia search failed"))
                return json.loads(bytes(response.get("content") or b"").decode("utf-8", errors="replace"))

            data = await asyncio.wait_for(asyncio.to_thread(fetch), timeout=deadline_s)
            results = self._format_search_results(data)
            self._last_observation_ts = time.time()
            return {
                "schema": SENSORY_RESULT_SCHEMA,
                "kind": "search",
                "ok": True,
                "trace_id": trace_id,
                "query": query,
                "source": "wikipedia",
                "source_url": url,
                "results": results,
                "result_count": len(results),
                "complete": True,
                "observed_at": self._last_observation_ts,
                "elapsed_s": round(time.monotonic() - started, 3),
                "observation_only": True,
                "requires_governance_for_effects": True,
            }
        except TimeoutError as exc:
            _record_sensory_degradation(
                exc,
                action="returned a search timeout receipt within the request deadline",
                severity="warning",
                extra={"trace_id": trace_id, "query": query, "deadline_s": deadline_s},
            )
            return self._refusal(
                "search", f"search exceeded its {deadline_s:.0f}s deadline", trace_id,
                query=query, elapsed_s=round(time.monotonic() - started, 3),
            )
        except SENSORY_RECOVERABLE_ERRORS as exc:
            _record_sensory_degradation(
                exc,
                action="returned search error result without crashing sensory actor",
                severity="warning",
                extra={"trace_id": trace_id, "query": query},
            )
            logger.error("❌ [%s] Wikipedia search failed: %s", trace_id[:8], exc)
            return self._refusal("search", str(exc), trace_id, query=query)

    @staticmethod
    def _format_search_results(data: Any) -> list[str]:
        if not isinstance(data, list) or len(data) < 4:
            return []
        titles = data[1] if isinstance(data[1], list) else []
        snippets = data[2] if isinstance(data[2], list) else []
        urls = data[3] if isinstance(data[3], list) else []
        return [
            f"{title}: {snippet} ({url})"
            for title, snippet, url in zip(titles, snippets, urls, strict=False)
        ]

    async def _handle_shutdown(self, payload: Any, trace_id: str) -> dict[str, Any]:
        """Shut down only for the supervisor, and only once per nonce.

        CP126 4dc9dc31: this was registered beside ordinary observation
        handlers and ignored its payload entirely, so any message named
        "shutdown" terminated the actor — no identity, capability, nonce,
        replay protection or durable acknowledgement.
        """
        payload = payload if isinstance(payload, dict) else {}
        token = str(payload.get("token") or "")
        nonce = str(payload.get("nonce") or "")
        reason = str(payload.get("reason") or "supervisor_request")[:200]

        if self._shutdown_token:
            if not hmac.compare_digest(token, self._shutdown_token):
                _record_sensory_degradation(
                    PermissionError("shutdown_token_mismatch"),
                    action="refused a shutdown message that did not carry the supervisor token",
                    severity="critical",
                    extra={"trace_id": trace_id},
                )
                logger.critical("🚫 [%s] Shutdown refused: bad token", trace_id[:8])
                return {"ok": False, "error": "shutdown_token_mismatch", "trace_id": trace_id}
            if not nonce:
                return {"ok": False, "error": "shutdown_nonce_required", "trace_id": trace_id}
            if nonce in self._used_shutdown_nonces:
                _record_sensory_degradation(
                    PermissionError("shutdown_nonce_replayed"),
                    action="refused a replayed shutdown message",
                    severity="error",
                    extra={"trace_id": trace_id},
                )
                return {"ok": False, "error": "shutdown_nonce_replayed", "trace_id": trace_id}
            self._used_shutdown_nonces.add(nonce)

        self._request_shutdown(reason)
        # Durable acknowledgement: the supervisor learns which request stopped
        # the actor, not merely that something did.
        return {
            "ok": True,
            "acknowledged": True,
            "trace_id": trace_id,
            "reason": reason,
            "pid": os.getpid(),
            "authenticated": bool(self._shutdown_token),
            "at": time.time(),
        }


def start_sensory_gate(connection: Any, *args: Any, **kwargs: Any) -> None:
    """Process entry point."""
    try:
        # SIGINT stays ignored so a terminal Ctrl-C reaches the supervisor
        # first, but SIGTERM must be able to stop the actor: CP126 b25cb82b
        # noted that with no signal path and no parent monitor, a lost
        # supervisor left this process alive holding a browser.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except SENSORY_RECOVERABLE_ERRORS as exc:
        _record_sensory_degradation(
            exc,
            action="continued sensory gate startup after signal handler setup failed",
            severity="debug",
        )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    actor = SensoryGateActor(
        connection,
        supervisor_pid=kwargs.get("supervisor_pid"),
        shutdown_token=str(kwargs.get("shutdown_token") or ""),
        authorized_principals=tuple(kwargs.get("authorized_principals") or ()),
    )
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _on_sigterm(signum: int, _frame: Any) -> None:
        logger.info("👁️ SensoryGate received signal %s; requesting shutdown", signum)
        loop.call_soon_threadsafe(actor._request_shutdown, f"signal_{signum}")

    for signum in (signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM)):
        try:
            signal.signal(signum, _on_sigterm)
        except (OSError, ValueError, RuntimeError) as exc:
            _record_sensory_degradation(
                exc,
                action="continued sensory gate startup without a signal shutdown path",
                severity="warning",
            )
    try:
        loop.run_until_complete(actor.run())
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
    finally:
        asyncio.set_event_loop(None)
        loop.close()
