"""Fetch a URL and return what came back.

LIVE GAP, 2026-08-20. Given an API endpoint and asked to work out how to call
it, she reached for ``web_search`` — the closest thing on offer — and answered
"let's make a request", because among seventy-five skills not one could make
one. Reading a document at a URL is the primitive under learning an unfamiliar
API, checking a company's own filings, and driving a service that has no
skill of its own; without it each of those degrades into searching for
somebody else's description of the thing.

Two rules keep this cheap and safe:

* **A read is a read.** GET and HEAD are ``read_only``. Every other method can
  change something on a server that is not ours, so they carry ``external_io``
  and reach the dispatch only when the turn is allowed that far.
* **Private address space is not the web.** A URL that resolves to loopback,
  a private range, or a cloud metadata endpoint is refused. The runtime's own
  API listens on loopback, and a fetch skill that can reach it is a way to
  drive Aura by asking her to read a link.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import socket
from typing import Any, Literal
from urllib.parse import urlencode, urlparse, urlsplit, urlunsplit

from pydantic import BaseModel, Field

from core.runtime.errors import record_degradation
from core.runtime.public_http_transport import request_public_http
from core.skills.base_skill import BaseSkill
from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT

#: Methods that only ask a server a question.
READ_METHODS: tuple[str, ...] = ("GET", "HEAD")

#: Methods that ask it to change something.
WRITE_METHODS: tuple[str, ...] = ("POST", "PUT", "PATCH", "DELETE")

#: Enough of a response to read and reason about, not enough to fill memory.
MAX_BODY_BYTES = 512 * 1024

#: Past this the caller wanted a download, which is a different skill.
MAX_TIMEOUT_SECONDS = 45.0

#: Cloud instance-metadata services, which hand out credentials to anything
#: inside the host that asks.
_METADATA_HOSTS = frozenset({"169.254.169.254", "metadata.google.internal", "metadata.goog"})


class HttpRequestInput(BaseModel):
    url: str = Field(..., description="Absolute http:// or https:// URL to request.")
    method: Literal["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"] = Field(
        "GET", description="HTTP method. GET and HEAD only read; the rest change server state."
    )
    params: dict[str, Any] | None = Field(
        None, description="Query-string parameters, appended to the URL."
    )
    headers: dict[str, str] | None = Field(None, description="Request headers.")
    json_body: dict[str, Any] | list[Any] | None = Field(
        None, description="JSON request body, for methods that take one."
    )
    timeout_seconds: float | None = Field(
        None, description=f"Seconds to wait, up to {MAX_TIMEOUT_SECONDS}."
    )


def _resolved_addresses(host: str) -> list[str]:
    """Every address the host resolves to, so none of them can be a surprise."""
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, OSError):
        return []
    return sorted({str(info[4][0]) for info in infos})


def _address_is_public(address: str) -> bool:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    return not (
        parsed.is_private
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_reserved
        or parsed.is_multicast
        or parsed.is_unspecified
    )


def check_url(url: str) -> tuple[str, str]:
    """The URL to fetch, or the reason it will not be fetched.

    Returns ``(url, "")`` when the request may proceed and ``("", reason)``
    when it may not. Resolution happens here rather than at request time, so a
    hostname that points into private space is refused by name.
    """
    text = str(url or "").strip()
    if not text:
        return "", "no URL given"
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"}:
        return "", f"only http and https are fetched, not {parsed.scheme or 'a bare path'}"
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return "", "the URL names no host"
    if host in _METADATA_HOSTS:
        return "", "that address hands out host credentials"
    addresses = _resolved_addresses(host)
    if not addresses:
        return "", f"{host} does not resolve"
    private = [address for address in addresses if not _address_is_public(address)]
    if private:
        return "", f"{host} resolves inside this machine's own network ({private[0]})"
    return text, ""


class HttpRequestSkill(BaseSkill):
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT

    name = "http_request"
    description = (
        "Fetch a URL over http or https and return the status, headers and body. "
        "Use this to read an API endpoint, a documentation page, a JSON feed or any "
        "web document directly, rather than searching for a description of it."
    )

    #: GET and HEAD ask a question; the rest change something on a server that
    #: is not ours, and the dispatch refuses them unless the turn reaches that
    #: far.
    ACTION_EFFECT_SCOPES = {
        "get": "read_only",
        "head": "read_only",
        "post": "external_io",
        "put": "external_io",
        "patch": "external_io",
        "delete": "external_io",
    }
    #: The skill's own scope is its worst action, as elsewhere; the per-action
    #: table above is what lets a GET be offered to a turn that may only read.
    effect_scope = "external_io"
    input_model = HttpRequestInput
    timeout_seconds = 50.0
    metabolic_cost = 1
    retry_safe = False

    async def execute(
        self, params: HttpRequestInput | dict[str, Any], context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if isinstance(params, dict):
            try:
                params = HttpRequestInput(**params)
            except (TypeError, ValueError) as exc:
                record_degradation("http_request", exc, severity="debug")
                return {"ok": False, "error": f"Invalid input: {exc}"}

        url, refusal = check_url(params.url)
        if refusal:
            return {"ok": False, "error": refusal, "url": str(params.url or "")}

        method = str(params.method or "GET").upper()
        timeout = min(float(params.timeout_seconds or 20.0), MAX_TIMEOUT_SECONDS)

        headers = {"User-Agent": "Aura/1.0 (+local research agent)"}
        headers.update({str(k): str(v) for k, v in (params.headers or {}).items()})

        if params.params:
            parsed = urlsplit(url)
            query = urlencode(params.params, doseq=True)
            url = urlunsplit(
                (
                    parsed.scheme,
                    parsed.netloc,
                    parsed.path,
                    "&".join(part for part in (parsed.query, query) if part),
                    parsed.fragment,
                )
            )

        body_payload: bytes | None = None
        if method in WRITE_METHODS and params.json_body is not None:
            body_payload = json.dumps(
                params.json_body,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")

        try:
            response = await request_public_http(
                method,
                url,
                headers=headers,
                data=body_payload,
                timeout_s=timeout,
                source=f"skill.http_request.{method.casefold()}",
                max_response_bytes=MAX_BODY_BYTES,
            )
            body = bytes(response.get("content", b""))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "http_request", exc, severity="info", action="the request did not complete"
            )
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "url": url}

        response_headers = {
            str(key).casefold(): str(value)
            for key, value in dict(response.get("headers") or {}).items()
        }
        content_type = response_headers.get("content-type", "").casefold()
        text = body.decode("utf-8", errors="replace")
        status = int(response.get("status_code") or 0)
        final_url = str(response.get("url") or url)
        bounded = str(response.get("error") or "").startswith("response_body_exceeds")
        result: dict[str, Any] = {
            "ok": bool(response.get("ok")) and status < 400,
            "url": final_url,
            "status": status,
            "content_type": content_type,
            "bytes": len(body),
            "truncated": bounded,
            "text": text,
        }
        if "json" in content_type or text.lstrip()[:1] in {"{", "["}:
            try:
                result["json"] = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                pass
        if not result["ok"]:
            result["error"] = str(response.get("error") or f"HTTP {status}")
            # What was actually requested. Without it a 400 is unattributable:
            # live 2026-08-20 the model narrated "something about the longitude
            # parameter" to the person because that was the only guess
            # available to it, and the log said only "HTTP 400".
            logging.getLogger("Skills.http_request").warning(
                "http_request %s %s -> %d; body begins %r",
                method,
                final_url[:300],
                status,
                text[:200],
            )
        return result
