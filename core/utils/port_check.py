import socket
import time
import logging

from core.runtime.network_gateway import get_network_gateway

logger = logging.getLogger("Aura.Utils.PortCheck")

_PORT_CHECK_ERRORS = (OSError, TimeoutError, socket.timeout, ConnectionRefusedError)
_HTTP_READY_ERRORS = (OSError, RuntimeError, TimeoutError, TypeError, ValueError)

def is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.connect((host, port))
            return True
        except _PORT_CHECK_ERRORS:
            return False

def wait_for_port(port: int, host: str = "127.0.0.1", timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_port_open(port, host):
            return True
        time.sleep(0.5)
    return False

def is_http_ready(port: int, host: str = "127.0.0.1", path: str = "/api/health") -> bool:
    url = f"http://{host}:{port}{path}"
    try:
        resp = get_network_gateway().request(
            "GET",
            url,
            timeout=3,
            source="port_check.is_http_ready",
            read_only=True,
            suppress_degradation=True,
        )
        status_code = int(resp.get("status_code", 0) or 0)
        return 0 < status_code < 500
    except _HTTP_READY_ERRORS as exc:
        logger.debug("HTTP readiness probe failed for %s: %s", url, exc)
        return False

def wait_for_http(port: int, host: str = "127.0.0.1", path: str = "/api/health", timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_http_ready(port, host, path):
            return True
        time.sleep(1.0)
    return False
