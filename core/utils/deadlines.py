import time


class Deadline:
    """Enterprise Deadline management for Aura.
    Ensures a single, consistent expiration time across multiple call layers.
    """
    def __init__(self, timeout: float | None = None, start_time: float | None = None) -> None:
        self._start_time = start_time or time.monotonic()
        self._timeout = timeout
        self._expiration = self._start_time + timeout if timeout is not None else None

    @property
    def remaining(self) -> float | None:
        """Returns remaining seconds until expiration."""
        if self._expiration is None:
            return None
        return max(0.0, self._expiration - time.monotonic())

    @property
    def is_expired(self) -> bool:
        """Checks if the deadline has passed."""
        if self._expiration is None:
            return False
        return time.monotonic() >= self._expiration

    def shield(self, buffer: float = 1.0) -> float:
        """Returns remaining time minus a safety buffer."""
        rem = self.remaining
        if rem is None:
            return 300.0 # Default fallback for unshielded tasks
        return max(0.1, rem - buffer)

    def __repr__(self) -> str:
        rem = self.remaining
        rem_str = f"{rem:.2f}s" if rem is not None else "INF"
        return f"<Deadline rem={rem_str}>"

def get_deadline(timeout: float | None = None) -> Deadline:
    """Factory to create a new deadline."""
    return Deadline(timeout)
