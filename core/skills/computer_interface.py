"""
Computer Interface Abstraction — Ported from computer-use-preview

Provides a common interface for different browser backends (Playwright,
Selenium, Cloud) so the Computer Use agent can swap backends without
logic changes.
"""
from __future__ import annotations


from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import TracebackType


@dataclass
class ComputerAction:
    action: str
    coordinate: tuple[int, int] | None = None
    text: str | None = None


class ComputerInterface(ABC):
    """Abstract base class for computer control interfaces."""

    @abstractmethod
    async def __aenter__(self) -> ComputerInterface:
        pass  # no-op: intentional

    @abstractmethod
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass  # no-op: intentional

    @abstractmethod
    async def screenshot(self) -> bytes:
        """Capture a screenshot and return as bytes."""
        pass  # no-op: intentional

    @abstractmethod
    async def click_at(self, x: int, y: int) -> None:
        """Click at the specified coordinates."""
        pass  # no-op: intentional

    @abstractmethod
    async def type_text(self, text: str) -> None:
        """Type text at the current cursor position."""
        pass  # no-op: intentional

    @abstractmethod
    async def type_text_at(self, x: int, y: int, text: str) -> None:
        """Click and type text."""
        pass  # no-op: intentional

    @abstractmethod
    async def key(self, key_combination: str) -> None:
        """Press a keyboard key or combination."""
        pass  # no-op: intentional

    @abstractmethod
    async def scroll(self, dx: int, dy: int) -> None:
        """Scroll the current view."""
        pass  # no-op: intentional

    @abstractmethod
    async def scroll_at(self, x: int, y: int, dx: int, dy: int) -> None:
        """Scroll at a specific coordinate."""
        pass  # no-op: intentional

    @abstractmethod
    async def drag(self, start_x: int, start_y: int, end_x: int, end_y: int) -> None:
        """Drag from start to end coordinate."""
        pass  # no-op: intentional

    @abstractmethod
    async def navigate(self, url: str) -> None:
        """Navigate to URL (if browser backend)."""
        pass  # no-op: intentional

    @abstractmethod
    async def get_url(self) -> str:
        """Get the current URL (if browser backend)."""
        pass  # no-op: intentional

    @abstractmethod
    async def get_html(self) -> str:
        """Get the current DOM HTML (if browser backend)."""
        pass  # no-op: intentional
