"""Core Sandbox Interface
Defines how Aura executes code in isolated environments.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    duration: float
    
class Sandbox(ABC):
    """Abstract base class for Sandboxes (Local, Docker, Cloud).
    """
    
    @abstractmethod
    def start(self) -> None:
        """Initialize the environment (e.g. create venv, start container)."""
        pass
        
    @abstractmethod
    def stop(self) -> None:
        """Teardown the environment."""
        pass
        
    @abstractmethod
    async def run_code(self, code: str, timeout: int = 30) -> ExecutionResult:
        """Execute a Python script in the sandbox (Async)."""
        pass
        
    @abstractmethod
    async def run_command(self, command: str, timeout: int = 30) -> ExecutionResult:
        """Execute a shell command in the sandbox (Async)."""
        pass
        
    @abstractmethod
    def read_file(self, path: str) -> str:
        """Read a file from the sandbox."""
        pass
        
    @abstractmethod
    def write_file(self, path: str, content: str) -> None:
        """Write a file to the sandbox."""
        pass
