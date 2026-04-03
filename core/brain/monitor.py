import logging
# core/brain/monitor.py
import asyncio
import psutil
import time
from typing import Callable, Optional

class SelfMonitor:
    """
    Lightweight monitor that periodically checks system health and calls hooks.
    """

    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self._running = False
        self._task = None
        self._on_high_memory: Optional[Callable[[float], None]] = None
        self._on_critical_memory: Optional[Callable[[float], None]] = None
        self._on_tick: Optional[Callable[[], None]] = None
        self.high_threshold = 80.0
        self.critical_threshold = 92.0

    def set_high_memory_hook(self, fn: Callable[[float], None]):
        self._on_high_memory = fn

    def set_critical_memory_hook(self, fn: Callable[[float], None]):
        self._on_critical_memory = fn

    def set_tick_hook(self, fn: Callable[[], None]):
        self._on_tick = fn

    async def _loop(self):
        while self._running:
            vm = psutil.virtual_memory()
            if vm.percent >= self.critical_threshold and self._on_critical_memory:
                try:
                    self._on_critical_memory(vm.percent)
                except Exception as _e:
                    logging.debug('Ignored Exception in monitor.py: %s', _e)
            elif vm.percent >= self.high_threshold and self._on_high_memory:
                try:
                    self._on_high_memory(vm.percent)
                except Exception as _e:
                    logging.debug('Ignored Exception in monitor.py: %s', _e)
            if self._on_tick:
                try:
                    self._on_tick()
                except Exception as _e:
                    logging.debug('Ignored Exception in monitor.py: %s', _e)
            await asyncio.sleep(self.interval)

    def start(self):
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._loop())

    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
