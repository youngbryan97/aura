from core.runtime.errors import record_degradation
import asyncio
import heapq
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional
from uuid import uuid4

logger = logging.getLogger("Aura.GlobalWorkspace")

@dataclass(order=True)
class WorkItem:
    priority: float
    ts: float = field(compare=False)
    id: str = field(compare=False)
    source: str = field(compare=False)
    payload: Dict[str, Any] = field(compare=False)
    reason: Optional[str] = field(compare=False)


class HistoryBuffer:
    """Fixed-size history with deque performance and list-like slices."""

    def __init__(self, maxlen: int, items: Optional[Iterable[WorkItem]] = None):
        self.maxlen = maxlen
        self._items = deque(items or [], maxlen=maxlen)

    def append(self, item: WorkItem) -> None:
        self._items.append(item)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[WorkItem]:
        return iter(self._items)

    def __getitem__(self, index):
        items = list(self._items)
        return items[index]

class GlobalWorkspace:
    """Central Attention Bus for Aura.
    Prioritizes and dispatches work items to subscribers.
    Implementation of Global Workspace Theory (GWT).
    """

    def __init__(self):
        self._queue = []
        self._subscribers: List[Callable[[WorkItem], Any]] = []
        self._lock = asyncio.Lock()
        self.max_history = 500
        self._history = HistoryBuffer(self.max_history) # Phase 16: Narrative History
        self._stop = False
        logger.info("Global Workspace initialized.")

    @property
    def history(self) -> HistoryBuffer:
        return self._history

    @history.setter
    def history(self, value: Iterable[WorkItem]):
        self._history = value if isinstance(value, HistoryBuffer) else HistoryBuffer(self.max_history, value)

    async def publish(self, priority: float, source: str, payload: Dict[str,Any], reason: Optional[str]=None):
        """Publish a new item to the workspace.
        priority: Lower value = Higher priority (heapq default).
        """
        # Phase 19.2: Priority Negotiation
        negotiated_priority = await self._negotiate_priority(priority, source, payload)
        
        async with self._lock:
            # Timestamp tie-breaker ensures FIFO for equal priority
            wi = WorkItem(
                priority=negotiated_priority, 
                ts=time.time(), 
                id=str(uuid4()), 
                source=source, 
                payload=payload, 
                reason=reason
            )
            heapq.heappush(self._queue, wi)
        return wi.id

    async def _negotiate_priority(self, priority: float, source: str, payload: Dict[str, Any]) -> float:
        """Resolve conflicts between Affective Urges and Strategic Goals."""
        try:
            from .container import ServiceContainer
            
            # 1. Fetch Affective State (Urges)
            affect = ServiceContainer.get("affect_engine", default=None)
            hunger = getattr(affect, "curiosity", 0.5) if affect else 0.5
            
            # 2. Fetch Strategic Importance
            is_strategic = payload.get("strategic", False) or source == "StrategicPlanner"
            
            # 3. Negotiation Logic
            if is_strategic:
                # Strategic goals are resistant to displacement but curiosity can 'interrupt'
                if hunger > 0.8:
                    logger.info("🧠 Negotiation: High curiosity is challenging strategic focus.")
                    return priority * 1.2 # Slightly degrade strategic priority
                return priority * 0.8 # Boost strategic priority
                
            if source == "CuriosityEngine":
                # Curiosity gets a boost if high hunger
                return priority * (1.0 - (hunger * 0.5))

            return priority
        except Exception as e:
            record_degradation('global_workspace', e)
            logger.debug("Negotiation error (defaulting to raw priority): %s", e)
            return priority

    def subscribe(self, handler: Callable[[WorkItem], Any]):
        """Register a handler to receive broadcasted work items."""
        self._subscribers.append(handler)

    async def get_status(self) -> Dict[str, Any]:
        """Get a snapshot of the workspace state."""
        async with self._lock:
            return {
                "queue_size": len(self._queue),
                "top_priority": self._queue[0].priority if self._queue else None,
                "top_source": self._queue[0].source if self._queue else None,
            }

    async def _dispatch_one(self, wi: WorkItem):
        # Dispatch to subscribers sequentially to avoid racey side effects
        # In a full GWT implementation, this would be a broadcast to all modules
        for s in list(self._subscribers):
            try:
                r = s(wi)
                if asyncio.iscoroutine(r):
                    await r
            except Exception as e:
                record_degradation('global_workspace', e)
                logger.error("Subscriber failed processing %s: %s", wi.id, e)
        
        # Phase 16: Append to history for summarization
        self._history.append(wi)

    async def run_loop(self, poll_interval: float = 0.1):
        """Main event loop for the workspace."""
        logger.info("Global Workspace loop starting...")
        while not self._stop:
            item = None
            async with self._lock:
                if self._queue:
                    item = heapq.heappop(self._queue)
            
            if item:
                await self._dispatch_one(item)
            else:
                await asyncio.sleep(poll_interval)
        logger.info("Global Workspace loop stopped.")

    async def stop(self):
        self._stop = True
