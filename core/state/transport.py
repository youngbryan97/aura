import asyncio
import logging

logger = logging.getLogger("State.Transport")

async def safe_commit(repo, state, reason, retries: int = 3):
    for attempt in range(retries):
        try:
            return await repo.commit(state, reason)
        except (BrokenPipeError, ConnectionError):
            if attempt == retries - 1:
                raise
            logger.warning("Broken pipe during commit, reconnecting (attempt %d/%d)", attempt + 1, retries)
            if hasattr(repo, "reconnect"):
                await repo.reconnect()
            await asyncio.sleep(0.25 * (attempt + 1))

async def safe_send(bus, topic, payload, retries: int = 3):
    for attempt in range(retries):
        try:
            return await bus.send(topic, payload)
        except (BrokenPipeError, ConnectionError):
            if attempt == retries - 1:
                raise
            logger.warning("Broken pipe during event send, reconnecting (attempt %d/%d)", attempt + 1, retries)
            if hasattr(bus, "reconnect"):
                await bus.reconnect()
            await asyncio.sleep(0.1 * (attempt + 1))
