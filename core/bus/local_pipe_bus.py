import json
import uuid
import logging
import asyncio
import time
import multiprocessing
import multiprocessing.connection
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional, Tuple
from core.bus.shared_mem_bus import SharedMemoryTransport

logger = logging.getLogger("Aura.LocalPipeBus")

class LocalPipeBus:
    """
    High-performance, zero-config intra-process communication using multiprocessing.Pipe.
    Refactored to use unidirectional pipe pairs to eliminate bidirectional deadlocks.
    ZENITH LOCKDOWN: Dedicated ThreadPoolExecutor for Pipe I/O to prevent starvation.
    """
    _LIVE_BUSES: "weakref.WeakSet[LocalPipeBus]" = weakref.WeakSet()
    _SHM_OFFLOAD_THRESHOLD_BYTES = 32 * 1024
    _SHM_SEGMENT_RETENTION_SECONDS = 20.0

    @staticmethod
    def _is_connection_pair(connection: Any) -> bool:
        return isinstance(connection, tuple) and len(connection) == 2

    @classmethod
    def shutdown_executor(cls) -> None:
        for bus in list(cls._LIVE_BUSES):
            bus._shutdown_executor()

    def __init__(self, is_child: bool = False, 
                 read_conn: Optional[multiprocessing.connection.Connection] = None, 
                 write_conn: Optional[multiprocessing.connection.Connection] = None,
                 start_reader: bool = True,
                 connection: Any = None):
        self.is_child = is_child
        self.start_reader = start_reader
        
        if self._is_connection_pair(connection):
            self.read_conn, self.write_conn = connection
        elif connection is not None:
            raise ValueError(
                "LocalPipeBus requires an explicit (read_conn, write_conn) transport pair; "
                "shared single-connection compatibility is no longer supported."
            )
        elif read_conn is not None and write_conn is not None:
            self.read_conn = read_conn
            self.write_conn = write_conn
        else:
            # Create two unidirectional pipes
            # pipe1: Parent Reads, Child Writes
            p_read, c_write = multiprocessing.Pipe(duplex=False)
            # pipe2: Child Reads, Parent Writes
            c_read, p_write = multiprocessing.Pipe(duplex=False)
            
            if is_child:
                self.read_conn = c_read
                self.write_conn = c_write
            else:
                self.read_conn = p_read
                self.write_conn = p_write

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._dispatcher_task: Optional[asyncio.Task] = None
        self._dispatch_queue: Optional[asyncio.Queue] = None
        self._handlers: Dict[str, Callable] = {}
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._is_running = False
        self._activity_callback: Optional[Callable[[], None]] = None
        self._pipe_broken = False
        self._outbound_shm_segments: Dict[str, Tuple[SharedMemoryTransport, float]] = {}
        self._LIVE_BUSES.add(self)

    def _get_executor(self) -> ThreadPoolExecutor:
        executor = self._executor
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="AuraPipeIO",
            )
            self._executor = executor
        return executor

    def _shutdown_executor(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None:
            return self._loop
        try:
            return asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "LocalPipeBus requires a running event loop. Start it from async boot/runtime code."
            ) from exc

    def _safe_close_connection(self, conn: Optional[multiprocessing.connection.Connection]) -> None:
        if conn is None:
            return
        try:
            conn.close()
        except Exception as exc:
            logger.debug("📡 LocalPipeBus: connection close skipped: %s", exc)

    def start(self):
        """Start the background reader task."""
        if self._is_running and self._reader_task and not self._reader_task.done():
            return
        
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._is_running = True
        if self.start_reader:
            self._dispatch_queue = asyncio.Queue(maxsize=256)
            self._dispatcher_task = loop.create_task(self._dispatch_loop())
            self._reader_task = loop.create_task(self._read_loop())
            logger.info("📡 LocalPipeBus reader ACTIVE (Child: %s)", self.is_child)
        else:
            logger.info("📡 LocalPipeBus ACTIVE (Manual Polling mode)")

    async def stop(self):
        """Stop the reader task."""
        self._is_running = False
        self._cancel_pending_requests(cancel=True)
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await asyncio.wait_for(self._reader_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass # Normal during shutdown
            except Exception as e:
                logger.error("📡 LocalPipeBus: Error during stop: %s", e)
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            try:
                await asyncio.wait_for(self._dispatcher_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                logger.debug("Suppressed bare exception")
                pass
            except Exception as e:
                logger.error("📡 LocalPipeBus: Dispatcher stop error: %s", e)
        self._cleanup_expired_shm_segments(force=True)
        if self.read_conn is self.write_conn:
            self._safe_close_connection(self.read_conn)
        else:
            self._safe_close_connection(self.read_conn)
            self._safe_close_connection(self.write_conn)
        self._shutdown_executor()

    def register_handler(self, msg_type: str, handler: Callable):
        """Register a handler for a specific message type."""
        self._handlers[msg_type] = handler

    def set_activity_callback(self, callback: Callable[[], None]):
        """Register a lightweight callback for inbound transport activity."""
        self._activity_callback = callback

    def _cleanup_expired_shm_segments(self, *, force: bool = False):
        if not self._outbound_shm_segments:
            return

        now = time.monotonic()
        expired = [
            name
            for name, (_, expires_at) in list(self._outbound_shm_segments.items())
            if force or expires_at <= now
        ]
        for name in expired:
            shm, _ = self._outbound_shm_segments.pop(name, (None, 0.0))
            if shm is None:
                continue
            try:
                shm.close()
            except Exception as e:
                logger.debug("📡 LocalPipeBus: SHM cleanup failed for %s: %s", name, e)

    def _retain_outbound_shm(self, shm: SharedMemoryTransport):
        self._cleanup_expired_shm_segments()
        self._outbound_shm_segments[shm.name] = (
            shm,
            time.monotonic() + self._SHM_SEGMENT_RETENTION_SECONDS,
        )

    async def _prepare_payload_for_transport(self, payload: Any) -> Any:
        """Serialize large payloads off-loop and offload them to SHM when needed."""
        if not isinstance(payload, (dict, list)):
            return payload

        serialized_payload = await asyncio.to_thread(json.dumps, payload)
        payload_bytes = serialized_payload.encode("utf-8")
        if len(payload_bytes) <= self._SHM_OFFLOAD_THRESHOLD_BYTES:
            return payload

        shm_name = f"shm_msg_{uuid.uuid4().hex[:8]}"
        shm = None
        try:
            shm = SharedMemoryTransport(shm_name, size=len(payload_bytes) + 1024)
            await shm.create()
            await asyncio.to_thread(shm.write_serialized, serialized_payload)
            self._retain_outbound_shm(shm)
            logger.debug("🚀 [SHM] Offloaded payload: %s (%d bytes)", shm_name, len(payload_bytes))
            return {"__shm__": shm_name}
        except Exception as e:
            if shm is not None:
                try:
                    shm.close()
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)
            logger.warning("⚠️ SHM offload failed, falling back to Pipe: %s", e)
            return payload

    async def send(self, msg_type: str, payload: Any, trace_id: Optional[str] = None):
        """Send a fire-and-forget message."""
        trace_id = trace_id or str(uuid.uuid4())
        msg = {
            "type": msg_type,
            "payload": payload,
            "trace_id": trace_id
        }
        try:
            # Pre-flight check to avoid BrokenPipeError hangs
            if self.write_conn.closed or getattr(self, '_pipe_broken', False):
                return  # Already closed — silently skip, no spam
            
            # Fast-fail check if connection is completely broken at the OS level
            try:
                # We can't poll writing, but we can check if it's explicitly broken if there's a quick way
                pass
            except Exception as _e:
                logger.debug('Ignored Exception in local_pipe_bus.py: %s', _e)

            msg["payload"] = await self._prepare_payload_for_transport(payload)
            raw_msg = await asyncio.to_thread(json.dumps, msg)
            # ZENITH LOCKDOWN: Use isolated pipe executor and hard 10s timeout
            await asyncio.wait_for(
                self.loop.run_in_executor(self._get_executor(), self.write_conn.send, raw_msg),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("📡 Pipe write TIMEOUT (10s) — connection may be saturated.")
        except (BrokenPipeError, EOFError, OSError, ConnectionResetError) as e:
            if not getattr(self, '_pipe_broken', False):
                self._pipe_broken = True
                logger.info("📡 Bus pipe closed (normal shutdown): %s", str(e)[:60])
            try:
                self._safe_close_connection(self.write_conn)
            except Exception:
                pass  # Already closed, expected
        except Exception as e:
            if self._is_running:
                logger.error("❌ Unexpected error in bus send: %s", e)

    async def request(self, msg_type: str, payload: Any, timeout: float = 5.0) -> Any:
        """Send a request and wait for a response."""
        request_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        
        future = self.loop.create_future()
        self._pending_requests[request_id] = future
        
        msg = {
            "type": msg_type,
            "payload": payload,
            "request_id": request_id,
            "trace_id": trace_id,
            "is_request": True
        }
        
        try:
            # Hardened connection check
            if self.write_conn.closed:
                 raise BrokenPipeError("Connection is closed")

            msg["payload"] = await self._prepare_payload_for_transport(payload)
            raw_msg = await asyncio.to_thread(json.dumps, msg)
            logger.debug("📡 Sending request: %s (ID: %s)", msg_type, request_id)
            
            # ZENITH LOCKDOWN: Use isolated pipe executor and hard 10s timeout on write
            await asyncio.wait_for(
                self.loop.run_in_executor(self._get_executor(), self.write_conn.send, raw_msg),
                timeout=min(timeout, 10.0)
            )
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            logger.warning("⏳ Bus request timed out: %s", msg_type)
            raise
        except (BrokenPipeError, EOFError, OSError) as e:
            # [GENESIS FIX] Immediately resolve the specific future to avoid hanging the caller
            if request_id in self._pending_requests:
                future = self._pending_requests.pop(request_id)
                if not future.done():
                    future.cancel()
            
            if not getattr(self, '_pipe_broken', False):
                self._pipe_broken = True
                logger.warning("📡 Bus request failed (Broken Pipe): %s", e)
            
            try:
                self._safe_close_connection(self.write_conn)
            except Exception as _e:
                logger.debug("📡 LocalPipeBus: Secondary error during request-failure close: %s", _e)
            raise

    async def _read_loop(self):
        """Internal reader loop using unidirectional read_conn."""
        while self._is_running:
            try:
                # ZENITH LOCKDOWN: Use isolated pipe executor for blocking recv()
                msg = await self.loop.run_in_executor(self._get_executor(), self.read_conn.recv)
                
                # Always handle potential JSON strings from other processes
                if isinstance(msg, str):
                    try:
                        msg = json.loads(msg)
                    except json.JSONDecodeError:
                        logger.error("🛑 Failed to parse bus message: %s...", msg[:100])
                        continue

                if not msg or not isinstance(msg, dict):
                    continue

                if self._activity_callback:
                    try:
                        self._activity_callback()
                    except Exception as callback_err:
                        logger.debug("LocalPipeBus activity callback failed: %s", callback_err)
                
                # SHM De-referencing
                payload = msg.get("payload")
                if isinstance(payload, dict) and "__shm__" in payload:
                    shm_name = payload["__shm__"]
                    try:
                        shm = SharedMemoryTransport(shm_name)
                        await asyncio.wait_for(shm.attach(), timeout=2.0)
                        msg["payload"] = await shm.read()
                        # Detach but don't unlink yet (let owner clean up or use a policy)
                        # Actually, for a single read, we should detach.
                        shm.close()
                        logger.debug("📥 Resolved SHM payload: %s", shm_name)
                    except Exception as e:
                        logger.error("❌ Failed to resolve SHM payload %s: %s", shm_name, e)
                        if msg.get("is_request") and "request_id" in msg:
                            err_resp = {
                                "response_to": msg["request_id"],
                                "payload": {"ok": False, "error": "shm_resolution_failed"},
                                "trace_id": msg.get("trace_id"),
                            }
                            raw_resp = json.dumps(err_resp)
                            await self.loop.run_in_executor(self._get_executor(), self.write_conn.send, raw_resp)
                        continue

                # Check if it's a response to a pending request
                if "response_to" in msg:
                    req_id = msg["response_to"]
                    future = self._pending_requests.pop(req_id, None)
                    if future and not future.done():
                        future.set_result(msg["payload"])
                    continue

                # Normal message or request
                msg_type = msg.get("type")
                if msg_type in self._handlers:
                    handler = self._handlers[msg_type]
                    if self._dispatch_queue is None:
                        self._dispatch_queue = asyncio.Queue(maxsize=256)
                    try:
                        await asyncio.wait_for(
                            self._dispatch_queue.put((handler, msg)),
                            timeout=1.0,
                        )
                    except asyncio.TimeoutError:
                        logger.warning("📡 Bus dispatch queue saturated. Dropping %s.", msg_type)
                        if msg.get("is_request") and "request_id" in msg:
                            err_resp = {
                                "response_to": msg["request_id"],
                                "payload": {"ok": False, "error": "dispatch_queue_saturated"},
                                "trace_id": msg.get("trace_id"),
                            }
                            raw_resp = json.dumps(err_resp)
                            await self.loop.run_in_executor(self._get_executor(), self.write_conn.send, raw_resp)
                else:
                    logger.debug("❓ Unhandled bus message type: %s", msg_type)

            except EOFError:
                logger.info("🔌 Bus connection closed by peer.")
                self._cancel_pending_requests(EOFError("Bus connection closed by peer"))
                break
            except (BrokenPipeError, OSError) as e:
                logger.error("🛑 Bus read error: %s", e)
                self._cancel_pending_requests(e)
                break
            except Exception as e:
                logger.exception("❌ Error in Bus read loop: %s", e)
                await asyncio.sleep(0.1)

    async def _dispatch_loop(self):
        """Process inbound messages in arrival order with bounded backpressure."""
        while self._is_running:
            try:
                if self._dispatch_queue is None:
                    await asyncio.sleep(0.05)
                    continue
                handler, msg = await self._dispatch_queue.get()
                try:
                    await self._handle_message(handler, msg)
                finally:
                    self._dispatch_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("❌ Error in Bus dispatch loop: %s", e)
                await asyncio.sleep(0.1)

    def _cancel_pending_requests(self, exception: Optional[Exception] = None, cancel: bool = False):
        """[GENESIS FIX] Ensure all awaiting requests are rejected immediately if the pipe dies."""
        for req_id, future in list(self._pending_requests.items()):
            if future.done():
                continue
            if cancel or exception is None:
                future.cancel()
            else:
                future.set_exception(exception)
        self._pending_requests.clear()

    async def _handle_message(self, handler: Callable, msg: Dict):
        """Wrap handler execution and handle responses."""
        try:
            result = handler(msg.get("payload"), msg.get("trace_id"))
            if asyncio.iscoroutine(result):
                result = await result
            
            # If it was a request, send back the result via write_conn
            if msg.get("is_request") and "request_id" in msg:
                resp = {
                    "response_to": msg["request_id"],
                    "payload": result,
                    "trace_id": msg.get("trace_id")
                }
                # Consistently use JSON
                raw_resp = json.dumps(resp)
                # ZENITH LOCKDOWN: Dedicated executor
                await self.loop.run_in_executor(self._get_executor(), self.write_conn.send, raw_resp)
        except Exception as e:
            logger.error("❌ Bus handler error (%s): %s", msg.get("type"), e)
            if msg.get("is_request") and "request_id" in msg:
                err_resp = {
                    "response_to": msg["request_id"],
                    "payload": {"error": str(e)},
                    "trace_id": msg.get("trace_id"),
                    "failed": True
                }
                # Consistently use JSON
                raw_err = json.dumps(err_resp)
                # ZENITH LOCKDOWN: Dedicated executor
                await self.loop.run_in_executor(self._get_executor(), self.write_conn.send, raw_err)
