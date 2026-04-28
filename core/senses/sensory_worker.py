from core.runtime.errors import record_degradation
import multiprocessing as mp
import logging
import time
import os
import sys
import queue

# Configure logging for the worker
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("core.senses.sensory_worker")


def _screen_capture_preflight_allowed() -> bool:
    """Avoid triggering macOS prompts from the isolated worker."""
    try:
        import Quartz  # type: ignore

        preflight = getattr(Quartz, "CGPreflightScreenCaptureAccess", None)
        if callable(preflight):
            return bool(preflight())
    except Exception as exc:
        record_degradation('sensory_worker', exc)
        logger.debug("Sensory worker Quartz preflight unavailable: %s", exc)
    return os.getenv("AURA_ASSUME_SCREEN_PERMISSION", "0") == "1"

def sensory_worker_loop(request_queue, response_queue):
    """
    Isolated process for Vision and Audio capturing.
    Prevents cv2/sounddevice memory corruption from taking down the brain.
    """
    logger.info("👀 [SENSORY] Isolated Worker started (PID: %d)", os.getpid())
    
    cv2 = None
    mss = None
    sd = None
    
    # Issue 29: Use collections.deque for efficient message buffering
    from collections import deque
    message_buffer = deque(maxlen=100)
    
    while True:
        try:
            # Issue 30: Reduce polling latency from 1.0s to 0.1s
            req = request_queue.get(timeout=0.1)
            if not req: break
            
            cmd = req.get("command")
            
            if cmd == "init_vision":
                try:
                    if not _screen_capture_preflight_allowed():
                        response_queue.put({"status": "error", "msg": "screen_permission_inactive"})
                        continue
                    import cv2 as _cv2
                    import mss as _mss
                    cv2, mss = _cv2, _mss
                    response_queue.put({"status": "ok"})
                except Exception as e:
                    record_degradation('sensory_worker', e)
                    response_queue.put({"status": "error", "msg": str(e)})
                
            elif cmd == "capture_screen":
                if not mss:
                    response_queue.put({"status": "error", "msg": "Vision not init"})
                    continue
                if not _screen_capture_preflight_allowed():
                    response_queue.put({"status": "error", "msg": "screen_permission_inactive"})
                    continue
                with mss.mss() as sct:
                    monitor = sct.monitors[1]
                    sct_img = sct.grab(monitor)
                    # Convert to minimal bytes to send over queue
                    response_queue.put({"status": "ok", "data": bytes(sct_img.raw)})
                
            elif cmd == "init_audio":
                try:
                    import sounddevice as _sd
                    sd = _sd
                    response_queue.put({"status": "ok"})
                except Exception as e:
                    record_degradation('sensory_worker', e)
                    response_queue.put({"status": "error", "msg": str(e)})

            elif cmd == "ping":
                response_queue.put({"status": "ok", "msg": "pong"})
                    
            elif cmd == "exit":
                break
            else:
                response_queue.put({"status": "error", "msg": f"Unknown command: {cmd}"})
                
        except (queue.Empty, Exception, KeyboardInterrupt) as e:
            if isinstance(e, (queue.Empty, KeyboardInterrupt)):
                if isinstance(e, KeyboardInterrupt): break
                continue
            logger.error("🛑 Sensory Worker Error: %s", e)

if __name__ == "__main__":
    # This worker is intended to be started via mp.Process from the client
    pass
