import multiprocessing as mp
import threading
import queue
import time
import uuid
import logging
from typing import Optional

from .mlx_vision_worker import _mlx_vision_worker_loop

logger = logging.getLogger("MLXVisionClient")

class MLXVisionClient:
    """
    Manages an isolated MLX vision model worker for multimodal inference.
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._process: Optional[mp.Process] = None
        self._req_q = mp.Queue(maxsize=10)
        self._res_q = mp.Queue(maxsize=10)
        self._lock = threading.Lock()
        self._pending_requests = {}
        self._listener_thread = None
        self._stop_event = threading.Event()
        self._init_done = False
        
    def start(self) -> None:
        with self._lock:
            if self._process is not None and self._process.is_alive():
                return
                
            logger.info(f"Starting MLX Vision Worker for {self.model_path}")
            # Ensure spawn method for MLX Metal compatibility
            ctx = mp.get_context("spawn") if hasattr(mp, "get_context") else mp
            
            self._process = ctx.Process(
                target=_mlx_vision_worker_loop,
                args=(self.model_path, self._req_q, self._res_q),
                daemon=True,
                name="MLX-Vision-Worker"
            )
            self._process.start()
            
            self._listener_thread = threading.Thread(target=self._listener_loop, daemon=True)
            self._listener_thread.start()
            
            # Wait for init
            start_time = time.time()
            while time.time() - start_time < 30.0:
                if self._init_done:
                    break
                time.sleep(0.1)
                
            if not self._init_done:
                logger.error("Vision worker failed to initialize within 30s")

    def _listener_loop(self):
        while not self._stop_event.is_set():
            try:
                msg = self._res_q.get(timeout=1.0)
                status = msg.get("status")
                action = msg.get("action")
                
                if status == "heartbeat":
                    continue
                    
                if action == "init":
                    self._init_done = True
                    continue
                    
                req_id = msg.get("id")
                if req_id and req_id in self._pending_requests:
                    self._pending_requests[req_id] = msg
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Vision listener error: {e}")

    def see(self, prompt: str, image_base64: str, max_tokens: int = 512, temp: float = 0.0) -> str:
        """
        Send a base64 image (with optional data:image prefix) and prompt to the vision model.
        Blocks until completion.
        """
        self.start()
        
        req_id = str(uuid.uuid4())
        self._pending_requests[req_id] = None
        
        self._req_q.put({
            "id": req_id,
            "action": "see",
            "prompt": prompt,
            "image_base64": image_base64,
            "max_tokens": max_tokens,
            "temp": temp
        })
        
        # Wait for response
        while self._pending_requests[req_id] is None:
            time.sleep(0.1)
            if self._process and not self._process.is_alive():
                raise RuntimeError("Vision worker crashed during inference")
                
        resp = self._pending_requests.pop(req_id)
        if resp.get("status") == "error":
            raise RuntimeError(f"Vision model error: {resp.get('message')}")
            
        return resp.get("response", "")

    def stop(self):
        self._stop_event.set()
        if self._req_q:
            try:
                self._req_q.put(None)
            except Exception:
                pass
        if self._process:
            self._process.join(timeout=3.0)
            if self._process.is_alive():
                self._process.terminate()
        logger.info("Vision worker stopped.")
