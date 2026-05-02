import multiprocessing as mp
import time
import os
import sys
import logging
import queue
import threading
from typing import Any

logger = logging.getLogger("MLXVisionWorker")

def _setup_worker_env():
    # Similar environment setup to mlx_worker.py for MLX
    os.environ["MLX_NUM_THREADS"] = "10"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["MLX_FORCE_SERIAL_COMPILE"] = "1"

_setup_worker_env()

class HeartbeatThread(threading.Thread):
    def __init__(self, res_q: mp.Queue):
        super().__init__(daemon=True)
        self.res_q = res_q
        self._stop_event = threading.Event()
        self._parent_pid = os.getppid()

    def run(self):
        while not self._stop_event.is_set():
            try:
                os.kill(self._parent_pid, 0)
            except OSError:
                os._exit(1)
            try:
                self.res_q.put({"status": "heartbeat", "timestamp": time.time(), "type": "mlx_vision_worker"}, block=False)
            except queue.Full:
                pass
            time.sleep(2.0)

    def stop(self):
        self._stop_event.set()

def _mlx_vision_worker_loop(model_path: str, req_q: mp.Queue, res_q: mp.Queue):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - VisionWorker - %(levelname)s - %(message)s')
    
    heartbeat = HeartbeatThread(res_q)
    heartbeat.start()
    
    try:
        import mlx.core as mx
        from mlx_vlm import load, generate
        from mlx_vlm.utils import load_config
        
        logger.info(f"Loading Vision Model: {model_path}")
        model, processor = load(model_path)
        config = load_config(model_path)
        logger.info("Vision Model loaded.")
        
        res_q.put({"status": "ok", "action": "init"})
    except Exception as e:
        logger.error(f"Failed to load vision model: {e}")
        res_q.put({"status": "error", "action": "init", "message": str(e)})
        return
        
    while True:
        try:
            job = req_q.get()
            if job is None:
                break
                
            action = job.get("action")
            if action == "see":
                prompt_text = job.get("prompt", "What is in this image?")
                image_base64 = job.get("image_base64")
                max_tokens = job.get("max_tokens", 512)
                temp = job.get("temp", 0.0)
                
                try:
                    # In mlx_vlm, we can use get_message_json from utils
                    from mlx_vlm.utils import get_message_json
                    from mlx_vlm.prompt_utils import apply_chat_template
                    
                    # mlx_vlm accepts plain text prompts and handles image insertion.
                    # Or we can format standard messages:
                    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
                    
                    try:
                        formatted_prompt = apply_chat_template(processor, config, messages)
                    except Exception:
                        formatted_prompt = prompt_text # Fallback
                        
                    response = generate(
                        model, processor,
                        prompt=formatted_prompt,
                        image=[image_base64],
                        verbose=False,
                        max_tokens=max_tokens,
                        temperature=temp
                    )
                    
                    if hasattr(response, 'text'):
                        text_output = response.text
                    else:
                        text_output = str(response)
                        
                    res_q.put({"status": "ok", "action": "see", "id": job.get("id"), "response": text_output})
                    
                except Exception as eval_e:
                    import traceback
                    err = f"{eval_e}\n{traceback.format_exc()}"
                    logger.error(f"Vision eval error: {err}")
                    res_q.put({"status": "error", "action": "see", "id": job.get("id"), "message": str(eval_e)})
                    
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Worker loop error: {e}")
