import logging
import multiprocessing as mp
import os
import queue
import threading
import time

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
            except queue.Full as _exc:
                logger.debug("Suppressed %s in core.brain.llm.mlx_vision_worker: %s", type(_exc).__name__, _exc)
            time.sleep(2.0)

    def stop(self):
        self._stop_event.set()

def _mlx_vision_worker_loop(model_path: str, req_q: mp.Queue, res_q: mp.Queue):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - VisionWorker - %(levelname)s - %(message)s')
    
    heartbeat = HeartbeatThread(res_q)
    heartbeat.start()
    
    try:
        import mlx.core  # noqa: F401
        from mlx_vlm import generate, load
        from mlx_vlm.utils import load_config
        
        logger.info("Loading Vision Model: %s", model_path)
        model, processor = load(model_path)
        config = load_config(model_path)
        logger.info("Vision Model loaded.")
        
        res_q.put({"status": "ok", "action": "init"})
    except (ImportError, AttributeError, RuntimeError) as e:
        logger.error("Failed to load vision model: %s", e)
        res_q.put({"status": "error", "action": "init", "message": str(e)})
        return
        
    worker_active = True
    while worker_active:
        try:
            job = req_q.get()
            if job is None:
                worker_active = False
                continue
                
            action = job.get("action")
            if action == "see":
                prompt_text = job.get("prompt", "What is in this image?")
                image_base64 = job.get("image_base64")
                max_tokens = job.get("max_tokens", 512)
                temp = job.get("temp", 0.0)
                
                try:
                    import base64 as _b64
                    import io as _io

                    from mlx_vlm.prompt_utils import apply_chat_template
                    from PIL import Image as _Image

                    # `generate` takes paths or PIL images, never base64. The
                    # previous code handed it the base64 string as if it were
                    # a path, so every call failed — and failed *fatally*,
                    # because the exception it raised was not in the handler
                    # below and killed the worker rather than the request.
                    image = _Image.open(
                        _io.BytesIO(_b64.b64decode(image_base64))
                    ).convert("RGB")

                    # The message needs an explicit image part, and the
                    # template needs to know how many. Without both, the
                    # prompt carries no image token, so the model is asked
                    # to describe a picture it was never shown — and answers
                    # anyway, from the question alone.
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt_text},
                            ],
                        }
                    ]

                    try:
                        formatted_prompt = apply_chat_template(
                            processor, config, messages, num_images=1
                        )
                    except (RuntimeError, AttributeError, TypeError, ValueError):
                        formatted_prompt = prompt_text  # Fallback

                    # No temperature kwarg: this build routes sampling
                    # elsewhere and rejects it, which is the other way the
                    # worker used to die mid-request.
                    response = generate(
                        model, processor,
                        prompt=formatted_prompt,
                        image=[image],
                        verbose=False,
                        max_tokens=max_tokens,
                    )

                    if hasattr(response, 'text'):
                        text_output = response.text
                    else:
                        text_output = str(response)

                    res_q.put({"status": "ok", "action": "see", "id": job.get("id"), "response": text_output})

                except Exception as eval_e:  # noqa: BLE001 - see below
                    # Deliberately broad. This is a worker process whose only
                    # job is to answer requests: any exception that escapes
                    # here kills it, and one malformed image would then take
                    # sight down for the rest of the session. A failed
                    # request must cost the request.
                    import traceback
                    err = f"{eval_e}\n{traceback.format_exc()}"
                    logger.error("Vision eval error: %s", err)
                    res_q.put({"status": "error", "action": "see", "id": job.get("id"), "message": str(eval_e)})
                    
        except KeyboardInterrupt:
            break
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.error("Worker loop error: %s", e)
