import sys
import json
import logging
import traceback
import time
import os
import multiprocessing as mp
import threading
from typing import Optional, Any
import numpy as np
import queue

from core.runtime.desktop_boot_safety import compute_mlx_cache_limit


def _strip_leading_chatml_prefix(text: str) -> str:
    cleaned = str(text or "")
    prefixes = (
        "<|im_start|>assistant\n",
        "<|im_start|>assistant",
        "<｜Assistant｜>",
        "Assistant:",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].lstrip("\n")
                changed = True
    return cleaned

class IPCWriterThread(threading.Thread):
    """
    ZENITH LOCKDOWN: Non-blocking IPC writer.
    Buffers responses in a local queue and writes to the multiprocessing pipe 
    in a dedicated thread to prevent blocking the main inference loop.
    """
    def __init__(self, mp_queue: mp.Queue):
        super().__init__(name="MLX-IPC-Writer", daemon=True)
        self.mp_queue = mp_queue
        self.local_queue = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()

    @staticmethod
    def _is_essential(item: Any) -> bool:
        if not isinstance(item, dict):
            return True
        status = item.get("status")
        return status not in {"heartbeat", "token"}

    def put(self, item):
        essential = self._is_essential(item)
        try:
            self.local_queue.put(item, block=False)
        except queue.Full:
            if essential:
                try:
                    # Never silently drop init/generation/error messages; bypass
                    # the local buffer when it is saturated with telemetry.
                    self.mp_queue.put(item, block=True, timeout=5.0)
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)
            # Drop non-essential telemetry if buffer is full.

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                item = self.local_queue.get(timeout=1.0)
                self.mp_queue.put(item)
            except queue.Empty:
                continue
            except Exception:
                break

class HeartbeatThread(threading.Thread):
    """
    ZENITH LOCKDOWN: Proactive Worker Heartbeat.
    Ensures the SupervisionTree sees this process as alive even during 
    massive 32B model loads or compilation stalls.
    """
    def __init__(self, writer: IPCWriterThread):
        super().__init__(name="MLX-Heartbeat", daemon=True)
        self.writer = writer
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            self.writer.put({"status": "heartbeat", "timestamp": time.time(), "type": "mlx_worker"})
            time.sleep(5.0)

# Set environment variables for MLX stability
def _setup_worker_env():
    import subprocess
    import platform
    import os
    
    # [PERFORMANCE] Fast-path: Use environment if already probed by parent
    cached_sdk = os.environ.get("AURA_SDK_PATH")
    if cached_sdk and os.path.exists(cached_sdk):
        os.environ["SDKROOT"] = cached_sdk
        print(f"Using cached SDK root: {cached_sdk}")
    else:
        try:
            sdk_path = subprocess.check_output(["xcrun", "--show-sdk-path"], timeout=2.0).decode().strip()
            import pathlib
            allowed_prefixes = ("/Library/", "/Applications/Xcode", "/usr/")
            if not any(sdk_path.startswith(pfx) for pfx in allowed_prefixes):
                raise RuntimeError(f"Suspicious SDK path rejected: {sdk_path}")
            os.environ["SDKROOT"] = sdk_path
            os.environ["AURA_SDK_PATH"] = sdk_path # Cache for subsequent spawns
        except Exception as e:
            print(f"⚠️ [MLX_WORKER_ENV] Failed to probe environment: {e}")
            return # Exit early if SDK probe fails critically

    try:
        ver_info = platform.mac_ver()
        release_str = ver_info[0]
        ver_parts = release_str.split(".")
        mac_ver = ".".join(ver_parts[:2])
        os.environ["MACOSX_DEPLOYMENT_TARGET"] = mac_ver
        
        sdk_path = os.environ.get("SDKROOT", "")
        sdk_inc = os.path.join(sdk_path, "usr", "include")
        cpp_inc = "/Library/Developer/CommandLineTools/usr/include/c++/v1"
        cpath_parts = []
        if sdk_path and os.path.exists(sdk_inc): cpath_parts.append(sdk_inc)
        if os.path.exists(cpp_inc): cpath_parts.append(cpp_inc)
        if cpath_parts:
            os.environ["CPATH"] = ":".join(cpath_parts + [os.environ.get("CPATH", "")]).strip(":")
    except Exception as e:
        print(f"⚠️ [MLX_WORKER_ENV] Failed to probe Mac version/CPATH: {e}")

    os.environ["MLX_NUM_THREADS"] = "10"   # M-series has 10+ perf cores
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["MLX_FORCE_SERIAL_COMPILE"] = "1"
    os.environ["METAL_COMPILER_TIMEOUT_MS"] = "30000"  # 32B model needs more shader compile time
    os.environ["METAL_DEVICE_WRAPPER_TYPE"] = "0"

_setup_worker_env()


def _clear_mlx_cache(mx_module: Any) -> None:
    try:
        mx_module.clear_cache()
    except Exception:
        try:
            mx_module.metal.clear_cache()
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

class JobWatchdog(threading.Thread):
    """
    Kills the worker process if a job is active but no tokens have been generated 
    within the timeout. This prevents 'Metal Stalls' from hanging the system.
    """
    def __init__(self, timeout=60.0):
        super().__init__(daemon=True)
        self.timeout = timeout
        self.last_activity = time.monotonic()
        self.active_job = False
        self._stop_event = threading.Event()

    def activity(self):
        self.last_activity = time.monotonic()

    def start_job(self):
        self.active_job = True
        self.last_activity = time.monotonic()

    def stop_job(self):
        self.active_job = False

    def run(self):
        while not self._stop_event.is_set():
            if self.active_job and (time.monotonic() - self.last_activity > self.timeout):
                print(f"🛑 [MLX_WATCHDOG] Job timeout triggered ({self.timeout}s). Self-terminating worker.")
                os._exit(1)
            time.sleep(1.0)

def _mlx_worker_loop(
    model_path: str, 
    request_queue: mp.Queue, 
    response_queue: mp.Queue, 
    device: str = "gpu",
    substrate_mem: Any = None
):
    """
    Runs in a FULLY ISOLATED native subprocess via ForkServer.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - MLXWorker - %(levelname)s - %(message)s',
        stream=sys.stderr
    )
    logger = logging.getLogger("MLXWorker")
    
    # ── Zenith Concurrency & Telemetry ──
    ipc_writer = IPCWriterThread(response_queue)
    ipc_writer.start()
    
    heartbeat = HeartbeatThread(ipc_writer)
    heartbeat.start()

    watchdog = JobWatchdog(timeout=90.0)  # 32B model needs more time on complex prompts
    watchdog.start()

    try:
        import mlx.core as mx
        from mlx_lm import load, generate
        try:
            from mlx_lm.sample_utils import make_sampler
        except ImportError:
            try: from mlx_lm.sample import make_sampler
            except ImportError: make_sampler = None
        
        logger.info("📡 [WORKER] Loading Core modules...")
        try:
            from .provider import LLMProvider
            from core.schemas import ChatStreamEvent
        except ImportError:
            logger.debug("[WORKER] Optional core module imports failed (non-fatal in spawn context)")
        
    except ImportError:
        logger.error("mlx-lm not installed in worker environment.")
        ipc_writer.put({"status": "error", "message": "mlx-lm missing"})
        return

    # VRAM Management
    if mx and device != "cpu":
        try:
            import psutil

            total_ram = psutil.virtual_memory().total
            limit = compute_mlx_cache_limit(total_ram)
            mx.set_cache_limit(limit)
            logger.info(f"Metal cache limit set to {limit // (1024**2)}MB")
        except Exception as e:
            try: mx.metal.set_cache_limit(1024 * 1024 * 1024 * 24)
            except Exception: pass

    # [PERFORMANCE] Metal probes shifted to after model load or triggered on demand
    # Initializing the model first is more critical for 'perceived' speed.

    # ZENITH: Local Concurrency Gate
    metal_semaphore = threading.Semaphore(1)

    # Load model with personality LoRA adapter if available
    try:
        adapter_path = os.environ.get("AURA_LORA_PATH")
        if not adapter_path:
            # Check default location
            _default_adapter = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "training", "adapters", "aura-personality", "adapters.safetensors"
            )
            if os.path.exists(_default_adapter):
                adapter_path = os.path.dirname(_default_adapter)
                logger.info(f"Found personality LoRA adapter: {adapter_path}")

        logger.info(f"Loading model: {model_path}")
        if adapter_path and os.path.isdir(adapter_path):
            logger.info(f"Loading with LoRA adapter: {adapter_path}")
            model, tokenizer = load(model_path, adapter_path=adapter_path)
            logger.info("Model loaded with Aura personality LoRA fused.")
        else:
            model, tokenizer = load(model_path)
            logger.info(f"Model loaded (no LoRA adapter).")

        # Attach Affective Steering
        try:
            from core.consciousness.affective_steering import get_steering_engine
            engine = get_steering_engine()
            engine.attach(model, tokenizer)
            if substrate_mem:
                engine.start_substrate_sync(shared_state=substrate_mem)
            logger.info("🎯 Affective Steering Engine ONLINE.")
        except Exception as se:
            logger.warning(f"Failed to attach steering: {se}")

        ipc_writer.put({"status": "ok", "action": "init", "device": device})
    except Exception as e:
        import traceback
        err_detail = f"{e}\n{traceback.format_exc()}"
        logger.error(f"Worker Init Error: {err_detail}")
        ipc_writer.put({"status": "error", "message": f"Init failed: {e}", "detail": err_detail})
        return

    while True:
        try:
            job = request_queue.get()
            if job is None: break
                
            action = job.get("action")
            if action == "generate":
                prompt = job.get("prompt")
                temp = job.get("temp", 0.7)
                top_p = job.get("top_p", 0.9)
                max_tokens = job.get("max_tokens", 512)
                schema = job.get("schema")
                
                # [v11.0 HARDENING] Structured Generation Overrides
                if schema:
                    temp = 0.0 # Force determinism for JSON
                    logger.info("🎯 [WORKER] Structured mode: temp=0.0 enforced.")
                
                kwargs = {"max_tokens": max_tokens}
                if make_sampler:
                    kwargs["sampler"] = make_sampler(temp=temp, top_p=top_p)
                
                # [v11.0 HARDENING] Logits Processors (JSON Enforcement)
                logits_processors = []
                if schema:
                    try:
                        brace_id = tokenizer.encode("{", add_special_tokens=False)[0]
                        def json_start_processor(tokens, logits):
                            if len(tokens) == 0:
                                # Force first token to be '{'
                                mask = mx.full_like(logits, -float("inf"))
                                mask[:, brace_id] = 0.0
                                return mask
                            return logits
                        logits_processors.append(json_start_processor)
                        logger.info("🎯 [WORKER] JSON start enforcement ACTIVE.")
                    except Exception as e:
                        logger.warning(f"Failed to setup JSON logits processor: {e}")
                
                if logits_processors:
                    kwargs["logits_processors"] = logits_processors
                
                stop_sequences = ["<|im_end|>", "<|im_start|>user", "<|im_start|>system", "<|im_start|>assistant", "User:", "Assistant:"]
                
                try:
                    from mlx_lm.generate import stream_generate
                    # : NO GPUSentinel here.
                    # GPUSentinel is a parent-process threading lock. In this isolated
                    # 'spawn' subprocess, it creates a SECOND serialization bottleneck
                    # on top of metal_semaphore, causing 30s GPU_TIMEOUT hangs.
                    # metal_semaphore(1) already serializes all GPU access in this worker.
                    
                    response_text = ""
                    total_generated_tokens = 0
                    
                    with metal_semaphore:
                        # Proactive cache clearing under memory pressure
                        if mx and device != "cpu":
                            try:
                                import psutil
                                if psutil.virtual_memory().percent > 90:  # 64GB — don't panic at 85%
                                    logger.warning("⚠️ High memory pressure detected in worker. Clearing MLX cache.")
                                    mx.clear_cache()
                            except Exception: pass

                        # [v11.5 HARDENING] Internal Worker Retries for Structured Leaks
                        # If a schema is present and we get an empty response, retry up to 2 times.
                        max_internal_retries = 2 if schema else 0
                        
                        for internal_attempt in range(max_internal_retries + 1):
                            watchdog.start_job()
                            try:
                                current_response = ""
                                token_count = 0
                                for response in stream_generate(model, tokenizer, prompt=prompt, **kwargs):
                                    watchdog.activity()
                                    token_count += 1
                                    current_response += response.text
                                    current_response = _strip_leading_chatml_prefix(current_response)
                                    
                                    # [AURA HARDENING] Prevent VRAM fragmentation
                                    if token_count % 10 == 0:
                                        _clear_mlx_cache(mx)
                                    
                                    # [AURA HARDENING] Absolute safety cap for local models
                                    if token_count > 2048:
                                        logger.warning("🏁 [WORKER] Hard token limit (2048) reached. Truncating.")
                                        break

                                    stop_hit = False
                                    for stop in stop_sequences:
                                        stop_index = current_response.find(stop)
                                        if stop_index > 0:
                                            current_response = current_response[:stop_index]
                                            stop_hit = True
                                            break
                                    if stop_hit:
                                        break
                                
                                response_text = current_response
                                total_generated_tokens = token_count
                                if response_text.strip() or not schema:
                                    break # Success or not a structured task
                                
                                logger.warning(f"⚠️ [WORKER] Empty structured response on attempt {internal_attempt + 1}. Retrying...")
                            finally:
                                watchdog.stop_job()
                            
                    if not response_text.strip():
                        logger.warning("⚠️ [WORKER] Generation yielded ZERO tokens. Prompt length: %d", len(prompt))
                        if len(prompt) > 2000:
                            logger.debug("Prompt snippet: %s...", prompt[:100])
                    
                    if mx and device != "cpu":
                        _clear_mlx_cache(mx)
                    
                    # : Tag with action: "generate" so client can distinguish
                    # from init/heartbeat responses unambiguously.
                    ipc_writer.put({
                        "id": job.get("id"),
                        "action": "generate",
                        "status": "ok", 
                        "text": response_text.strip(),
                        "tokens_used": total_generated_tokens
                    })
                except Exception as e:
                    logger.error(f"Generation failed: {e}")
                    ipc_writer.put({"status": "error", "action": "generate", "message": str(e)})
            
            elif action == "stream":
                prompt = job.get("prompt")
                temp = job.get("temp", 0.7)
                top_p = job.get("top_p", 0.9)
                max_tokens = job.get("max_tokens", 512)
                
                kwargs = {"max_tokens": max_tokens}
                if make_sampler:
                    kwargs["sampler"] = make_sampler(temp=temp, top_p=top_p)
                
                stop_sequences = ["<|im_end|>", "<|im_start|>user", "<|im_start|>system", "<|im_start|>assistant", "User:", "Assistant:"]
                
                try:
                    from mlx_lm.generate import stream_generate
                    # : NO GPUSentinel — same rationale as generate path.
                    
                    with metal_semaphore:
                        watchdog.start_job()
                        try:
                            full_text = ""
                            token_count = 0
                            for response in stream_generate(model, tokenizer, prompt=prompt, **kwargs):
                                watchdog.activity()
                                token_count += 1
                                token_text = response.text
                                full_text += token_text
                                full_text = _strip_leading_chatml_prefix(full_text)
                                ipc_writer.put({"status": "token", "text": token_text})
                                
                                # [AURA HARDENING] Prevent VRAM fragmentation
                                if token_count % 10 == 0:
                                    _clear_mlx_cache(mx)
                                
                                # [AURA HARDENING] Absolute safety cap for local models
                                if token_count > 1024:
                                    logger.warning("🏁 [WORKER] Hard token limit (1024) reached. Truncating.")
                                    break

                                stop_hit = False
                                for stop in stop_sequences:
                                    stop_index = full_text.find(stop)
                                    if stop_index > 0:
                                        full_text = full_text[:stop_index]
                                        stop_hit = True
                                        break
                                if stop_hit:
                                    break
                        finally:
                            watchdog.stop_job()
                    
                    ipc_writer.put({"status": "ok", "action": "stream_done"})
                    if mx and device != "cpu":
                        _clear_mlx_cache(mx)
                except Exception as e:
                    logger.error(f"Streaming failed: {e}")
                    ipc_writer.put({"status": "error", "action": "stream", "message": str(e)})

            elif action == "ping":
                if mx and device != "cpu":
                    _clear_mlx_cache(mx)
                ipc_writer.put({"status": "pong"})
                
            elif action == "clear_cache":
                if mx and device != "cpu":
                    _clear_mlx_cache(mx)
                ipc_writer.put({"status": "ok"})
                
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error("❌ [WORKER] Fatal error during initialization: %s\n%s", e, tb)
            ipc_writer.put({"status": "error", "message": "Init failed", "detail": tb})

if __name__ == "__main__":
    print("MLX Worker: Running in multiprocessing mode. Use mlx_client.py to launch.")
