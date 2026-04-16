import sys
import json
import logging
import traceback
import time
import os
import multiprocessing as mp
import threading
import copy
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
import numpy as np
import queue

from core.runtime.desktop_boot_safety import compute_mlx_cache_limit
from .model_registry import resolve_personality_adapter


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

    [STABILITY v51] Reduced interval from 5s → 2s for faster dead-worker
    detection.  Added parent-PID liveness check: if the parent process
    dies (crash, restart), the worker self-terminates to prevent orphans.
    """
    def __init__(self, writer: IPCWriterThread):
        super().__init__(name="MLX-Heartbeat", daemon=True)
        self.writer = writer
        self._stop_event = threading.Event()
        self._parent_pid = os.getppid()

    def stop(self):
        self._stop_event.set()

    def _parent_alive(self) -> bool:
        """Check if our parent process is still running."""
        try:
            os.kill(self._parent_pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def run(self):
        while not self._stop_event.is_set():
            # [STABILITY v51] Self-terminate if parent died — prevents orphan workers
            if not self._parent_alive():
                print(f"🛑 [MLX_HEARTBEAT] Parent process {self._parent_pid} is dead. Self-terminating orphaned worker.")
                os._exit(1)
            self.writer.put({"status": "heartbeat", "timestamp": time.time(), "type": "mlx_worker"})
            time.sleep(2.0)

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
    os.environ["METAL_COMPILER_TIMEOUT_MS"] = "60000"  # [FRONTIER UPGRADE] Extended for 32B model complex prompts
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


def _process_message_content(messages: list[dict[str, Any]]) -> None:
    """Normalize content for tokenizer.apply_chat_template()."""
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            text_fragments = [
                fragment.get("text", "")
                for fragment in content
                if isinstance(fragment, dict) and fragment.get("type") == "text"
            ]
            if len(text_fragments) != len(content):
                raise ValueError("Only text content fragments are supported in MLX worker chat templates.")
            message["content"] = "".join(text_fragments)
        elif content is None:
            message["content"] = ""


def _load_effective_context_window(model_path: str) -> int:
    path = Path(str(model_path))
    if not path.exists():
        return 32768

    config_path = path / "config.json"
    tokenizer_config_path = path / "tokenizer_config.json"

    max_position_embeddings = 0
    sliding_window = 0
    use_sliding_window = False
    tokenizer_model_max = 0

    try:
        if config_path.exists():
            config_payload = json.loads(config_path.read_text())
            max_position_embeddings = int(config_payload.get("max_position_embeddings") or 0)
            sliding_window = int(config_payload.get("sliding_window") or 0)
            use_sliding_window = bool(config_payload.get("use_sliding_window"))
    except Exception:
        max_position_embeddings = 0
        sliding_window = 0
        use_sliding_window = False

    try:
        if tokenizer_config_path.exists():
            tokenizer_payload = json.loads(tokenizer_config_path.read_text())
            tokenizer_model_max = int(tokenizer_payload.get("model_max_length") or 0)
    except Exception:
        tokenizer_model_max = 0

    if max_position_embeddings > 0:
        if use_sliding_window and sliding_window > max_position_embeddings:
            return max(sliding_window, max_position_embeddings)
        return max_position_embeddings
    if use_sliding_window and sliding_window > 0:
        return sliding_window
    if tokenizer_model_max > 0:
        return tokenizer_model_max
    return 32768


@dataclass
class _PromptCacheEntry:
    prompt_cache: list[Any]
    count: int


@dataclass
class _PromptCacheSearchResult:
    exact: list[int] | None
    shorter: list[int] | None
    longer: list[int] | None
    common_prefix: int


class _PromptCacheLRU:
    def __init__(self, max_size: int = 12):
        self.max_size = max_size
        self._cache: dict[int, dict[Any, Any]] = {}
        self._lru = deque()

    def clear(self) -> None:
        self._cache.clear()
        self._lru.clear()

    def _search(self, model_key: int, tokens: list[int]) -> _PromptCacheSearchResult:
        if model_key not in self._cache:
            return _PromptCacheSearchResult(None, None, None, 0)

        current = self._cache[model_key]
        last_cache_index = -1
        index = 0

        while index < len(tokens) and tokens[index] in current:
            current = current[tokens[index]]
            if "cache" in current:
                last_cache_index = index
            index += 1

        if last_cache_index == len(tokens) - 1:
            return _PromptCacheSearchResult(tokens, None, None, 0)

        shorter = tokens[: last_cache_index + 1] if last_cache_index > 0 else None
        longer = None
        common_prefix = index
        if index > 0 and last_cache_index <= 0:
            best = None
            stack = [(current, [])]
            while stack:
                node, extra = stack.pop()
                if "cache" in node:
                    if best is None or len(extra) < len(best):
                        best = extra
                else:
                    for tok in node:
                        stack.append((node[tok], extra + [tok]))
            if best is not None:
                longer = tokens[:index] + best

        return _PromptCacheSearchResult(None, shorter, longer, common_prefix)

    def _get(self, model_key: int, tokens: list[int]) -> _PromptCacheEntry:
        current = self._cache[model_key]
        for tok in tokens:
            current = current[tok]
        return current["cache"]

    def _delete(self, model_key: int, tokens: list[int]) -> None:
        path = [self._cache[model_key]]
        for tok in tokens:
            path.append(path[-1][tok])
        del path[-1]["cache"]
        for index in reversed(range(len(tokens))):
            prev_node, node, tok = path[index], path[index + 1], tokens[index]
            if len(node) > 0:
                break
            del prev_node[tok]

    def _extract(self, model_key: int, tokens: list[int]) -> _PromptCacheEntry:
        cache_entry = self._get(model_key, tokens)
        if cache_entry.count == 1:
            self._delete(model_key, tokens)
            try:
                self._lru.remove((model_key, tuple(tokens)))
            except ValueError:
                pass
            return cache_entry

        cache_entry.count -= 1
        return _PromptCacheEntry(copy.deepcopy(cache_entry.prompt_cache), 1)

    def fetch_nearest_cache(
        self,
        model_key: int,
        tokens: list[int],
        *,
        can_trim_prompt_cache: Any,
        trim_prompt_cache: Any,
    ) -> tuple[list[Any] | None, list[int]]:
        result = self._search(model_key, tokens)
        if result.exact is not None:
            cache_entry = self._extract(model_key, result.exact)
            return cache_entry.prompt_cache, []

        if result.shorter is not None:
            cache_entry = self._extract(model_key, result.shorter)
            prefix_len = len(result.shorter)
            return cache_entry.prompt_cache, tokens[prefix_len:]

        if result.longer is not None:
            cache_entry = self._get(model_key, result.longer)
            if can_trim_prompt_cache(cache_entry.prompt_cache):
                trimmed = _PromptCacheEntry(copy.deepcopy(cache_entry.prompt_cache), 1)
                prefix = min(len(tokens) - 1, result.common_prefix)
                num_to_trim = len(result.longer) - prefix
                trim_prompt_cache(trimmed.prompt_cache, num_to_trim)
                return trimmed.prompt_cache, tokens[prefix:]

        return None, tokens

    def insert_cache(self, model_key: int, tokens: list[int], prompt_cache: list[Any]) -> None:
        if model_key not in self._cache:
            self._cache[model_key] = {}
        current = self._cache[model_key]
        for tok in tokens:
            if tok not in current:
                current[tok] = {}
            current = current[tok]

        cache_key = (model_key, tuple(tokens))
        if "cache" in current:
            current["cache"].count += 1
            try:
                self._lru.remove(cache_key)
            except ValueError:
                pass
        else:
            current["cache"] = _PromptCacheEntry(prompt_cache, 1)

        self._lru.append(cache_key)
        if len(self._lru) > self.max_size:
            evict_model_key, evict_tokens = self._lru.popleft()
            self._delete(evict_model_key, list(evict_tokens))

class JobWatchdog(threading.Thread):
    """
    Kills the worker process if a job is active but no tokens have been generated 
    within the timeout. This prevents 'Metal Stalls' from hanging the system.

    [STABILITY v51] Reduced timeout from 240s → 90s. The 32B model's Metal
    shader compilation should complete within 60s on M5 hardware. If no token
    progress after 90s, the worker is stuck and must self-terminate so the
    parent can respawn it.
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

    watchdog = JobWatchdog(timeout=90.0)  # [STABILITY v51] Reduced from 240s. M5 Metal compilation should finish within 60s.
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

    # [STABILITY v53.7] LoRA adapter loading DISABLED.
    # The separate adapter causes intermittent float32 type errors on 8-bit
    # quantized models ("Function arguments must be trees of arrays or constants,
    # but received type float32"). Personality is enforced via 4-layer prompt
    # hardening (ChatML guard, message anchor, compact persona, post-gen filter).
    # The adapter weights are preserved for future use when MLX fixes the
    # quantization compatibility issue.
    try:
        logger.info(f"Loading model: {model_path}")
        model, tokenizer = load(model_path)
        logger.info("Model loaded (personality via prompt hardening, not LoRA).")

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
    # ZENITH: Prompt Cache LRU for massive speedup in multi-turn
    prompt_cache_lru = _PromptCacheLRU(max_size=12)

    while True:
        try:
            job = request_queue.get()
            if job is None: break
                
            action = job.get("action")
            if action == "generate":
                prompt = job.get("prompt")
                messages = job.get("messages")
                tools = job.get("tools")
                
                # [FRONTIER UPGRADE] Native Tool Templates
                if messages and hasattr(tokenizer, "apply_chat_template"):
                    try:
                        logger.info("🎯 [WORKER] Rendering native chat/tool template.")
                        prompt = tokenizer.apply_chat_template(
                            messages, 
                            tools=tools, 
                            add_generation_prompt=True, 
                            tokenize=False
                        )
                    except Exception as e:
                        logger.warning(f"❌ [WORKER] Native template compilation failed: {e}")

                temp = job.get("temp", 0.7)
                top_p = job.get("top_p", 0.9)
                max_tokens = job.get("max_tokens", 512)
                schema = job.get("schema")
                
                # [v11.0 HARDENING] Structured Generation Overrides
                if schema:
                    temp = 0.0 # Force determinism for JSON
                    logger.info("🎯 [WORKER] Structured mode: temp=0.0 enforced.")

                # Intelligence boosters: min_p sampling improves quality on smaller
                # models by filtering out low-probability tokens before top_p.
                # Repetition penalty reduces the stale/looping response pattern.
                min_p = job.get("min_p", 0.05)
                repetition_penalty = job.get("repetition_penalty", 1.1)
                kwargs = {"max_tokens": max_tokens}
                if make_sampler:
                    sampler_kwargs = {"temp": temp, "top_p": top_p}
                    try:
                        import inspect as _insp
                        _sparams = _insp.signature(make_sampler).parameters
                        if "min_p" in _sparams:
                            sampler_kwargs["min_p"] = min_p
                        if "repetition_penalty" in _sparams:
                            sampler_kwargs["repetition_penalty"] = repetition_penalty
                    except Exception:
                        pass
                    kwargs["sampler"] = make_sampler(**sampler_kwargs)

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
                                
                                # [FRONTIER UPGRADE] KV Prompt Caching Injection
                                tokens = tokenizer.encode(prompt)
                                import mlx_lm.utils as u
                                
                                def _can_trim(pc): return hasattr(u, "trim_prompt_cache")
                                def _do_trim(pc, num): 
                                    if hasattr(u, "trim_prompt_cache"): u.trim_prompt_cache(pc, num)
                                
                                model_key = id(model)
                                cache, remaining_tokens = prompt_cache_lru.fetch_nearest_cache(
                                    model_key, tokens, 
                                    can_trim_prompt_cache=_can_trim, 
                                    trim_prompt_cache=_do_trim
                                )
                                
                                gen_prompt = remaining_tokens if cache is not None else prompt
                                if cache is not None:
                                    kwargs["prompt_cache"] = cache

                                # Execute 
                                for response in stream_generate(model, tokenizer, prompt=gen_prompt, **kwargs):
                                    watchdog.activity()
                                    token_count += 1
                                    
                                    # Snag the prompt cache from the response if supported to save for next turn
                                    if hasattr(response, "prompt_cache") and response.prompt_cache is not None:
                                        prompt_cache_lru.insert_cache(model_key, tokens + [response.token], response.prompt_cache)
                                    
                                    current_response += response.text
                                    current_response = _strip_leading_chatml_prefix(current_response)

                                    if token_count == 1 or token_count % 16 == 0:
                                        ipc_writer.put({
                                            "id": job.get("id"),
                                            "action": "generate",
                                            "status": "progress",
                                            "tokens_generated": token_count,
                                            "timestamp": time.time(),
                                        })
                                    
                                    # [STABILITY v52] Explicit manual cache clearing to prevent MLX memory 
                                    # fragmentation mapping failures on long context 32B inferences.
                                    if mx and device != "cpu" and token_count % 32 == 0:
                                        _clear_mlx_cache(mx)
                                    
                                    # [FRONTIER UPGRADE] Absolute safety cap expanded so it never stops midway
                                    if token_count > 8192:
                                        logger.warning("🏁 [WORKER] Hard token limit (8192) reached. Truncating.")
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
                finally:
                    # [STABILITY v52] Guarantee VRAM gets purged after standard generation 
                    # completes or fails, ensuring pure state for next request.
                    if mx and device != "cpu":
                        _clear_mlx_cache(mx)
            
            elif action == "stream":
                prompt = job.get("prompt")
                temp = job.get("temp", 0.7)
                top_p = job.get("top_p", 0.9)
                max_tokens = job.get("max_tokens", 512)
                min_p = job.get("min_p", 0.05)
                repetition_penalty = job.get("repetition_penalty", 1.1)

                kwargs = {"max_tokens": max_tokens}
                if make_sampler:
                    sampler_kwargs = {"temp": temp, "top_p": top_p}
                    try:
                        import inspect as _insp2
                        _sparams2 = _insp2.signature(make_sampler).parameters
                        if "min_p" in _sparams2:
                            sampler_kwargs["min_p"] = min_p
                        if "repetition_penalty" in _sparams2:
                            sampler_kwargs["repetition_penalty"] = repetition_penalty
                    except Exception:
                        pass
                    kwargs["sampler"] = make_sampler(**sampler_kwargs)

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
                                ipc_writer.put(
                                    {
                                        "id": job.get("id"),
                                        "action": "stream",
                                        "status": "token",
                                        "text": token_text,
                                        "tokens_generated": token_count,
                                        "timestamp": time.time(),
                                    }
                                )
                                
                                # [AURA HARDENING] Prevent VRAM fragmentation
                                if token_count % 10 == 0:
                                    _clear_mlx_cache(mx)
                                
                                # [FRONTIER UPGRADE] Absolute safety cap natively expanded to frontier levels
                                if token_count > 8192:
                                    logger.warning("🏁 [WORKER] Hard token limit (8192) reached. Truncating.")
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
                except Exception as e:
                    logger.error(f"Streaming failed: {e}")
                    ipc_writer.put({"status": "error", "action": "stream", "message": str(e)})
                finally:
                    # [STABILITY v52] Guarantee VRAM gets purged after streaming
                    # completes or fails, ensuring pure state for next request.
                    if mx and device != "cpu":
                        _clear_mlx_cache(mx)

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
