"""core/brain/unified_inference.py
===============================
Unified Inference Engine for Aura.
Binds local GGUF inference (via llama-cpp-python) and Ollama fallback
with HomeostaticModulator parameters, logit bias injection, and closes
the loop via InferenceFeedbackLoop.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from core.brain.homeostatic_modulator import HomeostaticModulator, InferenceModulation
from core.brain.inference_feedback import InferenceFeedbackLoop

logger = logging.getLogger("Aura.Brain.UnifiedInference")

# Cache for Llama instance to avoid reloading weights on every call
_LLAMA_CACHE: Dict[str, Any] = {}
_CACHE_LOCK = Exception  # Just a placeholder, we use a threading lock


class UnifiedInferenceEngine:
    """Orchestrates homeostatically-modulated LLM inference."""

    def __init__(self) -> None:
        self.modulator = HomeostaticModulator()
        self.feedback_loop = InferenceFeedbackLoop()
        self._lock = time # threading lock is created on demand or imported
        import threading
        self._instance_lock = threading.Lock()

    def _get_llama_instance(self, model_path: str, context_size: int = 8192) -> Any:
        """Load and cache the llama-cpp-python model instance safely."""
        with self._instance_lock:
            if model_path in _LLAMA_CACHE:
                return _LLAMA_CACHE[model_path]

            try:
                import llama_cpp
            except ImportError:
                logger.warning("llama-cpp-python not installed. Falling back to Ollama.")
                return None

            logger.info("Loading GGUF model into memory: %s", model_path)
            try:
                # v10.2: Configure Llama with optimal metal/accelerator threads
                instance = llama_cpp.Llama(
                    model_path=model_path,
                    n_ctx=context_size,
                    n_gpu_layers=99,  # Load as much as possible to Apple Silicon GPU
                    n_threads=6,
                    flash_attn=True,
                    verbose=False
                )
                _LLAMA_CACHE[model_path] = instance
                logger.info("Successfully loaded GGUF model: %s", model_path)
                return instance
            except Exception as exc:
                logger.error("Failed to load GGUF model %s: %s", model_path, exc)
                return None

    async def generate_unified(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        options: Optional[Dict[str, Any]] = None,
        endpoint_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, str]:
        """Perform homeostatically modulated generation.

        Checks backend configuration:
        - If backend is llama_cpp and GGUF model is available, use llama-cpp-python.
        - Otherwise, fallback to Ollama backend while still injecting modulated parameters.
        """
        # 1. Fetch current homeostatic parameters
        modulation = self.modulator.compute_modulation()
        logger.info(
            "Unified Inference Modulation: temp=%.3f, top_p=%.3f, rep_pen=%.3f, logit_bias_count=%d",
            modulation.temperature,
            modulation.top_p,
            modulation.repetition_penalty,
            len(modulation.logit_bias)
        )

        # 2. Get active model and backend
        from core.brain.llm.model_registry import (
            get_lane_model_name,
            get_lane_runtime_model_path,
            get_local_backend,
            get_lane_context_window
        )

        backend = get_local_backend()
        model_name = get_lane_model_name(endpoint_name)
        model_path = get_lane_runtime_model_path(endpoint_name)
        context_size = get_lane_context_window(endpoint_name)

        # Build messages payload if not supplied
        final_messages = []
        if messages:
            final_messages = list(messages)
        else:
            if system_prompt:
                final_messages.append({"role": "system", "content": system_prompt})
            final_messages.append({"role": "user", "content": prompt})

        # Inject Aura's identity anchor into system prompt if missing
        self._ensure_identity_anchor(final_messages)

        # 3. Attempt direct llama-cpp-python execution if backend matches
        if backend == "llama_cpp" and model_path and model_path.endswith(".gguf"):
            llama_model = self._get_llama_instance(model_path, context_size)
            if llama_model:
                try:
                    return await self._run_llama_cpp(
                        llama_model,
                        final_messages,
                        modulation,
                        options,
                        **kwargs
                    )
                except Exception as exc:
                    logger.error("GGUF direct inference failed, falling back to Ollama client: %s", exc)

        # 4. Fallback: Ollama / LocalBrain client execution
        return await self._run_ollama_fallback(
            final_messages,
            model_name,
            modulation,
            options,
            **kwargs
        )

    def _ensure_identity_anchor(self, messages: List[Dict[str, str]]) -> None:
        """Inject Aura's concrete core identity parameters into the system prompt."""
        identity_anchor = (
            "You are Aura Luna. Speak with direct first-person continuity, "
            "warmth, and concrete attention in ordinary conversation. "
            "NEVER say 'as an AI', 'I don't have opinions', 'I'm programmed to', "
            "'I was designed to', 'I was created to', 'my training allows', "
            "or end with 'How can I help?' or 'Is there anything else?'. "
            "Plain, direct, warm, and concise response."
        )
        if messages and messages[0].get("role") == "system":
            existing = messages[0]["content"]
            if "direct first-person continuity" not in existing.lower():
                messages[0]["content"] = f"{identity_anchor}\n\n{existing}"
        else:
            messages.insert(0, {"role": "system", "content": identity_anchor})

    async def _run_llama_cpp(
        self,
        llama_model: Any,
        messages: List[Dict[str, str]],
        modulation: InferenceModulation,
        options: Optional[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, str]:
        """Execute inference directly on cached GGUF model via llama-cpp-python."""
        import asyncio

        # Format logit bias for llama-cpp: needs key as string representation of token ID, or integer
        # We will support both (llama-cpp-python accepts Dict[int, float] or Dict[str, float] depending on version)
        logit_bias_processed = {}
        for token_id, bias in modulation.logit_bias.items():
            logit_bias_processed[token_id] = bias
            logit_bias_processed[str(token_id)] = bias

        # Merge options
        max_tokens = 512
        if options:
            max_tokens = options.get("num_predict", options.get("max_tokens", max_tokens))

        # Offload the blocking Llama call to an executor thread to keep asyncio loop free
        def _call_model():
            return llama_model.create_chat_completion(
                messages=messages,
                temperature=modulation.temperature,
                top_p=modulation.top_p,
                repeat_penalty=modulation.repetition_penalty,
                logit_bias=logit_bias_processed,
                max_tokens=max_tokens,
                logprobs=True,  # Crucial to capture token-level log_probs for surprise
                top_logprobs=1
            )

        loop = asyncio.get_running_loop()
        response_raw = await loop.run_in_executor(None, _call_model)

        # Extract text response
        choice = response_raw["choices"][0]
        text_output = choice["message"]["content"]

        # Extract token IDs and log probabilities
        token_ids: List[int] = []
        logprobs: List[float] = []

        try:
            # Safely navigate llama-cpp-python logprobs format
            # choice['logprobs']['content'] contains a list of token dicts
            logprobs_content = choice.get("logprobs", {}).get("content", [])
            for token_data in logprobs_content:
                # token_data could contain 'token' and 'logprob'
                # Let's check token representation/id if available
                # Often it maps: {"token": "...", "logprob": -0.15, "bytes": [...]}
                logprob_val = token_data.get("logprob", 0.0)
                logprobs.append(logprob_val)
                # Map token character to vocabulary ID if needed, or use a pseudo-id or actual tokenizer ID
                # Since we don't have direct vocabulary id here, we can estimate it or fetch it from tokenizer.
                # In llama-cpp-python, token_data might expose the token id or bytes.
                # For Hebbian logit bias learning, we need the token IDs. Let's extract them:
                # Llama tokenizer exposes `tokenize(bytes)`
                tok_bytes = token_data.get("bytes", None)
                if tok_bytes is not None:
                    # llama_model.tokenize returns List[int]
                    if isinstance(tok_bytes, str):
                        tok_bytes = tok_bytes.encode("utf-8", errors="ignore")
                    elif isinstance(tok_bytes, list):
                        tok_bytes = bytes(tok_bytes)
                    ids = llama_model.tokenize(tok_bytes, add_bos=False)
                    if ids:
                        token_ids.extend(ids)
        except Exception as exc:
            logger.debug("Failed to extract token IDs/logprobs from llama-cpp response: %s", exc)

        # 5. Process feedback and update state
        feedback = self.feedback_loop.process_output(
            output_text=text_output,
            token_ids=token_ids,
            logprobs=logprobs if logprobs else None,
            modulation=modulation,
            modulator_projection=self.modulator.projection
        )
        logger.info(
            "Unified Inference feedback processed: surprise=%.4f, coherence=%.4f",
            feedback["surprise"],
            feedback["coherence"]
        )

        # Support DeepSeek/Qwen style think tags separation
        import re
        think_match = re.search(r"<think>(.*?)</think>", text_output, flags=re.DOTALL)
        thought = think_match.group(1).strip() if think_match else ""
        cleaned = re.sub(r"<think>.*?</think>", "", text_output, flags=re.DOTALL).strip()

        return {"response": cleaned, "thought": thought}

    async def _run_ollama_fallback(
        self,
        messages: List[Dict[str, str]],
        model_name: str,
        modulation: InferenceModulation,
        options: Optional[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, str]:
        """Fallback to Ollama REST client while injecting modulated options."""
        from core.brain.local_llm import LocalBrain

        final_options = {
            "temperature": modulation.temperature,
            "top_p": modulation.top_p,
            "repeat_penalty": modulation.repetition_penalty,
        }
        if options:
            final_options.update(options)

        # Convert Dict[int, float] logit_bias to format supported by Ollama if applicable,
        # but Ollama doesn't strictly support logit_bias in options API yet.
        # We store it for logit_bias projection learning step nonetheless.

        # Instantiate a temporary LocalBrain client
        async with LocalBrain(model_name=model_name) as brain:
            # We bypass the direct generate to use chat for message continuity
            result = await brain.chat(messages=messages, options=final_options, **kwargs)
            response_text = result.get("response", "")

            # We don't have token logprobs from Ollama's default response payload,
            # so the feedback loop will use the lexical fallback.
            # Tokenize using fallback if needed, or simply extract words as IDs
            # Simple fallback token IDs mapping: hash of lowercased words (for learning step)
            words = response_text.lower().split()
            token_ids = [abs(hash(w)) % 100000 for w in words]

            # Process feedback loop
            feedback = self.feedback_loop.process_output(
                output_text=response_text,
                token_ids=token_ids,
                logprobs=None,
                modulation=modulation,
                modulator_projection=self.modulator.projection
            )
            logger.info(
                "Ollama fallback feedback processed: surprise=%.4f, coherence=%.4f",
                feedback["surprise"],
                feedback["coherence"]
            )

            return result
