import logging
import sys

logger = logging.getLogger("LLM.MLX")

class MLXClient:
    """
    Client for running local LLMs on Apple Silicon via mlx_lm.
    """
    def __init__(self, model_path="mlx-community/Mistral-7B-Instruct-v0.3-4bit"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        try:
            from mlx_lm import load
            import mlx.core as mx
            logger.info(f"Loading MLX Model: {self.model_path}...")
            self.model, self.tokenizer = load(self.model_path)
            logger.info("MLX Model Loaded Successfully.")
        except ImportError:
            logger.error("mlx-lm not installed. Run 'pip install mlx-lm mlx'.")
        except Exception as e:
            logger.error(f"Failed to load MLX model: {e}")

    @staticmethod
    def _extract_think_segments(text: str) -> tuple[str, str]:
        import re
        thoughts = []
        for m in re.finditer(r'<think>(.*?)</think>', text, flags=re.DOTALL):
            thought_text = m.group(1).strip()
            if thought_text:
                thoughts.append(thought_text)
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        cleaned = cleaned.replace('</think>', '').replace('<think>', '')
        return cleaned.strip(), "\n\n".join(thoughts)

    def call(self, prompt: str, system_prompt: str = None, max_tokens: int = 2048, **kwargs) -> dict:
        if not self.model:
            return {"ok": False, "error": "MLX Model not loaded (pkg missing?)"}

        try:
            from mlx_lm import generate
            
            # Simple prompt assembly if system prompt provided
            if system_prompt:
                formatted_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                formatted_prompt = prompt
                
            temp = kwargs.get("temperature", 0.7)
            logger.info(f"Generating MLX response (max {max_tokens} tokens)...")
            
            response_text = generate(
                self.model, 
                self.tokenizer, 
                prompt=formatted_prompt, 
                max_tokens=max_tokens, 
                temp=temp,
                verbose=False
            )
            
            cleaned, thought = self._extract_think_segments(response_text)
            return {"ok": True, "text": cleaned, "thought": thought}
            
        except Exception as e:
            logger.error(f"MLX Generation Error: {e}")
            return {"ok": False, "error": str(e)}

    # v15: Streaming support
    async def call_stream(self, prompt: str, system_prompt: str = None, max_tokens: int = 2048, **kwargs):
        if not self.model:
            yield "Error: MLX Model not loaded"
            return
            
        try:
            from mlx_lm import generate_step
            import mlx.core as mx
            
            if system_prompt:
                formatted_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                formatted_prompt = prompt
                
            temp = kwargs.get("temperature", 0.7)
            
            # Simple non-blocking wrapper around sync generator (MLX is CPU/GPU locked)
            # A true prod impl would run this in a threadpool
            in_think_block = False
            
            for (token, _), _ in zip(generate_step(mx.array(self.tokenizer.encode(formatted_prompt)), self.model, temp), range(max_tokens)):
                token_str = self.tokenizer.decode([token])
                
                # Real-time think block filtering
                if in_think_block:
                    if "</think>" in token_str:
                        in_think_block = False
                        parts = token_str.split("</think>")
                        if len(parts) > 1 and parts[1]:
                            yield parts[1]
                    continue
                    
                if "<think>" in token_str:
                    in_think_block = True
                    parts = token_str.split("<think>")
                    if parts[0]:
                        yield parts[0]
                    continue
                    
                if token_str:
                    yield token_str
                    
        except Exception as e:
            logger.error(f"MLX Streaming Error: {e}")
            yield f"\n[MLX Generation Failed: {str(e)}]"
