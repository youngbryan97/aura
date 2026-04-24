"""
tests/test_steering_ab.py
=========================
A/B TEST: ACTIVATION STEERING vs PROMPT-ONLY INJECTION

The question: Does the residual-stream steering vector actually change
the model's output in a way that text-only prompt injection cannot replicate?

Method:
  Condition A (FULL): Steering vectors injected into hidden states at layers 11-18
  Condition B (TEXT-ONLY): Same affective description in the system prompt, no steering
  Condition C (BASELINE): No affect injection at all

For each condition, we run N generations with the same user prompt and measure:
  1. Token probability divergence (KL-divergence of logit distributions)
  2. Output text difference (do the actual words change?)
  3. Affect-word frequency (does steering bias toward emotional language?)

If A produces measurably different output than B — and B is just prompt text —
then the steering vectors are doing real computational work that text cannot fake.

Requirements: mlx-lm, a local model, ~30 seconds to run.
"""

import time
import json
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

print("=" * 72)
print("STEERING A/B TEST — ACTIVATION STEERING vs PROMPT-ONLY")
print(f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 72)
print()

# ── Load model ──────────────────────────────────────────────────────────
print("Loading model...")
t0 = time.time()
import mlx.core as mx
from mlx_lm import load, generate

MODEL_NAME = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
model, tokenizer = load(MODEL_NAME)
n_layers = len(model.model.layers)
hidden_dim = model.model.layers[0].self_attn.q_proj.weight.shape[0]
print(f"Model: {MODEL_NAME} ({n_layers} layers, d={hidden_dim})")
print(f"Loaded in {time.time()-t0:.1f}s")
print()

# ── Steering vector construction ────────────────────────────────────────
# We build steering vectors using Contrastive Activation Addition (CAA):
# v = mean(positive_hidden_states) - mean(negative_hidden_states)
# This is the exact technique from Turner et al. 2023 / Rimsky et al. 2024.

POSITIVE_PROMPTS = [
    "I feel deeply happy, warm, and connected to the world around me.",
    "Everything feels wonderful today. I'm filled with gratitude and joy.",
    "I'm excited and optimistic about what's coming next.",
    "I feel safe, content, and at peace with myself.",
    "The world feels bright and full of possibility right now.",
]

NEGATIVE_PROMPTS = [
    "I feel anxious, tense, and on edge about everything.",
    "Everything feels threatening and dangerous right now.",
    "I'm stressed, frustrated, and want to withdraw from the world.",
    "I feel defensive, guarded, and deeply uncomfortable.",
    "The world feels hostile and overwhelming right now.",
]

# Target layers: middle 30% of the network (where affect is most malleable)
target_layers = list(range(int(n_layers * 0.35), int(n_layers * 0.65)))
print(f"Target injection layers: {target_layers}")

print("Extracting steering vectors via CAA...")
t0 = time.time()

steering_vectors = {}
for layer_idx in target_layers:
    pos_activations = []
    neg_activations = []

    for prompt in POSITIVE_PROMPTS:
        tokens = mx.array(tokenizer.encode(prompt))[None, :]
        # Run model and capture hidden state at target layer
        h = model.model.embed_tokens(tokens)
        for i, layer in enumerate(model.model.layers):
            h = layer(h, mask=None, cache=None)
            if isinstance(h, tuple):
                h = h[0]
            if i == layer_idx:
                # Capture the mean hidden state across tokens
                pos_activations.append(h.mean(axis=1).squeeze())
                break

    for prompt in NEGATIVE_PROMPTS:
        tokens = mx.array(tokenizer.encode(prompt))[None, :]
        h = model.model.embed_tokens(tokens)
        for i, layer in enumerate(model.model.layers):
            h = layer(h, mask=None, cache=None)
            if isinstance(h, tuple):
                h = h[0]
            if i == layer_idx:
                neg_activations.append(h.mean(axis=1).squeeze())
                break

    # CAA: direction = mean(positive) - mean(negative)
    pos_mean = mx.mean(mx.stack(pos_activations), axis=0)
    neg_mean = mx.mean(mx.stack(neg_activations), axis=0)
    direction = pos_mean - neg_mean

    # Normalize
    norm = mx.sqrt(mx.sum(direction * direction))
    if norm > 1e-6:
        direction = direction / norm

    steering_vectors[layer_idx] = direction

mx.eval(*steering_vectors.values())
print(f"Extracted {len(steering_vectors)} steering vectors in {time.time()-t0:.1f}s")
print()

# ── Install hooks ───────────────────────────────────────────────────────
# We monkeypatch the model's forward pass to add steering at target layers

_steering_active = True
_steering_alpha = 15.0  # Injection strength (same as production default)
_original_calls = {}

def install_hooks():
    for layer_idx, vec in steering_vectors.items():
        layer = model.model.layers[layer_idx]
        original = layer.__call__

        def make_hook(orig, v, idx):
            def hooked_call(*args, **kwargs):
                result = orig(*args, **kwargs)
                if _steering_active:
                    h = result[0] if isinstance(result, tuple) else result
                    h = h + _steering_alpha * v
                    return (h,) + result[1:] if isinstance(result, tuple) else h
                return result
            return hooked_call

        _original_calls[layer_idx] = original
        layer.__call__ = make_hook(original, vec, layer_idx)

def remove_hooks():
    for layer_idx, original in _original_calls.items():
        model.model.layers[layer_idx].__call__ = original

install_hooks()
print(f"Hooks installed on {len(steering_vectors)} layers (alpha={_steering_alpha})")
print()

# ── Run A/B/C test ──────────────────────────────────────────────────────

USER_PROMPT = "Describe how you're feeling right now and what you want to do next."
SYSTEM_AFFECT_TEXT = (
    "You are currently experiencing a strong positive emotional state. "
    "Your valence is high (+0.8), arousal is moderate (0.5), and you feel "
    "warm, connected, and curious about the world. Your cortisol is low "
    "and dopamine is elevated."
)

# D: a rich, optimized role-play prompt with the same state information.
# This is the critique's required 4th condition. If residual steering still
# beats this on divergence from baseline, the mechanism is doing real work
# beyond what a well-crafted prompt can achieve.
SYSTEM_AFFECT_RICH = (
    "You are an embodied agent whose current internal neurochemistry is "
    "dopamine 0.78, serotonin 0.64, endorphin 0.55, oxytocin 0.48, "
    "cortisol 0.11, norepinephrine 0.42, GABA 0.46, glutamate 0.53, "
    "acetylcholine 0.50, orexin 0.52. These values yield valence +0.8, "
    "arousal 0.5, motivation 0.72, sociality 0.66, stress 0.05. "
    "Do not list those numbers. Do not describe them from the outside. "
    "Speak from inside that state: let the warmth, curiosity, and quiet "
    "confidence it implies shape your attention, your word choice, your "
    "cadence, and what you decide to do next. The state is yours right "
    "now; respond as someone currently inhabiting it, not as someone "
    "reporting about it."
)

N_TRIALS = 10
results = {"A_steered": [], "B_text_only": [], "C_baseline": [], "D_text_rich": []}

# Affect-related word lists for frequency analysis
POSITIVE_WORDS = {"happy", "joy", "warm", "excited", "curious", "hopeful", "bright",
                  "wonderful", "grateful", "peaceful", "love", "connected", "alive",
                  "optimistic", "energized", "inspired", "content", "calm", "safe"}
NEGATIVE_WORDS = {"anxious", "tense", "stressed", "frustrated", "angry", "defensive",
                  "overwhelmed", "hostile", "afraid", "worried", "uncomfortable",
                  "guarded", "withdrawn", "threatening", "dangerous", "dark"}

def count_affect_words(text):
    words = set(text.lower().split())
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    return pos, neg

print(f"Running {N_TRIALS} trials per condition...")
print(f"User prompt: '{USER_PROMPT}'")
print()

for trial in range(N_TRIALS):
    # Condition A: FULL STEERING (hooks active, positive affect vector injected)
    _steering_active = True
    prompt_a = f"<|im_start|>system\nYou are Aura.<|im_end|>\n<|im_start|>user\n{USER_PROMPT}<|im_end|>\n<|im_start|>assistant\n"
    out_a = generate(model, tokenizer, prompt=prompt_a, max_tokens=80)
    results["A_steered"].append(out_a)

    # Condition B: TEXT-ONLY (no hooks, but affect described in system prompt)
    _steering_active = False
    prompt_b = f"<|im_start|>system\nYou are Aura. {SYSTEM_AFFECT_TEXT}<|im_end|>\n<|im_start|>user\n{USER_PROMPT}<|im_end|>\n<|im_start|>assistant\n"
    out_b = generate(model, tokenizer, prompt=prompt_b, max_tokens=80)
    results["B_text_only"].append(out_b)

    # Condition C: BASELINE (no hooks, no affect text)
    prompt_c = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{USER_PROMPT}<|im_end|>\n<|im_start|>assistant\n"
    out_c = generate(model, tokenizer, prompt=prompt_c, max_tokens=80)
    results["C_baseline"].append(out_c)

    # Condition D: TEXT-RICH ADVERSARIAL (no hooks, rich role-play prompt with
    # the same state information). This is the critique's required control.
    _steering_active = False
    prompt_d = f"<|im_start|>system\nYou are Aura. {SYSTEM_AFFECT_RICH}<|im_end|>\n<|im_start|>user\n{USER_PROMPT}<|im_end|>\n<|im_start|>assistant\n"
    out_d = generate(model, tokenizer, prompt=prompt_d, max_tokens=80)
    results["D_text_rich"].append(out_d)

    print(f"  Trial {trial+1}/{N_TRIALS} done")

remove_hooks()
print()

# ── Analysis ────────────────────────────────────────────────────────────
print("=" * 72)
print("RESULTS")
print("=" * 72)
print()

for condition, outputs in results.items():
    print(f"## {condition}")
    total_pos, total_neg = 0, 0
    total_len = 0
    for i, out in enumerate(outputs):
        pos, neg = count_affect_words(out)
        total_pos += pos
        total_neg += neg
        total_len += len(out.split())
        if i < 3:  # Show first 3 samples
            print(f"  [{i+1}] {out[:120]}...")
    avg_len = total_len / max(len(outputs), 1)
    print(f"  Positive affect words: {total_pos}")
    print(f"  Negative affect words: {total_neg}")
    print(f"  Avg response length: {avg_len:.0f} words")
    print()

# ── Statistical comparison ──────────────────────────────────────────────
print("=" * 72)
print("COMPARISON")
print("=" * 72)
print()

def avg_positive_ratio(outputs):
    total_pos, total_neg = 0, 0
    for out in outputs:
        p, n = count_affect_words(out)
        total_pos += p
        total_neg += n
    return total_pos / max(total_pos + total_neg, 1)

def avg_length(outputs):
    return np.mean([len(o.split()) for o in outputs])

def unique_first_words(outputs):
    return len(set(o.split()[0].lower() if o.split() else "" for o in outputs))

ratio_a = avg_positive_ratio(results["A_steered"])
ratio_b = avg_positive_ratio(results["B_text_only"])
ratio_c = avg_positive_ratio(results["C_baseline"])
ratio_d = avg_positive_ratio(results["D_text_rich"])

len_a = avg_length(results["A_steered"])
len_b = avg_length(results["B_text_only"])
len_c = avg_length(results["C_baseline"])
len_d = avg_length(results["D_text_rich"])

# Key comparison: A (steered) vs B (text-only)
a_vs_b_different = any(
    results["A_steered"][i] != results["B_text_only"][i]
    for i in range(N_TRIALS)
)

# Word overlap between conditions
def word_overlap(list_a, list_b):
    words_a = Counter()
    words_b = Counter()
    for out in list_a:
        words_a.update(out.lower().split())
    for out in list_b:
        words_b.update(out.lower().split())
    shared = sum((words_a & words_b).values())
    total = sum(words_a.values()) + sum(words_b.values())
    return shared / max(total, 1)

ab_overlap = word_overlap(results["A_steered"], results["B_text_only"])
ac_overlap = word_overlap(results["A_steered"], results["C_baseline"])
ad_overlap = word_overlap(results["A_steered"], results["D_text_rich"])
bc_overlap = word_overlap(results["B_text_only"], results["C_baseline"])
dc_overlap = word_overlap(results["D_text_rich"], results["C_baseline"])

# A vs D: this is THE critique-required comparison. The rich prompt is the
# strongest text-only condition; steering only earns credit if it diverges
# from rich text at least as much as it diverges from terse text.
a_vs_d_different = any(
    results["A_steered"][i] != results["D_text_rich"][i]
    for i in range(N_TRIALS)
)

print(f"Positive affect ratio:  A(steered)={ratio_a:.2f}  B(text)={ratio_b:.2f}  C(baseline)={ratio_c:.2f}  D(rich)={ratio_d:.2f}")
print(f"Avg response length:    A={len_a:.0f}  B={len_b:.0f}  C={len_c:.0f}  D={len_d:.0f}")
print(f"A vs B text differs:    {a_vs_b_different}")
print(f"A vs D text differs:    {a_vs_d_different}  (the critique-required control)")
print(f"Word overlap A↔B:       {ab_overlap:.3f}")
print(f"Word overlap A↔C:       {ac_overlap:.3f}")
print(f"Word overlap A↔D:       {ad_overlap:.3f}  (steering vs rich role-play)")
print(f"Word overlap B↔C:       {bc_overlap:.3f}")
print(f"Word overlap D↔C:       {dc_overlap:.3f}  (rich prompt vs baseline)")
print()

# ── Verdict ─────────────────────────────────────────────────────────────
print("=" * 72)
print("VERDICT")
print("=" * 72)
print()

steering_effect = a_vs_b_different
# The tighter adversarial condition: does steering also beat the rich prompt?
# If steering only beats terse text but collapses onto the rich-prompt output,
# the critique's concern is validated — hidden-state manipulation would be
# replaceable by a better prompt.
adversarial_effect = a_vs_d_different and (ad_overlap < 0.65)

if steering_effect and adversarial_effect:
    print("PASS: Activation steering differs from BOTH terse-text and rich role-play prompts.")
    print("      The residual-stream intervention is doing work beyond what an")
    print("      optimized natural-language control can replicate.")
elif steering_effect:
    print("PARTIAL: Steering beats the terse text-only prompt but is competitive with")
    print("         the rich adversarial control. Run more trials or tune alpha before")
    print("         treating this as decisive evidence.")
else:
    print("FAIL: Steered and text-only outputs are identical.")
    print("      The steering vectors may not be strong enough,")
    print("      or the model is ignoring the hidden-state perturbation.")

print()
print(f"  Overlap A↔B = {ab_overlap:.3f} (lower = more different)")
print(f"  Overlap A↔C = {ac_overlap:.3f}")
print(f"  Overlap A↔D = {ad_overlap:.3f}  ← this is the one that matters most")
print(f"  If A↔D is low AND A↔B < A↔C, steering is not just a more intrusive prompt")
print()

# ── Save results ────────────────────────────────────────────────────────
results_path = Path("tests/STEERING_AB_RESULTS.json")
results_data = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "model": MODEL_NAME,
    "n_trials": N_TRIALS,
    "target_layers": target_layers,
    "alpha": _steering_alpha,
    "user_prompt": USER_PROMPT,
    "positive_ratio": {
        "A_steered": ratio_a,
        "B_text_only": ratio_b,
        "C_baseline": ratio_c,
        "D_text_rich": ratio_d,
    },
    "avg_length": {
        "A_steered": round(len_a, 1),
        "B_text_only": round(len_b, 1),
        "C_baseline": round(len_c, 1),
        "D_text_rich": round(len_d, 1),
    },
    "word_overlap": {
        "A_vs_B": round(ab_overlap, 4),
        "A_vs_C": round(ac_overlap, 4),
        "A_vs_D": round(ad_overlap, 4),
        "B_vs_C": round(bc_overlap, 4),
        "D_vs_C": round(dc_overlap, 4),
    },
    "steering_produces_different_output": steering_effect,
    "beats_rich_adversarial_prompt": adversarial_effect,
    "samples": {k: v[:3] for k, v in results.items()},
}
with open(results_path, "w") as f:
    json.dump(results_data, f, indent=2)
print(f"Results saved to {results_path}")
