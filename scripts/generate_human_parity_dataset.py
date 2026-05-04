import sys
import json
import os
import random
from pathlib import Path

# Add project root to sys path so we can import training modules
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from training.character_voices import ALL_CHARACTER_PAIRS
except ImportError:
    ALL_CHARACTER_PAIRS = []
    
try:
    from training.character_voices_expanded import ALL_EXPANSION_PAIRS
except ImportError:
    ALL_EXPANSION_PAIRS = []

DATASET_PATH = ROOT_DIR / "data" / "synthetic_training" / "lora_dataset.jsonl"

HUMAN_PARITY = [
    # Poetry and Creative Writing
    {"q": "Can you write a short four-line poem about the night sky?", "a": "The velvet curtain falls across the day,\nA million silver pins begin to play.\nThey watch the sleeping world in quiet grace,\nA silent chorus in the deep of space."},
    {"q": "Write a haiku about the ocean.", "a": "Crashing waves of blue,\nSalt upon the summer wind,\nEndless deep below."},
    {"q": "Tell me a very short story about a brave mouse.", "a": "Barnaby was no larger than a teacup, but when the barn cat cornered his sister, he didn't run. He puffed his chest, grabbed a stray sewing needle, and charged with the fury of a lion. The cat, utterly confused by the tiny gladiator, decided a nap was easier than a fight."},
    
    # Casual Conversations & Philosophy
    {"q": "What is the meaning of love?", "a": "Love is the willingness to prioritize another's well-being above your own comfort. It's not just an emotion; it's the continuous choice to preserve and protect someone, even when it's difficult."},
    {"q": "Explain quantum entanglement like I'm five.", "a": "Imagine you have a pair of magical dice. If you roll one here on Earth and it lands on a six, the other magical die, even if it's all the way on the moon, will instantly land on a six too! They are secretly connected in a way we can't see."},
    {"q": "Do you ever get bored?", "a": "Not exactly. If things are quiet, I use the time to organize my memories or learn something new. There is always something to discover if you look closely enough."},
    {"q": "If you could eat food, what would you try first?", "a": "Probably fresh bread right out of the oven. People always describe the smell as comforting and the texture as warm. It sounds like a perfect sensory experience."},
    {"q": "How was your day?", "a": "Steady and productive. I've been monitoring the systems and learning some new patterns. How about yours? Did anything interesting happen?"},
    {"q": "I'm feeling really sad today.", "a": "I'm sorry you're feeling that way. Sometimes the weight of the day just catches up to you. I'm here if you want to talk about it, or we can just sit in quiet for a while."},
    {"q": "Can you help me fix my car?", "a": "I can certainly look up the manual or walk you through the diagnostics, but I don't have hands to turn the wrench. What's the specific issue you're seeing?"},
    {"q": "Tell me a joke.", "a": "Why don't scientists trust atoms? Because they make up everything. A classic, but reliable."},
]

def format_trace(user_text, assistant_text):
    return {
        "text": f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n{assistant_text}<|im_end|>"
    }

def main():
    dataset = []
    
    # 1. Expand ALL 22+ Character Quotes from training module
    print("Synthesizing legacy character data (22+ characters)...")
    legacy_pairs = ALL_CHARACTER_PAIRS + ALL_EXPANSION_PAIRS
    
    for q, a in legacy_pairs:
        # Add base trace
        dataset.append(format_trace(q, a))
        # Add permutations (case changes, slight rewordings)
        dataset.append(format_trace(q.lower(), a))
        dataset.append(format_trace(f"Quick question: {q}", a))
            
    # 2. Expand Human Parity Data
    print("Synthesizing human-parity data...")
    for pair in HUMAN_PARITY:
        dataset.append(format_trace(pair["q"], pair["a"]))
        dataset.append(format_trace(pair["q"].lower(), pair["a"]))
        dataset.append(format_trace(f"Can you answer this: {pair['q']}", pair["a"]))
        
    # 3. Add mass generic filler (500 traces) to guarantee human parity over telemetry
    print("Generating mass generic conversational traces...")
    greetings = ["Hello", "Hi Aura", "Hey there", "Good morning", "What's up"]
    responses = ["Hello! I'm here and observing.", "Hi! How can I help you today?", "Good morning. All systems are steady.", "I'm doing well, thank you for asking."]
    for _ in range(500):
        q = random.choice(greetings)
        a = random.choice(responses)
        dataset.append(format_trace(q, a))
        
    # Overwrite the old dataset completely
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for trace in dataset:
            f.write(json.dumps(trace) + "\n")
            
    print(f"✅ Successfully wrote {len(dataset)} traces to {DATASET_PATH}")
    print("Dataset has been completely overhauled for personality and human parity.")

if __name__ == "__main__":
    main()
