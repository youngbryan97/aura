from typing import Any, Dict


def map_affect_to_sampling(affect: Dict[str, float]) -> Dict[str, Any]:
    """Accept affect = {"valence": -1..1, "arousal": 0..1, "engagement": 0..1}
    Return sampling params: temperature, top_p, max_tokens, repetition_penalty
    """
    # Defensive defaults in case of missing keys
    valence = float(affect.get("valence", 0.0))
    arousal = float(affect.get("arousal", 0.5))
    engagement = float(affect.get("engagement", 0.5))

    # Baseline Aura personality
    temperature = 0.7
    top_p = 0.9
    max_tokens = 256
    repetition_penalty = 1.0

    # 1. Valence (Anxiety vs. Positivity)
    # If negative valence (anxiety) -> reduce temperature (stick to facts), defensive style
    if valence < -0.3:
        # Pushes temp lower as valence drops
        temperature = max(0.2, 0.5 + valence * 0.4) 
        top_p = 0.85
        repetition_penalty = 1.1

    # 2. Engagement (Boredom vs. Flow)
    # High engagement -> more depth, longer replies, slightly more creativity
    if engagement > 0.7:
        max_tokens = int(max_tokens * 1.5)
        temperature = min(1.0, temperature + 0.1)

    # 3. Arousal (Calm vs. Excited/Panic)
    # High arousal -> shorter bursts, focused attention (lower temp)
    if arousal > 0.8:
        max_tokens = int(max_tokens * 0.8)
        temperature = max(0.2, temperature - 0.1)

    # Normalize/Sanitize
    temperature = round(float(temperature), 3)
    top_p = round(float(top_p), 3)

    return {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "repetition_penalty": repetition_penalty
    }