import re

def test_reflex(message):
    print(f"Testing message: {repr(message)}")
    if "[EMBODIED CONTROL CONTRACT]" not in message:
        print("Contract not found.")
        return None

    content = message.split("[EMBODIED CONTROL CONTRACT]")[0].strip()
    print(f"Extracted content: {repr(content)}")
    
    patterns = {
        r"^[^\n?]{1,80}$": "[SOMATIC:key=' ']",
    }

    for pattern, response in patterns.items():
        if re.search(pattern, content, re.IGNORECASE):
            print(f"MATCH: {pattern} -> {response}")
            return response
    print("NO MATCH.")
    return None

# Exact string from log (simulated)
raw_screen = "You are lucky!  Full moon tonight.                                              \n                   "
message = f"{raw_screen}\n\n[EMBODIED CONTROL CONTRACT] Somatic reflex matcher v3 ACTIVE."

test_reflex(message)
