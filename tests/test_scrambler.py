import sys
sys.path.append('.')
from core.memory.black_hole import encode_payload, decode_payload

def test():
    key = "event-horizon-7734"
    text = "The universe remembers everything. " * 50  # Make it long enough for compression to work well
    
    print(f"Original length: {len(text)}")
    result = encode_payload(text, key)
    print(f"Encoded bits length: {len(result['encoded'])}")
    print(f"Compression ratio: {result['ratio']}%")
    
    decoded = decode_payload(result['encoded'], key)
    
    if decoded == text:
        print("Success! Decoded matches original.")
    else:
        print("Error: Decoded text does not match.")
        print(f"Expected: {text[:50]}...")
        if decoded:
            print(f"Got: {decoded[:50]}...")
        else:
            print("Got: None")

if __name__ == "__main__":
    test()
