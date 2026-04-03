################################################################################


import sys
import os
import time
import numpy as np

# Add core to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.memory_vector import VectorMemory

def test_vector_memory():
    print("Initializing Vector Memory...")
    # Use a temp path
    mem = VectorMemory(path="autonomy_engine/data/test_vectors.json", local_first=True)
    
    # Clear previous
    mem.clear()
    
    print("Generative Dummy Data...")
    # Mocking the embedding generation to avoid calling Ollama/OpenAI for speed
    # We will manually inject vectors
    
    # 3 Distinct concepts
    vec_a = [1.0, 0.0, 0.0] # Concept A
    vec_b = [0.0, 1.0, 0.0] # Concept B
    vec_c = [0.0, 0.0, 1.0] # Concept C
    vec_ab = [0.7, 0.7, 0.0] # Mixed A and B
    
    mem.add("Apple", vector=vec_a)
    mem.add("Banana", vector=vec_b)
    mem.add("Car", vector=vec_c)
    mem.add("Fruit Salad", vector=vec_ab)
    
    print(f"Items in memory: {len(mem.items)}")
    print(f"Matrix shape: {mem.matrix.shape if mem.matrix is not None else 'None'}")
    
    # Test Search
    print("\nSearching for 'Apple' (Expect Apple, Fruit Salad)...")
    results = mem.search(vector=vec_a, k=3)
    
    for res in results:
        print(f"  - {res['text']} (Score: {res['similarity']:.4f})")
        
    # Verify Matrix Math
    # Apple (1,0,0) dot Apple (1,0,0) = 1.0
    # Apple (1,0,0) dot Fruit Salad (0.7, 0.7, 0) = 0.7 (approx, after normalization)
    
    assert results[0]['text'] == "Apple"
    assert results[0]['similarity'] > 0.99
    
    print("\n✅ Matrix Search Verified.")

if __name__ == "__main__":
    test_vector_memory()


##
