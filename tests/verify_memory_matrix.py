import tempfile
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.memory.sqlite_vector_store import SQLiteVectorStore


def test_vector_memory():
    print("Initializing SQLite vector memory...")
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteVectorStore(
            Path(tmp) / "vectors.sqlite3",
            collection_name="matrix_contract",
        )

        print("Generating deterministic vector data...")
        vec_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        vec_c = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        vec_ab = np.array([0.7, 0.7, 0.0], dtype=np.float32)

        store.upsert("apple", "Apple", vec_a)
        store.upsert("banana", "Banana", vec_b)
        store.upsert("car", "Car", vec_c)
        store.upsert("fruit_salad", "Fruit Salad", vec_ab)

        print(f"Items in memory: {store.count()}")
        print("\nSearching for Apple vector (expect Apple, Fruit Salad)...")
        results = store.query(vec_a, limit=3)

        for res in results:
            print(f"  - {res.content} (Score: {res.score:.4f})")

        assert [res.content for res in results[:2]] == ["Apple", "Fruit Salad"]
        assert results[0].score > 0.99
        assert 0.69 < results[1].score < 0.72

    print("\nMatrix Search Verified.")


if __name__ == "__main__":
    test_vector_memory()
