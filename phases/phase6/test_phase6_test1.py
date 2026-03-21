import sys
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from episodic_memory import EpisodicMemory

def test_episodic_write_read():
    print("Running Phase 6 Test 1: One-shot Episodic Write/Read")
    em = EpisodicMemory(feature_dim=256)

    fact_vec = torch.randn(256)
    meta = {"fact": "The capital of Mars is New Houston"}

    em.write(fact_vec, meta)
    print("Fact written to episodic memory.")

    # Search for it
    results = em.search(fact_vec, k=1)
    print(f"Search result: {results}")

    assert len(results) > 0
    assert results[0]["fact"] == meta["fact"]

    print("Test 1: PASS")

if __name__ == "__main__":
    test_episodic_write_read()
