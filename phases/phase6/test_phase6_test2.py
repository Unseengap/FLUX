import sys
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from semantic_memory import SemanticMemory

def test_semantic_protection():
    print("Running Phase 6 Test 2: Semantic Memory Protection")
    # Mock field
    sm = SemanticMemory(field=None)

    # Placeholder for protection logic
    sm.protect_attractor(101)
    assert 101 in sm.mature_attractors
    print("Attractor protection verified")

    print("Test 2: PASS")

if __name__ == "__main__":
    test_semantic_protection()
