"""
Phase 6 Test 1: One-Shot Episodic Write/Read

Verifies:
- Write a fact to episodic memory with one-shot encoding
- Retrieve it correctly after 100+ subsequent writes
- Retrieval accuracy >= 90% across a test set

Uses Phase 6 checkpoint components.
"""

import sys
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'phases' / 'phase1'))
sys.path.insert(0, str(ROOT / 'phases' / 'phase6'))

from flux_utils import load_checkpoint, get_device, PhaseResults
from cse import ContinuousSemanticEncoder
from working_memory import WorkingMemory
from episodic_memory import EpisodicMemory


def test_episodic_write_read():
    """Test one-shot episodic write and read accuracy."""
    print("=" * 60)
    print("  Phase 6 Test 1: One-Shot Episodic Write/Read")
    print("=" * 60)

    device = get_device()

    # ── Load checkpoint ──
    ckpt6 = load_checkpoint(6)
    ckpt1 = load_checkpoint(1)

    # Rebuild CSE
    cse = ContinuousSemanticEncoder(**ckpt1.get('config', {}))
    cse.load_state_dict(ckpt1['state_dict'])
    cse = cse.to(device).eval()

    # Rebuild working memory (for compression layer)
    cfg = ckpt6.get('config', {})
    wm = WorkingMemory(
        window_size=cfg.get('window_size', 512),
        wave_dim=cfg.get('wave_dim', 432),
        feature_dim=cfg.get('feature_dim', 256),
    ).to(device)
    wm.load_state_memory(ckpt6['working_memory_state'])
    wm.eval()

    # Fresh episodic memory
    em = EpisodicMemory(feature_dim=cfg.get('feature_dim', 256))

    # ── Test facts ──
    facts = [
        "The capital of France is Paris",
        "Water boils at 100 degrees Celsius",
        "The earth orbits the sun in 365 days",
        "FLUX uses resonance fields instead of weight matrices",
        "Gravitational relevance achieves O(log n) complexity",
        "Thermodynamic learning requires no backpropagation",
        "Causal geometry nodes store both WHAT and WHY",
        "Working memory is session-scoped and rolling",
        "Episodic memory uses FAISS for fast retrieval",
        "Semantic memory protected by energy barriers",
        "The speed of light is approximately 300000 km/s",
        "DNA carries genetic information",
        "Python was created by Guido van Rossum",
        "The Milky Way is a spiral galaxy",
        "Oxygen has atomic number 8",
    ]

    print(f"\n  Writing {len(facts)} facts to episodic memory...")

    # One-shot write all facts
    for fact in facts:
        with torch.no_grad():
            wave = cse.encode(fact)
            vec = wm.compress(wave.full.mean(dim=0).to(device).unsqueeze(0)).squeeze(0)
        em.write(vec, fact=fact, causal_source="test1")

    # Write 100 distractor entries to simulate noise
    print("  Writing 100 distractor entries...")
    for i in range(100):
        noise = torch.randn(cfg.get('feature_dim', 256))
        em.write(noise, fact=f"distractor_{i}", causal_source="noise")

    # ── Retrieval test ──
    print(f"\n  Testing retrieval accuracy after {em.size} total entries...")
    correct = 0
    for fact in facts:
        with torch.no_grad():
            wave = cse.encode(fact)
            q_vec = wm.compress(wave.full.mean(dim=0).to(device).unsqueeze(0)).squeeze(0)
        results = em.search(q_vec, k=1)
        if results and results[0][0].fact == fact:
            correct += 1
            print(f"    ✓ '{fact[:50]}...'")
        else:
            retrieved = results[0][0].fact if results else "NONE"
            print(f"    ✗ '{fact[:50]}...' → got '{retrieved[:50]}'")

    accuracy = correct / len(facts)
    print(f"\n  Retrieval accuracy: {correct}/{len(facts)} = {accuracy:.1%}")
    print(f"  Threshold: ≥ 90%")

    passed = accuracy >= 0.9
    if passed:
        print("  ✓ Test 1: PASS")
    else:
        print("  ✗ Test 1: FAIL")

    assert passed, f"Episodic retrieval accuracy {accuracy:.1%} < 90%"
    print("\nTest 1: PASS")


if __name__ == "__main__":
    test_episodic_write_read()
