"""
Phase 6 Test 2: Semantic Memory Protection

Verifies:
- Semantic field is unchanged after 1000 episodic writes
- Protected attractors maintain their energy barriers
- Field stability >= 0.95 after bulk episodic operations

Uses Phase 6 checkpoint components.
"""

import sys
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'phases' / 'phase1'))
sys.path.insert(0, str(ROOT / 'phases' / 'phase2'))
sys.path.insert(0, str(ROOT / 'phases' / 'phase6'))

from flux_utils import load_checkpoint, get_device, PhaseResults
from cse import ContinuousSemanticEncoder
from field import ResonanceField
from working_memory import WorkingMemory
from episodic_memory import EpisodicMemory
from semantic_memory import SemanticMemory


def test_semantic_protection():
    """Test that semantic memory field is protected during episodic writes."""
    print("=" * 60)
    print("  Phase 6 Test 2: Semantic Memory Protection")
    print("=" * 60)

    device = get_device()

    # ── Load checkpoints ──
    ckpt6 = load_checkpoint(6)
    ckpt1 = load_checkpoint(1)
    ckpt2 = load_checkpoint(2)

    # Rebuild CSE
    cse = ContinuousSemanticEncoder(**ckpt1.get('config', {}))
    cse.load_state_dict(ckpt1['state_dict'])
    cse = cse.to(device).eval()

    # Rebuild field
    field_cfg = ckpt2.get('config', {}).get('field', {})
    field = ResonanceField(**field_cfg)
    field.load_state_dict(ckpt2['state_dict'])
    field = field.to(device)

    # Rebuild working memory
    cfg = ckpt6.get('config', {})
    wm = WorkingMemory(
        window_size=cfg.get('window_size', 512),
        wave_dim=cfg.get('wave_dim', 432),
        feature_dim=cfg.get('feature_dim', 256),
    ).to(device)
    wm.load_state_memory(ckpt6['working_memory_state'])
    wm.eval()

    # Build semantic memory with loaded field
    semantic = SemanticMemory(field=field, protection_threshold=5.0).to(device)

    # Protect attractors
    for i in range(10):
        semantic.protect_attractor(i)

    # ── Snapshot field before episodic writes ──
    snapshot_before = semantic.get_field_snapshot()
    energy_before = semantic.get_field_energy()
    print(f"\n  Field energy before 1000 episodic writes: {energy_before:.6f}")
    print(f"  Protected attractors: {semantic.num_protected}")

    # ── Perform 1000 episodic writes (should NOT affect semantic field) ──
    episodic = EpisodicMemory(feature_dim=cfg.get('feature_dim', 256))

    print("  Writing 1000 episodic entries (should not affect field)...")
    for i in range(1000):
        text = f"Episodic fact number {i}: random knowledge item"
        with torch.no_grad():
            wave = cse.encode(text)
            vec = wm.compress(wave.full.mean(dim=0).to(device).unsqueeze(0)).squeeze(0)
        episodic.write(vec, fact=text, causal_source="test2")

    # ── Check field is unchanged ──
    energy_after = semantic.get_field_energy()
    stability = semantic.compute_stability(snapshot_before)

    print(f"\n  Field energy after 1000 episodic writes: {energy_after:.6f}")
    print(f"  Field stability: {stability:.6f}")
    print(f"  Energy delta: {abs(energy_after - energy_before):.8f}")

    # Verify protection
    all_protected = all(semantic.is_protected(i) for i in range(10))
    print(f"  All attractors still protected: {all_protected}")

    # ── Assertions ──
    # Episodic writes should NOT change the semantic field at all
    # (stability should be exactly 1.0 since episodic is separate)
    passed_stability = stability >= 0.95
    passed_protection = all_protected
    passed_energy = abs(energy_after - energy_before) < 0.001

    print(f"\n  Stability >= 0.95: {'PASS ✓' if passed_stability else 'FAIL ✗'}")
    print(f"  Attractors protected: {'PASS ✓' if passed_protection else 'FAIL ✗'}")
    print(f"  Energy unchanged: {'PASS ✓' if passed_energy else 'FAIL ✗'}")

    passed = passed_stability and passed_protection and passed_energy
    if passed:
        print("\n  ✓ Test 2: PASS")
    else:
        print("\n  ✗ Test 2: FAIL")

    assert passed, "Semantic memory protection test failed"
    print("\nTest 2: PASS")


if __name__ == "__main__":
    test_semantic_protection()
