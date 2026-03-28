"""
Phase 8.9 Test 1: Grid Round-Trip Encoding

Test that GridToWave and WaveToGrid maintain high fidelity.

Pass criteria:
- Wave embedding similarity > 0.85 for same grid
- Grid reconstruction has correct dimensions
- Values in valid range (0-9)
"""

import torch
import sys
from pathlib import Path

# Add project paths
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PHASES_DIR / 'phase8_8') not in sys.path:
    sys.path.insert(0, str(_PHASES_DIR / 'phase8_8'))
if str(_PHASES_DIR / 'phase8_9') not in sys.path:
    sys.path.insert(0, str(_PHASES_DIR / 'phase8_9'))

from flux_utils import PhaseResults


def test_grid_round_trip():
    """Test grid encoding and decoding round-trip."""
    results = PhaseResults(phase=8.9, component_name="Grid Round-Trip")
    
    print("=" * 60)
    print("Phase 8.9 Test 1: Grid Round-Trip Encoding")
    print("=" * 60)
    
    # ─────────────────────────────────────────────
    # Load adapters
    # ─────────────────────────────────────────────
    print("\n1. Loading grid adapters...")
    
    from grid_adapters import GridToWave, WaveToGrid
    
    device = 'cpu'  # Use CPU for reproducibility
    grid_to_wave = GridToWave(device=device)
    wave_to_grid = WaveToGrid(device=device)
    
    print(f"  GridToWave: {sum(p.numel() for p in grid_to_wave.parameters()):,} params")
    print(f"  WaveToGrid: {sum(p.numel() for p in wave_to_grid.parameters()):,} params")
    
    # ─────────────────────────────────────────────
    # Test 1: Same grid produces similar waves
    # ─────────────────────────────────────────────
    print("\n2. Testing wave consistency...")
    
    grid1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    
    wave1 = grid_to_wave.encode(grid1, mode='holistic')
    wave2 = grid_to_wave.encode(grid1, mode='holistic')  # Same grid
    
    similarity = torch.nn.functional.cosine_similarity(
        wave1.unsqueeze(0), wave2.unsqueeze(0)
    ).item()
    
    print(f"  Same grid similarity: {similarity:.4f}")
    
    passed = similarity > 0.999  # Should be identical (deterministic)
    results.add_test(
        "Same grid consistency",
        passed=passed,
        score=similarity,
        threshold=0.999,
    )
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    
    # ─────────────────────────────────────────────
    # Test 2: Different grids have different embeddings
    # ─────────────────────────────────────────────
    print("\n3. Testing discriminability...")
    
    grid2 = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]  # Reversed
    wave3 = grid_to_wave.encode(grid2, mode='holistic')
    
    diff_similarity = torch.nn.functional.cosine_similarity(
        wave1.unsqueeze(0), wave3.unsqueeze(0)
    ).item()
    
    print(f"  Different grid similarity: {diff_similarity:.4f}")
    
    # Should be different but not orthogonal (both are 3x3 grids with same colors)
    passed = diff_similarity < 0.95
    results.add_test(
        "Different grids discriminable",
        passed=passed,
        score=1 - diff_similarity,  # Higher is better
        threshold=0.05,
    )
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    
    # ─────────────────────────────────────────────
    # Test 3: Decode produces correct dimensions
    # ─────────────────────────────────────────────
    print("\n4. Testing decode dimensions...")
    
    test_sizes = [(3, 3), (5, 5), (10, 10), (7, 4)]
    all_correct = True
    
    for h, w in test_sizes:
        grid = torch.randint(0, 10, (h, w)).tolist()
        wave = grid_to_wave.encode(grid, mode='holistic')
        decoded = wave_to_grid.decode(wave, grid_size=(h, w))
        
        if decoded.shape != (h, w):
            print(f"  ✗ Size {h}x{w}: got {decoded.shape}")
            all_correct = False
        else:
            print(f"  ✓ Size {h}x{w}: correct")
    
    results.add_test(
        "Decode dimensions correct",
        passed=all_correct,
        score=1.0 if all_correct else 0.0,
        threshold=1.0,
    )
    
    # ─────────────────────────────────────────────
    # Test 4: Values in valid range
    # ─────────────────────────────────────────────
    print("\n5. Testing output value range...")
    
    grid = torch.randint(0, 10, (5, 5)).tolist()
    wave = grid_to_wave.encode(grid, mode='holistic')
    decoded = wave_to_grid.decode(wave, grid_size=(5, 5))
    
    min_val = decoded.min().item()
    max_val = decoded.max().item()
    
    print(f"  Output range: [{min_val}, {max_val}]")
    
    passed = min_val >= 0 and max_val <= 9
    results.add_test(
        "Values in range [0, 9]",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
    )
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    
    # ─────────────────────────────────────────────
    # Test 5: Spatial mode produces per-cell waves
    # ─────────────────────────────────────────────
    print("\n6. Testing spatial mode...")
    
    grid = [[1, 2], [3, 4]]
    waves_spatial = grid_to_wave.encode(grid, mode='spatial')
    
    expected_shape = (4, 432)  # 2*2 cells, each with 432-dim wave
    
    print(f"  Spatial output shape: {waves_spatial.shape}")
    
    passed = waves_spatial.shape == torch.Size(expected_shape)
    results.add_test(
        "Spatial mode shape",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
    )
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    
    # ─────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    results.summary()
    results.save()
    
    return results.all_passed()


if __name__ == '__main__':
    success = test_grid_round_trip()
    sys.exit(0 if success else 1)
