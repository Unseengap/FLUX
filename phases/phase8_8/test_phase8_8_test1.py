"""
Phase 8.8 Test 1: Adapter Loading and Basic Operations

Tests:
1. .flx file can be downloaded/loaded
2. Adapters initialize correctly
3. Text encoding produces correct shapes
4. Grid encoding produces correct shapes
5. Wave similarity is meaningful
"""

import sys
from pathlib import Path

# Setup paths
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'phases' / 'phase8_8'))

for i in range(1, 9):
    p = ROOT / 'phases' / f'phase{i}'
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
import torch.nn.functional as F
from flux_utils import get_device


def test_flx_loading():
    """Test that .flx file can be loaded."""
    print("\n  Test 1: FLX Loading...")
    
    from flx_loader import verify_flx, DEFAULT_FLX_PATH
    
    success, msg = verify_flx(DEFAULT_FLX_PATH)
    assert success, f"FLX verification failed: {msg}"
    print(f"    ✓ {msg}")


def test_adapter_initialization():
    """Test that adapters can be initialized."""
    print("\n  Test 2: Adapter Initialization...")
    
    device = get_device()
    
    from grid_adapters import GridToWave, WaveToGrid
    from text_adapters import WaveToText
    
    grid_to_wave = GridToWave(device=device)
    wave_to_grid = WaveToGrid(device=device)
    wave_to_text = WaveToText(device=device)
    
    assert grid_to_wave is not None
    assert wave_to_grid is not None
    assert wave_to_text is not None
    
    print("    ✓ All adapters initialized")


def test_grid_encoding_shapes():
    """Test that grid encoding produces correct shapes."""
    print("\n  Test 3: Grid Encoding Shapes...")
    
    device = get_device()
    from grid_adapters import GridToWave
    
    encoder = GridToWave(device=device)
    
    # Test holistic mode
    grid = [[1, 2], [3, 4]]
    wave_h = encoder.encode(grid, mode='holistic')
    assert wave_h.shape == torch.Size([432]), f"Expected [432], got {wave_h.shape}"
    print(f"    ✓ Holistic: {list(wave_h.shape)}")
    
    # Test spatial mode
    wave_s = encoder.encode(grid, mode='spatial')
    assert wave_s.shape == torch.Size([4, 432]), f"Expected [4, 432], got {wave_s.shape}"
    print(f"    ✓ Spatial: {list(wave_s.shape)}")
    
    # Test larger grid
    large = [[i % 10 for j in range(30)] for i in range(30)]
    wave_large = encoder.encode(large, mode='holistic')
    assert wave_large.shape == torch.Size([432])
    print(f"    ✓ 30×30 grid encoded")


def test_wave_similarity():
    """Test that wave similarity is meaningful."""
    print("\n  Test 4: Wave Similarity...")
    
    device = get_device()
    from grid_adapters import GridToWave
    
    encoder = GridToWave(device=device)
    
    # Same grid should have similarity ~1.0
    grid_a = [[1, 2], [3, 4]]
    wave_a1 = encoder.encode(grid_a)
    wave_a2 = encoder.encode(grid_a)
    
    sim_same = F.cosine_similarity(wave_a1.unsqueeze(0), wave_a2.unsqueeze(0)).item()
    assert sim_same > 0.99, f"Same grid similarity {sim_same} < 0.99"
    print(f"    ✓ Same grid similarity: {sim_same:.4f}")
    
    # Different grids should have lower similarity
    grid_b = [[5, 6], [7, 8]]
    wave_b = encoder.encode(grid_b)
    
    sim_diff = F.cosine_similarity(wave_a1.unsqueeze(0), wave_b.unsqueeze(0)).item()
    assert sim_diff < 1.0, f"Different grid similarity {sim_diff} should be < 1.0"
    print(f"    ✓ Different grid similarity: {sim_diff:.4f}")


def test_delta_extraction():
    """Test that delta waves can be extracted."""
    print("\n  Test 5: Delta Extraction...")
    
    device = get_device()
    from grid_adapters import GridToWave
    
    encoder = GridToWave(device=device)
    
    input_grid = [[0, 0], [0, 0]]
    output_grid = [[1, 1], [1, 1]]
    
    in_wave, out_wave, delta = encoder.encode_pair(input_grid, output_grid)
    
    assert in_wave.shape == torch.Size([432])
    assert out_wave.shape == torch.Size([432])
    assert delta.shape == torch.Size([432])
    
    # Delta should be non-trivial
    delta_norm = delta.norm().item()
    assert delta_norm > 0.1, f"Delta norm {delta_norm} too small"
    print(f"    ✓ Delta wave extracted: norm={delta_norm:.4f}")


def main():
    print("\n" + "="*60)
    print("  Phase 8.8 Test 1: Adapter Loading and Operations")
    print("="*60)
    
    tests = [
        ("FLX Loading", test_flx_loading),
        ("Adapter Init", test_adapter_initialization),
        ("Grid Encoding Shapes", test_grid_encoding_shapes),
        ("Wave Similarity", test_wave_similarity),
        ("Delta Extraction", test_delta_extraction),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"    ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "-"*60)
    print(f"  Results: {passed}/{len(tests)} tests passed")
    
    if failed == 0:
        print("  ✓ ALL TESTS PASSED")
    else:
        print(f"  ✗ {failed} tests failed")
    
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
