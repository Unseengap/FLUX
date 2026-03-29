#!/usr/bin/env python3
"""
test_phase12_test1.py — Test: Grid Rendering + Vision Input

Tests:
1. Grid rendering with ARC colors
2. Ice field overlay
3. Exploration mass overlay
4. Agent position marker
5. ASCII fallback rendering
"""

import sys
from pathlib import Path
import numpy as np

# Add project root and phase12
PHASE_DIR = Path(__file__).parent
PROJECT_ROOT = PHASE_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PHASE_DIR))

from flux_utils import PhaseResults, get_device


def test_grid_renderer():
    """Test GridRenderer with various configurations."""
    from grid_renderer import GridRenderer, PIL_AVAILABLE, ARC_COLORS
    
    passed = 0
    total = 0
    
    # Test 1: PIL availability
    total += 1
    if PIL_AVAILABLE:
        print("  ✓ PIL/Pillow available")
        passed += 1
        renderer = GridRenderer(cell_size=10)
    else:
        print("  ⚠ PIL not available, skipping image tests")
        return passed, total
    
    # Test 2: Basic grid rendering
    total += 1
    try:
        grid = [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [0, 0, 1, 0, 0],
            [2, 0, 0, 0, 3],
            [0, 4, 0, 5, 0],
        ]
        img = renderer.render(grid)
        assert img.size == (50, 50)  # 5 cells * 10 pixels
        assert img.mode == 'RGB'
        print(f"  ✓ Basic grid rendering: {img.size}")
        passed += 1
    except Exception as e:
        print(f"  ✗ Basic grid rendering failed: {e}")
    
    # Test 3: Agent position marker
    total += 1
    try:
        img = renderer.render(grid, position=(2, 2))
        assert img is not None
        # Check that center pixel area is different (agent marker)
        pixels = np.array(img)
        center_pixel = pixels[25, 25]  # Center of grid
        assert not np.all(center_pixel == 0), "Agent marker should be visible"
        print(f"  ✓ Agent position marker rendered")
        passed += 1
    except Exception as e:
        print(f"  ✗ Agent position marker failed: {e}")
    
    # Test 4: Ice field overlay
    total += 1
    try:
        import torch
        ice_field = torch.zeros(5, 5)
        ice_field[1, 1] = 20.0  # High curiosity
        ice_field[3, 3] = 15.0
        
        img = renderer.render(grid, ice_field=ice_field)
        assert img is not None
        print(f"  ✓ Ice field overlay rendered")
        passed += 1
    except Exception as e:
        print(f"  ✗ Ice field overlay failed: {e}")
    
    # Test 5: Exploration mass overlay
    total += 1
    try:
        import torch
        mass = torch.zeros(5, 5)
        mass[0, :] = 5.0  # Explored top row
        mass[1, :] = 3.0
        
        img = renderer.render(grid, exploration_mass=mass)
        assert img is not None
        print(f"  ✓ Exploration mass overlay rendered")
        passed += 1
    except Exception as e:
        print(f"  ✗ Exploration mass overlay failed: {e}")
    
    # Test 6: Comparison rendering
    total += 1
    try:
        grid_before = [[0, 1, 1], [0, 0, 0], [2, 2, 0]]
        grid_after = [[0, 1, 2], [0, 0, 0], [2, 2, 0]]
        
        img = renderer.render_comparison(
            grid_before, grid_after,
            position_before=(1, 1),
            position_after=(1, 2),
        )
        assert img.width > 30, "Comparison should be wider"
        print(f"  ✓ Comparison rendering: {img.size}")
        passed += 1
    except Exception as e:
        print(f"  ✗ Comparison rendering failed: {e}")
    
    return passed, total


def test_ascii_renderer():
    """Test ASCIIRenderer fallback."""
    from grid_renderer import ASCIIRenderer
    
    passed = 0
    total = 0
    
    renderer = ASCIIRenderer()
    
    # Test 1: Basic ASCII rendering
    total += 1
    try:
        grid = [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [0, 0, 1, 0, 0],
            [2, 0, 0, 0, 3],
            [0, 4, 0, 5, 0],
        ]
        ascii_out = renderer.render(grid)
        assert len(ascii_out) > 0
        assert '\n' in ascii_out
        print(f"  ✓ ASCII rendering: {len(ascii_out)} chars")
        passed += 1
    except Exception as e:
        print(f"  ✗ ASCII rendering failed: {e}")
    
    # Test 2: Position marker
    total += 1
    try:
        ascii_out = renderer.render(grid, position=(2, 2))
        assert '@' in ascii_out, "Position marker should be @"
        print(f"  ✓ ASCII position marker: found @")
        passed += 1
    except Exception as e:
        print(f"  ✗ ASCII position marker failed: {e}")
    
    # Test 3: Ice overlay symbols
    total += 1
    try:
        import torch
        ice = torch.zeros(5, 5)
        ice[1, 3] = 15.0  # High curiosity
        
        ascii_out = renderer.render(grid, ice_field=ice)
        assert '!' in ascii_out or '*' in ascii_out, "Ice should show ! or *"
        print(f"  ✓ ASCII ice symbols rendered")
        passed += 1
    except Exception as e:
        print(f"  ✗ ASCII ice symbols failed: {e}")
    
    # Test 4: Legend generation
    total += 1
    try:
        legend = renderer.render_legend()
        assert '@' in legend
        assert '!' in legend
        print(f"  ✓ Legend generated: {len(legend)} chars")
        passed += 1
    except Exception as e:
        print(f"  ✗ Legend generation failed: {e}")
    
    return passed, total


def test_normalize_grid():
    """Test grid normalization helper."""
    from grid_renderer import normalize_grid
    import torch
    
    passed = 0
    total = 0
    
    # Test 1: List input
    total += 1
    try:
        grid = [[1, 2], [3, 4]]
        result = normalize_grid(grid)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        print(f"  ✓ List normalization")
        passed += 1
    except Exception as e:
        print(f"  ✗ List normalization failed: {e}")
    
    # Test 2: Tensor input
    total += 1
    try:
        grid = torch.tensor([[1, 2], [3, 4]])
        result = normalize_grid(grid)
        assert isinstance(result, np.ndarray)
        print(f"  ✓ Tensor normalization")
        passed += 1
    except Exception as e:
        print(f"  ✗ Tensor normalization failed: {e}")
    
    # Test 3: Dict input
    total += 1
    try:
        frame = {'frame': [[1, 2], [3, 4]]}
        result = normalize_grid(frame)
        assert result.shape == (2, 2)
        print(f"  ✓ Dict normalization")
        passed += 1
    except Exception as e:
        print(f"  ✗ Dict normalization failed: {e}")
    
    return passed, total


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  Phase 12 Test 1: Grid Rendering + Vision Input")
    print("=" * 60 + "\n")
    
    results = PhaseResults(phase=12, component_name="Grid Renderer")
    
    total_passed = 0
    total_tests = 0
    
    # Run test suites
    print("Testing GridRenderer...")
    passed, total = test_grid_renderer()
    total_passed += passed
    total_tests += total
    results.add_test("GridRenderer", passed=passed == total, 
                     score=passed/total if total > 0 else 0, threshold=0.8)
    
    print("\nTesting ASCIIRenderer...")
    passed, total = test_ascii_renderer()
    total_passed += passed
    total_tests += total
    results.add_test("ASCIIRenderer", passed=passed == total,
                     score=passed/total if total > 0 else 0, threshold=0.8)
    
    print("\nTesting normalize_grid...")
    passed, total = test_normalize_grid()
    total_passed += passed
    total_tests += total
    results.add_test("normalize_grid", passed=passed == total,
                     score=passed/total if total > 0 else 0, threshold=0.8)
    
    # Summary
    print("\n" + "-" * 40)
    print(f"Total: {total_passed}/{total_tests} tests passed")
    
    overall = total_passed / total_tests if total_tests > 0 else 0
    results.add_test("Overall", passed=overall >= 0.8, score=overall, threshold=0.8)
    
    if overall >= 0.8:
        print("✓ Test 1 PASSED")
    else:
        print("✗ Test 1 FAILED")
    
    # Save results
    results.save()
    print(f"\nResults saved")
    
    return 0 if overall >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
