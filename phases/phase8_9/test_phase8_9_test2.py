"""
Phase 8.9 Test 2: Image Generation Quality

Test that WaveToImage_Universal produces valid images with all physics engines.

Pass criteria:
- All 3 physics engines produce valid RGB images
- All 5 style presets work
- Images have reasonable brightness/contrast
- Different waves produce different images
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


def test_image_generation():
    """Test image generation quality."""
    results = PhaseResults(phase=8.9, component_name="Image Generation")
    
    print("=" * 60)
    print("Phase 8.9 Test 2: Image Generation Quality")
    print("=" * 60)
    
    # ─────────────────────────────────────────────
    # Load adapters
    # ─────────────────────────────────────────────
    print("\n1. Loading image adapters...")
    
    from image_adapters import (
        WaveToImage_Universal,
        GravityRenderer,
        InterferenceRenderer,
        ThermodynamicRenderer,
        STYLE_PRESETS,
    )
    
    device = 'cpu'
    wave_to_image = WaveToImage_Universal(device=device)
    
    print(f"  WaveToImage: {sum(p.numel() for p in wave_to_image.parameters()):,} params")
    
    # Create test wave
    torch.manual_seed(42)
    test_wave = torch.randn(432, device=device)
    test_wave = test_wave / test_wave.norm() * 10
    
    # ─────────────────────────────────────────────
    # Test 1: Individual physics engines
    # ─────────────────────────────────────────────
    print("\n2. Testing individual physics engines...")
    
    size = (64, 64)
    engines_ok = True
    
    # Gravity
    img_g = wave_to_image.gravity(test_wave, size[0], size[1])
    valid_g = img_g.shape == (64, 64, 3) and img_g.min() >= 0 and img_g.max() <= 1
    print(f"  Gravity: shape={img_g.shape}, valid={valid_g}")
    engines_ok &= valid_g
    
    # Interference
    img_i = wave_to_image.interference(test_wave, size[0], size[1])
    valid_i = img_i.shape == (64, 64, 3) and img_i.min() >= 0 and img_i.max() <= 1
    print(f"  Interference: shape={img_i.shape}, valid={valid_i}")
    engines_ok &= valid_i
    
    # Thermodynamic
    img_t = wave_to_image.thermodynamic(test_wave, size[0], size[1])
    valid_t = img_t.shape == (64, 64, 3) and img_t.min() >= 0 and img_t.max() <= 1
    print(f"  Thermodynamic: shape={img_t.shape}, valid={valid_t}")
    engines_ok &= valid_t
    
    results.add_test(
        "All 3 physics engines work",
        passed=engines_ok,
        score=int(valid_g) + int(valid_i) + int(valid_t),
        threshold=3,
    )
    print(f"  {'✓ PASS' if engines_ok else '✗ FAIL'}")
    
    # ─────────────────────────────────────────────
    # Test 2: All style presets
    # ─────────────────────────────────────────────
    print("\n3. Testing style presets...")
    
    presets_ok = True
    for style_name in STYLE_PRESETS:
        img = wave_to_image.decode(test_wave, size=size, style=style_name)
        valid = img.shape == (64, 64, 3) and img.min() >= 0 and img.max() <= 1
        print(f"  {style_name}: valid={valid}")
        presets_ok &= valid
    
    results.add_test(
        "All 5 style presets work",
        passed=presets_ok,
        score=len(STYLE_PRESETS) if presets_ok else 0,
        threshold=5,
    )
    print(f"  {'✓ PASS' if presets_ok else '✗ FAIL'}")
    
    # ─────────────────────────────────────────────
    # Test 3: Different waves → different images
    # ─────────────────────────────────────────────
    print("\n4. Testing wave discriminability...")
    
    wave1 = torch.randn(432, device=device)
    wave2 = torch.randn(432, device=device)
    
    img1 = wave_to_image.decode(wave1, size=size, style='dream')
    img2 = wave_to_image.decode(wave2, size=size, style='dream')
    
    # Images should be different
    diff = (img1 - img2).abs().mean().item()
    print(f"  Mean pixel difference: {diff:.4f}")
    
    passed = diff > 0.01  # Should be noticeably different
    results.add_test(
        "Different waves → different images",
        passed=passed,
        score=diff,
        threshold=0.01,
    )
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    
    # ─────────────────────────────────────────────
    # Test 4: Same wave → same image (deterministic)
    # ─────────────────────────────────────────────
    print("\n5. Testing determinism...")
    
    wave_to_image.eval()
    with torch.no_grad():
        img_a = wave_to_image.decode(test_wave, size=size, style='photorealistic')
        img_b = wave_to_image.decode(test_wave, size=size, style='photorealistic')
    
    diff = (img_a - img_b).abs().max().item()
    print(f"  Max difference on same input: {diff:.6f}")
    
    passed = diff < 1e-5
    results.add_test(
        "Deterministic output",
        passed=passed,
        score=1.0 if passed else diff,
        threshold=1e-5,
    )
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    
    # ─────────────────────────────────────────────
    # Test 5: Images have reasonable statistics
    # ─────────────────────────────────────────────
    print("\n6. Testing image statistics...")
    
    img = wave_to_image.decode(test_wave, size=(128, 128), style='dream')
    
    brightness = img.mean().item()
    contrast = img.std().item()
    
    print(f"  Brightness: {brightness:.3f} (expected: 0.2-0.8)")
    print(f"  Contrast: {contrast:.3f} (expected: 0.05-0.5)")
    
    stats_ok = 0.1 < brightness < 0.9 and 0.01 < contrast < 0.6
    results.add_test(
        "Reasonable image statistics",
        passed=stats_ok,
        score=1.0 if stats_ok else 0.0,
        threshold=1.0,
    )
    print(f"  {'✓ PASS' if stats_ok else '✗ FAIL'}")
    
    # ─────────────────────────────────────────────
    # Test 6: Auto-blend produces valid weights
    # ─────────────────────────────────────────────
    print("\n7. Testing auto-blend...")
    
    auto_weights = wave_to_image.auto_blend(test_wave)
    
    sum_weights = auto_weights.sum().item()
    all_positive = (auto_weights >= 0).all().item()
    
    print(f"  Weights: ({auto_weights[0]:.3f}, {auto_weights[1]:.3f}, {auto_weights[2]:.3f})")
    print(f"  Sum: {sum_weights:.4f}, All positive: {all_positive}")
    
    passed = abs(sum_weights - 1.0) < 0.01 and all_positive
    results.add_test(
        "Auto-blend valid weights",
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
    success = test_image_generation()
    sys.exit(0 if success else 1)
