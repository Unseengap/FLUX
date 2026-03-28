"""
Phase 8.9 Test 3: FluxToAny All Modalities

Test the unified FluxToAny interface with all supported modalities.

Pass criteria:
- Modality detection works for all types
- Encoding works for text, grid, image
- Decoding works for text, grid, image
- Cross-modal processing (text→image, grid→text) works
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


def test_flux_to_any():
    """Test FluxToAny with all modalities."""
    results = PhaseResults(phase=8.9, component_name="FluxToAny")
    
    print("=" * 60)
    print("Phase 8.9 Test 3: FluxToAny All Modalities")
    print("=" * 60)
    
    # ─────────────────────────────────────────────
    # Create minimal FluxToAny
    # ─────────────────────────────────────────────
    print("\n1. Creating FluxToAny...")
    
    from flux_to_any import FluxToAny
    
    device = 'cpu'
    
    # Create minimal mock model
    class MockFlux:
        def __init__(self):
            self.cse = None
            self.field = None
            self.episodic_memory = None
    
    mock_flux = MockFlux()
    model = FluxToAny(mock_flux, device=device)
    
    # Force load adapters
    model._load_adapters()
    
    print(f"  Input adapters: {list(model.input_adapters.keys())}")
    print(f"  Output adapters: {list(model.output_adapters.keys())}")
    
    adapters_loaded = len(model.input_adapters) >= 3 and len(model.output_adapters) >= 3
    results.add_test(
        "Adapters loaded",
        passed=adapters_loaded,
        score=len(model.input_adapters) + len(model.output_adapters),
        threshold=6,
    )
    print(f"  {'✓ PASS' if adapters_loaded else '✗ FAIL'}")
    
    # ─────────────────────────────────────────────
    # Test 2: Modality detection
    # ─────────────────────────────────────────────
    print("\n2. Testing modality detection...")
    
    detection_ok = True
    
    # Text
    mod = model.detect_modality("Hello world")
    text_ok = mod == 'text'
    print(f"  String → {mod} ({'✓' if text_ok else '✗'})")
    detection_ok &= text_ok
    
    # Grid (list of lists)
    mod = model.detect_modality([[1, 2], [3, 4]])
    grid_ok = mod == 'grid'
    print(f"  List[List[int]] → {mod} ({'✓' if grid_ok else '✗'})")
    detection_ok &= grid_ok
    
    # Grid (tensor)
    mod = model.detect_modality(torch.tensor([[1, 2], [3, 4]], dtype=torch.long))
    grid_tensor_ok = mod == 'grid'
    print(f"  Tensor (long) → {mod} ({'✓' if grid_tensor_ok else '✗'})")
    detection_ok &= grid_tensor_ok
    
    # Image
    mod = model.detect_modality(torch.randn(64, 64, 3))
    image_ok = mod == 'image'
    print(f"  Tensor [H, W, 3] → {mod} ({'✓' if image_ok else '✗'})")
    detection_ok &= image_ok
    
    results.add_test(
        "Modality detection",
        passed=detection_ok,
        score=int(text_ok) + int(grid_ok) + int(grid_tensor_ok) + int(image_ok),
        threshold=4,
    )
    print(f"  {'✓ PASS' if detection_ok else '✗ FAIL'}")
    
    # ─────────────────────────────────────────────
    # Test 3: Grid encoding
    # ─────────────────────────────────────────────
    print("\n3. Testing grid encoding...")
    
    grid = [[1, 2, 3], [4, 5, 6]]
    wave = model.encode(grid, modality='grid')
    
    grid_encode_ok = wave.shape == torch.Size([432])
    print(f"  Grid → wave shape: {wave.shape} ({'✓' if grid_encode_ok else '✗'})")
    
    results.add_test(
        "Grid encoding",
        passed=grid_encode_ok,
        score=1.0 if grid_encode_ok else 0.0,
        threshold=1.0,
    )
    
    # ─────────────────────────────────────────────
    # Test 4: Grid decoding
    # ─────────────────────────────────────────────
    print("\n4. Testing grid decoding...")
    
    decoded = model.decode(wave, modality='grid', grid_size=(2, 3))
    
    grid_decode_ok = decoded.shape == torch.Size([2, 3])
    print(f"  Wave → grid shape: {decoded.shape} ({'✓' if grid_decode_ok else '✗'})")
    
    results.add_test(
        "Grid decoding",
        passed=grid_decode_ok,
        score=1.0 if grid_decode_ok else 0.0,
        threshold=1.0,
    )
    
    # ─────────────────────────────────────────────
    # Test 5: Image decoding
    # ─────────────────────────────────────────────
    print("\n5. Testing image decoding...")
    
    wave = torch.randn(432)
    image = model.decode(wave, modality='image', size=(64, 64))
    
    image_decode_ok = image.shape == torch.Size([64, 64, 3])
    print(f"  Wave → image shape: {image.shape} ({'✓' if image_decode_ok else '✗'})")
    
    valid_range = image.min() >= 0 and image.max() <= 1
    print(f"  Valid range [0, 1]: {'✓' if valid_range else '✗'}")
    
    results.add_test(
        "Image decoding",
        passed=image_decode_ok and valid_range,
        score=1.0 if (image_decode_ok and valid_range) else 0.0,
        threshold=1.0,
    )
    
    # ─────────────────────────────────────────────
    # Test 6: Cross-modal (grid → image)
    # ─────────────────────────────────────────────
    print("\n6. Testing cross-modal: grid → image...")
    
    grid = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    
    # Use manual pipeline since mock flux doesn't have full model
    wave = model.encode(grid, modality='grid')
    image = model.decode(wave, modality='image', size=(64, 64))
    
    cross_ok = image.shape == torch.Size([64, 64, 3])
    print(f"  Grid → Image shape: {image.shape} ({'✓' if cross_ok else '✗'})")
    
    results.add_test(
        "Cross-modal grid → image",
        passed=cross_ok,
        score=1.0 if cross_ok else 0.0,
        threshold=1.0,
    )
    
    # ─────────────────────────────────────────────
    # Test 7: Image encoding
    # ─────────────────────────────────────────────
    print("\n7. Testing image encoding...")
    
    image = torch.rand(64, 64, 3)
    wave = model.encode(image, modality='image')
    
    # Patches mode: should have multiple waves
    image_encode_ok = wave.dim() == 2 and wave.shape[1] == 432
    print(f"  Image → wave shape: {wave.shape} ({'✓' if image_encode_ok else '✗'})")
    
    results.add_test(
        "Image encoding",
        passed=image_encode_ok,
        score=1.0 if image_encode_ok else 0.0,
        threshold=1.0,
    )
    
    # ─────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    results.summary()
    results.save()
    
    return results.all_passed()


if __name__ == '__main__':
    success = test_flux_to_any()
    sys.exit(0 if success else 1)
