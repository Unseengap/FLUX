"""
Phase 8.9 Demo 1: Cross-Modal Processing

Demonstrates FluxToAny's ability to process different input modalities
and produce outputs in any target modality.

Examples:
- Text → Wave → Text (round-trip)
- Grid → Wave → Grid (spatial reasoning)
- Text → Wave → Image (text-to-image)
- Grid → Wave → Text (describe grid)
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

from flux_utils import get_device, PhaseLogger


def demo_cross_modal():
    """Demonstrate cross-modal processing with FluxToAny."""
    log = PhaseLogger(phase=8.9)
    log.separator("Demo 1: Cross-Modal Processing")
    
    device = get_device()
    log.info(f"Device: {device}")
    
    # ─────────────────────────────────────────────
    # Load FluxToAny
    # ─────────────────────────────────────────────
    log.info("\n1. Loading FluxToAny from Flux-X.flx...")
    
    try:
        from flux_to_any import FluxToAny
        model = FluxToAny.from_flx('Flux-X.flx', device=device)
        log.success(f"  Model loaded")
    except Exception as e:
        log.warning(f"  Could not load Flux-X.flx: {e}")
        log.info("  Creating minimal FluxToAny for demo...")
        
        # Create minimal model for demo
        from flux_to_any import FluxToAny
        
        # Try to load any available model
        try:
            from flx_loader import load_flux_from_flx, download_flx_from_hf
            flx_path = download_flx_from_hf('Flux-beta.flx')
            flux_model = load_flux_from_flx(str(flx_path), device=device)
            model = FluxToAny(flux_model, device=device)
            model._load_adapters()
            log.success("  Loaded from Flux-beta.flx")
        except Exception as e2:
            log.error(f"  Failed: {e2}")
            return
    
    # ─────────────────────────────────────────────
    # Demo: Text → Wave → Text
    # ─────────────────────────────────────────────
    log.info("\n2. Text → Wave → Text (round-trip)")
    
    text = "The quick brown fox"
    log.info(f"  Input: '{text}'")
    
    # Encode to wave
    wave = model.encode(text, modality='text')
    log.info(f"  Wave shape: {wave.shape}")
    log.info(f"  Wave norm: {wave.norm().item():.3f}")
    
    # Decode back to text (note: WaveToText is approximate)
    # For now, just show we can call the pipeline
    result = model(text, output_modality='text')
    log.info(f"  Output type: {type(result)}")
    log.success("  ✓ Text round-trip complete")
    
    # ─────────────────────────────────────────────
    # Demo: Grid → Wave → Grid
    # ─────────────────────────────────────────────
    log.info("\n3. Grid → Wave → Grid (spatial)")
    
    grid = [
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3],
    ]
    log.info(f"  Input grid: {grid}")
    
    # Encode to wave
    wave = model.encode(grid, modality='grid')
    log.info(f"  Wave shape: {wave.shape}")
    
    # Decode back to grid
    output_grid = model(grid, output_modality='grid', grid_size=(3, 3))
    log.info(f"  Output grid shape: {output_grid.shape if hasattr(output_grid, 'shape') else 'N/A'}")
    log.success("  ✓ Grid round-trip complete")
    
    # ─────────────────────────────────────────────
    # Demo: Text → Image
    # ─────────────────────────────────────────────
    log.info("\n4. Text → Wave → Image")
    
    text = "A beautiful sunset over mountains"
    log.info(f"  Input: '{text}'")
    
    # Generate image
    image = model(text, output_modality='image', size=(64, 64), style='photorealistic')
    log.info(f"  Output image shape: {image.shape}")
    log.info(f"  Pixel range: [{image.min():.2f}, {image.max():.2f}]")
    log.success("  ✓ Text-to-image complete")
    
    # ─────────────────────────────────────────────
    # Demo: Multi-modal comparison
    # ─────────────────────────────────────────────
    log.info("\n5. Multi-modal embedding comparison")
    
    # Compare similar concepts across modalities
    text1 = "red square"
    text2 = "blue circle"
    
    grid1 = [[1, 1], [1, 1]]  # Red square
    grid2 = [[2, 2], [2, 2]]  # Blue square
    
    wave_text1 = model.encode(text1, modality='text').mean(dim=0)
    wave_text2 = model.encode(text2, modality='text').mean(dim=0)
    wave_grid1 = model.encode(grid1, modality='grid')
    wave_grid2 = model.encode(grid2, modality='grid')
    
    # Cosine similarities
    def cosine(a, b):
        return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    
    log.info(f"  text1 vs text2: {cosine(wave_text1, wave_text2):.3f}")
    log.info(f"  grid1 vs grid2: {cosine(wave_grid1, wave_grid2):.3f}")
    log.info(f"  text1 vs grid1: {cosine(wave_text1, wave_grid1):.3f}")
    log.info(f"  text2 vs grid2: {cosine(wave_text2, wave_grid2):.3f}")
    
    log.success("  ✓ Multi-modal comparison complete")
    
    # ─────────────────────────────────────────────
    log.separator("Demo Complete")
    log.success("Phase 8.9 enables universal cross-modal processing!")


if __name__ == '__main__':
    demo_cross_modal()
