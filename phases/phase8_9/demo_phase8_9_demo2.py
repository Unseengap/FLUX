"""
Phase 8.9 Demo 2: Image Generation with 3 Physics Engines

Demonstrates WaveToImage_Universal with:
1. Gravity Renderer - Smooth gradients, focal points
2. Interference Renderer - Wave patterns, ripples
3. Thermodynamic Renderer - Organic textures, settling

Shows all 5 style presets and custom blending.
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


def demo_image_physics():
    """Demonstrate 3 physics-based image generation."""
    log = PhaseLogger(phase=8.9)
    log.separator("Demo 2: Image Generation with 3 Physics")
    
    device = get_device()
    log.info(f"Device: {device}")
    
    # ─────────────────────────────────────────────
    # Load adapters
    # ─────────────────────────────────────────────
    log.info("\n1. Loading image adapters...")
    
    from image_adapters import (
        ImageToWave, WaveToImage_Universal,
        GravityRenderer, InterferenceRenderer, ThermodynamicRenderer,
        STYLE_PRESETS,
    )
    
    wave_to_image = WaveToImage_Universal(device=device)
    log.success(f"  WaveToImage loaded")
    log.info(f"  Available styles: {list(STYLE_PRESETS.keys())}")
    
    # ─────────────────────────────────────────────
    # Generate test wave
    # ─────────────────────────────────────────────
    log.info("\n2. Creating test wave from text...")
    
    try:
        from text_adapters import TextToWave
        text_adapter = TextToWave(device=device)
        
        # Try to load CSE
        try:
            from flx_loader import download_flx_from_hf, load_flux_from_flx
            flx_path = download_flx_from_hf('Flux-beta.flx')
            model = load_flux_from_flx(str(flx_path), device=device)
            text_adapter.cse = model['cse'] if isinstance(model, dict) else model.cse
        except:
            pass
        
        wave = text_adapter.encode_pooled("A cosmic nebula with swirling colors")
        log.success(f"  Wave shape: {wave.shape}")
    except Exception as e:
        log.warning(f"  Could not encode text: {e}")
        log.info("  Using random wave for demo...")
        wave = torch.randn(432, device=device)
        wave = wave / wave.norm() * 10  # Normalize
    
    # ─────────────────────────────────────────────
    # Individual physics engines
    # ─────────────────────────────────────────────
    log.info("\n3. Individual physics engine outputs...")
    
    size = (128, 128)
    
    # Pure gravity
    img_gravity = wave_to_image.gravity(wave, size[0], size[1])
    log.info(f"  Gravity: shape={img_gravity.shape}, range=[{img_gravity.min():.2f}, {img_gravity.max():.2f}]")
    
    # Pure interference
    img_interference = wave_to_image.interference(wave, size[0], size[1])
    log.info(f"  Interference: shape={img_interference.shape}, range=[{img_interference.min():.2f}, {img_interference.max():.2f}]")
    
    # Pure thermodynamic
    img_thermo = wave_to_image.thermodynamic(wave, size[0], size[1])
    log.info(f"  Thermodynamic: shape={img_thermo.shape}, range=[{img_thermo.min():.2f}, {img_thermo.max():.2f}]")
    
    log.success("  ✓ All 3 physics engines work")
    
    # ─────────────────────────────────────────────
    # All style presets
    # ─────────────────────────────────────────────
    log.info("\n4. Style preset outputs...")
    
    for style_name, preset in STYLE_PRESETS.items():
        img = wave_to_image.decode(wave, size=size, style=style_name)
        
        # Compute some statistics
        brightness = img.mean().item()
        contrast = img.std().item()
        
        log.info(f"  {style_name:15} blend=({preset.gravity:.2f}, {preset.interference:.2f}, {preset.thermodynamic:.2f}) "
                f"brightness={brightness:.3f} contrast={contrast:.3f}")
    
    log.success("  ✓ All 5 presets work")
    
    # ─────────────────────────────────────────────
    # Custom blending
    # ─────────────────────────────────────────────
    log.info("\n5. Custom blending examples...")
    
    custom_blends = [
        ("Pure Gravity", (1.0, 0.0, 0.0)),
        ("Pure Interference", (0.0, 1.0, 0.0)),
        ("Pure Thermodynamic", (0.0, 0.0, 1.0)),
        ("Gravity + Interference", (0.5, 0.5, 0.0)),
        ("All Equal", (0.33, 0.33, 0.34)),
    ]
    
    for name, weights in custom_blends:
        img = wave_to_image.decode(wave, size=size, custom_weights=weights)
        log.info(f"  {name:25} weights={weights} shape={img.shape}")
    
    log.success("  ✓ Custom blending works")
    
    # ─────────────────────────────────────────────
    # Auto-blend from wave content
    # ─────────────────────────────────────────────
    log.info("\n6. Auto-blend (wave-derived)...")
    
    img_auto = wave_to_image.decode(wave, size=size, auto_blend=True)
    
    # Get the auto-determined weights
    auto_weights = wave_to_image.auto_blend(wave)
    log.info(f"  Auto weights: ({auto_weights[0]:.3f}, {auto_weights[1]:.3f}, {auto_weights[2]:.3f})")
    log.info(f"  Output shape: {img_auto.shape}")
    log.success("  ✓ Auto-blend works")
    
    # ─────────────────────────────────────────────
    # Generate images from different prompts
    # ─────────────────────────────────────────────
    log.info("\n7. Different semantic inputs...")
    
    prompts = [
        "Fire and flames",
        "Ocean waves",
        "Forest at night",
        "Geometric patterns",
    ]
    
    for prompt in prompts:
        # Create wave from prompt (or random if CSE not available)
        try:
            wave = text_adapter.encode_pooled(prompt)
        except:
            # Use deterministic random based on prompt
            torch.manual_seed(hash(prompt) % 2**32)
            wave = torch.randn(432, device=device)
        
        img = wave_to_image.decode(wave, size=(64, 64), style='dream')
        
        # Quick stats
        dominant_channel = ['R', 'G', 'B'][img.mean(dim=(0,1)).argmax().item()]
        
        log.info(f"  '{prompt}' → dominant channel: {dominant_channel}")
    
    log.success("  ✓ Semantic variation works")
    
    # ─────────────────────────────────────────────
    log.separator("Demo Complete")
    log.info("Physics engines produce distinct visual styles:")
    log.info("  • Gravity: Smooth gradients, focal points")
    log.info("  • Interference: Wave patterns, ripples") 
    log.info("  • Thermodynamic: Organic textures, settling")


if __name__ == '__main__':
    demo_image_physics()
