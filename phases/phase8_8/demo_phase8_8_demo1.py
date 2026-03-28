"""
Phase 8.8 Demo 1: Load Flux-beta.flx and Test Adapters

Demonstrates:
- Downloading .flx from HuggingFace Hub
- Loading FLUXLarge model
- Text and grid encoding
- Cross-modal wave space operations
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
from flux_utils import get_device


def main():
    print("\n" + "="*60)
    print("  Phase 8.8 Demo 1: Load .flx and Test Adapters")
    print("="*60)
    
    device = get_device()
    print(f"\n  Device: {device}")
    
    # ─────────────────────────────────────────────
    # Step 1: Load Flux-beta.flx
    # ─────────────────────────────────────────────
    print("\n  Step 1: Loading Flux-beta.flx...")
    
    from flx_loader import load_flux_from_flx, DEFAULT_FLX_PATH
    
    model = load_flux_from_flx(
        path=DEFAULT_FLX_PATH,
        device=device,
        verbose=True,
        auto_download=True,  # Download from HF if missing
    )
    
    stats = model.get_stats()
    print(f"\n  ✓ Model loaded: {stats.total_params:,} params")
    
    # ─────────────────────────────────────────────
    # Step 2: Initialize Adapters
    # ─────────────────────────────────────────────
    print("\n  Step 2: Initializing adapters...")
    
    from text_adapters import TextToWave
    from grid_adapters import GridToWave, WaveToGrid
    
    text_to_wave = TextToWave(device=device, cse=model.cse)
    grid_to_wave = GridToWave(device=device)
    wave_to_grid = WaveToGrid(device=device)
    
    print("  ✓ Adapters initialized")
    
    # ─────────────────────────────────────────────
    # Step 3: Demo Text Encoding
    # ─────────────────────────────────────────────
    print("\n  Step 3: Text encoding demo...")
    
    texts = [
        "Hello, World!",
        "FLUX encodes any text as semantic waves.",
        "こんにちは",
    ]
    
    for text in texts:
        wave = text_to_wave.encode(text)
        print(f'    "{text}" → wave {list(wave.shape)}')
    
    # ─────────────────────────────────────────────
    # Step 4: Demo Grid Encoding
    # ─────────────────────────────────────────────
    print("\n  Step 4: Grid encoding demo...")
    
    grid = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    
    wave_h = grid_to_wave.encode(grid, mode='holistic')
    wave_s = grid_to_wave.encode(grid, mode='spatial')
    
    print(f"    3×3 grid → holistic wave {list(wave_h.shape)}")
    print(f"    3×3 grid → spatial waves {list(wave_s.shape)}")
    
    # ─────────────────────────────────────────────
    # Step 5: Demo Transformation Extraction
    # ─────────────────────────────────────────────
    print("\n  Step 5: Transformation extraction (for ARC)...")
    
    input_grid = [[1, 0], [0, 0]]
    output_grid = [[1, 1], [1, 1]]
    
    in_wave, out_wave, delta = grid_to_wave.encode_pair(input_grid, output_grid)
    
    print(f"    Input:  {input_grid}")
    print(f"    Output: {output_grid}")
    print(f"    Delta wave norm: {delta.norm().item():.4f}")
    
    # ─────────────────────────────────────────────
    # Step 6: Demo Full Pipeline
    # ─────────────────────────────────────────────
    print("\n  Step 6: Full pipeline (query → generate)...")
    
    query = "What is the capital of France?"
    print(f'    Query: "{query}"')
    
    # Encode
    query_wave = text_to_wave.encode(query)
    print(f"    Encoded: {list(query_wave.shape)}")
    
    # Generate
    with torch.no_grad():
        response = model.generate(query, max_length=50, temperature=0.8)
    print(f"    Response: \"{response[:80]}...\"")
    
    print("\n" + "="*60)
    print("  ✓ Phase 8.8 Demo 1 Complete!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
