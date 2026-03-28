"""
Phase 10 Demo 3: Multimodal Extension Points

Demonstrates the architecture for extending FLUX to
multiple output modalities using the same wave representation.

The insight: WaveGenerator produces modality-agnostic semantic waves.
Different "last-mile" decoders convert to:
- WaveToText: UTF-8 text
- WaveToImage: RGB images (Phase 8.8)
- WaveToAudio: Audio waveforms (future)
- WaveToMol: Molecular structures (future)

Run: python demo_phase10_demo3.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'phases' / 'phase10'))
sys.path.insert(0, str(ROOT_DIR / 'phases' / 'phase8_8'))
sys.path.insert(0, str(ROOT_DIR / 'phases' / 'phase8'))

from flux_utils import get_device


def demo_multimodal():
    """Demonstrate multimodal extension architecture."""
    import torch
    from wave_modules import WaveGenerator, WaveToText
    from task_router import TaskRouter
    
    print("\n" + "="*70)
    print("  FLUX Phase 10: Multimodal Extension Points")
    print("  One Wave Representation → Multiple Output Modalities")
    print("="*70)
    
    device = get_device()
    print(f"\n  Device: {device}")
    
    # Build wave modules
    print("\n  Building wave modules...")
    
    wave_generator = WaveGenerator(
        wave_dim=432,
        field_features=512,
    ).to(device)
    
    wave_to_text = WaveToText(
        wave_dim=432,
        hidden_dim=256,
    ).to(device)
    
    router = TaskRouter()
    
    print("  ✓ WaveGenerator built")
    print("  ✓ WaveToText built")
    print("  ✓ TaskRouter built")
    
    print("\n" + "-"*70)
    print("  THE MULTIMODAL ARCHITECTURE")
    print("-"*70)
    
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                     FLUXHybrid                                  │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │     Input Prompt                                                │
    │          │                                                      │
    │          ▼                                                      │
    │     ┌─────────┐     ┌──────────────┐                           │
    │     │   CSE   │ ──▶ │  Field Query │                           │
    │     └─────────┘     └──────────────┘                           │
    │                            │                                    │
    │                            ▼                                    │
    │                    ┌───────────────┐                           │
    │                    │ WaveGenerator │  ← Modality-agnostic      │
    │                    └───────────────┘                           │
    │                            │                                    │
    │                            ▼                                    │
    │            ┌─────── [N, 432] waves ───────┐                    │
    │            │               │               │                    │
    │            ▼               ▼               ▼                    │
    │     ┌───────────┐   ┌───────────┐   ┌───────────┐             │
    │     │WaveToText │   │WaveToImage│   │WaveToAudio│             │
    │     │  (text)   │   │  (RGB)    │   │  (audio)  │             │
    │     └───────────┘   └───────────┘   └───────────┘             │
    │            │               │               │                    │
    │            ▼               ▼               ▼                    │
    │         "Hello"      [3,256,256]     [48000,]                  │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    print("-"*70)
    print("  MODALITY DETECTION")
    print("-"*70)
    
    # Demo modality detection
    detection_tests = [
        "Explain quantum computing",
        "Draw a picture of a sunset over the ocean",
        "Generate an image of a futuristic city",
        "Speak this text aloud",
        "Create a molecule for aspirin",
        "Write Python code for sorting",
    ]
    
    print("\n  Testing modality detection:\n")
    
    for prompt in detection_tests:
        detected = router.detect_modality(prompt)
        mode = router.route(prompt, output_modality=detected)
        
        icon = {
            'text': '📝',
            'image': '🖼️',
            'audio': '🔊',
            'mol': '🧬',
        }.get(detected, '❓')
        
        print(f"  {icon} \"{prompt[:40]}...\"")
        print(f"      Modality: {detected}, Mode: {mode}")
    
    print("\n" + "-"*70)
    print("  WAVE GENERATION DEMO")
    print("-"*70)
    
    # Generate some waves
    print("\n  Generating semantic waves from field context...")
    
    field_context = torch.randn(512, device=device) * 0.1  # Mock field context
    
    result = wave_generator.generate(
        field_context=field_context,
        max_waves=10,
        temperature=0.8,
    )
    
    print(f"\n  Generated {result.n_steps} semantic waves")
    print(f"  Stop reason: {result.stop_reason}")
    print(f"  Average confidence: {result.confidences.mean():.3f}")
    
    # Decode to text
    print("\n  Converting waves to text via WaveToText...")
    
    text_output = wave_to_text.decode_sequence(
        result.waves,
        temperature=0.8,
    )
    
    print(f"  Output: \"{text_output[:100]}...\"")
    
    print("\n" + "-"*70)
    print("  EXTENSION POINTS (Future Work)")
    print("-"*70)
    
    print("""
    The same waves can be decoded to different modalities:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │  MODALITY      │  DECODER         │  STATUS                    │
    ├─────────────────────────────────────────────────────────────────┤
    │  Text          │  WaveToText      │  ✓ Implemented             │
    │  Image         │  WaveToImage     │  ○ Phase 8.8 (partial)     │
    │  Audio         │  WaveToAudio     │  ○ Planned                 │
    │  Molecules     │  WaveToMol       │  ○ Planned                 │
    │  3D Scenes     │  WaveTo3D        │  ○ Future                  │
    │  Music         │  WaveToMusic     │  ○ Future                  │
    └─────────────────────────────────────────────────────────────────┘
    
    Key insight: The WaveGenerator learns to produce semantic waves.
    Each WaveToX decoder learns to interpret those waves for its modality.
    
    Training a new modality only requires:
    1. A WaveToX decoder architecture
    2. Paired wave-modality data
    3. Standard supervised training
    
    The core FLUX model (CSE + Field + WaveGenerator) stays frozen.
    """)
    
    print("\n" + "="*70)
    print("  ✓ Demo complete")
    print("="*70)


if __name__ == '__main__':
    demo_multimodal()
