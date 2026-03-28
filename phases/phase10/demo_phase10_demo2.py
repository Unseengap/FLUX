"""
Phase 10 Demo 2: Wave vs Byte Comparison

Side-by-side comparison of wave-mode and byte-mode generation
on the same prompts, showing:
- Speed difference (~6x faster for wave)
- Output quality differences
- Use case recommendations

Run: python demo_phase10_demo2.py
"""

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'phases' / 'phase10'))
sys.path.insert(0, str(ROOT_DIR / 'phases' / 'phase8_8'))
sys.path.insert(0, str(ROOT_DIR / 'phases' / 'phase8'))

from flux_utils import get_device


def demo_wave_vs_byte():
    """Compare wave and byte modes side by side."""
    import torch
    from flux_hybrid import FLUXHybrid
    
    print("\n" + "="*70)
    print("  FLUX Phase 10: Wave vs Byte Comparison")
    print("="*70)
    
    device = get_device()
    print(f"\n  Device: {device}")
    
    # Build model
    print("\n  Building FLUXHybrid model...")
    
    config = {
        'wave_dim': 432,
        'field_h': 32,
        'field_w': 32,
        'field_d': 32,
        'field_features': 512,
    }
    
    model = FLUXHybrid(config=config, device=device)
    model._init_wave_modules()
    model = model.to(device)
    model.eval()
    
    stats = model.get_stats()
    print(f"  ✓ Model built ({stats['total_params']:,} params)")
    
    # Test prompts
    prompts = [
        "The future of artificial intelligence is",
        "Explain quantum computing:",
        "Write a haiku about nature:",
        "def fibonacci(n):",
        "The capital of France is",
    ]
    
    print("\n" + "-"*70)
    print("  SIDE-BY-SIDE COMPARISON")
    print("-"*70)
    
    wave_times = []
    byte_times = []
    
    for prompt in prompts:
        print(f"\n  📝 Prompt: \"{prompt}\"")
        print("  " + "─"*60)
        
        # Wave mode
        print("\n  🌊 WAVE MODE (fast, semantic)")
        start = time.time()
        wave_response = model.generate(prompt, max_length=50, mode='wave', temperature=0.7)
        wave_time = time.time() - start
        wave_times.append(wave_time)
        
        wave_text = wave_response.text.replace('\n', ' ')[:70] if wave_response.text else "(no output)"
        print(f"     Time: {wave_time:.3f}s")
        print(f"     Steps: {wave_response.n_steps}")
        print(f"     Confidence: {wave_response.confidence:.2f}")
        print(f"     Output: \"{wave_text}\"")
        
        # Byte mode
        print("\n  🔢 BYTE MODE (precise)")
        start = time.time()
        byte_response = model.generate(prompt, max_length=50, mode='byte', temperature=0.7)
        byte_time = time.time() - start
        byte_times.append(byte_time)
        
        byte_text = byte_response.text.replace('\n', ' ')[:70] if byte_response.text else "(no output)"
        print(f"     Time: {byte_time:.3f}s")
        print(f"     Steps: {byte_response.n_steps}")
        print(f"     Output: \"{byte_text}\"")
        
        # Comparison
        if wave_time > 0 and byte_time > 0:
            if wave_time < byte_time:
                speedup = byte_time / wave_time
                print(f"\n  ⚡ Wave is {speedup:.1f}x faster")
            else:
                speedup = wave_time / byte_time
                print(f"\n  ⚡ Byte is {speedup:.1f}x faster")
    
    # Overall statistics
    print("\n" + "="*70)
    print("  OVERALL STATISTICS")
    print("="*70)
    
    avg_wave = sum(wave_times) / len(wave_times)
    avg_byte = sum(byte_times) / len(byte_times)
    overall_speedup = avg_byte / max(avg_wave, 0.001)
    
    print(f"\n  Average times:")
    print(f"    Wave mode: {avg_wave:.3f}s")
    print(f"    Byte mode: {avg_byte:.3f}s")
    print(f"    Speedup:   {overall_speedup:.1f}x")
    
    print(f"\n  Recommended use cases:")
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │  🌊 WAVE MODE (fast)      │  🔢 BYTE MODE (precise)     │")
    print("  ├─────────────────────────────────────────────────────────┤")
    print("  │  • Chat & conversation    │  • Code generation          │")
    print("  │  • Explanations           │  • URLs & technical refs    │")
    print("  │  • Summaries              │  • Exact quotes             │")
    print("  │  • Creative writing       │  • Structured data          │")
    print("  │  • Brainstorming          │  • Mathematical formulas    │")
    print("  └─────────────────────────────────────────────────────────┘")
    
    print("\n" + "="*70)
    print("  ✓ Comparison complete")
    print("="*70)


if __name__ == '__main__':
    demo_wave_vs_byte()
