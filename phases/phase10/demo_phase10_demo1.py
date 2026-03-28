"""
Phase 10 Demo 1: Hybrid Routing in Action

Demonstrates the TaskRouter making real-time decisions
between wave-mode and byte-mode generation based on
prompt characteristics.

Run: python demo_phase10_demo1.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'phases' / 'phase10'))
sys.path.insert(0, str(ROOT_DIR / 'phases' / 'phase8_8'))
sys.path.insert(0, str(ROOT_DIR / 'phases' / 'phase8'))

from flux_utils import get_device


def demo_hybrid_routing():
    """Demonstrate hybrid routing in action."""
    import torch
    from flux_hybrid import FLUXHybrid
    
    print("\n" + "="*70)
    print("  FLUX Phase 10: Hybrid Routing Demo")
    print("  Wave-mode (fast) vs Byte-mode (precise)")
    print("="*70)
    
    device = get_device()
    print(f"\n  Device: {device}")
    
    # Build model
    print("\n  Building FLUXHybrid model...")
    
    # Try loading from .flx first
    flx_paths = [
        ROOT_DIR / 'checkpoints' / 'Flux-capable.flx',
        ROOT_DIR / 'checkpoints' / 'Flux-beta.flx',
    ]
    
    model = None
    for flx_path in flx_paths:
        if flx_path.exists():
            try:
                model = FLUXHybrid.from_flx(str(flx_path), device=device, verbose=True)
                print(f"  ✓ Loaded from {flx_path.name}")
                break
            except Exception as e:
                print(f"  ⚠ Failed to load {flx_path.name}: {e}")
    
    if model is None:
        # Build minimal model
        config = {
            'wave_dim': 432,
            'field_h': 16,
            'field_w': 16,
            'field_d': 16,
            'field_features': 512,
        }
        model = FLUXHybrid(config=config, device=device)
        model._init_wave_modules()
        model = model.to(device)
        print("  ✓ Built fresh model")
    
    model.eval()
    stats = model.get_stats()
    print(f"  Total params: {stats['total_params']:,}")
    print(f"  Wave mode available: {model.wave_mode_available}")
    
    # Demo prompts with expected routing
    demo_prompts = [
        # Wave-expected (semantic/conversational)
        ("Explain artificial intelligence", "Semantic explanation"),
        ("What is the meaning of life?", "Philosophical question"),
        ("Tell me about the solar system", "Educational content"),
        ("Summarize machine learning", "Summary request"),
        
        # Byte-expected (precision needed)
        ("```python\ndef add(a, b):\n    return a + b\n```", "Code block"),
        ("def factorial(n):", "Python function"),
        ("https://github.com/example/flux", "URL"),
        ("Write a function that checks if a number is prime", "Code request"),
        
        # Edge cases
        ("Hello!", "Short greeting"),
        ("Can you help me understand how neural networks process information over time?", "Long question"),
    ]
    
    print("\n" + "-"*70)
    print("  ROUTING DEMONSTRATIONS")
    print("-"*70)
    
    for prompt, description in demo_prompts:
        print(f"\n  📝 {description}")
        print(f"     Prompt: \"{prompt[:50]}{'...' if len(prompt) > 50 else ''}\"")
        
        # Generate with auto routing
        response = model.generate(prompt, max_length=40, mode='auto')
        
        # Display routing decision
        mode_icon = "🌊" if response.mode == 'wave' else "🔢"
        print(f"     {mode_icon} Mode: {response.mode.upper()}")
        print(f"     📊 Reason: {response.mode_reason}")
        print(f"     ⏱  Time: {response.generation_time:.3f}s")
        
        if response.text:
            display_text = response.text.replace('\n', ' ')[:60]
            print(f"     💬 Output: \"{display_text}...\"")
        
        if response.mode == 'wave' and response.confidence > 0:
            print(f"     📈 Confidence: {response.confidence:.2f}")
    
    # Show routing statistics
    print("\n" + "-"*70)
    print("  ROUTING STATISTICS")
    print("-"*70)
    
    router_stats = model.task_router.get_stats()
    total = router_stats['total_routes']
    
    print(f"\n  Total decisions: {total}")
    print(f"  Wave mode: {router_stats['route_counts']['wave']} ({100*router_stats['wave_ratio']:.0f}%)")
    print(f"  Byte mode: {router_stats['route_counts']['byte']} ({100*router_stats['byte_ratio']:.0f}%)")
    
    print("\n  Reason breakdown:")
    for reason, count in sorted(router_stats['reason_counts'].items(), key=lambda x: -x[1]):
        print(f"    • {reason}: {count}")
    
    print("\n" + "="*70)
    print("  ✓ Demo complete")
    print("="*70)


if __name__ == '__main__':
    demo_hybrid_routing()
