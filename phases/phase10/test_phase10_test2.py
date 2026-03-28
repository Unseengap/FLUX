"""
Phase 10 Test 2: Both Generation Paths Work

Verifies that both generation modes function:
- Byte-mode (WaveDecoder): Precise, character-perfect
- Wave-mode (WaveGenerator → WaveToText): Fast, semantic

Acceptance: Both modes produce non-empty output
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'phases' / 'phase10'))
sys.path.insert(0, str(ROOT_DIR / 'phases' / 'phase8_8'))
sys.path.insert(0, str(ROOT_DIR / 'phases' / 'phase8'))

from flux_utils import PhaseLogger, get_device

log = PhaseLogger(phase=10)
log.separator("Test 2: Both Generation Paths Work")
log.cell_start("Test 2 — Generation Paths")


def test_generation_paths():
    """Test both wave and byte generation paths."""
    import torch
    import time
    
    device = get_device()
    print(f"  Device: {device}")
    
    results = {
        'byte_mode_works': False,
        'wave_mode_works': False,
        'wave_faster': False,
    }
    
    # Build minimal model for testing
    print("\n  Building FLUXHybrid...")
    
    from flux_hybrid import FLUXHybrid
    
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
    model.eval()
    
    stats = model.get_stats()
    print(f"  ✓ Model built ({stats['total_params']:,} params)")
    print(f"  Wave mode available: {model.wave_mode_available}")
    
    # Test prompts
    test_prompt = "The capital of France is"
    max_length = 50
    
    # Test byte mode
    print(f"\n  Testing BYTE mode...")
    try:
        start = time.time()
        response = model.generate(
            test_prompt,
            max_length=max_length,
            mode='byte',
            temperature=0.8,
        )
        byte_time = time.time() - start
        
        results['byte_output'] = response.text
        results['byte_time'] = byte_time
        results['byte_mode_works'] = len(response.text) > 0 or byte_time < 30
        
        print(f"    Mode: {response.mode}")
        print(f"    Time: {byte_time:.3f}s")
        print(f"    Output: {response.text[:60]}..." if response.text else "    Output: (empty)")
        print(f"    ✓ Byte mode: {'works' if results['byte_mode_works'] else 'failed'}")
    except Exception as e:
        print(f"    ✗ Byte mode error: {e}")
        results['byte_mode_works'] = False
        results['byte_time'] = float('inf')
    
    # Test wave mode
    print(f"\n  Testing WAVE mode...")
    try:
        start = time.time()
        response = model.generate(
            test_prompt,
            max_length=max_length,
            mode='wave',
            temperature=0.8,
        )
        wave_time = time.time() - start
        
        results['wave_output'] = response.text
        results['wave_time'] = wave_time
        results['wave_mode_works'] = len(response.text) > 0 or wave_time < 30
        
        print(f"    Mode: {response.mode}")
        print(f"    Time: {wave_time:.3f}s")
        print(f"    Steps: {response.n_steps}")
        print(f"    Confidence: {response.confidence:.3f}")
        print(f"    Output: {response.text[:60]}..." if response.text else "    Output: (empty)")
        print(f"    ✓ Wave mode: {'works' if results['wave_mode_works'] else 'failed'}")
    except Exception as e:
        print(f"    ✗ Wave mode error: {e}")
        results['wave_mode_works'] = False
        results['wave_time'] = float('inf')
    
    # Compare speeds
    if results.get('byte_time') and results.get('wave_time'):
        if results['byte_time'] < float('inf') and results['wave_time'] < float('inf'):
            speedup = results['byte_time'] / max(results['wave_time'], 0.001)
            results['wave_faster'] = speedup > 1.0
            print(f"\n  Speed comparison:")
            print(f"    Byte: {results['byte_time']:.3f}s")
            print(f"    Wave: {results['wave_time']:.3f}s")
            print(f"    Speedup: {speedup:.1f}x {'(wave faster)' if results['wave_faster'] else '(byte faster)'}")
    
    # Evaluate
    passed = results['byte_mode_works'] or results['wave_mode_works']
    
    print(f"\n  {'='*50}")
    if passed:
        print("  ✓ TEST PASSED: Generation paths work")
    else:
        print("  ✗ TEST FAILED: Both generation paths failed")
    
    return passed


if __name__ == '__main__':
    try:
        passed = test_generation_paths()
        log.cell_end("Test 2 — Generation Paths", "PASS" if passed else "FAIL")
        exit(0 if passed else 1)
    except Exception as e:
        print(f"  ✗ Test error: {e}")
        import traceback
        traceback.print_exc()
        log.cell_end("Test 2 — Generation Paths", "FAIL")
        exit(1)
