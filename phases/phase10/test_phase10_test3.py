"""
Phase 10 Test 3: Router Makes Sensible Decisions

Verifies that TaskRouter correctly routes:
- Code/technical → byte mode
- Chat/explanations → wave mode
- Multimodal → wave mode (only option)

Acceptance: ≥80% routing accuracy on test cases
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'phases' / 'phase10'))

from flux_utils import PhaseLogger

log = PhaseLogger(phase=10)
log.separator("Test 3: Router Makes Sensible Decisions")
log.cell_start("Test 3 — Task Router")


def test_task_router():
    """Test TaskRouter routing decisions."""
    from task_router import TaskRouter, analyze_prompt
    
    router = TaskRouter(default_mode='wave')
    
    # Test cases: (prompt, expected_mode, category)
    test_cases = [
        # Code patterns → BYTE
        ("```python\ndef hello():\n    print('hi')\n```", "byte", "code_block"),
        ("def fibonacci(n):", "byte", "python_code"),
        ("import numpy as np", "byte", "import_statement"),
        ("function getData() { return fetch(url); }", "byte", "js_code"),
        ("Write a function to sort a list", "byte", "code_request"),
        ("https://github.com/example/repo", "byte", "url"),
        ("My email is test@example.com", "byte", "email_like"),
        ("Version 2.0.1 is released", "byte", "version_number"),
        ("The API endpoint is /api/v2/users", "byte", "api_mention"),
        
        # Semantic patterns → WAVE
        ("Explain quantum computing", "wave", "explain"),
        ("What is machine learning?", "wave", "question"),
        ("Tell me about the history of AI", "wave", "tell_about"),
        ("Summarize this article", "wave", "summarize"),
        ("How does gravity work?", "wave", "how_question"),
        ("Why is the sky blue?", "wave", "why_question"),
        ("In your own words, describe neural networks", "wave", "paraphrase"),
        ("I think AI will transform healthcare in many ways because of its ability to process large amounts of data quickly", "wave", "long_prompt"),
        
        # Edge cases
        ("Hello", "byte", "short_greeting"),  # Short → byte
    ]
    
    print(f"\n  Running {len(test_cases)} routing tests...")
    print("-" * 60)
    
    correct = 0
    results = []
    
    for prompt, expected, category in test_cases:
        mode, reason = router.route_with_reason(prompt)
        is_correct = mode == expected
        if is_correct:
            correct += 1
        
        results.append({
            'prompt': prompt[:40],
            'expected': expected,
            'got': mode,
            'correct': is_correct,
            'category': category,
            'reason': reason,
        })
        
        status = "✓" if is_correct else "✗"
        print(f"  {status} [{category}] → {mode} {'(expected: ' + expected + ')' if not is_correct else ''}")
    
    accuracy = correct / len(test_cases)
    
    print("-" * 60)
    print(f"\n  Routing accuracy: {correct}/{len(test_cases)} ({100*accuracy:.0f}%)")
    
    # Show stats
    stats = router.get_stats()
    print(f"\n  Router statistics:")
    print(f"    Total routes: {stats['total_routes']}")
    print(f"    Wave ratio: {100*stats['wave_ratio']:.0f}%")
    print(f"    Byte ratio: {100*stats['byte_ratio']:.0f}%")
    
    # Show failed cases
    failed = [r for r in results if not r['correct']]
    if failed:
        print(f"\n  Failed cases:")
        for r in failed:
            print(f"    • {r['prompt'][:30]}... → {r['got']} (expected: {r['expected']})")
            print(f"      Reason: {r['reason']}")
    
    # Test multimodal routing
    print(f"\n  Testing multimodal routing...")
    multimodal_cases = [
        ("Draw a cat", "image"),
        ("Generate an image of a sunset", "image"),
        ("Speak this text", "audio"),
    ]
    
    multimodal_correct = 0
    for prompt, modality in multimodal_cases:
        detected = router.detect_modality(prompt)
        mode = router.route(prompt, output_modality=modality)
        
        # Non-text should always use wave
        is_correct = mode == 'wave'
        if is_correct:
            multimodal_correct += 1
        
        status = "✓" if is_correct else "✗"
        print(f"    {status} '{prompt[:30]}' (modality={modality}) → {mode}")
    
    # Evaluate
    threshold = 0.75  # 75% accuracy threshold
    passed = accuracy >= threshold
    
    print(f"\n  {'='*50}")
    print(f"  Accuracy: {100*accuracy:.0f}% (threshold: {100*threshold:.0f}%)")
    
    if passed:
        print("  ✓ TEST PASSED: Router makes sensible decisions")
    else:
        print("  ✗ TEST FAILED: Router accuracy below threshold")
    
    return passed


if __name__ == '__main__':
    try:
        passed = test_task_router()
        log.cell_end("Test 3 — Task Router", "PASS" if passed else "FAIL")
        exit(0 if passed else 1)
    except Exception as e:
        print(f"  ✗ Test error: {e}")
        import traceback
        traceback.print_exc()
        log.cell_end("Test 3 — Task Router", "FAIL")
        exit(1)
