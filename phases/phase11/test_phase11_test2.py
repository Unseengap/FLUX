"""
test_phase11_test2.py — Context Injection Test

Verifies that retrieved memories are correctly injected
into the LLM's context, enabling memory-augmented generation.

Test criteria:
- Text injection produces valid tokenized input
- Soft prompt injection adds correct number of tokens
- Memory content is preserved in injected context
"""

import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
from flux_utils import PhaseResults, get_device

from context_injector import ContextInjector, InjectorConfig, InjectionMode, SimpleTextInjector


def test_context_injection():
    """Test context injection mechanisms."""
    
    results = PhaseResults(phase=11, component_name="Context Injection")
    
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Phase 11 Test 2: Context Injection")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    all_passed = True
    
    # ── Test 1: Simple Text Injector ──
    print("Test 1: Simple Text Injector")
    print("-" * 50)
    
    injector = SimpleTextInjector(max_memories=5)
    
    # Mock tokenizer
    class MockTokenizer:
        def __call__(self, text, **kwargs):
            return {
                'input_text': text,
                'length': len(text),
                'truncated': len(text) > kwargs.get('max_length', 9999),
            }
    
    # Test injection
    memory_texts = [
        "The capital of France is Paris",
        "Python is a programming language",
        "FLUX uses gravitational relevance",
    ]
    
    result = injector.inject(
        input_text="What is the capital of France?",
        memory_texts=memory_texts,
        tokenizer=MockTokenizer(),
    )
    
    # Verify memories are in output
    text = result['input_text']
    memories_present = all(mem in text for mem in memory_texts)
    input_present = "What is the capital of France?" in text
    
    print(f"  Memories included: {memories_present}")
    print(f"  Input included: {input_present}")
    print(f"  Output length: {result['length']}")
    
    test1_passed = memories_present and input_present
    results.add_test("Text Injection", passed=test1_passed, score=1.0 if test1_passed else 0.0, threshold=1.0)
    all_passed = all_passed and test1_passed
    
    # ── Test 2: Context Injector with Soft Prompts ──
    print("\nTest 2: Soft Prompt Injection")
    print("-" * 50)
    
    config = InjectorConfig(
        mode=InjectionMode.SOFT_PROMPT,
        wave_dim=432,
        llm_hidden_dim=4096,
        soft_prompt_length=16,
    )
    injector = ContextInjector(config)
    injector.to(device)
    
    # Test memory processing
    memory_waves = torch.randn(5, 432, device=device)
    input_embeds = torch.randn(2, 100, 4096, device=device)
    attention_mask = torch.ones(2, 100, device=device)
    
    output = injector(
        input_embeds=input_embeds,
        memory_waves=memory_waves,
        attention_mask=attention_mask,
    )
    
    # Verify shapes
    expected_seq_len = 100 + 16  # original + soft prompts
    actual_seq_len = output['inputs_embeds'].shape[1]
    
    print(f"  Input seq length: 100")
    print(f"  Soft prompt length: 16")
    print(f"  Output seq length: {actual_seq_len}")
    print(f"  Expected: {expected_seq_len}")
    
    test2_passed = actual_seq_len == expected_seq_len
    results.add_test("Soft Prompt Injection", passed=test2_passed, score=1.0 if test2_passed else 0.0, threshold=1.0)
    all_passed = all_passed and test2_passed
    
    # ── Test 3: Memory Processing ──
    print("\nTest 3: Memory Processing")
    print("-" * 50)
    
    query_wave = torch.randn(432, device=device)
    processed = injector.process_memories(memory_waves, query_wave)
    
    print(f"  Input waves: {memory_waves.shape}")
    print(f"  Query wave: {query_wave.shape}")
    print(f"  Processed: {processed.shape}")
    
    expected_shape = (5, 4096)  # num_memories, llm_hidden_dim
    test3_passed = tuple(processed.shape) == expected_shape
    print(f"  Expected shape: {expected_shape}")
    print(f"  Shape correct: {test3_passed}")
    
    results.add_test("Memory Processing", passed=test3_passed, score=1.0 if test3_passed else 0.0, threshold=1.0)
    all_passed = all_passed and test3_passed
    
    # ── Test 4: Gradient Flow ──
    print("\nTest 4: Gradient Flow Through Injector")
    print("-" * 50)
    
    memory_waves = torch.randn(5, 432, device=device, requires_grad=True)
    input_embeds = torch.randn(2, 100, 4096, device=device, requires_grad=True)
    
    output = injector(
        input_embeds=input_embeds,
        memory_waves=memory_waves,
    )
    
    loss = output['inputs_embeds'].sum()
    loss.backward()
    
    has_grad = memory_waves.grad is not None and memory_waves.grad.abs().sum() > 0
    print(f"  Gradients flow to memory waves: {has_grad}")
    
    results.add_test("Gradient Flow", passed=has_grad, score=1.0 if has_grad else 0.0, threshold=1.0)
    all_passed = all_passed and has_grad
    
    # ── Summary ──
    results.save()
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ TEST PASSED — Context injection verified")
    else:
        print("✗ TEST FAILED — Some checks did not pass")
    print(f"{'='*60}")
    
    return all_passed


if __name__ == "__main__":
    success = test_context_injection()
    sys.exit(0 if success else 1)
