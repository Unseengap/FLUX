"""
demo_phase11_demo1.py — Chat with FLUX Memory

Demonstrates the FLUX-Augmented LLM in action:
- Teach facts one-shot
- Retrieve from memory
- Generate fluent responses

This shows the core value proposition:
An LLM with persistent memory and one-shot learning.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from flux_utils import get_device


def demo_chat_with_memory():
    """Demo: Interactive chat with FLUX memory."""
    
    print(f"\n{'='*60}")
    print(f"Phase 11 Demo 1: Chat with FLUX Memory")
    print(f"{'='*60}\n")
    
    device = get_device()
    print(f"Device: {device}")
    
    # Check if we can import full model
    try:
        from flux_augmented_llm import FLUXAugmentedLLM, FLUXAugmentedConfig
        
        print("\nInitializing FLUX-Augmented LLM...")
        print("(This will download the LLM if not cached)")
        
        config = FLUXAugmentedConfig(
            llm_name="Qwen/Qwen2.5-3B-Instruct",
            load_in_4bit=True,
            top_k_retrieval=5,
        )
        
        model = FLUXAugmentedLLM(config, device=device)
        
        # ── Teach some facts ──
        print("\n" + "="*60)
        print("PHASE 1: Teaching Facts (One-Shot Learning)")
        print("="*60)
        
        facts = [
            "The FLUX architecture uses gravitational relevance instead of attention",
            "FLUX achieves O(log n) retrieval using spatial trees",
            "The wave dimension in FLUX is 432",
            "Attractors form at energy minima in the resonance field",
            "FLUX was created by Dectrick McGee",
        ]
        
        for fact in facts:
            model.teach(fact, verify=True)
        
        print(f"\n✓ Taught {len(facts)} facts")
        print(f"Memory size: {len(model.memory)}")
        
        # ── Chat with memory ──
        print("\n" + "="*60)
        print("PHASE 2: Chat with Memory Retrieval")
        print("="*60)
        
        questions = [
            "What is the wave dimension used in FLUX?",
            "How does FLUX achieve fast retrieval?",
            "Who created the FLUX architecture?",
        ]
        
        for question in questions:
            print(f"\n[User]: {question}")
            
            # Show retrieved memories
            texts, scores = model.retrieve(question)
            if texts:
                print(f"[Memory]: Retrieved {len(texts)} relevant memories")
                for i, (text, score) in enumerate(zip(texts[:2], scores[:2])):
                    print(f"  [{score:.2f}] {text[:60]}...")
            
            # Generate response
            response = model.chat(question, max_new_tokens=100)
            print(f"[Assistant]: {response}")
        
        # ── Test knowledge persistence ──
        print("\n" + "="*60)
        print("PHASE 3: Knowledge Persistence Test")
        print("="*60)
        
        # Teach a unique fact
        unique_fact = "The secret code is FLUX-42-GRAVITY"
        print(f"\nTeaching unique fact: {unique_fact}")
        model.teach(unique_fact)
        
        # Query it back
        response = model.chat("What is the secret code?", max_new_tokens=50)
        print(f"\n[User]: What is the secret code?")
        print(f"[Assistant]: {response}")
        
        if "FLUX-42-GRAVITY" in response or "42" in response:
            print("\n✓ Memory retrieval working — unique fact recalled!")
        else:
            print("\n⚠ Fact may not have been retrieved (check context)")
        
        # ── Summary ──
        print("\n" + "="*60)
        print("DEMO COMPLETE")
        print("="*60)
        print(f"\nSystem stats:")
        stats = model.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n✓ Demo 1 completed successfully")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nNote: This demo requires:")
        print("  - transformers library")
        print("  - bitsandbytes (for 4-bit quantization)")
        print("  - GPU with ~8GB VRAM")
        print("\nRun on Kaggle with T4 GPU for best results.")
        return False


if __name__ == "__main__":
    success = demo_chat_with_memory()
    sys.exit(0 if success else 1)
