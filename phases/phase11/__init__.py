"""
Phase 11: FLUX-Augmented LLM — Universal Model Assembly

This phase introduces the ability to:
1. Connect any pretrained LLM to FLUX's memory system
2. Give LLMs infinite context via O(log n) retrieval
3. Enable one-shot learning without fine-tuning
4. Achieve zero catastrophic forgetting when adding knowledge
5. Create portable AI systems with .flx format

Key Components:
- ModelBridge: Abstract base for connecting any model to FLUX
- LLMBridge: Specific bridge for HuggingFace language models  
- ContextInjector: Inject retrieved memories into LLM context
- FLUXAugmentedLLM: Full system combining FLUX + LLM
"""

from .model_bridge import ModelBridge, BridgeConfig, TestBridge
from .llm_bridge import LLMBridge, LLMBridgeConfig, LightweightLLMBridge
from .context_injector import ContextInjector, InjectorConfig, SimpleTextInjector
from .flux_augmented_llm import FLUXAugmentedLLM, FLUXAugmentedConfig

__all__ = [
    # Base
    'ModelBridge',
    'BridgeConfig',
    'TestBridge',
    
    # LLM
    'LLMBridge',
    'LLMBridgeConfig',
    'LightweightLLMBridge',
    
    # Injection
    'ContextInjector',
    'InjectorConfig',
    'SimpleTextInjector',
    
    # Main
    'FLUXAugmentedLLM',
    'FLUXAugmentedConfig',
]
