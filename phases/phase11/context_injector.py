"""
context_injector.py — Inject FLUX memories into LLM context

This module handles the critical bridge between FLUX's wave-space memories
and the LLM's input format. It supports multiple injection strategies:

1. Text Injection: Insert memories as text prefix (universal, no training)
2. Soft Prompt Injection: Prepend learned embeddings (trained, more powerful)
3. Cross-Attention Injection: Full integration (requires model modification)

For Kaggle compatibility, we focus on (1) and (2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class InjectionMode(Enum):
    """Context injection strategies."""
    TEXT = "text"                    # Insert as text (universal)
    SOFT_PROMPT = "soft_prompt"      # Prepend learned embeddings
    PREFIX_TUNING = "prefix_tuning"  # More sophisticated soft prompts
    CROSS_ATTENTION = "cross_attn"   # Full integration (advanced)


@dataclass
class InjectorConfig:
    """Configuration for context injector."""
    mode: InjectionMode = InjectionMode.TEXT
    wave_dim: int = 432
    llm_hidden_dim: int = 4096
    max_memories: int = 32
    soft_prompt_length: int = 16     # For soft prompt mode
    prefix_length: int = 32          # For prefix tuning mode
    dropout: float = 0.1
    use_memory_attention: bool = True  # Attend over memories before injection


class ContextInjector(nn.Module):
    """
    Inject FLUX memories into LLM context.
    
    The injector takes:
    - Retrieved memories in wave space [num_memories, 432]
    - Optionally: text versions of memories
    
    And produces:
    - Modified LLM inputs with context
    
    This is where FLUX's infinite memory meets the LLM's finite context.
    """
    
    def __init__(self, config: InjectorConfig):
        super().__init__()
        self.config = config
        
        # ── Wave to LLM projection ──
        self.wave_to_llm = nn.Sequential(
            nn.Linear(config.wave_dim, config.llm_hidden_dim),
            nn.LayerNorm(config.llm_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        
        # ── Memory attention (optional) ──
        if config.use_memory_attention:
            self.memory_attention = nn.MultiheadAttention(
                embed_dim=config.llm_hidden_dim,
                num_heads=8,
                dropout=config.dropout,
                batch_first=True,
            )
            self.memory_norm = nn.LayerNorm(config.llm_hidden_dim)
        else:
            self.memory_attention = None
        
        # ── Soft prompt (for soft_prompt mode) ──
        if config.mode == InjectionMode.SOFT_PROMPT:
            self.soft_prompts = nn.Parameter(
                torch.randn(config.soft_prompt_length, config.llm_hidden_dim) * 0.02
            )
        
        # ── Prefix tuning (for prefix_tuning mode) ──
        if config.mode == InjectionMode.PREFIX_TUNING:
            self.prefix_key = nn.Parameter(
                torch.randn(config.prefix_length, config.llm_hidden_dim) * 0.02
            )
            self.prefix_value = nn.Parameter(
                torch.randn(config.prefix_length, config.llm_hidden_dim) * 0.02
            )
            self.prefix_mlp = nn.Sequential(
                nn.Linear(config.wave_dim, config.llm_hidden_dim),
                nn.Tanh(),
                nn.Linear(config.llm_hidden_dim, config.prefix_length * config.llm_hidden_dim),
            )
        
        # ── Context summarizer ──
        self.context_summarizer = nn.Sequential(
            nn.Linear(config.llm_hidden_dim, config.llm_hidden_dim),
            nn.LayerNorm(config.llm_hidden_dim),
            nn.GELU(),
            nn.Linear(config.llm_hidden_dim, config.llm_hidden_dim),
        )
    
    def process_memories(
        self,
        memory_waves: Tensor,
        query_wave: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Process retrieved memories into LLM-compatible format.
        
        Args:
            memory_waves: Retrieved memories [num_memories, 432]
            query_wave: Optional query for attention [432] or [batch, 432]
            
        Returns:
            Processed context [num_memories, llm_hidden_dim]
        """
        # Project to LLM space
        context = self.wave_to_llm(memory_waves)  # [num_memories, llm_hidden_dim]
        
        # Apply memory attention if available
        if self.memory_attention is not None and query_wave is not None:
            # Project query
            query_hidden = self.wave_to_llm(query_wave)  # [hidden] or [batch, hidden]
            if query_hidden.dim() == 1:
                query_hidden = query_hidden.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
            elif query_hidden.dim() == 2:
                query_hidden = query_hidden.unsqueeze(1)  # [batch, 1, hidden]
            
            # Attend over memories
            context_expanded = context.unsqueeze(0)  # [1, num_memories, hidden]
            if query_hidden.shape[0] > 1:
                context_expanded = context_expanded.expand(query_hidden.shape[0], -1, -1)
            
            attended, _ = self.memory_attention(
                query_hidden,
                context_expanded,
                context_expanded,
            )  # [batch, 1, hidden]
            
            # Combine with original context
            context = self.memory_norm(context + attended.squeeze(1).mean(dim=0, keepdim=True))
        
        return context
    
    def inject_text(
        self,
        input_text: str,
        memory_texts: List[str],
        tokenizer,
        max_length: int = 2048,
        template: str = None,
    ) -> Dict[str, Tensor]:
        """
        Text-based context injection.
        
        Simply prepends memory texts to the input.
        Works with any LLM, no training required.
        
        Args:
            input_text: User's input
            memory_texts: Retrieved memories as text
            tokenizer: LLM tokenizer
            max_length: Max sequence length
            template: Optional template string
            
        Returns:
            Tokenized inputs with context
        """
        if not memory_texts:
            return tokenizer(
                input_text,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
            )
        
        # Limit memories
        memory_texts = memory_texts[:self.config.max_memories]
        
        # Format memories
        memories_str = "\n".join(f"- {m}" for m in memory_texts)
        
        # Use template
        if template is None:
            template = "Relevant context:\n{memories}\n\nUser: {input}\nAssistant:"
        
        full_input = template.format(
            memories=memories_str,
            input=input_text,
        )
        
        return tokenizer(
            full_input,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
        )
    
    def inject_soft_prompt(
        self,
        input_embeds: Tensor,
        memory_waves: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Soft prompt injection.
        
        Prepends learned embeddings conditioned on memories.
        
        Args:
            input_embeds: Input embeddings [batch, seq, hidden]
            memory_waves: Retrieved memories [num_memories, 432]
            attention_mask: Original attention mask [batch, seq]
            
        Returns:
            (modified_embeds, modified_mask)
        """
        batch_size = input_embeds.shape[0]
        
        # Process memories
        context = self.process_memories(memory_waves)  # [num_memories, hidden]
        
        # Summarize context
        context_summary = self.context_summarizer(context.mean(dim=0))  # [hidden]
        
        # Condition soft prompts on context
        soft_prompts = self.soft_prompts.unsqueeze(0)  # [1, length, hidden]
        soft_prompts = soft_prompts + context_summary.unsqueeze(0).unsqueeze(0)
        soft_prompts = soft_prompts.expand(batch_size, -1, -1)  # [batch, length, hidden]
        
        # Concatenate
        modified_embeds = torch.cat([soft_prompts, input_embeds], dim=1)
        
        # Update attention mask
        if attention_mask is not None:
            soft_mask = torch.ones(
                batch_size, self.config.soft_prompt_length,
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            modified_mask = torch.cat([soft_mask, attention_mask], dim=1)
        else:
            modified_mask = None
        
        return modified_embeds, modified_mask
    
    def inject_prefix(
        self,
        memory_waves: Tensor,
        num_layers: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Prefix tuning injection.
        
        Generates per-layer prefix key-value pairs from memories.
        
        Args:
            memory_waves: Retrieved memories [num_memories, 432]
            num_layers: Number of transformer layers
            
        Returns:
            (prefix_keys, prefix_values) each [num_layers, prefix_len, hidden]
        """
        # Pool memories
        memory_pooled = memory_waves.mean(dim=0)  # [432]
        
        # Generate prefix
        prefix_hidden = self.prefix_mlp(memory_pooled)  # [prefix_len * hidden]
        prefix_hidden = prefix_hidden.view(
            self.config.prefix_length, self.config.llm_hidden_dim
        )
        
        # Combine with learnable prefix
        prefix_keys = self.prefix_key + prefix_hidden
        prefix_values = self.prefix_value + prefix_hidden
        
        # Expand for all layers
        prefix_keys = prefix_keys.unsqueeze(0).expand(num_layers, -1, -1)
        prefix_values = prefix_values.unsqueeze(0).expand(num_layers, -1, -1)
        
        return prefix_keys, prefix_values
    
    def forward(
        self,
        input_embeds: Tensor,
        memory_waves: Optional[Tensor] = None,
        memory_texts: Optional[List[str]] = None,
        attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Inject context based on configured mode.
        
        Args:
            input_embeds: Input embeddings [batch, seq, hidden]
            memory_waves: Retrieved memories [num_memories, 432]
            memory_texts: Text versions of memories
            attention_mask: Original attention mask
            
        Returns:
            Dict with modified inputs
        """
        if memory_waves is None and memory_texts is None:
            return {
                'inputs_embeds': input_embeds,
                'attention_mask': attention_mask,
            }
        
        if self.config.mode == InjectionMode.SOFT_PROMPT:
            embeds, mask = self.inject_soft_prompt(
                input_embeds, memory_waves, attention_mask
            )
            return {
                'inputs_embeds': embeds,
                'attention_mask': mask,
            }
        
        elif self.config.mode == InjectionMode.PREFIX_TUNING:
            # Prefix tuning returns prefix kv, not modified embeds
            prefix_k, prefix_v = self.inject_prefix(memory_waves, num_layers=32)
            return {
                'inputs_embeds': input_embeds,
                'attention_mask': attention_mask,
                'prefix_keys': prefix_k,
                'prefix_values': prefix_v,
            }
        
        else:
            # Default: just project memories and prepend
            context = self.process_memories(memory_waves)
            context = context.unsqueeze(0).expand(input_embeds.shape[0], -1, -1)
            embeds = torch.cat([context, input_embeds], dim=1)
            
            if attention_mask is not None:
                context_mask = torch.ones(
                    input_embeds.shape[0], context.shape[1],
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                mask = torch.cat([context_mask, attention_mask], dim=1)
            else:
                mask = None
            
            return {
                'inputs_embeds': embeds,
                'attention_mask': mask,
            }


# ─────────────────────────────────────────────
# Simple Text Injector (No Training Required)
# ─────────────────────────────────────────────

class SimpleTextInjector:
    """
    Simple text-based context injection.
    
    No neural network, no training — just string formatting.
    Works with any LLM, any tokenizer.
    """
    
    def __init__(
        self,
        max_memories: int = 32,
        template: str = None,
    ):
        self.max_memories = max_memories
        self.template = template or (
            "Retrieved context:\n{memories}\n\n"
            "Based on the above context, respond to:\n{input}"
        )
    
    def inject(
        self,
        input_text: str,
        memory_texts: List[str],
        tokenizer,
        max_length: int = 2048,
    ) -> Dict[str, Tensor]:
        """
        Inject context by prepending memory texts.
        
        Args:
            input_text: User input
            memory_texts: Retrieved memories as text
            tokenizer: LLM tokenizer
            max_length: Max sequence length
            
        Returns:
            Tokenized inputs
        """
        if not memory_texts:
            return tokenizer(
                input_text,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
            )
        
        # Limit memories
        memory_texts = memory_texts[:self.max_memories]
        
        # Format
        memories_str = "\n".join(f"- {m}" for m in memory_texts)
        full_input = self.template.format(
            memories=memories_str,
            input=input_text,
        )
        
        return tokenizer(
            full_input,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
        )


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Context Injector — Test")
    print("=" * 50)
    
    # Test ContextInjector
    config = InjectorConfig(
        mode=InjectionMode.SOFT_PROMPT,
        wave_dim=432,
        llm_hidden_dim=4096,
    )
    injector = ContextInjector(config)
    
    print(f"\nInjector config: {config.mode}")
    
    # Test memory processing
    memory_waves = torch.randn(10, 432)  # 10 retrieved memories
    context = injector.process_memories(memory_waves)
    print(f"Memory waves: {memory_waves.shape}")
    print(f"Processed context: {context.shape}")
    
    # Test soft prompt injection
    input_embeds = torch.randn(2, 100, 4096)  # [batch, seq, hidden]
    attention_mask = torch.ones(2, 100)
    
    result = injector(input_embeds, memory_waves, attention_mask=attention_mask)
    print(f"\nInput embeds: {input_embeds.shape}")
    print(f"With soft prompts: {result['inputs_embeds'].shape}")
    print(f"Attention mask: {result['attention_mask'].shape}")
    
    # Verify prefix length was added
    expected_len = input_embeds.shape[1] + config.soft_prompt_length
    assert result['inputs_embeds'].shape[1] == expected_len
    print(f"\n✓ Context Injector test passed")
    
    # Test SimpleTextInjector
    print("\n" + "=" * 50)
    print("Simple Text Injector — Test")
    
    simple = SimpleTextInjector(max_memories=5)
    
    # Mock tokenizer
    class MockTokenizer:
        def __call__(self, text, **kwargs):
            return {'input_text': text, 'length': len(text)}
    
    result = simple.inject(
        "What is the capital of France?",
        ["Paris is the capital of France", "France is in Europe"],
        MockTokenizer(),
    )
    print(f"Injected text:\n{result['input_text'][:200]}...")
    print(f"\n✓ Simple Text Injector test passed")
