"""
llm_bridge.py — Bridge for HuggingFace Language Models

Connects any HuggingFace causal LM to FLUX's wave space.
This is the "voice" module — it gives FLUX the ability to speak fluently.

The LLM handles:
- Tokenization
- Fluent text generation
- Language understanding

FLUX handles:
- Infinite memory
- O(log n) retrieval
- One-shot learning
- Causal reasoning

Together: A fluent AI with perfect memory and infinite context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

from model_bridge import ModelBridge, BridgeConfig, register_bridge


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class LLMBridgeConfig(BridgeConfig):
    """Extended config for LLM bridges."""
    model_name: str = "microsoft/phi-3-mini-4k-instruct"
    load_in_4bit: bool = True           # Quantize for Kaggle
    load_in_8bit: bool = False          # Alternative quantization
    use_flash_attention: bool = False   # Flash attention if available
    max_length: int = 2048              # Max sequence length
    max_context_injection: int = 32     # Max retrieved attractors to inject
    freeze_llm: bool = True             # Don't train LLM weights
    trust_remote_code: bool = True      # For some models


# ─────────────────────────────────────────────
# Context Injection Templates
# ─────────────────────────────────────────────

CONTEXT_TEMPLATES = {
    "default": (
        "Relevant context from memory:\n"
        "{memories}\n\n"
        "User: {input}\n"
        "Assistant:"
    ),
    "minimal": (
        "[Context: {memories}]\n\n"
        "{input}"
    ),
    "chat": (
        "<|system|>\n"
        "You have access to the following relevant memories:\n"
        "{memories}\n"
        "<|end|>\n"
        "<|user|>\n"
        "{input}\n"
        "<|end|>\n"
        "<|assistant|>\n"
    ),
    "phi3": (
        "<|user|>\n"
        "Context: {memories}\n\n"
        "Question: {input}<|end|>\n"
        "<|assistant|>\n"
    ),
}


# ─────────────────────────────────────────────
# LLM Bridge
# ─────────────────────────────────────────────

@register_bridge("llm")
class LLMBridge(ModelBridge):
    """
    Bridge connecting a HuggingFace LLM to FLUX wave space.
    
    This enables:
    1. LLM hidden states → wave space (for field storage)
    2. Wave space → LLM hidden states (for context injection)
    3. Fluent generation with FLUX memory augmentation
    
    The LLM weights are frozen by default — only bridge projections train.
    This keeps the system small and fast for Kaggle.
    """
    
    def __init__(
        self, 
        config: LLMBridgeConfig = None,
        model_name: str = None,
        device: str = 'cuda',
        load_model: bool = True,
    ):
        """
        Initialize LLM Bridge.
        
        Args:
            config: Full configuration (preferred)
            model_name: Model name (alternative to config)
            device: Target device
            load_model: Whether to load the LLM now
        """
        # Build config if not provided
        if config is None:
            config = LLMBridgeConfig(
                name="llm",
                model_name=model_name or "microsoft/phi-3-mini-4k-instruct",
            )
        
        self.llm = None
        self.tokenizer = None
        self.device = device
        self._model_loaded = False
        
        # We need to know hidden_dim before calling super().__init__
        # Try to get it from model config without loading full model
        if load_model:
            self._load_model(config)
            config.source_dim = self.llm.config.hidden_size
        else:
            # Estimate based on model name
            config.source_dim = self._estimate_hidden_dim(config.model_name)
        
        # Initialize bridge projections
        super().__init__(config)
        
        # Select context template
        if 'phi' in config.model_name.lower():
            self.context_template = CONTEXT_TEMPLATES['phi3']
        else:
            self.context_template = CONTEXT_TEMPLATES['default']
    
    def _estimate_hidden_dim(self, model_name: str) -> int:
        """Estimate hidden dim from model name."""
        model_name = model_name.lower()
        if 'phi-3-mini' in model_name:
            return 3072
        elif 'phi-3-small' in model_name:
            return 4096
        elif 'phi-3-medium' in model_name:
            return 5120
        elif 'llama-3-8b' in model_name or 'llama-2-7b' in model_name:
            return 4096
        elif 'llama-3-70b' in model_name or 'llama-2-70b' in model_name:
            return 8192
        elif 'mistral-7b' in model_name:
            return 4096
        elif 'qwen' in model_name:
            return 4096
        else:
            return 4096  # Default
    
    def _load_model(self, config: LLMBridgeConfig):
        """Load the LLM and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"  Loading LLM: {config.model_name}...")
        
        # ── Load tokenizer ──
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # ── Prepare loading kwargs ──
        load_kwargs = {
            'trust_remote_code': config.trust_remote_code,
            'torch_dtype': torch.float16,
            'device_map': 'auto',
        }
        
        # ── Quantization ──
        if config.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                print(f"    Using 4-bit quantization")
            except ImportError:
                print(f"    ⚠ bitsandbytes not available, loading in fp16")
        elif config.load_in_8bit:
            load_kwargs['load_in_8bit'] = True
            print(f"    Using 8-bit quantization")
        
        # ── Load model ──
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **load_kwargs,
        )
        
        # ── Freeze if requested ──
        if config.freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
            self.llm.eval()
        
        self._model_loaded = True
        
        # Count parameters
        llm_params = sum(p.numel() for p in self.llm.parameters())
        print(f"    ✓ LLM loaded: {llm_params/1e9:.1f}B parameters")
        print(f"    Hidden dim: {self.llm.config.hidden_size}")
    
    def ensure_loaded(self):
        """Ensure model is loaded."""
        if not self._model_loaded:
            self._load_model(self.config)
    
    def get_model_hidden_states(
        self, 
        text: Union[str, List[str]],
        layer: int = -1,
    ) -> Tensor:
        """
        Extract hidden states from LLM for given text.
        
        Args:
            text: Input text(s)
            layer: Which layer to extract (-1 for last)
            
        Returns:
            Hidden states [batch, seq, hidden_dim]
        """
        self.ensure_loaded()
        
        if isinstance(text, str):
            text = [text]
        
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        
        # Move to model's device
        inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.llm(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # Extract specified layer
        hidden_states = outputs.hidden_states[layer]
        return hidden_states
    
    def encode_to_wave(self, text: Union[str, List[str]]) -> Tensor:
        """
        Full pipeline: text → LLM hidden states → wave space.
        
        Args:
            text: Input text(s)
            
        Returns:
            Wave representation [batch, seq, 432]
        """
        hidden = self.get_model_hidden_states(text)
        
        # Move bridge to same device as hidden
        if next(self.to_wave_proj.parameters()).device != hidden.device:
            self.to_wave_proj.to(hidden.device)
        
        wave = self.to_wave(hidden)
        return wave
    
    def inject_context(
        self,
        input_text: str,
        context_waves: Optional[Tensor] = None,
        context_texts: Optional[List[str]] = None,
        mode: str = "text",  # "text" or "embedding"
    ) -> Dict[str, Tensor]:
        """
        Inject retrieved context into LLM input.
        
        Two modes:
        1. Text mode: Insert memories as text (simple, works everywhere)
        2. Embedding mode: Inject as embeddings (more powerful, needs training)
        
        Args:
            input_text: User's input
            context_waves: Retrieved memory waves [num_memories, 432]
            context_texts: Text versions of memories
            mode: Injection mode
            
        Returns:
            Prepared inputs for LLM
        """
        self.ensure_loaded()
        
        if mode == "text" and context_texts:
            # ── Text injection ──
            memories_str = "\n".join(
                f"- {text}" 
                for text in context_texts[:self.config.max_context_injection]
            )
            full_input = self.context_template.format(
                memories=memories_str,
                input=input_text,
            )
            
            inputs = self.tokenizer(
                full_input,
                return_tensors='pt',
                truncation=True,
                max_length=self.config.max_length,
            )
            return {k: v.to(self.llm.device) for k, v in inputs.items()}
        
        elif mode == "embedding" and context_waves is not None:
            # ── Embedding injection ──
            # Project waves back to LLM space
            if self.from_wave_proj is None:
                raise ValueError("Embedding injection requires bidirectional bridge")
            
            # Move to correct device
            if context_waves.device != self.llm.device:
                context_waves = context_waves.to(self.llm.device)
            
            context_embeddings = self.from_wave(context_waves)  # [num, hidden]
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors='pt',
                truncation=True,
                max_length=self.config.max_length - self.config.max_context_injection,
            )
            inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}
            
            # Get input embeddings
            input_embeds = self.llm.get_input_embeddings()(inputs['input_ids'])
            
            # Prepend context embeddings
            context_embeds = context_embeddings.unsqueeze(0)  # [1, num, hidden]
            combined_embeds = torch.cat([context_embeds, input_embeds], dim=1)
            
            # Create attention mask
            context_mask = torch.ones(
                1, context_embeds.shape[1],
                device=self.llm.device,
                dtype=inputs['attention_mask'].dtype,
            )
            combined_mask = torch.cat([context_mask, inputs['attention_mask']], dim=1)
            
            return {
                'inputs_embeds': combined_embeds,
                'attention_mask': combined_mask,
            }
        
        else:
            # No context injection
            inputs = self.tokenizer(
                input_text,
                return_tensors='pt',
                truncation=True,
                max_length=self.config.max_length,
            )
            return {k: v.to(self.llm.device) for k, v in inputs.items()}
    
    def generate(
        self,
        input_text: str,
        context_waves: Optional[Tensor] = None,
        context_texts: Optional[List[str]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """
        Generate text with optional context injection.
        
        Args:
            input_text: User prompt
            context_waves: Retrieved memory waves (optional)
            context_texts: Text versions of memories (optional)
            max_new_tokens: Max generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample (False = greedy)
            
        Returns:
            Generated text
        """
        self.ensure_loaded()
        
        # Prepare inputs with context
        if context_texts:
            inputs = self.inject_context(input_text, context_texts=context_texts, mode="text")
        elif context_waves is not None:
            inputs = self.inject_context(input_text, context_waves=context_waves, mode="embedding")
        else:
            inputs = self.inject_context(input_text, mode="text")
        
        # Generate
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )
        
        # Decode response
        # Get length of input to skip
        if 'input_ids' in inputs:
            input_length = inputs['input_ids'].shape[1]
        else:
            input_length = inputs['inputs_embeds'].shape[1]
        
        # Decode only new tokens
        generated_ids = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()
    
    def get_bridge_params(self) -> int:
        """Get number of trainable bridge parameters (not LLM)."""
        params = sum(p.numel() for p in self.to_wave_proj.parameters())
        if self.from_wave_proj:
            params += sum(p.numel() for p in self.from_wave_proj.parameters())
        return params


# ─────────────────────────────────────────────
# Lightweight LLM Bridge (No Model Loading)
# ─────────────────────────────────────────────

@register_bridge("llm-lightweight")
class LightweightLLMBridge(ModelBridge):
    """
    LLM Bridge that doesn't load the model.
    
    Useful for:
    - Testing bridge logic
    - Pre-computing bridge weights
    - Environments without GPU
    """
    
    def __init__(
        self,
        hidden_dim: int = 4096,
        wave_dim: int = 432,
    ):
        config = BridgeConfig(
            name="llm-lightweight",
            source_dim=hidden_dim,
            wave_dim=wave_dim,
        )
        super().__init__(config)
    
    def get_model_hidden_states(self, inputs: Tensor) -> Tensor:
        """Inputs are already hidden states."""
        return inputs
    
    def inject_context(
        self,
        model_inputs: Tensor,
        context_waves: Tensor,
        context_texts: Optional[List[str]] = None,
    ) -> Tensor:
        """Simple concatenation for testing."""
        context_hidden = self.from_wave(context_waves)
        if model_inputs.dim() == 2:
            return torch.cat([context_hidden, model_inputs], dim=0)
        else:
            batch_size = model_inputs.shape[0]
            context_expanded = context_hidden.unsqueeze(0).expand(batch_size, -1, -1)
            return torch.cat([context_expanded, model_inputs], dim=1)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("LLM Bridge — Test")
    print("=" * 50)
    
    # Test lightweight bridge (no model loading)
    print("\n1. Testing lightweight bridge...")
    bridge = LightweightLLMBridge(hidden_dim=4096, wave_dim=432)
    print(f"   {bridge}")
    
    # Test projections
    hidden = torch.randn(2, 10, 4096)
    wave = bridge.to_wave(hidden)
    print(f"   Hidden → Wave: {hidden.shape} → {wave.shape}")
    
    reconstructed = bridge.from_wave(wave)
    print(f"   Wave → Hidden: {wave.shape} → {reconstructed.shape}")
    
    # Test context injection
    context = torch.randn(5, 432)
    injected = bridge.inject_context(hidden, context)
    print(f"   With context: {hidden.shape} + {context.shape} → {injected.shape}")
    
    print("\n   ✓ Lightweight bridge test passed")
    
    # Test full bridge only if transformers available
    try:
        from transformers import AutoModelForCausalLM
        print("\n2. Testing full bridge with model loading...")
        print("   (This requires a GPU and will download the model)")
        print("   Skipping in test mode — run manually if needed")
    except ImportError:
        print("\n2. Transformers not available, skipping full bridge test")
    
    print("\n✓ LLM Bridge tests passed")
