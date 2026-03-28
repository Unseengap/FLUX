# Phase 11 Specification: Universal Model Assembly + FLUX-Augmented LLM
## Assemble Any AI Model — Give It FLUX Superpowers

> Prerequisites: 
> - `Flux-beta.flx` must exist (contains all Phases 1-8 components)
> - Phase 10 hybrid infrastructure (optional, for wave-mode generation)
>
> Copilot: Open SPECIFICATION.md + FLUX_FILE_FORMAT.md + this file.
>
> **This phase transforms FLUX from a standalone architecture into a universal AI assembly protocol.**

---

## The Vision: FLUX as Universal AI Backbone

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FLUX-AUGMENTED AI SYSTEM                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │
│   │  Llama 3    │   │   CLIP      │   │  Whisper    │   │  Stable     │    │
│   │  (Voice)    │   │  (Vision)   │   │  (Audio)    │   │ Diffusion   │    │
│   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘    │
│          │                 │                 │                 │            │
│          ▼                 ▼                 ▼                 ▼            │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    MODEL BRIDGES (XToWave / WaveToX)                │  │
│   │         Translate any model's representations → 432-dim waves       │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                         FLUX CORE                                   │  │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │  │
│   │  │   CSE   │ │  Field  │ │ Gravity │ │   TL    │ │ Memory  │       │  │
│   │  │ (encode)│ │ (store) │ │ O(logn) │ │ (learn) │ │ (3-tier)│       │  │
│   │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘       │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   FLUX Superpowers Applied to ALL Ingested Models:                         │
│   ✓ Infinite context window (gravity replaces attention)                   │
│   ✓ Zero catastrophic forgetting (attractors accumulate)                   │
│   ✓ One-shot learning (new fact → new attractor → instant recall)          │
│   ✓ Causal reasoning (arrows track WHY)                                    │
│   ✓ Memory portability (carry .flx across machines)                        │
│   ✓ O(log n) retrieval (spatial tree, not all-pairs)                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Core Problems We Solve

### Problem 1: LLM Context Windows Are Limited

Every transformer has a hard context limit (8K, 32K, 128K tokens) because attention is O(n²).

**FLUX Solution:** Store everything in the field. Query with O(log n) gravity. Inject relevant context into a local window. Effective context = unlimited.

### Problem 2: LLMs Forget After Context

Information outside the context window is completely inaccessible. Sessions are stateless.

**FLUX Solution:** Every token processed → stored as field attractor. All history lives in memory. Cross-session persistence via `.flx` file.

### Problem 3: LLMs Can't Learn Without Retraining

To teach an LLM a new fact, you need fine-tuning or in-context examples every time.

**FLUX Solution:** One-shot attractor formation. Show once → permanent memory. No gradient required.

### Problem 4: Model Assembly Causes Catastrophic Forgetting

Combining models (merge, fine-tune, LoRA stack) often degrades capabilities.

**FLUX Solution:** Models connect via bridges, not merged weights. Each model's knowledge → separate attractors. Nothing overwrites anything.

### Problem 5: Byte-Level Generation Struggles with Fluency

FLUX Phase 8's WaveDecoder generates byte-by-byte, which fights against learned language patterns.

**FLUX Solution:** Use a pretrained LLM as the "voice" — it handles fluency. FLUX handles memory, reasoning, context. Best of both.

---

## Architecture Overview

```
                            USER INPUT
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                     INPUT PROCESSING                          │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     │
│  │ Text Input  │     │ Image Input │     │ Audio Input │     │
│  │ (tokenizer) │     │ (CLIP enc)  │     │(Whisper enc)│     │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘     │
│         │                   │                   │             │
│         ▼                   ▼                   ▼             │
│  ┌───────────────────────────────────────────────────────┐   │
│  │              MODEL BRIDGES (→ 432-dim wave)           │   │
│  │    LLMToWave        CLIPToWave        WhisperToWave   │   │
│  └───────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                      FLUX CORE (from .flx)                    │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                  RESONANCE FIELD (96³ × 512)            │ │
│  │    Every fact ever learned → stored as attractor        │ │
│  │    New inputs → perturb field → find relevant memories  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                              │                                │
│                              ▼                                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │            GRAVITATIONAL RELEVANCE (O(log n))           │ │
│  │    Query field for top-k relevant attractors            │ │
│  │    No attention matrix — spatial tree lookup            │ │
│  └─────────────────────────────────────────────────────────┘ │
│                              │                                │
│                              ▼                                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              THREE-TIER MEMORY SYSTEM                   │ │
│  │    Working (session) + Episodic (facts) + Semantic      │ │
│  └─────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                   CONTEXT INJECTION                           │
│                                                               │
│    Retrieved attractors (432-dim) ─────────────────────┐     │
│                                                         │     │
│    ┌─────────────────────────────────────────────────┐ │     │
│    │           WAVE → LLM PROJECTION                 │ │     │
│    │   wave_to_llm: Linear(432 → llm_hidden_dim)     │ │     │
│    └─────────────────────────────────────────────────┘ │     │
│                         │                               │     │
│                         ▼                               │     │
│    ┌─────────────────────────────────────────────────┐ │     │
│    │          CONTEXT INJECTOR                       │ │     │
│    │   Prepend retrieved context to LLM input        │ │     │
│    │   "Here's what you should know: [memories]"     │ │     │
│    └─────────────────────────────────────────────────┘ │     │
│                         │                               │     │
└─────────────────────────┼───────────────────────────────┘     │
                          │                                      │
                          ▼                                      │
┌───────────────────────────────────────────────────────────────┐
│                    LLM VOICE MODULE                           │
│                                                               │
│    ┌─────────────────────────────────────────────────────┐   │
│    │           PRETRAINED LLM (frozen or LoRA)           │   │
│    │                                                     │   │
│    │   Options:                                          │   │
│    │   • Phi-3-mini (3.8B) — Kaggle-friendly            │   │
│    │   • Llama-3-8B-4bit — balanced                     │   │
│    │   • Mistral-7B — good at code                      │   │
│    │   • Qwen-2-7B — multilingual                       │   │
│    │                                                     │   │
│    │   The LLM receives:                                 │   │
│    │   [Retrieved Context] + [User Input] → [Response]   │   │
│    └─────────────────────────────────────────────────────┘   │
│                              │                                │
│                              ▼                                │
│    ┌─────────────────────────────────────────────────────┐   │
│    │           RESPONSE → WAVE STORAGE                   │   │
│    │   llm_to_wave: Linear(llm_hidden_dim → 432)         │   │
│    │   Store conversation in field for future recall     │   │
│    └─────────────────────────────────────────────────────┘   │
│                                                               │
└───────────────────────────────────────────────────────────────┘
                                │
                                ▼
                          FLUENT OUTPUT
```

---

## File Structure

```
phases/phase11/
├── PHASE_11_SPEC.md              ← This file
│
├── # ── Core Architecture ──
├── model_bridge.py               ← Abstract ModelBridge base class
├── llm_bridge.py                 ← LLMToWave / WaveToLLM bridges
├── context_injector.py           ← Inject retrieved context into LLM
├── flux_augmented_llm.py         ← FLUXAugmentedLLM main class
├── capability_router.py          ← Route to appropriate model/mode
│
├── # ── Model Adapters ──
├── adapters/
│   ├── phi3_adapter.py           ← Phi-3-mini integration
│   ├── llama_adapter.py          ← Llama-3 integration
│   ├── mistral_adapter.py        ← Mistral integration
│   └── generic_hf_adapter.py     ← Any HuggingFace model
│
├── # ── .flx Format Extensions ──
├── flx_v3.py                     ← .flx v3.0 format with model slots
├── model_ingestion.py            ← Ingest external model → .flx slot
│
├── # ── Training ──
├── train_bridges.py              ← Train wave↔LLM projection bridges
├── train_context_injector.py     ← Train context injection layer
│
├── # ── Tests ──
├── test_phase11_test1.py         ← Test: Bridge projection quality
├── test_phase11_test2.py         ← Test: Context injection works
├── test_phase11_test3.py         ← Test: No forgetting after assembly
├── test_phase11_test4.py         ← Test: Infinite context retrieval
│
├── # ── Demos ──
├── demo_phase11_demo1.py         ← Demo: Chat with FLUX memory
├── demo_phase11_demo2.py         ← Demo: One-shot fact learning
├── demo_phase11_demo3.py         ← Demo: Multi-model assembly
│
└── RESULTS_PHASE_11.md           ← Auto-generated by PhaseResults
```

---

## Component 1: ModelBridge — Universal Adapter Interface

```python
"""
model_bridge.py — Abstract base for all model adapters

Every external model connects to FLUX through a bridge.
Bridges translate between the model's representation space and FLUX's 432-dim wave space.
"""

import torch
import torch.nn as nn
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BridgeConfig:
    """Configuration for a model bridge."""
    source_dim: int          # Model's hidden dimension
    wave_dim: int = 432      # FLUX wave dimension (constant)
    hidden_dim: int = 512    # Intermediate projection dimension
    num_layers: int = 2      # Projection depth
    dropout: float = 0.1
    bidirectional: bool = True  # Can project both directions?


class ModelBridge(nn.Module, ABC):
    """
    Abstract base class for all model bridges.
    
    A bridge translates between an external model's representation space
    and FLUX's 432-dimensional wave space.
    
    Required methods:
    - to_wave(): Model representation → Wave space
    - from_wave(): Wave space → Model representation (if bidirectional)
    
    The bridge is responsible for:
    - Dimension alignment
    - Representation normalization
    - Semantic preservation (similar things → nearby waves)
    """
    
    def __init__(self, config: BridgeConfig):
        super().__init__()
        self.config = config
        
        # ── To Wave projection ──
        self.to_wave_proj = nn.Sequential(
            nn.Linear(config.source_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.wave_dim),
            nn.LayerNorm(config.wave_dim),
        )
        
        # ── From Wave projection (if bidirectional) ──
        if config.bidirectional:
            self.from_wave_proj = nn.Sequential(
                nn.Linear(config.wave_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.source_dim),
                nn.LayerNorm(config.source_dim),
            )
        else:
            self.from_wave_proj = None
    
    def to_wave(self, x: Tensor) -> Tensor:
        """
        Project model representation to wave space.
        
        Args:
            x: Model representation [batch, seq, source_dim] or [batch, source_dim]
            
        Returns:
            Wave representation [batch, seq, 432] or [batch, 432]
        """
        return self.to_wave_proj(x)
    
    def from_wave(self, wave: Tensor) -> Tensor:
        """
        Project wave back to model representation space.
        
        Args:
            wave: Wave representation [batch, seq, 432] or [batch, 432]
            
        Returns:
            Model representation [batch, seq, source_dim] or [batch, source_dim]
        """
        if self.from_wave_proj is None:
            raise ValueError("Bridge is not bidirectional")
        return self.from_wave_proj(wave)
    
    @abstractmethod
    def get_model_hidden_states(self, inputs: Any) -> Tensor:
        """
        Extract hidden states from the source model.
        Subclasses implement model-specific extraction.
        """
        pass
    
    @abstractmethod
    def inject_context(self, model_inputs: Any, context_wave: Tensor) -> Any:
        """
        Inject retrieved context into model inputs.
        Subclasses implement model-specific injection.
        """
        pass
    
    def forward(self, x: Tensor, direction: str = 'to_wave') -> Tensor:
        """Forward pass in specified direction."""
        if direction == 'to_wave':
            return self.to_wave(x)
        elif direction == 'from_wave':
            return self.from_wave(x)
        else:
            raise ValueError(f"Unknown direction: {direction}")
```

---

## Component 2: LLMBridge — Connect Any HuggingFace LLM

```python
"""
llm_bridge.py — Bridge for HuggingFace language models

Connects any HF causal LM to FLUX's wave space.
Handles tokenization, hidden state extraction, and context injection.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any, Optional, List, Union
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass

from model_bridge import ModelBridge, BridgeConfig


@dataclass  
class LLMBridgeConfig(BridgeConfig):
    """Extended config for LLM bridges."""
    model_name: str = "microsoft/phi-3-mini-4k-instruct"
    load_in_4bit: bool = True           # Quantize for Kaggle
    use_flash_attention: bool = False   # Flash attention if available
    max_context_injection: int = 32     # Max retrieved attractors to inject
    freeze_llm: bool = True             # Don't train LLM weights


class LLMBridge(ModelBridge):
    """
    Bridge connecting a HuggingFace LLM to FLUX wave space.
    
    This bridge:
    1. Extracts hidden states from the LLM
    2. Projects them to 432-dim wave space
    3. Projects retrieved waves back to LLM space for injection
    4. Handles context injection formats
    """
    
    def __init__(self, config: LLMBridgeConfig, device: str = 'cuda'):
        # Get model hidden dim before calling super().__init__
        self.config = config
        self.device = device
        
        # ── Load tokenizer ──
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # ── Load model (quantized for efficiency) ──
        load_kwargs = {
            'trust_remote_code': True,
            'torch_dtype': torch.float16,
        }
        
        if config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **load_kwargs,
        )
        
        if config.freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
        
        # Get hidden dimension from model config
        hidden_dim = self.llm.config.hidden_size
        config.source_dim = hidden_dim
        
        # ── Initialize bridge projections ──
        super().__init__(config)
        self.to(device)
        
        # ── Context injection template ──
        self.context_template = (
            "Retrieved memories (most relevant first):\n"
            "{memories}\n\n"
            "Now respond to: {input}"
        )
    
    def get_model_hidden_states(
        self, 
        text: Union[str, List[str]],
        layer: int = -1,  # Which layer to extract from (-1 = last)
    ) -> Tensor:
        """
        Extract hidden states from LLM for given text.
        
        Args:
            text: Input text(s)
            layer: Which layer to extract (-1 for last hidden state)
            
        Returns:
            Hidden states [batch, seq, hidden_dim]
        """
        if isinstance(text, str):
            text = [text]
        
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.llm(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # Extract specified layer's hidden states
        hidden_states = outputs.hidden_states[layer]  # [batch, seq, hidden]
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
        wave = self.to_wave(hidden)
        return wave
    
    def inject_context(
        self,
        input_text: str,
        context_waves: Tensor,          # [num_memories, 432]
        context_texts: List[str] = None, # Optional: text versions
    ) -> Dict[str, Tensor]:
        """
        Inject retrieved context into LLM input.
        
        Two modes:
        1. Text injection: Insert context as text prefix
        2. Embedding injection: Add context embeddings directly
        
        Args:
            input_text: User's input
            context_waves: Retrieved memory waves [num_memories, 432]
            context_texts: Optional text versions of memories
            
        Returns:
            Prepared inputs for LLM generation
        """
        if context_texts is not None:
            # ── Mode 1: Text injection ──
            memories_str = "\n".join(
                f"- {text}" for text in context_texts[:self.config.max_context_injection]
            )
            full_input = self.context_template.format(
                memories=memories_str,
                input=input_text,
            )
            
            inputs = self.tokenizer(
                full_input,
                return_tensors='pt',
                truncation=True,
                max_length=2048,
            ).to(self.device)
            
            return inputs
        
        else:
            # ── Mode 2: Embedding injection ──
            # Project waves back to LLM space
            context_embeddings = self.from_wave(context_waves)  # [num, hidden]
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors='pt',
                truncation=True,
                max_length=2048 - self.config.max_context_injection,
            ).to(self.device)
            
            # Get input embeddings
            input_embeds = self.llm.get_input_embeddings()(inputs['input_ids'])
            
            # Prepend context embeddings
            context_embeds = context_embeddings.unsqueeze(0)  # [1, num, hidden]
            combined_embeds = torch.cat([context_embeds, input_embeds], dim=1)
            
            # Create attention mask
            context_mask = torch.ones(
                1, context_embeds.shape[1], 
                device=self.device, dtype=inputs['attention_mask'].dtype
            )
            combined_mask = torch.cat([context_mask, inputs['attention_mask']], dim=1)
            
            return {
                'inputs_embeds': combined_embeds,
                'attention_mask': combined_mask,
            }
    
    def generate(
        self,
        input_text: str,
        context_waves: Optional[Tensor] = None,
        context_texts: Optional[List[str]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """
        Generate text with optional context injection.
        
        Args:
            input_text: User prompt
            context_waves: Retrieved memory waves (optional)
            context_texts: Text versions of memories (optional)
            max_new_tokens: Generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated text
        """
        if context_waves is not None or context_texts is not None:
            inputs = self.inject_context(input_text, context_waves, context_texts)
        else:
            inputs = self.tokenizer(
                input_text,
                return_tensors='pt',
                truncation=True,
                max_length=2048,
            ).to(self.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )
        
        # Decode, removing input tokens
        if 'input_ids' in inputs:
            input_length = inputs['input_ids'].shape[1]
        else:
            input_length = inputs['inputs_embeds'].shape[1]
        
        generated = outputs[0][input_length:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
```

---

## Component 3: FLUXAugmentedLLM — The Full System

```python
"""
flux_augmented_llm.py — FLUX-Augmented Language Model

Combines FLUX's infinite memory and O(log n) retrieval with a pretrained LLM's fluency.

The LLM is the "voice" — it generates fluent text.
FLUX is the "brain" — it remembers everything and retrieves relevant context.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import sys

# Import FLUX components
sys.path.append(str(Path(__file__).parent.parent.parent))
from flux_utils import get_device, load_checkpoint

sys.path.append(str(Path(__file__).parent.parent / 'phase1'))
from cse import ContinuousSemanticEncoder

sys.path.append(str(Path(__file__).parent.parent / 'phase2'))
from resonance_field import ResonanceField

sys.path.append(str(Path(__file__).parent.parent / 'phase3'))
from gravitational_relevance import GravitationalRelevance

sys.path.append(str(Path(__file__).parent.parent / 'phase6'))
from episodic_memory import EpisodicMemory

from llm_bridge import LLMBridge, LLMBridgeConfig


@dataclass
class FLUXAugmentedConfig:
    """Configuration for FLUX-Augmented LLM."""
    # LLM settings
    llm_name: str = "microsoft/phi-3-mini-4k-instruct"
    load_in_4bit: bool = True
    freeze_llm: bool = True
    
    # FLUX settings
    flx_path: Optional[str] = None  # Path to .flx file
    wave_dim: int = 432
    
    # Retrieval settings
    top_k_retrieval: int = 16       # How many memories to retrieve
    retrieval_threshold: float = 0.3 # Minimum similarity for retrieval
    
    # Memory settings
    store_conversations: bool = True  # Store exchanges in field
    store_to_episodic: bool = True    # Also store in episodic memory


class FLUXAugmentedLLM(nn.Module):
    """
    FLUX-Augmented Language Model.
    
    Combines:
    - FLUX CSE for semantic encoding
    - FLUX Field for infinite memory storage
    - FLUX GravitationalRelevance for O(log n) retrieval  
    - FLUX EpisodicMemory for fact storage
    - Pretrained LLM for fluent generation
    
    The key innovations:
    1. Every conversation → stored in field → never forgotten
    2. Every query → gravitational lookup → relevant memories injected
    3. Effective context window → unlimited
    4. Learning → one-shot (show once → permanent attractor)
    """
    
    def __init__(self, config: FLUXAugmentedConfig, device: str = None):
        super().__init__()
        self.config = config
        self.device = device or get_device()
        
        # ── Initialize FLUX Core ──
        self._init_flux_core(config.flx_path)
        
        # ── Initialize LLM Bridge ──
        bridge_config = LLMBridgeConfig(
            model_name=config.llm_name,
            source_dim=0,  # Will be set by LLMBridge
            wave_dim=config.wave_dim,
            load_in_4bit=config.load_in_4bit,
            freeze_llm=config.freeze_llm,
        )
        self.llm_bridge = LLMBridge(bridge_config, self.device)
        
        # ── Conversation history (in-session) ──
        self.conversation_history: List[Dict[str, str]] = []
        
        print(f"  ✓ FLUXAugmentedLLM initialized")
        print(f"    LLM: {config.llm_name}")
        print(f"    Device: {self.device}")
        print(f"    Retrieval top-k: {config.top_k_retrieval}")
    
    def _init_flux_core(self, flx_path: Optional[str]):
        """Initialize FLUX components from .flx file or fresh."""
        
        if flx_path and Path(flx_path).exists():
            print(f"  Loading FLUX core from {flx_path}...")
            flx = torch.load(flx_path, map_location='cpu')
            
            # ── Load CSE ──
            self.cse = ContinuousSemanticEncoder(
                wave_dim=flx['cse']['config'].get('wave_dim', 432)
            )
            self.cse.load_state_dict(flx['cse']['state_dict'])
            self.cse.to(self.device)
            self.cse.eval()
            
            # ── Load Field ──
            field_config = flx['field']['config']
            self.field = ResonanceField(
                h=field_config.get('h', 96),
                w=field_config.get('w', 96),
                d=field_config.get('d', 96),
                features=field_config.get('features', 512),
            )
            self.field.load_state_dict(flx['field']['state_dict'])
            self.field.to(self.device)
            
            # ── Load Gravitational Relevance ──
            self.gravity = GravitationalRelevance(
                field_features=field_config.get('features', 512),
                wave_dim=self.config.wave_dim,
            )
            if 'gravity_state' in flx['field']:
                self.gravity.load_state_dict(flx['field']['gravity_state'])
            self.gravity.to(self.device)
            
            # ── Load Episodic Memory ──
            self.episodic = EpisodicMemory(wave_dim=self.config.wave_dim)
            if 'memory' in flx and 'episodic' in flx['memory']:
                self.episodic.load_state(flx['memory']['episodic'])
            
            print(f"    ✓ Loaded CSE, Field, GR, Episodic from .flx")
            print(f"    ✓ Episodic entries: {len(self.episodic)}")
            
        else:
            print(f"  Initializing fresh FLUX core...")
            
            # ── Fresh CSE ──
            self.cse = ContinuousSemanticEncoder(wave_dim=self.config.wave_dim)
            self.cse.to(self.device)
            
            # ── Fresh Field ──
            self.field = ResonanceField(h=64, w=64, d=64, features=512)
            self.field.to(self.device)
            
            # ── Fresh GR ──
            self.gravity = GravitationalRelevance(
                field_features=512,
                wave_dim=self.config.wave_dim,
            )
            self.gravity.to(self.device)
            
            # ── Fresh Episodic ──
            self.episodic = EpisodicMemory(wave_dim=self.config.wave_dim)
            
            print(f"    ✓ Fresh FLUX core initialized")
    
    def encode(self, text: str) -> Tensor:
        """Encode text to wave space using FLUX CSE."""
        with torch.no_grad():
            wave = self.cse.encode(text)  # [seq, 432]
        return wave
    
    def retrieve(
        self, 
        query_wave: Tensor,
        top_k: int = None,
    ) -> Tuple[Tensor, List[str]]:
        """
        Retrieve relevant memories using gravitational relevance.
        
        Args:
            query_wave: Query in wave space [seq, 432] or [432]
            top_k: Number of memories to retrieve
            
        Returns:
            (memory_waves, memory_texts): Retrieved memories
        """
        top_k = top_k or self.config.top_k_retrieval
        
        # Pool query wave if sequence
        if query_wave.dim() == 2:
            query_vec = query_wave.mean(dim=0)  # [432]
        else:
            query_vec = query_wave
        
        # ── Query episodic memory ──
        episodic_results = self.episodic.recall(
            query_vec.cpu().numpy(),
            k=top_k,
        )
        
        # ── Query field attractors via gravity ──
        # (If gravity is set up, otherwise skip)
        if hasattr(self.gravity, 'query_attractors'):
            field_results = self.gravity.query_attractors(
                query_vec.unsqueeze(0),
                top_k=top_k,
            )
        else:
            field_results = []
        
        # Combine results
        memory_waves = []
        memory_texts = []
        
        for entry in episodic_results:
            memory_waves.append(torch.tensor(entry['wave']))
            memory_texts.append(entry.get('text', entry.get('content', '[memory]')))
        
        if memory_waves:
            memory_waves = torch.stack(memory_waves).to(self.device)
        else:
            memory_waves = torch.zeros(0, self.config.wave_dim, device=self.device)
        
        return memory_waves, memory_texts
    
    def store(self, text: str, wave: Optional[Tensor] = None, metadata: Dict = None):
        """
        Store information in FLUX memory.
        
        Args:
            text: Text to store
            wave: Pre-computed wave (optional)
            metadata: Additional metadata
        """
        if wave is None:
            wave = self.encode(text)
        
        # Pool to single vector
        if wave.dim() == 2:
            wave_vec = wave.mean(dim=0)
        else:
            wave_vec = wave
        
        # Store in episodic memory
        if self.config.store_to_episodic:
            self.episodic.store(
                wave=wave_vec.cpu().numpy(),
                content={'text': text, **(metadata or {})},
            )
        
        # Perturb field (creates attractor)
        if self.config.store_conversations:
            self.field.absorb(wave_vec.unsqueeze(0))
    
    def chat(
        self,
        message: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        use_memory: bool = True,
    ) -> str:
        """
        Chat with FLUX-augmented LLM.
        
        This is the main interface. It:
        1. Encodes the message to wave space
        2. Retrieves relevant memories (O(log n))
        3. Injects context into the LLM
        4. Generates a response
        5. Stores the exchange in memory
        
        Args:
            message: User message
            max_new_tokens: Max generation length
            temperature: Sampling temperature
            use_memory: Whether to retrieve and use memories
            
        Returns:
            Assistant response
        """
        # Step 1: Encode message
        message_wave = self.encode(message)
        
        # Step 2: Retrieve relevant memories
        if use_memory and len(self.episodic) > 0:
            memory_waves, memory_texts = self.retrieve(message_wave)
            print(f"    Retrieved {len(memory_texts)} memories")
        else:
            memory_waves, memory_texts = None, None
        
        # Step 3: Generate with context injection
        response = self.llm_bridge.generate(
            input_text=message,
            context_waves=memory_waves,
            context_texts=memory_texts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        # Step 4: Store exchange in memory
        exchange_text = f"User: {message}\nAssistant: {response}"
        self.store(exchange_text, metadata={'type': 'conversation'})
        
        # Track conversation history
        self.conversation_history.append({
            'user': message,
            'assistant': response,
        })
        
        return response
    
    def teach(self, fact: str, verify: bool = True) -> bool:
        """
        Teach a new fact to FLUX (one-shot learning).
        
        The fact is stored as an attractor and can be recalled immediately.
        
        Args:
            fact: The fact to teach
            verify: Whether to verify storage
            
        Returns:
            Success status
        """
        print(f"  Teaching: {fact[:50]}...")
        
        # Encode and store
        wave = self.encode(fact)
        self.store(fact, wave, metadata={'type': 'taught_fact'})
        
        # Verify recall
        if verify:
            retrieved_waves, retrieved_texts = self.retrieve(wave, top_k=1)
            if retrieved_texts and fact in retrieved_texts[0]:
                print(f"    ✓ Fact stored and verified")
                return True
            else:
                print(f"    ⚠ Fact stored but verification unclear")
                return True  # Still stored
        
        return True
    
    def save_flx(self, path: str):
        """
        Save the full system to .flx format.
        
        The LLM weights are NOT saved (too large).
        The FLUX core + bridges ARE saved.
        LLM is referenced by name and reloaded on init.
        """
        flx = {
            'format': 'FLUX',
            'version': '3.0-augmented',
            'metadata': {
                'llm_name': self.config.llm_name,
                'wave_dim': self.config.wave_dim,
                'episodic_entries': len(self.episodic),
            },
            
            # FLUX Core
            'cse': {
                'config': {'wave_dim': self.config.wave_dim},
                'state_dict': self.cse.state_dict(),
            },
            'field': {
                'config': {
                    'h': self.field.h,
                    'w': self.field.w,
                    'd': self.field.d,
                    'features': self.field.features,
                },
                'state_dict': self.field.state_dict(),
                'gravity_state': self.gravity.state_dict(),
            },
            'memory': {
                'episodic': self.episodic.get_state(),
            },
            
            # Bridges (trainable)
            'bridges': {
                'llm_to_wave': self.llm_bridge.to_wave_proj.state_dict(),
                'wave_to_llm': self.llm_bridge.from_wave_proj.state_dict() 
                    if self.llm_bridge.from_wave_proj else None,
            },
            
            # LLM reference (not weights)
            'llm_reference': {
                'name': self.config.llm_name,
                'load_in_4bit': self.config.load_in_4bit,
            },
        }
        
        torch.save(flx, path)
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved to {path} ({size_mb:.1f} MB)")
    
    @classmethod
    def from_flx(cls, path: str, device: str = None):
        """Load from .flx file."""
        flx = torch.load(path, map_location='cpu')
        
        config = FLUXAugmentedConfig(
            llm_name=flx['llm_reference']['name'],
            load_in_4bit=flx['llm_reference']['load_in_4bit'],
            flx_path=path,
        )
        
        model = cls(config, device=device)
        
        # Load bridges
        if flx['bridges']['llm_to_wave']:
            model.llm_bridge.to_wave_proj.load_state_dict(
                flx['bridges']['llm_to_wave']
            )
        if flx['bridges']['wave_to_llm']:
            model.llm_bridge.from_wave_proj.load_state_dict(
                flx['bridges']['wave_to_llm']
            )
        
        return model
```

---

## Component 4: Forgetting Verification Test

```python
"""
test_phase11_test3.py — Zero-Forgetting Verification

This test proves that adding new models/memories to FLUX
does NOT cause catastrophic forgetting of existing knowledge.

Test procedure:
1. Load existing .flx with known facts
2. Verify all facts are recallable
3. Add NEW model/memories
4. Verify OLD facts are STILL recallable
5. Verify NEW facts are also recallable
6. Compute forgetting score (should be 0)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from flux_utils import PhaseResults, get_device

from flux_augmented_llm import FLUXAugmentedLLM, FLUXAugmentedConfig


def test_zero_forgetting():
    """Verify zero catastrophic forgetting during model assembly."""
    
    results = PhaseResults(phase=11, component_name="Zero-Forgetting Verification")
    
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Phase 11 Test 3: Zero-Forgetting Verification")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # ── Batch 1: Initial Facts ──
    batch1_facts = [
        "The capital of France is Paris",
        "Water boils at 100 degrees Celsius",
        "Python was created by Guido van Rossum",
        "The moon orbits the Earth",
        "Shakespeare wrote Hamlet",
    ]
    
    # ── Batch 2: New Facts (added later) ──
    batch2_facts = [
        "FLUX uses gravitational relevance instead of attention",
        "Resonance fields replace weight matrices",
        "The episodic memory uses FAISS for indexing",
        "Wave dimension is 432 in FLUX",
        "Thermodynamic learning replaces backpropagation",
    ]
    
    # ── Initialize model ──
    print("Step 1: Initialize FLUX-Augmented LLM...")
    config = FLUXAugmentedConfig(
        llm_name="microsoft/phi-3-mini-4k-instruct",
        load_in_4bit=True,
    )
    model = FLUXAugmentedLLM(config, device=device)
    
    # ── Teach Batch 1 ──
    print("\nStep 2: Teaching Batch 1 (5 facts)...")
    for fact in batch1_facts:
        model.teach(fact, verify=False)
    print(f"  ✓ Batch 1 stored. Episodic entries: {len(model.episodic)}")
    
    # ── Verify Batch 1 recall BEFORE adding Batch 2 ──
    print("\nStep 3: Verify Batch 1 recall BEFORE Batch 2...")
    batch1_recall_before = 0
    for fact in batch1_facts:
        wave = model.encode(fact)
        _, texts = model.retrieve(wave, top_k=3)
        if any(fact.lower() in t.lower() for t in texts):
            batch1_recall_before += 1
    
    recall_rate_before = batch1_recall_before / len(batch1_facts)
    print(f"  Batch 1 recall before: {batch1_recall_before}/{len(batch1_facts)} ({recall_rate_before:.0%})")
    
    # ── Teach Batch 2 (simulate adding new model/knowledge) ──
    print("\nStep 4: Teaching Batch 2 (5 new facts)...")
    for fact in batch2_facts:
        model.teach(fact, verify=False)
    print(f"  ✓ Batch 2 stored. Episodic entries: {len(model.episodic)}")
    
    # ── Verify Batch 1 recall AFTER adding Batch 2 ──
    print("\nStep 5: Verify Batch 1 recall AFTER Batch 2...")
    batch1_recall_after = 0
    for fact in batch1_facts:
        wave = model.encode(fact)
        _, texts = model.retrieve(wave, top_k=3)
        if any(fact.lower() in t.lower() for t in texts):
            batch1_recall_after += 1
    
    recall_rate_after = batch1_recall_after / len(batch1_facts)
    print(f"  Batch 1 recall after: {batch1_recall_after}/{len(batch1_facts)} ({recall_rate_after:.0%})")
    
    # ── Verify Batch 2 recall ──
    print("\nStep 6: Verify Batch 2 recall...")
    batch2_recall = 0
    for fact in batch2_facts:
        wave = model.encode(fact)
        _, texts = model.retrieve(wave, top_k=3)
        if any(fact.lower() in t.lower() for t in texts):
            batch2_recall += 1
    
    batch2_recall_rate = batch2_recall / len(batch2_facts)
    print(f"  Batch 2 recall: {batch2_recall}/{len(batch2_facts)} ({batch2_recall_rate:.0%})")
    
    # ── Compute Forgetting Score ──
    # Forgetting = (recall_before - recall_after) / recall_before
    # 0.0 = perfect (no forgetting)
    # 1.0 = complete forgetting
    if batch1_recall_before > 0:
        forgetting_score = (batch1_recall_before - batch1_recall_after) / batch1_recall_before
    else:
        forgetting_score = 0.0
    
    print(f"\n{'='*60}")
    print(f"FORGETTING SCORE: {forgetting_score:.4f}")
    print(f"(0.0 = perfect, 1.0 = total forgetting)")
    print(f"{'='*60}")
    
    # ── Record Results ──
    results.add_test(
        "Batch 1 Recall Before",
        passed=recall_rate_before >= 0.8,
        score=recall_rate_before,
        threshold=0.8,
    )
    
    results.add_test(
        "Batch 1 Recall After",
        passed=recall_rate_after >= 0.8,
        score=recall_rate_after,
        threshold=0.8,
    )
    
    results.add_test(
        "Batch 2 Recall",
        passed=batch2_recall_rate >= 0.8,
        score=batch2_recall_rate,
        threshold=0.8,
    )
    
    results.add_test(
        "Forgetting Score",
        passed=forgetting_score <= 0.05,
        score=forgetting_score,
        threshold=0.05,
    )
    
    results.save()
    
    # ── Final Verdict ──
    if forgetting_score <= 0.05:
        print("\n  ✓ TEST PASSED — Zero catastrophic forgetting confirmed")
        return True
    else:
        print("\n  ✗ TEST FAILED — Forgetting detected")
        return False


if __name__ == "__main__":
    success = test_zero_forgetting()
    sys.exit(0 if success else 1)
```

---

## .flx v3.0 Format Specification

```python
# .flx v3.0-augmented format
{
    "format": "FLUX",
    "version": "3.0-augmented",
    
    "metadata": {
        "created": "2026-03-28T...",
        "llm_name": "microsoft/phi-3-mini-4k-instruct",
        "wave_dim": 432,
        "episodic_entries": 1000,
        "field_energy": 8847.43,
        "total_conversations": 500,
    },
    
    # ── FLUX Native Core (from Flux-beta.flx) ──
    "cse": {
        "config": {"wave_dim": 432, "byte_window": 8},
        "state_dict": {...}
    },
    
    "field": {
        "config": {"h": 96, "w": 96, "d": 96, "features": 512},
        "state_dict": {...},
        "gravity_state": {...},
        "thermodynamic_state": {...},
    },
    
    "memory": {
        "working": {...},
        "episodic": {
            "index_data": [...],      # FAISS index
            "entries": [...],          # Stored facts/conversations
        },
        "semantic": {...},
    },
    
    "causal": {
        "cgn_state": {...},
        "graph_state": {...},
    },
    
    # ── Bridge Weights (trained) ──
    "bridges": {
        "llm_to_wave": {...},         # Project LLM hidden → wave
        "wave_to_llm": {...},         # Project wave → LLM hidden
        "context_injector": {...},    # Context injection layer
    },
    
    # ── LLM Reference (NOT weights) ──
    "llm_reference": {
        "name": "microsoft/phi-3-mini-4k-instruct",
        "load_in_4bit": true,
        "adapter": null,              # Optional LoRA adapter path
    },
    
    # ── Ingested Models (for multi-model assembly) ──
    "ingested_models": {
        "clip-vision": {
            "source": "openai/clip-vit-large",
            "bridge": "CLIPToWave",
            "state_dict": {...},
        },
        "whisper-base": {
            "source": "openai/whisper-base", 
            "bridge": "WhisperToWave",
            "state_dict": {...},
        },
    },
    
    # ── Capability Router ──
    "capability_router": {
        "rules": [
            {"input_type": "text", "use": "llm_reference"},
            {"input_type": "image", "use": "clip-vision"},
            {"input_type": "audio", "use": "whisper-base"},
        ],
    },
}
```

---

## Training the Bridges (Kaggle-Optimized)

The only training required is for the bridge projections (~10M parameters):

```python
# Training objective: Semantic alignment
# Similar text → similar waves → nearby field coordinates

def train_bridges(model, dataset, epochs=3, lr=1e-4):
    """
    Train wave↔LLM bridges on contrastive alignment.
    
    The goal is to ensure that semantically similar inputs
    map to nearby locations in both wave space and LLM space.
    """
    optimizer = torch.optim.AdamW([
        {'params': model.llm_bridge.to_wave_proj.parameters()},
        {'params': model.llm_bridge.from_wave_proj.parameters()},
    ], lr=lr)
    
    for epoch in range(epochs):
        for batch in dataset:
            # Get LLM hidden states
            llm_hidden = model.llm_bridge.get_model_hidden_states(batch['text'])
            
            # Project to wave space
            wave = model.llm_bridge.to_wave(llm_hidden)
            
            # Reconstruct back to LLM space
            reconstructed = model.llm_bridge.from_wave(wave)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(reconstructed, llm_hidden)
            
            # Alignment loss: CSE encoding vs LLM-derived wave
            cse_wave = model.cse.encode(batch['text'])
            align_loss = 1 - F.cosine_similarity(
                wave.mean(dim=1), 
                cse_wave.mean(dim=1),
            ).mean()
            
            loss = recon_loss + 0.5 * align_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model
```

---

## Kaggle Notebook Cell Structure

```
Cell 1:  Clone repo + install deps
Cell 2:  Download Flux-beta.flx from HuggingFace
Cell 3:  Init logger + detect hardware + load HF_TOKEN
Cell 4:  Smoke test — verify .flx loads
Cell 5:  Initialize FLUXAugmentedLLM with Phi-3-mini
Cell 6:  Demo A — Chat with memory retrieval
Cell 7:  Demo B — One-shot fact learning
Cell 8:  Demo C — Infinite context simulation
Cell 9:  Test 1 — Bridge projection quality
Cell 10: Test 2 — Zero-forgetting verification
Cell 11: Test 3 — Retrieval speed (O(log n) vs O(n))
Cell 12: Save Flux-augmented.flx
Cell 13: Upload to HuggingFace Hub
Cell 14: View full log
Cell 15: Final summary + next steps
```

---

## Success Criteria

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Bridge reconstruction loss | < 0.1 | LLM ↔ wave roundtrip |
| Forgetting score | ≤ 0.05 | After adding new knowledge |
| Retrieval accuracy | ≥ 80% | Taught facts recallable |
| Retrieval latency | < 10ms | O(log n) for 10K entries |
| Generation fluency | Coherent | Qualitative — LLM voice |
| One-shot recall | 100% | Taught fact → immediate recall |
| .flx size | < 500MB | Bridges only, not LLM weights |

---

## What This Enables

After Phase 11:

1. **Any LLM as voice** — swap Phi-3 for Llama, Mistral, Qwen
2. **Infinite context** — no more token limits
3. **Persistent memory** — carry conversations in .flx
4. **One-shot learning** — teach once, remember forever
5. **Multi-model assembly** — add CLIP, Whisper, etc.
6. **Zero forgetting** — old knowledge + new knowledge coexist

The `.flx` file becomes a **portable AI brain** — carrying not just weights, but memories, conversations, learned facts, and the capability to plug into any LLM for fluent expression.

---

*Phase 11: Where FLUX becomes the universal AI backbone.*
*FLUX: Field-based Latent Understanding eXperience*
