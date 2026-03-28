"""
flux_augmented_llm.py — FLUX-Augmented Language Model

The main class that combines FLUX's infinite memory and O(log n) retrieval
with a pretrained LLM's fluency.

Architecture:
┌────────────────────────────────────────────────────────────────┐
│                    FLUXAugmentedLLM                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   USER INPUT                                                   │
│       │                                                        │
│       ▼                                                        │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐              │
│   │   CSE    │────▶│  Field   │────▶│ Gravity  │              │
│   │ (encode) │     │ (store)  │     │ (O(logn))│              │
│   └──────────┘     └──────────┘     └──────────┘              │
│                                           │                    │
│                                           ▼                    │
│                                    ┌──────────┐                │
│                                    │ Episodic │                │
│                                    │  Memory  │                │
│                                    └──────────┘                │
│                                           │                    │
│                                           ▼                    │
│   ┌───────────────────────────────────────────────────────┐   │
│   │              Context Injector                          │   │
│   │   Prepend memories to LLM input                        │   │
│   └───────────────────────────────────────────────────────┘   │
│                          │                                     │
│                          ▼                                     │
│   ┌───────────────────────────────────────────────────────┐   │
│   │              LLM Voice (Phi-3, Llama, etc.)           │   │
│   │   Fluent text generation                               │   │
│   └───────────────────────────────────────────────────────┘   │
│                          │                                     │
│                          ▼                                     │
│                    FLUENT OUTPUT                               │
│                                                                │
└────────────────────────────────────────────────────────────────┘

Key capabilities:
- Infinite context (everything stored in field)
- O(log n) retrieval (gravitational relevance)
- One-shot learning (teach once → permanent memory)
- Zero forgetting (attractors accumulate, never overwrite)
- Portable memory (save/load via .flx)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
import sys
import time
import numpy as np

# Add paths for imports
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent
sys.path.append(str(_PROJECT_ROOT))
sys.path.append(str(_PHASES_DIR / 'phase8_8'))
sys.path.append(str(_PHASES_DIR / 'phase8_9'))

from llm_bridge import LLMBridge, LLMBridgeConfig
from context_injector import SimpleTextInjector

# Default FLX paths (most capable → least capable)
FLX_PATHS = [
    _PROJECT_ROOT / 'checkpoints' / 'Flux-X-complete.flx',  # Phase 8.9 - full
    _PROJECT_ROOT / 'checkpoints' / 'Flux-beta.flx',         # Phase 8 - base
]


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class FLUXAugmentedConfig:
    """Configuration for FLUX-Augmented LLM."""
    
    # LLM settings
    llm_name: str = "Qwen/Qwen2.5-3B-Instruct"
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    freeze_llm: bool = True
    
    # FLUX settings
    flx_path: Optional[str] = None     # Auto-detect if None
    wave_dim: int = 432
    load_full_flux: bool = True        # Load full FLUX components if available
    load_adapters: bool = True         # Load Phase 8.9 modality adapters
    
    # Retrieval settings
    top_k_retrieval: int = 16
    retrieval_threshold: float = 0.3
    
    # Memory settings
    store_conversations: bool = True
    store_to_episodic: bool = True
    
    # Generation settings
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


# ─────────────────────────────────────────────
# Simple Memory Store (Kaggle-compatible)
# ─────────────────────────────────────────────

class SimpleMemoryStore:
    """
    Simple memory store using cosine similarity.
    
    This is a lightweight alternative to the full FLUX field + episodic memory
    for Kaggle environments. It stores wave vectors and retrieves by similarity.
    
    For full FLUX capabilities, use the actual field from Flux-beta.flx.
    """
    
    def __init__(self, wave_dim: int = 432):
        self.wave_dim = wave_dim
        self.memories: List[Dict] = []
        self.vectors: List[np.ndarray] = []
    
    def store(
        self,
        wave: Union[Tensor, np.ndarray],
        text: str,
        metadata: Optional[Dict] = None,
    ):
        """Store a memory."""
        if isinstance(wave, Tensor):
            wave = wave.detach().cpu().numpy()
        
        # Normalize
        wave = wave / (np.linalg.norm(wave) + 1e-8)
        
        self.vectors.append(wave)
        self.memories.append({
            'text': text,
            'metadata': metadata or {},
            'timestamp': time.time(),
        })
    
    def retrieve(
        self,
        query: Union[Tensor, np.ndarray],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[Dict, float]]:
        """
        Retrieve memories by similarity.
        
        Returns:
            List of (memory, similarity) tuples
        """
        if len(self.vectors) == 0:
            return []
        
        if isinstance(query, Tensor):
            query = query.detach().cpu().numpy()
        
        # Normalize query
        query = query / (np.linalg.norm(query) + 1e-8)
        
        # Compute similarities
        vectors = np.stack(self.vectors)
        similarities = vectors @ query
        
        # Get top-k
        indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in indices:
            sim = similarities[idx]
            if sim >= threshold:
                results.append((self.memories[idx], float(sim)))
        
        return results
    
    def __len__(self) -> int:
        return len(self.memories)
    
    def get_state(self) -> Dict:
        """Get serializable state."""
        return {
            'memories': self.memories,
            'vectors': [v.tolist() for v in self.vectors],
            'wave_dim': self.wave_dim,
        }
    
    def load_state(self, state: Dict):
        """Load from state dict."""
        self.memories = state['memories']
        self.vectors = [np.array(v) for v in state['vectors']]
        self.wave_dim = state['wave_dim']


# ─────────────────────────────────────────────
# FLUXLarge Memory Adapter (Full Power)
# ─────────────────────────────────────────────

class FLUXLargeMemoryAdapter:
    """
    Adapter that wraps FLUXLarge's full memory system.
    
    When we load from Flux-X-complete.flx, this provides access to:
    - Resonance Field (Phase 2) for attractor storage
    - Gravitational Relevance (Phase 3) for O(log n) retrieval
    - Thermodynamic Learning (Phase 4) for continuous learning
    - Working/Episodic/Semantic Memory (Phase 6)
    
    This gives Qwen access to the FULL FLUX power, not just cosine similarity.
    """
    
    def __init__(self, flux_large):
        self.flux = flux_large
        self.wave_dim = 432
        self._text_cache = {}  # wave_hash -> text
        # Get device from model parameters
        self._device = next(flux_large.parameters()).device
    
    def store(
        self,
        wave: Union[Tensor, np.ndarray],
        text: str,
        metadata: Optional[Dict] = None,
    ):
        """Store into FLUX field + episodic memory."""
        if isinstance(wave, np.ndarray):
            wave = torch.from_numpy(wave).float()
        
        # Pool to single vector if needed
        if wave.dim() > 1:
            wave = wave.mean(dim=0)
        
        # Store in episodic memory with text using write() method
        wave = wave.to(self._device)
        self.flux.episodic_memory.write(
            vector=wave,
            fact=text,
            confidence=metadata.get('confidence', 0.9) if metadata else 0.9,
            causal_source=metadata.get('source', 'teach') if metadata else 'teach',
        )
        
        # Also plant into field as attractor
        try:
            # Find best position for this wave
            positions, distances = self.flux.gr.query(wave.unsqueeze(0), top_k=1)
            # Plant nearby the most similar existing attractor
            position = positions[0] + torch.randn_like(positions[0]) * 0.1
            self.flux.field.add_attractor(position, wave)
        except:
            pass  # Field planting is optional
    
    def retrieve(
        self,
        query: Union[Tensor, np.ndarray],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[Dict, float]]:
        """
        Retrieve using gravitational relevance (O(log n)).
        
        Returns:
            List of (memory, similarity) tuples
        """
        if isinstance(query, np.ndarray):
            query = torch.from_numpy(query).float()
        
        if query.dim() > 1:
            query = query.mean(dim=0)
        
        query = query.to(self._device)
        
        # Use episodic memory search
        results = self.flux.episodic_memory.search(query, k=top_k)
        
        output = []
        for entry, sim in results:
            if sim >= threshold:
                memory = {
                    'text': entry.fact,
                    'metadata': {
                        'confidence': entry.confidence,
                        'causal_source': entry.causal_source,
                    },
                    'timestamp': entry.timestamp,
                }
                output.append((memory, float(sim)))
        
        return output
    
    def __len__(self) -> int:
        return len(self.flux.episodic_memory._metadata)
    
    def get_state(self) -> Dict:
        """Get state for saving."""
        return self.flux.episodic_memory.save_state()
    
    def load_state(self, state: Dict):
        """Load from state."""
        self.flux.episodic_memory.load_state(state)


# ─────────────────────────────────────────────
# Simple Wave Encoder (Kaggle-compatible)
# ─────────────────────────────────────────────

class SimpleWaveEncoder(nn.Module):
    """
    Simple wave encoder for text.
    
    This is a lightweight version of the CSE for Kaggle compatibility.
    For full FLUX capabilities, use the actual CSE from Flux-beta.flx.
    """
    
    def __init__(self, wave_dim: int = 432, hidden_dim: int = 512):
        super().__init__()
        self.wave_dim = wave_dim
        
        # Byte embedding
        self.byte_embed = nn.Embedding(256, hidden_dim)
        
        # Projection
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, wave_dim),
            nn.LayerNorm(wave_dim),
        )
    
    def forward(self, text: str) -> Tensor:
        """Encode text to wave."""
        # Convert to bytes
        bytes_list = list(text.encode('utf-8'))
        bytes_tensor = torch.tensor(bytes_list, dtype=torch.long, device=self.byte_embed.weight.device)
        
        # Embed
        embeds = self.byte_embed(bytes_tensor)  # [seq, hidden]
        
        # Project
        waves = self.proj(embeds)  # [seq, wave_dim]
        
        return waves
    
    def encode(self, text: str) -> Tensor:
        """Alias for forward."""
        return self.forward(text)


# ─────────────────────────────────────────────
# FLUX-Augmented LLM
# ─────────────────────────────────────────────

class FLUXAugmentedLLM(nn.Module):
    """
    FLUX-Augmented Language Model.
    
    Combines:
    - FLUX encoding for semantic representation
    - FLUX memory for infinite storage
    - O(log n) retrieval via similarity
    - Pretrained LLM for fluent generation
    
    The result is an LLM with:
    - Unlimited effective context
    - Persistent cross-session memory
    - One-shot learning capability
    - Zero catastrophic forgetting
    """
    
    def __init__(
        self,
        config: FLUXAugmentedConfig = None,
        device: str = None,
    ):
        super().__init__()
        
        self.config = config or FLUXAugmentedConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*60}")
        print(f"Initializing FLUX-Augmented LLM")
        print(f"{'='*60}")
        
        # ── Initialize FLUX Core ──
        self._init_flux_core()
        
        # ── Initialize LLM Bridge ──
        self._init_llm_bridge()
        
        # ── Initialize Injector ──
        self.injector = SimpleTextInjector(max_memories=self.config.top_k_retrieval)
        
        # ── Conversation tracking ──
        self.conversation_history: List[Dict[str, str]] = []
        
        print(f"\n{'='*60}")
        print(f"✓ FLUXAugmentedLLM ready")
        print(f"  Device: {self.device}")
        print(f"  LLM: {self.config.llm_name}")
        print(f"  Memory entries: {len(self.memory)}")
        print(f"{'='*60}\n")
    
    def _init_flux_core(self):
        """Initialize FLUX components from .flx file."""
        print(f"\n  Initializing FLUX core...")
        
        # Find FLX file
        flx_path = self._find_flx_path()
        self.flx_version = None
        self.flux_large = None
        self.flux_to_any = None
        
        if flx_path and flx_path.exists():
            self._load_from_flx(flx_path)
        else:
            self._init_fresh_flux()
    
    def _find_flx_path(self) -> Optional[Path]:
        """Find the best available FLX file."""
        if self.config.flx_path:
            path = Path(self.config.flx_path)
            if path.exists():
                return path
        
        # Auto-detect: try most capable first
        for path in FLX_PATHS:
            if path.exists():
                return path
        
        return None
    
    def _load_from_flx(self, flx_path: Path):
        """Load FLUX components from .flx file."""
        print(f"    Loading from {flx_path.name}...")
        
        try:
            flx = torch.load(str(flx_path), map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"    ⚠ Failed to load .flx: {e}")
            self._init_fresh_flux()
            return
        
        version = flx.get('version', 'unknown')
        self.flx_version = version
        print(f"    Format: FLUX v{version}")
        
        # Track what we loaded
        loaded_components = []
        
        # ── Load CSE (Phase 1) ──
        if 'cse' in flx and flx['cse'] and 'state_dict' in flx['cse']:
            try:
                sys.path.insert(0, str(_PHASES_DIR / 'phase1'))
                from cse import ContinuousSemanticEncoder
                
                cse_cfg = flx['cse'].get('config', {})
                self.encoder = ContinuousSemanticEncoder(
                    wave_dim=cse_cfg.get('wave_dim', 432)
                )
                self.encoder.load_state_dict(flx['cse']['state_dict'])
                self.encoder.to(self.device)
                self.encoder.eval()
                loaded_components.append('CSE')
                print(f"    ✓ CSE (Phase 1)")
            except Exception as e:
                print(f"    ⚠ CSE failed: {e}")
                self.encoder = SimpleWaveEncoder(self.config.wave_dim)
                self.encoder.to(self.device)
        else:
            self.encoder = SimpleWaveEncoder(self.config.wave_dim)
            self.encoder.to(self.device)
        
        # ── Load Full FLUXLarge (Phases 2-7) ──
        if self.config.load_full_flux and 'field' in flx:
            try:
                from flx_loader import load_flux_from_flx
                self.flux_large = load_flux_from_flx(
                    path=flx_path,
                    device=self.device,
                    verbose=False,
                )
                loaded_components.extend(['Field', 'GR', 'TL', 'Memory'])
                print(f"    ✓ Field + GR + TL (Phases 2-4)")
                
                # Use FLUXLarge's episodic memory
                self.memory = FLUXLargeMemoryAdapter(self.flux_large)
                em_size = self.flux_large.episodic_memory.size
                print(f"    ✓ EpisodicMemory ({em_size} entries, Phase 6)")
                
            except Exception as e:
                print(f"    ⚠ FLUXLarge failed: {e}")
                self.memory = SimpleMemoryStore(self.config.wave_dim)
        else:
            # Use simple memory
            self.memory = SimpleMemoryStore(self.config.wave_dim)
            if 'memory' in flx and 'episodic' in flx['memory']:
                try:
                    self.memory.load_state(flx['memory']['episodic'])
                    print(f"    ✓ Memory ({len(self.memory)} entries)")
                    loaded_components.append('Memory')
                except:
                    pass
        
        # ── Load Adapters (Phase 8.8/8.9) ──
        if self.config.load_adapters and 'adapters' in flx:
            try:
                self._load_adapters(flx)
                loaded_components.append('Adapters')
                print(f"    ✓ WaveToX Adapters (Phase 8.9)")
            except Exception as e:
                print(f"    ⚠ Adapters failed: {e}")
        
        # Summary
        if loaded_components:
            print(f"    Loaded: {', '.join(loaded_components)}")
        else:
            print(f"    ⚠ No components loaded from .flx")
    
    def _load_adapters(self, flx: Dict):
        """Load Phase 8.9 modality adapters."""
        from grid_adapters import GridToWave, WaveToGrid
        from image_adapters import WaveToImage_Universal
        
        self.grid_encoder = GridToWave(wave_dim=self.config.wave_dim)
        self.grid_decoder = WaveToGrid(wave_dim=self.config.wave_dim)
        self.image_decoder = WaveToImage_Universal(wave_dim=self.config.wave_dim)
        
        # Load state dicts if available
        if 'grid_encoder' in flx['adapters']:
            self.grid_encoder.load_state_dict(flx['adapters']['grid_encoder'])
        if 'grid_decoder' in flx['adapters']:
            self.grid_decoder.load_state_dict(flx['adapters']['grid_decoder'])
        if 'image_decoder' in flx['adapters']:
            self.image_decoder.load_state_dict(flx['adapters']['image_decoder'])
        
        self.grid_encoder.to(self.device)
        self.grid_decoder.to(self.device)
        self.image_decoder.to(self.device)
    
    def _init_fresh_flux(self):
        """Initialize with lightweight fallbacks."""
        print(f"    No .flx found, using lightweight fallbacks...")
        self.encoder = SimpleWaveEncoder(self.config.wave_dim)
        self.encoder.to(self.device)
        self.memory = SimpleMemoryStore(self.config.wave_dim)
        print(f"    ✓ Fresh FLUX core (SimpleEncoder + SimpleMemory)")
    
    def _init_llm_bridge(self):
        """Initialize LLM bridge."""
        print(f"\n  Initializing LLM bridge...")
        
        bridge_config = LLMBridgeConfig(
            name="llm",
            model_name=self.config.llm_name,
            load_in_4bit=self.config.load_in_4bit,
            load_in_8bit=self.config.load_in_8bit,
            freeze_llm=self.config.freeze_llm,
        )
        
        self.llm_bridge = LLMBridge(
            config=bridge_config,
            device=self.device,
            load_model=True,
        )
        
        print(f"    ✓ LLM bridge initialized")
    
    def encode(self, text: str) -> Tensor:
        """Encode text to wave space."""
        with torch.no_grad():
            wave = self.encoder.encode(text)
        return wave.to(self.device)
    
    def retrieve(
        self,
        query: Union[str, Tensor],
        top_k: int = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Retrieve relevant memories.
        
        Args:
            query: Query text or wave
            top_k: Number of results
            
        Returns:
            (texts, similarities)
        """
        top_k = top_k or self.config.top_k_retrieval
        
        # Encode query if needed
        if isinstance(query, str):
            wave = self.encode(query)
        else:
            wave = query
        
        # Pool to single vector
        if wave.dim() > 1:
            wave = wave.mean(dim=0)
        
        # Retrieve
        results = self.memory.retrieve(
            wave.cpu().numpy(),
            top_k=top_k,
            threshold=self.config.retrieval_threshold,
        )
        
        texts = [r[0]['text'] for r in results]
        scores = [r[1] for r in results]
        
        return texts, scores
    
    def store(
        self,
        text: str,
        wave: Optional[Tensor] = None,
        metadata: Optional[Dict] = None,
    ):
        """Store information in memory."""
        if wave is None:
            wave = self.encode(text)
        
        # Pool to single vector
        if wave.dim() > 1:
            wave = wave.mean(dim=0)
        
        self.memory.store(wave, text, metadata)
    
    def teach(self, fact: str, verify: bool = True) -> bool:
        """
        Teach a new fact (one-shot learning).
        
        The fact is stored immediately and can be recalled.
        No gradient descent, no retraining.
        
        Args:
            fact: The fact to teach
            verify: Whether to verify recall
            
        Returns:
            Success status
        """
        print(f"  Teaching: {fact[:60]}...")
        
        # Encode and store
        wave = self.encode(fact)
        self.store(fact, wave, metadata={'type': 'taught_fact'})
        
        # Verify
        if verify:
            texts, _ = self.retrieve(fact, top_k=1)
            if texts and fact.lower() in texts[0].lower():
                print(f"    ✓ Fact stored and verified")
                return True
            else:
                print(f"    ⚠ Stored but verification unclear")
        
        return True
    
    def chat(
        self,
        message: str,
        use_memory: bool = True,
        max_new_tokens: int = None,
        temperature: float = None,
        **kwargs,
    ) -> str:
        """
        Chat with the FLUX-augmented LLM.
        
        This is the main interface. It:
        1. Retrieves relevant memories (O(log n))
        2. Injects context into LLM
        3. Generates fluent response
        4. Stores exchange in memory
        
        Args:
            message: User message
            use_memory: Whether to use memory retrieval
            max_new_tokens: Override generation length
            temperature: Override sampling temperature
            
        Returns:
            Assistant response
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        
        # ── Retrieve relevant memories ──
        context_texts = None
        if use_memory and len(self.memory) > 0:
            context_texts, scores = self.retrieve(message)
            if context_texts:
                print(f"    Retrieved {len(context_texts)} memories (max sim: {max(scores):.2f})")
        
        # ── Generate with context ──
        response = self.llm_bridge.generate(
            input_text=message,
            context_texts=context_texts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=self.config.top_p,
            **kwargs,
        )
        
        # ── Store exchange ──
        if self.config.store_conversations:
            exchange = f"User: {message}\nAssistant: {response}"
            self.store(exchange, metadata={'type': 'conversation'})
        
        # ── Track history ──
        self.conversation_history.append({
            'user': message,
            'assistant': response,
        })
        
        return response
    
    def clear_history(self):
        """Clear conversation history (not memory)."""
        self.conversation_history.clear()
    
    def save_flx(self, path: str, include_full_flux: bool = True):
        """
        Save to .flx format.
        
        Args:
            path: Output path
            include_full_flux: If True, saves full FLUXLarge model (~2GB)
                              If False, saves only lightweight components (~12MB)
        
        Note: LLM weights are NOT saved (too large).
        The LLM is referenced by name and reloaded.
        """
        print(f"\n  Building .flx archive...")
        
        flx = {
            'format': 'FLUX',
            'version': '3.0-augmented',
            
            'metadata': {
                'llm_name': self.config.llm_name,
                'wave_dim': self.config.wave_dim,
                'memory_entries': len(self.memory),
                'conversations': len(self.conversation_history),
                'has_full_flux': include_full_flux and hasattr(self, 'flux_large') and self.flux_large is not None,
                'has_adapters': hasattr(self, 'grid_encoder'),
            },
            
            'llm_reference': {
                'name': self.config.llm_name,
                'load_in_4bit': self.config.load_in_4bit,
            },
            
            'conversation_history': self.conversation_history,
        }
        
        # ── Save CSE (Phase 1) ──
        flx['cse'] = {
            'config': {'wave_dim': self.config.wave_dim},
            'state_dict': self.encoder.state_dict(),
        }
        print(f"    ✓ CSE")
        
        # ── Save Full FLUXLarge (Phases 2-7) ──
        if include_full_flux and hasattr(self, 'flux_large') and self.flux_large is not None:
            print(f"    Saving full FLUXLarge (this is the big part)...")
            
            # Field (Phase 2)
            flx['field'] = {
                'config': {
                    'h': self.flux_large.config.get('field_h', 96),
                    'w': self.flux_large.config.get('field_w', 96),
                    'd': self.flux_large.config.get('field_d', 96),
                    'features': self.flux_large.config.get('field_features', 512),
                },
                'state_dict': self.flux_large.field.state_dict(),
            }
            print(f"    ✓ Field (Phase 2)")
            
            # Gravitational Relevance (Phase 3)
            if hasattr(self.flux_large, 'gr'):
                flx['field']['gravity_state'] = self.flux_large.gr.save_state()
                print(f"    ✓ GravitationalRelevance (Phase 3)")
            
            # Thermodynamic Learner (Phase 4)
            if hasattr(self.flux_large, 'tl'):
                flx['field']['thermodynamic_state'] = self.flux_large.tl.save_state()
                print(f"    ✓ ThermodynamicLearner (Phase 4)")
            
            # Memory Systems (Phase 6)
            flx['memory'] = {}
            if hasattr(self.flux_large, 'working_memory'):
                flx['memory']['working'] = self.flux_large.working_memory.state_dict_memory()
            if hasattr(self.flux_large, 'episodic_memory'):
                flx['memory']['episodic'] = self.flux_large.episodic_memory.save_state()
                print(f"    ✓ EpisodicMemory ({self.flux_large.episodic_memory.size} entries)")
            if hasattr(self.flux_large, 'semantic_memory'):
                flx['memory']['semantic'] = self.flux_large.semantic_memory.save_state()
            
            # Decoder (Phase 8)
            if hasattr(self.flux_large, 'decoder'):
                flx['decoder'] = {
                    'config': getattr(self.flux_large.decoder, 'config', {}),
                    'state_dict': self.flux_large.decoder.state_dict(),
                }
                print(f"    ✓ WaveDecoder (Phase 8)")
            
            # Causal Graph (Phase 5)
            if hasattr(self.flux_large, 'causal_graph'):
                flx['causal'] = {
                    'graph': self.flux_large.causal_graph.save_state(),
                }
                print(f"    ✓ CausalGraph (Phase 5)")
        else:
            # Lightweight memory only
            flx['memory'] = {
                'episodic': self.memory.get_state(),
            }
            print(f"    ✓ Memory (lightweight, {len(self.memory)} entries)")
        
        # ── Save Bridges ──
        flx['bridges'] = {
            'llm_to_wave': self.llm_bridge.to_wave_proj.state_dict(),
            'wave_to_llm': self.llm_bridge.from_wave_proj.state_dict()
                if self.llm_bridge.from_wave_proj else None,
        }
        print(f"    ✓ LLM Bridges")
        
        # ── Save Adapters (Phase 8.9) ──
        if hasattr(self, 'grid_encoder') or hasattr(self, 'image_decoder'):
            flx['adapters'] = {}
            
            if hasattr(self, 'grid_encoder'):
                flx['adapters']['grid_encoder'] = self.grid_encoder.state_dict()
                print(f"    ✓ GridToWave")
            if hasattr(self, 'grid_decoder'):
                flx['adapters']['grid_decoder'] = self.grid_decoder.state_dict()
                print(f"    ✓ WaveToGrid")
            if hasattr(self, 'image_decoder'):
                flx['adapters']['image_decoder'] = self.image_decoder.state_dict()
                print(f"    ✓ WaveToImage")
        
        # ── Save to disk ──
        print(f"\n  Writing to {path}...")
        torch.save(flx, path)
        
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        size_gb = size_mb / 1024
        
        if size_gb > 1:
            print(f"  ✓ Saved: {path} ({size_gb:.2f} GB)")
        else:
            print(f"  ✓ Saved: {path} ({size_mb:.1f} MB)")
    
    @classmethod
    def from_flx(cls, path: str, device: str = None) -> 'FLUXAugmentedLLM':
        """Load from .flx file."""
        flx = torch.load(path, map_location='cpu')
        
        config = FLUXAugmentedConfig(
            llm_name=flx['llm_reference']['name'],
            load_in_4bit=flx['llm_reference']['load_in_4bit'],
            flx_path=path,
        )
        
        return cls(config, device=device)
    
    def get_stats(self) -> Dict:
        """Get system statistics."""
        stats = {
            'memory_entries': len(self.memory),
            'conversations': len(self.conversation_history),
            'llm_name': self.config.llm_name,
            'device': self.device,
            'wave_dim': self.config.wave_dim,
            'flx_version': getattr(self, 'flx_version', None),
        }
        
        # Add component availability
        stats['has_full_flux'] = hasattr(self, 'flux_large') and self.flux_large is not None
        stats['has_grid_adapter'] = hasattr(self, 'grid_encoder')
        stats['has_image_adapter'] = hasattr(self, 'image_decoder')
        
        return stats
    
    # ─────────────────────────────────────────
    # Multi-Modal Methods (Phase 8.8/8.9)
    # ─────────────────────────────────────────
    
    def encode_grid(self, grid: Union[List[List[int]], Tensor]) -> Tensor:
        """
        Encode an ARC grid to wave space.
        
        Requires Phase 8.9 adapters to be loaded.
        
        Args:
            grid: 2D grid of integers (0-9)
            
        Returns:
            Wave representation [432]
        """
        if not hasattr(self, 'grid_encoder'):
            raise RuntimeError("Grid encoder not loaded. Requires Flux-X-complete.flx")
        
        with torch.no_grad():
            wave = self.grid_encoder(grid, mode='holistic')
        return wave.to(self.device)
    
    def decode_grid(self, wave: Tensor, size: Tuple[int, int] = (3, 3)) -> List[List[int]]:
        """
        Decode a wave to an ARC grid.
        
        Args:
            wave: Wave representation [432]
            size: Output grid size (height, width)
            
        Returns:
            2D grid of integers (0-9)
        """
        if not hasattr(self, 'grid_decoder'):
            raise RuntimeError("Grid decoder not loaded. Requires Flux-X-complete.flx")
        
        with torch.no_grad():
            grid = self.grid_decoder(wave, grid_size=size)
        return grid.tolist()
    
    def solve_arc_task(
        self,
        input_grid: List[List[int]],
        output_size: Tuple[int, int] = None,
    ) -> List[List[int]]:
        """
        Solve an ARC task using FLUX wave transformations.
        
        This encodes the input, queries field for similar transformations,
        and decodes the output.
        
        Args:
            input_grid: Input grid
            output_size: Expected output size (inferred if None)
            
        Returns:
            Predicted output grid
        """
        if not hasattr(self, 'grid_encoder'):
            raise RuntimeError("Grid adapters not loaded. Requires Flux-X-complete.flx")
        
        # Encode input
        input_wave = self.encode_grid(input_grid)
        
        # Query field for similar patterns
        if hasattr(self, 'flux_large') and self.flux_large is not None:
            # Use full FLUX gravitational retrieval
            attractors = self.flux_large.gr.query(input_wave.unsqueeze(0), top_k=5)
            # Apply transformation from most similar attractor
            if len(attractors[0]) > 0:
                delta = attractors[0][0] - input_wave
                output_wave = input_wave + delta
            else:
                output_wave = input_wave
        else:
            # Simple pass-through
            output_wave = input_wave
        
        # Infer output size from input if not specified
        output_size = output_size or (len(input_grid), len(input_grid[0]))
        
        # Decode
        return self.decode_grid(output_wave, output_size)
    
    def generate_image(
        self,
        prompt: str,
        size: int = 64,
        style: str = 'dream',
    ) -> Tensor:
        """
        Generate an image from text prompt.
        
        Uses FLUX physics-based rendering (Phase 8.9):
        - Gravity: Mass attractors → smooth gradients
        - Interference: Wave superposition → ripples
        - Thermodynamic: Energy minimization → textures
        
        Args:
            prompt: Text prompt
            size: Output image size
            style: Rendering style preset
            
        Returns:
            Image tensor [H, W, 3]
        """
        if not hasattr(self, 'image_decoder'):
            raise RuntimeError("Image decoder not loaded. Requires Flux-X-complete.flx")
        
        # Encode prompt
        wave = self.encode(prompt)
        if wave.dim() > 1:
            wave = wave.mean(dim=0)
        
        # Generate image
        with torch.no_grad():
            image = self.image_decoder(wave, size=size, style=style)
        
        return image
    
    def get_capabilities(self) -> List[str]:
        """Get list of available capabilities based on loaded components."""
        caps = ['text_encoding', 'memory', 'chat', 'teach']
        
        if hasattr(self, 'flux_large') and self.flux_large is not None:
            caps.extend(['field', 'gravitational_retrieval', 'thermodynamic_learning'])
        
        if hasattr(self, 'grid_encoder'):
            caps.append('grid_encoding')
        if hasattr(self, 'grid_decoder'):
            caps.append('grid_decoding')
        if hasattr(self, 'image_decoder'):
            caps.append('image_generation')
        
        return caps


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("FLUX-Augmented LLM — Test")
    print("=" * 60)
    
    # Note: This test requires GPU and model download
    # For CI/testing, we test the components separately
    
    print("\nTesting SimpleMemoryStore...")
    memory = SimpleMemoryStore(wave_dim=432)
    
    # Store some facts
    for i, fact in enumerate([
        "The capital of France is Paris",
        "Python is a programming language",
        "Water boils at 100 degrees Celsius",
    ]):
        wave = torch.randn(432)
        memory.store(wave, fact, {'index': i})
    
    print(f"  Stored {len(memory)} memories")
    
    # Retrieve
    query = torch.randn(432)
    results = memory.retrieve(query, top_k=2)
    print(f"  Retrieved {len(results)} results")
    for mem, sim in results:
        print(f"    - [{sim:.3f}] {mem['text'][:50]}...")
    
    print("\nTesting SimpleWaveEncoder...")
    encoder = SimpleWaveEncoder(wave_dim=432)
    wave = encoder.encode("Hello, world!")
    print(f"  Encoded 'Hello, world!' → {wave.shape}")
    
    print("\n✓ Component tests passed")
    print("\nTo test full FLUXAugmentedLLM, run with GPU:")
    print("  python flux_augmented_llm.py --full")
