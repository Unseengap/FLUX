"""
model_bridge.py — Abstract base for all model adapters

Every external model connects to FLUX through a bridge.
Bridges translate between the model's representation space and FLUX's 432-dim wave space.

This is the foundation for universal model assembly — any AI model, any size,
any family can be integrated without catastrophic forgetting because bridges
connect models via the field, not merged weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
import sys
from pathlib import Path

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class BridgeConfig:
    """Configuration for a model bridge."""
    name: str = "generic"        # Bridge identifier
    source_dim: int = 768        # Model's hidden dimension
    wave_dim: int = 432          # FLUX wave dimension (constant)
    hidden_dim: int = 512        # Intermediate projection dimension
    num_layers: int = 2          # Projection depth
    dropout: float = 0.1
    bidirectional: bool = True   # Can project both directions?
    normalize_output: bool = True
    use_residual: bool = True


# ─────────────────────────────────────────────
# Abstract Base Class
# ─────────────────────────────────────────────

class ModelBridge(nn.Module, ABC):
    """
    Abstract base class for all model bridges.
    
    A bridge translates between an external model's representation space
    and FLUX's 432-dimensional wave space. This enables:
    
    1. Any model's knowledge → stored as FLUX field attractors
    2. FLUX memories → injected into any model's context
    3. Multiple models → connected through shared field (no weight merging)
    4. Zero forgetting → each model has separate bridge, shared field
    
    Required methods:
    - to_wave(): Model representation → Wave space
    - from_wave(): Wave space → Model representation (if bidirectional)
    - get_model_hidden_states(): Extract representations from source model
    - inject_context(): Add retrieved context to model inputs
    
    The bridge is responsible for:
    - Dimension alignment (source_dim ↔ 432)
    - Representation normalization
    - Semantic preservation (similar things → nearby waves)
    """
    
    def __init__(self, config: BridgeConfig):
        super().__init__()
        self.config = config
        
        # ── To Wave projection (source_dim → 432) ──
        if config.num_layers == 1:
            self.to_wave_proj = nn.Sequential(
                nn.Linear(config.source_dim, config.wave_dim),
                nn.LayerNorm(config.wave_dim) if config.normalize_output else nn.Identity(),
            )
        else:
            layers = []
            in_dim = config.source_dim
            for i in range(config.num_layers - 1):
                out_dim = config.hidden_dim if i < config.num_layers - 2 else config.hidden_dim
                layers.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ])
                in_dim = out_dim
            layers.extend([
                nn.Linear(in_dim, config.wave_dim),
                nn.LayerNorm(config.wave_dim) if config.normalize_output else nn.Identity(),
            ])
            self.to_wave_proj = nn.Sequential(*layers)
        
        # ── From Wave projection (432 → source_dim) ──
        if config.bidirectional:
            if config.num_layers == 1:
                self.from_wave_proj = nn.Sequential(
                    nn.Linear(config.wave_dim, config.source_dim),
                    nn.LayerNorm(config.source_dim) if config.normalize_output else nn.Identity(),
                )
            else:
                layers = []
                in_dim = config.wave_dim
                for i in range(config.num_layers - 1):
                    out_dim = config.hidden_dim
                    layers.extend([
                        nn.Linear(in_dim, out_dim),
                        nn.LayerNorm(out_dim),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                    ])
                    in_dim = out_dim
                layers.extend([
                    nn.Linear(in_dim, config.source_dim),
                    nn.LayerNorm(config.source_dim) if config.normalize_output else nn.Identity(),
                ])
                self.from_wave_proj = nn.Sequential(*layers)
        else:
            self.from_wave_proj = None
        
        # ── Optional residual connections ──
        if config.use_residual and config.source_dim == config.wave_dim:
            self.use_residual = True
        else:
            self.use_residual = False
    
    def to_wave(self, x: Tensor) -> Tensor:
        """
        Project model representation to wave space.
        
        Args:
            x: Model representation 
               - [batch, seq, source_dim] for sequences
               - [batch, source_dim] for pooled
               - [source_dim] for single vector
            
        Returns:
            Wave representation with same batch/seq structure but dim=432
        """
        output = self.to_wave_proj(x)
        if self.use_residual:
            output = output + x[..., :self.config.wave_dim]
        return output
    
    def from_wave(self, wave: Tensor) -> Tensor:
        """
        Project wave back to model representation space.
        
        Args:
            wave: Wave representation [batch, seq, 432] or [batch, 432] or [432]
            
        Returns:
            Model representation with same structure but dim=source_dim
        """
        if self.from_wave_proj is None:
            raise ValueError(f"Bridge '{self.config.name}' is not bidirectional")
        
        output = self.from_wave_proj(wave)
        if self.use_residual:
            output = output + wave[..., :self.config.source_dim]
        return output
    
    @abstractmethod
    def get_model_hidden_states(self, inputs: Any) -> Tensor:
        """
        Extract hidden states from the source model.
        
        Subclasses implement model-specific extraction logic.
        
        Args:
            inputs: Model-specific input format
            
        Returns:
            Hidden states tensor [batch, seq, source_dim]
        """
        pass
    
    @abstractmethod
    def inject_context(
        self, 
        model_inputs: Any, 
        context_waves: Tensor,
        context_texts: Optional[List[str]] = None,
    ) -> Any:
        """
        Inject retrieved context into model inputs.
        
        Subclasses implement model-specific injection logic.
        
        Args:
            model_inputs: Original model inputs
            context_waves: Retrieved memories in wave space [num_memories, 432]
            context_texts: Optional text versions of memories
            
        Returns:
            Modified model inputs with context
        """
        pass
    
    def encode_to_wave(self, inputs: Any) -> Tensor:
        """
        Full pipeline: inputs → model hidden states → wave space.
        
        Args:
            inputs: Model-specific input format
            
        Returns:
            Wave representation [batch, seq, 432]
        """
        hidden = self.get_model_hidden_states(inputs)
        wave = self.to_wave(hidden)
        return wave
    
    def forward(self, x: Tensor, direction: str = 'to_wave') -> Tensor:
        """Forward pass in specified direction."""
        if direction == 'to_wave':
            return self.to_wave(x)
        elif direction == 'from_wave':
            return self.from_wave(x)
        else:
            raise ValueError(f"Unknown direction: {direction}. Use 'to_wave' or 'from_wave'")
    
    def get_param_count(self) -> int:
        """Count trainable parameters in bridge."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"source_dim={self.config.source_dim}, "
            f"wave_dim={self.config.wave_dim}, "
            f"bidirectional={self.config.bidirectional}, "
            f"params={self.get_param_count():,})"
        )


# ─────────────────────────────────────────────
# Bridge Registry
# ─────────────────────────────────────────────

_BRIDGE_REGISTRY: Dict[str, type] = {}


def register_bridge(name: str):
    """Decorator to register a bridge class."""
    def decorator(cls):
        _BRIDGE_REGISTRY[name] = cls
        return cls
    return decorator


def get_bridge(name: str) -> type:
    """Get a bridge class by name."""
    if name not in _BRIDGE_REGISTRY:
        raise ValueError(f"Unknown bridge: {name}. Available: {list(_BRIDGE_REGISTRY.keys())}")
    return _BRIDGE_REGISTRY[name]


def list_bridges() -> List[str]:
    """List all registered bridges."""
    return list(_BRIDGE_REGISTRY.keys())


# ─────────────────────────────────────────────
# Simple Test Bridge (for testing)
# ─────────────────────────────────────────────

@register_bridge("test")
class TestBridge(ModelBridge):
    """Simple bridge for testing — no actual model."""
    
    def __init__(self, source_dim: int = 768, wave_dim: int = 432):
        config = BridgeConfig(
            name="test",
            source_dim=source_dim,
            wave_dim=wave_dim,
        )
        super().__init__(config)
    
    def get_model_hidden_states(self, inputs: Tensor) -> Tensor:
        """For testing, inputs ARE the hidden states."""
        return inputs
    
    def inject_context(
        self, 
        model_inputs: Tensor, 
        context_waves: Tensor,
        context_texts: Optional[List[str]] = None,
    ) -> Tensor:
        """For testing, just concatenate."""
        context_hidden = self.from_wave(context_waves)
        if model_inputs.dim() == 2:
            # [batch, dim] → prepend context
            return torch.cat([context_hidden.mean(dim=0, keepdim=True), model_inputs], dim=0)
        else:
            # [batch, seq, dim] → prepend context
            batch_size = model_inputs.shape[0]
            context_expanded = context_hidden.unsqueeze(0).expand(batch_size, -1, -1)
            return torch.cat([context_expanded, model_inputs], dim=1)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Model Bridge — Test")
    print("=" * 50)
    
    # Test the test bridge
    bridge = TestBridge(source_dim=768, wave_dim=432)
    print(f"\n{bridge}")
    
    # Test forward pass
    x = torch.randn(2, 10, 768)  # [batch, seq, source_dim]
    wave = bridge.to_wave(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Wave shape: {wave.shape}")
    
    # Test roundtrip
    reconstructed = bridge.from_wave(wave)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Test context injection
    context = torch.randn(5, 432)  # 5 retrieved memories
    injected = bridge.inject_context(x, context)
    print(f"Injected shape: {injected.shape}")
    
    print("\n✓ Model Bridge test passed")
