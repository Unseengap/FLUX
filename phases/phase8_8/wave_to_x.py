"""
Phase 8.8: WaveToX Base Classes + Registry

The Universal Modality Interface for FLUX.
Every modality adapter implements this interface.
The FLUX core never imports specific adapters — it uses the registry.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any, Optional, Type, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass


# ─────────────────────────────────────────────
# Data Types
# ─────────────────────────────────────────────

@dataclass
class ModalitySpec:
    """Specification for a modality."""
    name: str                    # 'text', 'grid', 'image', 'audio'
    wave_dim: int = 432          # FLUX wave dimension (constant)
    supports_sequence: bool = True   # Can handle variable-length input?
    discrete: bool = False       # Discrete tokens (grid/text) vs continuous
    spatial: bool = False        # Has 2D/3D structure (grid/image)
    temporal: bool = False       # Has time dimension (audio/video)


MODALITY_SPECS = {
    'text': ModalitySpec('text', supports_sequence=True, discrete=True),
    'grid': ModalitySpec('grid', supports_sequence=False, discrete=True, spatial=True),
    'image': ModalitySpec('image', supports_sequence=True, discrete=False, spatial=True),
    'audio': ModalitySpec('audio', supports_sequence=True, discrete=False, temporal=True),
}


# ─────────────────────────────────────────────
# Abstract Base Classes
# ─────────────────────────────────────────────

class XToWave(nn.Module, ABC):
    """
    Input adapter: Modality → Wave space.
    
    Converts any input modality into FLUX's 432-dim semantic waves.
    The output can be:
    - [432] single vector (for grid-as-whole)
    - [seq, 432] sequence (for text, patches)
    """
    
    def __init__(self, modality: str, wave_dim: int = 432):
        super().__init__()
        self.modality = modality
        self.wave_dim = wave_dim
        self.spec = MODALITY_SPECS.get(modality, ModalitySpec(modality))
    
    @abstractmethod
    def encode(self, x: Any) -> Tensor:
        """
        Encode input to wave space.
        
        Args:
            x: Input in native format (str, grid tensor, image tensor, etc.)
        
        Returns:
            Tensor of shape [432] or [seq, 432]
        """
        pass
    
    def forward(self, x: Any) -> Tensor:
        return self.encode(x)


class WaveToX(nn.Module, ABC):
    """
    Output adapter: Wave space → Modality.
    
    Converts FLUX's 432-dim semantic waves into any output modality.
    The input can be:
    - [432] single vector
    - [seq, 432] sequence of waves
    """
    
    def __init__(self, modality: str, wave_dim: int = 432):
        super().__init__()
        self.modality = modality
        self.wave_dim = wave_dim
        self.spec = MODALITY_SPECS.get(modality, ModalitySpec(modality))
    
    @abstractmethod
    def decode(self, waves: Tensor, **kwargs) -> Any:
        """
        Decode waves to output modality.
        
        Args:
            waves: [432] or [seq, 432] wave tensor
            **kwargs: Modality-specific options (temperature, grid_size, etc.)
        
        Returns:
            Output in native format (str, grid tensor, image tensor, etc.)
        """
        pass
    
    def forward(self, waves: Tensor, **kwargs) -> Any:
        return self.decode(waves, **kwargs)


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

_INPUT_ADAPTERS: Dict[str, Type[XToWave]] = {}
_OUTPUT_ADAPTERS: Dict[str, Type[WaveToX]] = {}


def register_input_adapter(modality: str):
    """Decorator to register an input adapter."""
    def decorator(cls: Type[XToWave]):
        _INPUT_ADAPTERS[modality] = cls
        return cls
    return decorator


def register_output_adapter(modality: str):
    """Decorator to register an output adapter."""
    def decorator(cls: Type[WaveToX]):
        _OUTPUT_ADAPTERS[modality] = cls
        return cls
    return decorator


def get_input_adapter(modality: str, **kwargs) -> XToWave:
    """Get input adapter for modality."""
    if modality not in _INPUT_ADAPTERS:
        raise ValueError(f"No input adapter for modality: {modality}")
    return _INPUT_ADAPTERS[modality](**kwargs)


def get_output_adapter(modality: str, **kwargs) -> WaveToX:
    """Get output adapter for modality."""
    if modality not in _OUTPUT_ADAPTERS:
        raise ValueError(f"No output adapter for modality: {modality}")
    return _OUTPUT_ADAPTERS[modality](**kwargs)


def list_adapters() -> Dict[str, Dict[str, bool]]:
    """List all registered adapters."""
    return {
        'input': {m: True for m in _INPUT_ADAPTERS},
        'output': {m: True for m in _OUTPUT_ADAPTERS},
    }


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("WaveToX Base Classes — Module Check")
    print(f"  Modality specs: {list(MODALITY_SPECS.keys())}")
    print(f"  Input adapters: {list(_INPUT_ADAPTERS.keys())}")
    print(f"  Output adapters: {list(_OUTPUT_ADAPTERS.keys())}")
    print("  ✓ Base module ready")
