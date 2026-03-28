"""
Phase 8.9: FluxToAny — Unified Multi-Modal Model

The universal FLUX interface. Load once, process any modality.

Usage:
    model = FluxToAny.from_flx('Flux-X.flx')
    
    # Text → Text
    output = model('Hello world', output_modality='text')
    
    # Grid → Grid (ARC task)
    output = model(input_grid, output_modality='grid', grid_size=(3, 3))
    
    # Text → Image
    output = model('A sunset', output_modality='image', style='photorealistic')
    
    # Image → Text (description)
    output = model(image_tensor, output_modality='text')
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import sys
from pathlib import Path

# Add project paths
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PHASES_DIR / 'phase8_8') not in sys.path:
    sys.path.insert(0, str(_PHASES_DIR / 'phase8_8'))
if str(_PHASES_DIR / 'phase8_9') not in sys.path:
    sys.path.insert(0, str(_PHASES_DIR / 'phase8_9'))


# ─────────────────────────────────────────────
# FluxToAny Output
# ─────────────────────────────────────────────

@dataclass
class FluxOutput:
    """Output from FluxToAny."""
    output: Any                  # The decoded output in target modality
    wave: Tensor                 # The intermediate wave representation
    input_modality: str          # Detected input modality
    output_modality: str         # Requested output modality
    attractors: Optional[list] = None   # Retrieved attractors (if used)
    memories: Optional[list] = None     # Retrieved memories (if used)


# ─────────────────────────────────────────────
# FluxToAny Model
# ─────────────────────────────────────────────

class FluxToAny(nn.Module):
    """
    Universal FLUX interface.
    
    Architecture:
        Input → XToWave → [432-dim wave] → Field/Memory → WaveToX → Output
        
    The core FLUX model (CSE, Field, Memory, CGN) operates purely in wave space.
    FluxToAny wraps it with modality adapters for universal I/O.
    """
    
    def __init__(
        self,
        flux_model: nn.Module,
        device: str = 'cpu',
    ):
        super().__init__()
        self.flux = flux_model
        self.device = device
        
        # Input adapters: modality → wave
        self.input_adapters = nn.ModuleDict()
        
        # Output adapters: wave → modality
        self.output_adapters = nn.ModuleDict()
        
        # Load adapters lazily
        self._adapters_loaded = False
    
    def _load_adapters(self):
        """Load all modality adapters."""
        if self._adapters_loaded:
            return
        
        # Import adapters
        from text_adapters import TextToWave, WaveToText
        from grid_adapters import GridToWave, WaveToGrid
        from image_adapters import ImageToWave, WaveToImage_Universal
        from audio_adapters import AudioToWave, WaveToAudio
        
        # Text
        text_to_wave = TextToWave(device=self.device)
        if hasattr(self.flux, 'cse'):
            text_to_wave.cse = self.flux.cse
        self.input_adapters['text'] = text_to_wave
        self.output_adapters['text'] = WaveToText(device=self.device)
        
        # Grid
        self.input_adapters['grid'] = GridToWave(device=self.device)
        self.output_adapters['grid'] = WaveToGrid(device=self.device)
        
        # Image
        self.input_adapters['image'] = ImageToWave(device=self.device)
        self.output_adapters['image'] = WaveToImage_Universal(device=self.device)
        
        # Audio (stubs)
        self.input_adapters['audio'] = AudioToWave(device=self.device)
        self.output_adapters['audio'] = WaveToAudio(device=self.device)
        
        self._adapters_loaded = True
    
    @classmethod
    def from_flx(cls, path_or_name: str, device: str = 'cpu') -> 'FluxToAny':
        """
        Load FluxToAny from .flx file.
        
        Args:
            path_or_name: Path to .flx file or 'Flux-X.flx' for HF download
            device: Target device
        
        Returns:
            FluxToAny model ready for inference
        """
        # Try to load from file or HuggingFace
        try:
            from flx_loader import load_flux_from_flx, download_flx_from_hf
        except ImportError:
            # Fallback to phase8_5
            sys.path.insert(0, str(_PHASES_DIR / 'phase8_5'))
            from flx_loader import load_flux_from_flx, download_flx_from_hf
        
        # Check if file exists locally
        flx_path = Path(path_or_name)
        if not flx_path.exists():
            # Try checkpoints directory
            checkpoint_path = _PROJECT_ROOT / 'checkpoints' / path_or_name
            if checkpoint_path.exists():
                flx_path = checkpoint_path
            else:
                # Download from HuggingFace
                print(f"Downloading {path_or_name} from HuggingFace...")
                flx_path = download_flx_from_hf(path_or_name)
        
        # Load model
        flux_model = load_flux_from_flx(str(flx_path), device=device)
        
        # Create FluxToAny wrapper
        model = cls(flux_model, device=device)
        model._load_adapters()
        
        return model
    
    def detect_modality(self, x: Any) -> str:
        """
        Auto-detect input modality.
        
        Args:
            x: Input data
        
        Returns:
            Modality string: 'text', 'grid', 'image', 'audio'
        """
        if isinstance(x, str):
            return 'text'
        
        if isinstance(x, list):
            # Check if it's a grid (list of lists of ints)
            if all(isinstance(row, list) for row in x):
                if all(isinstance(v, int) for row in x for v in row):
                    return 'grid'
        
        if isinstance(x, Tensor):
            if x.dim() == 2 and x.dtype in [torch.long, torch.int]:
                return 'grid'
            if x.dim() == 3 and x.shape[-1] == 3:
                return 'image'
            if x.dim() == 3 and x.shape[0] == 3:
                return 'image'
            if x.dim() == 1:
                return 'audio'
        
        raise ValueError(f"Cannot detect modality for input type: {type(x)}")
    
    def encode(self, x: Any, modality: Optional[str] = None) -> Tensor:
        """
        Encode input to wave space.
        
        Args:
            x: Input in any supported modality
            modality: Override modality detection
        
        Returns:
            [seq, 432] or [432] wave tensor
        """
        self._load_adapters()
        
        if modality is None:
            modality = self.detect_modality(x)
        
        adapter = self.input_adapters.get(modality)
        if adapter is None:
            raise ValueError(f"No input adapter for modality: {modality}")
        
        return adapter.encode(x)
    
    def decode(
        self,
        wave: Tensor,
        modality: str,
        **kwargs,
    ) -> Any:
        """
        Decode wave to output modality.
        
        Args:
            wave: [432] or [seq, 432] wave tensor
            modality: Target output modality
            **kwargs: Modality-specific options
        
        Returns:
            Output in target modality
        """
        self._load_adapters()
        
        adapter = self.output_adapters.get(modality)
        if adapter is None:
            raise ValueError(f"No output adapter for modality: {modality}")
        
        return adapter.decode(wave, **kwargs)
    
    def process_wave(
        self,
        wave: Tensor,
        use_field: bool = True,
        use_memory: bool = True,
        k_attractors: int = 4,
        k_memories: int = 3,
    ) -> Tuple[Tensor, list, list]:
        """
        Process wave through FLUX core (field + memory).
        
        Args:
            wave: [432] or [seq, 432] input wave
            use_field: Query resonance field for attractors
            use_memory: Query episodic memory
            k_attractors: Number of attractors to retrieve
            k_memories: Number of memories to retrieve
        
        Returns:
            (processed_wave, attractors, memories)
        """
        # Pool to single vector if sequence
        if wave.dim() == 2:
            pooled = wave.mean(dim=0)
        else:
            pooled = wave
        
        attractors = []
        memories = []
        
        # Query field
        if use_field and hasattr(self.flux, 'field'):
            try:
                field_result = self.flux.field.query(pooled.unsqueeze(0), k=k_attractors)
                if isinstance(field_result, tuple):
                    attractors = list(field_result[0])  # (waves, similarities)
                else:
                    attractors = list(field_result)
            except Exception as e:
                print(f"Field query failed: {e}")
        
        # Query memory
        if use_memory and hasattr(self.flux, 'episodic_memory'):
            try:
                mem_results = self.flux.episodic_memory.search(pooled, k=k_memories)
                memories = mem_results
            except Exception as e:
                print(f"Memory query failed: {e}")
        
        # Blend wave with retrieved context
        processed = pooled.clone()
        
        if attractors:
            for attr in attractors[:2]:  # Top 2 attractors
                if isinstance(attr, Tensor):
                    processed = processed + 0.1 * attr.to(self.device)
        
        return processed, attractors, memories
    
    def forward(
        self,
        x: Any,
        output_modality: str = 'text',
        input_modality: Optional[str] = None,
        use_field: bool = True,
        use_memory: bool = True,
        **output_kwargs,
    ) -> FluxOutput:
        """
        Process input and produce output in target modality.
        
        Args:
            x: Input in any supported modality
            output_modality: Target output modality
            input_modality: Override input modality detection
            use_field: Query resonance field
            use_memory: Query episodic memory
            **output_kwargs: Passed to output adapter
        
        Returns:
            FluxOutput with decoded result and metadata
        """
        self._load_adapters()
        
        # Detect modality
        if input_modality is None:
            input_modality = self.detect_modality(x)
        
        # Encode
        wave = self.encode(x, modality=input_modality)
        
        # Process through FLUX core
        processed, attractors, memories = self.process_wave(
            wave,
            use_field=use_field,
            use_memory=use_memory,
        )
        
        # Decode
        output = self.decode(processed, modality=output_modality, **output_kwargs)
        
        return FluxOutput(
            output=output,
            wave=processed,
            input_modality=input_modality,
            output_modality=output_modality,
            attractors=attractors,
            memories=memories,
        )
    
    def __call__(self, x: Any, output_modality: str = 'text', **kwargs) -> Any:
        """Convenience: return just the output."""
        result = self.forward(x, output_modality=output_modality, **kwargs)
        return result.output


# ─────────────────────────────────────────────
# Quick Load Function
# ─────────────────────────────────────────────

def load_flux_universal(
    model_name: str = 'Flux-X.flx',
    device: str = 'cpu',
) -> FluxToAny:
    """
    Load FluxToAny from HuggingFace or local file.
    
    Args:
        model_name: 'Flux-X.flx', 'Flux-beta.flx', or path
        device: Target device
    
    Returns:
        FluxToAny model
    """
    return FluxToAny.from_flx(model_name, device=device)
