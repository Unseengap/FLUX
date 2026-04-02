"""
Phase 8.9: Multi-Modal Adapters

Image and audio adapters for cross-modal wave conversion.

Key components:
- ImageAdapter: Image ↔ Wave conversion via CLIP bridge
- AudioAdapter: Audio ↔ Wave conversion via Whisper bridge
- FluxToAny: Universal output adapter for any modality

Usage:
    from phases.phase8_9 import ImageAdapter, AudioAdapter
    
    image_adapter = ImageAdapter(wave_dim=432, clip_dim=768)
    wave = image_adapter.encode(image)  # PIL.Image → [432]
    
    audio_adapter = AudioAdapter(wave_dim=432)
    wave = audio_adapter.encode(audio_path)  # audio → [432]
"""

try:
    from .image_adapters import (
        ImageAdapter,
        ImageToWave,
        WaveToImage,
        CLIPBridge,
    )
    from .audio_adapters import (
        AudioAdapter,
        AudioToWave,
        WaveToAudio,
        WhisperBridge,
    )
    from .flux_to_any import (
        FluxToAny,
        ModalityRouter,
        detect_output_modality,
    )
except ImportError:
    # Allow partial imports for standalone execution
    pass

__all__ = [
    # Image adapters
    'ImageAdapter',
    'ImageToWave',
    'WaveToImage',
    'CLIPBridge',
    
    # Audio adapters
    'AudioAdapter',
    'AudioToWave',
    'WaveToAudio',
    'WhisperBridge',
    
    # Universal output
    'FluxToAny',
    'ModalityRouter',
    'detect_output_modality',
]
