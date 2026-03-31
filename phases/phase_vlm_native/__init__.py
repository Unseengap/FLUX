"""
Phase VLM Native: Self-contained VLM for FLUX.

This phase replaces the HuggingFace-dependent VLM with a native implementation:
- SVD-compressed weights (~50% smaller)
- Embedded architecture code (no trust_remote_code)
- Full control over inference
- FLUX wave integration (432D ↔ hidden)

Usage:
    from phases.phase_vlm_native import load_native_vlm_from_flx
    
    model = torch.load('Flux-Apex-V1.flx', map_location='cpu')
    vlm = load_native_vlm_from_flx(model)
    
    # Generate
    output = vlm.generate(input_ids, max_new_tokens=100)
"""

from .vlm_svd import (
    SVDConfig,
    compress_vlm_weights,
    decompress_vlm_weights,
    embed_vlm_svd_in_flx,
    load_vlm_from_flx_svd,
)

from .vlm_architecture import (
    NativeVLMConfig,
    NativeVLM,
    load_native_vlm_from_flx,
)

__all__ = [
    # SVD compression
    'SVDConfig',
    'compress_vlm_weights',
    'decompress_vlm_weights',
    'embed_vlm_svd_in_flx',
    'load_vlm_from_flx_svd',
    
    # Native architecture
    'NativeVLMConfig',
    'NativeVLM',
    'load_native_vlm_from_flx',
]

__version__ = '1.0.0'
