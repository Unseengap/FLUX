"""
FLUX-LM: Vocabulary-Free Wave-Based Language Model

Combines Phase 1 (CSE) + Phase 1.5 (CWC) + WavePredictor for autoregressive generation.
Target: ~124M parameters (GPT-2 small equivalent)
   
Components:
    - CSELarge: Scaled Continuous Semantic Encoder (10M params)
    - CWCLarge: Scaled Causal Wave Chainer with order awareness (5M params)
    - WavePredictor: Transformer backbone for next-wave prediction (100M params)
    - WaveDecoderLarge: Scaled wave→byte decoder (9M params)
    - FluxLM: Combined model for training and generation
"""

from .cse_large import CSELarge, CSE_L_CONFIG
from .cwc_large import CWCLarge, CWC_L_CONFIG
from .wave_predictor import WavePredictor, WAVE_PREDICTOR_CONFIG
from .wave_decoder_large import WaveDecoderLarge, WAVE_DECODER_L_CONFIG
from .flux_lm import FluxLM, FLUX_LM_CONFIG

__all__ = [
    'CSELarge',
    'CWCLarge', 
    'WavePredictor',
    'WaveDecoderLarge',
    'FluxLM',
    'CSE_L_CONFIG',
    'CWC_L_CONFIG',
    'WAVE_PREDICTOR_CONFIG',
    'WAVE_DECODER_L_CONFIG',
    'FLUX_LM_CONFIG',
]

__version__ = '1.0.0'
