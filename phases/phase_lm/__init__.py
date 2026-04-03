"""
FLUX-LM: Vocabulary-Free Wave-Based Language Model

Combines Phase 1 (CSE) + Phase 1.5 (CWC) + WavePredictor for autoregressive generation.

Components:
    Base (~141M params):
    - CSELarge: Scaled Continuous Semantic Encoder (12.7M params)
    - CWCLarge: Scaled Causal Wave Chainer with order awareness (28.2M params)
    - WavePredictor: Transformer backbone for next-wave prediction (86.6M params)
    - WaveDecoderLarge: Scaled wave→byte decoder (13.9M params)
    - FluxLM: Combined model for training and generation
    
    Universal (Multi-Domain, Scalable 141M → 500M → 1B → 3B):
    - FluxLMUniversal: Multi-domain model with special tokens
    - MultiDomainDataset: Dataset for combined domain training
    - Extended vocabulary: 320 (256 bytes + 64 special tokens)
"""

from .cse_large import CSELarge, CSE_L_CONFIG
from .cwc_large import CWCLarge, CWC_L_CONFIG
from .wave_predictor import WavePredictor, WAVE_PREDICTOR_CONFIG
from .wave_decoder_large import WaveDecoderLarge, WAVE_DECODER_L_CONFIG
from .flux_lm import FluxLM, FLUX_LM_CONFIG

# Universal model (multi-domain)
from .flux_lm_universal import (
    FluxLMUniversal,
    GenerationConfigUniversal,
    FLUX_LM_UNIVERSAL_CONFIG,
    FLUX_LM_500M_CONFIG,
    FLUX_LM_1B_CONFIG,
    FLUX_LM_3B_CONFIG,
    DomainEmbedding,
)

# Multi-domain data loading
from .multi_domain_data import (
    SPECIAL_TOKENS,
    VOCAB_SIZE,
    DATASET_CONFIGS,
    MultiDomainDataset,
    DomainDataset,
    load_all_datasets,
    create_dataloaders,
    encode_special,
    decode_special,
)

__all__ = [
    # Base model
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
    
    # Universal model
    'FluxLMUniversal',
    'GenerationConfigUniversal',
    'FLUX_LM_UNIVERSAL_CONFIG',
    'FLUX_LM_500M_CONFIG',
    'FLUX_LM_1B_CONFIG',
    'FLUX_LM_3B_CONFIG',
    'DomainEmbedding',
    
    # Multi-domain data
    'SPECIAL_TOKENS',
    'VOCAB_SIZE',
    'DATASET_CONFIGS',
    'MultiDomainDataset',
    'DomainDataset',
    'load_all_datasets',
    'create_dataloaders',
    'encode_special',
    'decode_special',
]

__version__ = '2.0.0'

