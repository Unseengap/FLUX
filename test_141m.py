#!/usr/bin/env python3
"""
Test the Flux-LM-141M checkpoint.

This is the ORIGINAL FluxLM model (~124M params) trained on WikiText-103.
Uses: CSE-Large + CWC-Large + WavePredictor + WaveDecoder-Large
NOT the multi-domain universal model.
"""

import sys
sys.path.insert(0, 'phases/phase_lm')
import torch

# Use the ORIGINAL FluxLM class, NOT FluxLMUniversal
from flux_lm import FluxLM, GenerationConfig

# Load the 141M model
print('=' * 60)
print('Loading Flux-LM-141M (Original WikiText-103 Model)')
print('=' * 60)

model = FluxLM.load('checkpoints/Flux-LM-141M.pt', device='cpu')
params = model.count_parameters()

print(f'\nModel loaded successfully!')
print(f'Total parameters: {params["total"]:,}')
print(f'\nParameter breakdown:')
for name, count in params.items():
    if name != 'total':
        print(f'  {name}: {count:,}')

# Quick generation test
print('\n' + '=' * 60)
print('Generation Test')
print('=' * 60)

test_prompts = [
    'The scientist discovered',
    'In the year 2024',
    'The city of Paris',
    'Once upon a time',
]

model.eval()
for prompt in test_prompts:
    print(f'\nPrompt: {repr(prompt)}')
    try:
        output = model.generate(
            prompt,
            GenerationConfig(
                max_new_bytes=80,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.1,
            )
        )
        print(f'Output: {repr(output)}')
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

print('\n' + '=' * 60)
print('Test complete!')
print('=' * 60)
