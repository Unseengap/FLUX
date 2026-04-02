#!/usr/bin/env python3
"""Test FLUX-LM components."""

import torch
from cse_large import CSELarge, count_parameters
from cwc_large import CWCLarge
from wave_predictor import WavePredictor
from wave_decoder_large import WaveDecoderLarge

def fmt(n):
    if n >= 1e6: return f'{n/1e6:.2f}M'
    if n >= 1e3: return f'{n/1e3:.2f}K'
    return str(n)

print('=' * 50)
print('FLUX-LM Component Parameter Counts')
print('=' * 50)

cse = CSELarge()
cwc = CWCLarge()
pred = WavePredictor()
dec = WaveDecoderLarge()

cse_p = count_parameters(cse)
cwc_p = count_parameters(cwc)
pred_p = count_parameters(pred)
dec_p = count_parameters(dec)
total = cse_p + cwc_p + pred_p + dec_p

print(f'CSE-Large:        {fmt(cse_p):>12}')
print(f'CWC-Large:        {fmt(cwc_p):>12}')
print(f'WavePredictor:    {fmt(pred_p):>12}')
print(f'WaveDecoder-L:    {fmt(dec_p):>12}')
print('-' * 30)
print(f'TOTAL:            {fmt(total):>12}')
print('=' * 50)

# Quick forward test
print()
print('Testing forward pass...')
text = 'Hello, world!'
wave = cse.encode(text)
print(f'  CSE: text -> wave {wave.full.shape}')

causal = cwc(wave.full)
print(f'  CWC: wave -> causal {causal.shape}')

pred_out, _ = pred(causal.unsqueeze(0))
print(f'  Predictor: causal -> next {pred_out.shape}')

logits = dec(pred_out)
print(f'  Decoder: next -> logits {logits.shape}')

print()
print('✓ All components working!')
