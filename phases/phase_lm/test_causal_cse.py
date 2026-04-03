#!/usr/bin/env python3
"""Test that CSE is fully causal - waves don't change when context grows."""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cse_large import CSELarge

def test_causal_cse():
    """Verify adding bytes doesn't change existing position waves."""
    cse = CSELarge()
    cse.eval()  # CRITICAL: Disable dropout for deterministic output!
    print('Testing causal CSE (eval mode)...')
    print('=' * 60)
    
    text1 = 'The scientist'
    text2 = 'The scientist discovered'
    
    bytes1 = cse.text_to_bytes(text1)
    bytes2 = cse.text_to_bytes(text2)
    
    # Debug: Check windows are the same
    print(f'\nWindow size: {cse.byte_window}')
    windows1 = cse.bytes_to_windows(bytes1)
    windows2 = cse.bytes_to_windows(bytes2)
    
    pos = 5
    print(f'\nWindow at position {pos}:')
    print(f'  Text1 window: {windows1[pos].tolist()}')
    print(f'  Text2 window: {windows2[pos].tolist()}')
    win_match = torch.equal(windows1[pos], windows2[pos])
    print(f'  Windows match: {win_match}')
    
    # Debug: Step through pipeline manually
    with torch.no_grad():
        # Step 1: Embed windows
        emb1 = cse.byte_embed(windows1.unsqueeze(0)) + cse.window_pos
        emb2 = cse.byte_embed(windows2.unsqueeze(0)) + cse.window_pos
        emb1_flat = emb1.reshape(1, len(text1), -1)
        emb2_flat = emb2.reshape(1, len(text2), -1)
        
        emb_sim = F.cosine_similarity(emb1_flat[0, pos:pos+1], emb2_flat[0, pos:pos+1]).item()
        print(f'\nAfter embedding:')
        print(f'  Position {pos} similarity: {emb_sim:.6f}')
        
        # Step 2: Conv bank
        feat1 = cse.conv_bank(emb1_flat)
        feat2 = cse.conv_bank(emb2_flat)
        
        feat_sim = F.cosine_similarity(feat1[0, pos:pos+1], feat2[0, pos:pos+1]).item()
        print(f'\nAfter conv bank:')
        print(f'  Position {pos} similarity: {feat_sim:.6f}')
        
        # Step 3: Wave projections (before concat)
        phon1 = torch.tanh(cse.wave_projections['phonetic'](feat1))
        phon2 = torch.tanh(cse.wave_projections['phonetic'](feat2))
        phon_sim = F.cosine_similarity(phon1[0, pos:pos+1], phon2[0, pos:pos+1]).item()
        print(f'\nAfter wave projections (phonetic):')
        print(f'  Position {pos} similarity: {phon_sim:.6f}')
        
        # Full pipeline
        wave1 = cse.encode_bytes(bytes1)
        wave2 = cse.encode_bytes(bytes2)
    
    # Check if causal
    print(f'\n{"="*60}')
    print(f'Text 1: "{text1}" ({len(text1)} chars)')
    print(f'Text 2: "{text2}" ({len(text2)} chars)')
    print(f'\nFinal position-by-position wave similarity:')
    
    all_stable = True
    for i in range(min(len(text1), wave1.full.shape[0])):
        sim = F.cosine_similarity(
            wave1.full[i:i+1], 
            wave2.full[i:i+1]
        ).item()
        status = '✓' if sim > 0.999 else '✗'
        print(f'  Position {i:2d}: {sim:.6f} {status}')
        if sim < 0.999:
            all_stable = False
    
    print('=' * 60)
    if all_stable:
        print('✓ SUCCESS: CSE is fully CAUSAL!')
        return True
    else:
        print('✗ FAILED: CSE is still non-causal!')
        return False

if __name__ == '__main__':
    success = test_causal_cse()
    sys.exit(0 if success else 1)
