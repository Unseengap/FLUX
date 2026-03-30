#!/usr/bin/env python3
"""Test flux_format.py voice config changes."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phases.phase2.flux_format import (
    VoiceConfig, 
    FLUXRuntimeConfig, 
    COMPONENT_REGISTRY,
    GenerationConfig
)

print('✓ Imports successful')

# Test VoiceConfig
vc = VoiceConfig()
print(f'✓ VoiceConfig: enabled={vc.enabled}, model_type={vc.model_type}')

# Test GenerationConfig
gc = GenerationConfig()
print(f'✓ GenerationConfig: voice_primary={gc.voice_primary}, mode={gc.generation_mode}')

# Test FLUXRuntimeConfig
config = FLUXRuntimeConfig()
print(f'✓ FLUXRuntimeConfig: voice.enabled={config.voice.enabled}')

# Test to_dict/from_dict
d = config.to_dict()
print(f'✓ to_dict: {len(d)} sections')
assert 'voice' in d

config2 = FLUXRuntimeConfig.from_dict(d)
print(f'✓ from_dict: voice.enabled={config2.voice.enabled}')

# Test COMPONENT_REGISTRY
print(f'✓ COMPONENT_REGISTRY: {len(COMPONENT_REGISTRY)} components')
assert 'voice' in COMPONENT_REGISTRY
assert 'voice_thinker' in COMPONENT_REGISTRY
print(f'  voice components: {[k for k in COMPONENT_REGISTRY if k.startswith("voice")]}')

print()
print('All tests passed!')
