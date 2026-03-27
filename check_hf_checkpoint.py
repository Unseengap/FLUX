"""Check the Phase 8 checkpoint on HuggingFace Hub for decoder_state_dict."""

import os
import torch
from huggingface_hub import hf_hub_download

HF_REPO_ID = "UnseenGAP/FLUX"

# Load token from .env if not in environment
if not os.environ.get('HF_TOKEN'):
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        for line in open(env_path):
            if line.startswith('hf_token='):
                os.environ['HF_TOKEN'] = line.strip().split('=', 1)[1]

HF_TOKEN = os.environ.get('HF_TOKEN', '')
assert HF_TOKEN, 'Set HF_TOKEN env var or add hf_token= to .env'

print("Downloading phase8.phase.pt from HuggingFace Hub...")
path = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename="checkpoints/phase8.phase.pt",
    token=HF_TOKEN,
)
print(f"  Downloaded to: {path}")

size_mb = os.path.getsize(path) / 1e6
print(f"  Size: {size_mb:.1f} MB")

print("\nLoading checkpoint...")
ckpt = torch.load(path, map_location='cpu', weights_only=False)

print(f"\nTop-level keys:")
for k in sorted(ckpt.keys()):
    v = ckpt[k]
    if isinstance(v, dict):
        print(f"  {k}: dict with {len(v)} entries")
    elif hasattr(v, 'shape'):
        print(f"  {k}: tensor {v.shape}")
    else:
        print(f"  {k}: {type(v).__name__} = {str(v)[:100]}")

# ── Check decoder ──
if 'decoder_state_dict' in ckpt:
    keys = list(ckpt['decoder_state_dict'].keys())
    print(f"\n✓ decoder_state_dict FOUND — {len(keys)} weight tensors")
    has_prefix = any(k.startswith('_orig_mod.') for k in keys)
    print(f"  Has _orig_mod. prefix (torch.compile): {has_prefix}")
    for k in keys[:10]:
        shape = ckpt['decoder_state_dict'][k].shape if hasattr(ckpt['decoder_state_dict'][k], 'shape') else '?'
        print(f"    {k}: {shape}")
    if len(keys) > 10:
        print(f"    ... ({len(keys) - 10} more)")
else:
    print("\n✗ decoder_state_dict NOT FOUND in HF checkpoint!")

# ── Check metrics ──
metrics = ckpt.get('metrics', {})
if metrics:
    print(f"\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

# ── Check config ──
config = ckpt.get('config', {})
if config:
    print(f"\nConfig (selected):")
    for k in ['field_features', 'field_h', 'field_w', 'field_d', 'wave_dim']:
        if k in config:
            print(f"  {k}: {config[k]}")

print(f"\n{'='*60}")
print(f"  phase: {ckpt.get('phase', '?')}")
print(f"  timestamp: {ckpt.get('timestamp', '?')}")
print(f"  learning_steps: {ckpt.get('learning_steps', '?')}")
print(f"{'='*60}")
