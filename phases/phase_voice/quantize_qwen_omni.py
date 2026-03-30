"""
Phase Voice: Quantize Qwen2.5-Omni for FLUX embedding.

This script:
1. Downloads Qwen2.5-Omni-7B from HuggingFace
2. Quantizes to 4-bit using bitsandbytes
3. Extracts and saves state dict for .flx embedding
4. Embeds tokenizer data

Output: checkpoints/qwen_omni_4bit.pt (~2.8 GB)

Usage:
    python phases/phase_voice/quantize_qwen_omni.py

Requirements:
    pip install transformers bitsandbytes accelerate
"""

import sys
from pathlib import Path
import torch
from datetime import datetime

# Add project root
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from flux_utils import PhaseLogger, get_device

log = PhaseLogger(phase=99)  # Phase Voice
log.separator("Phase Voice: Quantize Qwen2.5-Omni")

CHECKPOINT_DIR = ROOT / 'checkpoints'
CHECKPOINT_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = CHECKPOINT_DIR / 'qwen_omni_4bit.pt'
MODEL_ID = 'Qwen/Qwen2.5-Omni-7B'


def download_and_quantize():
    """Download Qwen-Omni and quantize to 4-bit."""
    
    print("\n[1] Checking dependencies...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
        from transformers import BitsAndBytesConfig
        import bitsandbytes as bnb
        print("  ✓ transformers, bitsandbytes available")
    except ImportError as e:
        print(f"  ✗ Missing dependency: {e}")
        print("  Run: pip install transformers bitsandbytes accelerate")
        return None
    
    device = get_device()
    print(f"  Device: {device}")
    
    # Check if already exists
    if OUTPUT_PATH.exists():
        size_mb = OUTPUT_PATH.stat().st_size / 1e6
        print(f"\n[!] Output already exists: {OUTPUT_PATH} ({size_mb:.1f} MB)")
        response = input("    Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("    Skipping quantization.")
            return None
    
    print(f"\n[2] Downloading {MODEL_ID}...")
    print("    This will download ~22 GB on first run.")
    print("    Please be patient...")
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    try:
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("  ✓ Model loaded with 4-bit quantization")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )
        print("  ✓ Tokenizer loaded")
        
        # Try to load processor (for audio/vision)
        try:
            processor = AutoProcessor.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
            )
            print("  ✓ Processor loaded (audio/vision)")
        except Exception as e:
            print(f"  ⚠ Processor not loaded: {e}")
            processor = None
            
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return None
    
    print("\n[3] Extracting state dict...")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Get state dict
    state_dict = model.state_dict()
    print(f"  State dict keys: {len(state_dict)}")
    
    # Organize by module
    voice_state = {
        'format': 'FLUX_VOICE',
        'version': '1.0',
        'base_model': MODEL_ID,
        'quantization': '4bit',
        'timestamp': datetime.now().isoformat(),
        'total_params': total_params,
        
        'config': {
            'model_config': model.config.to_dict() if hasattr(model.config, 'to_dict') else {},
        },
        
        'thinker': {},
        'talker': {},
        'token2wav': {},
        
        'tokenizer': {
            'vocab_size': tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer),
            'special_tokens': dict(tokenizer.special_tokens_map) if hasattr(tokenizer, 'special_tokens_map') else {},
        },
    }
    
    # Sort weights into modules
    print("\n[4] Organizing weights by module...")
    
    for key, tensor in state_dict.items():
        if key.startswith('thinker.'):
            voice_state['thinker'][key] = tensor.cpu()
        elif key.startswith('talker.'):
            voice_state['talker'][key] = tensor.cpu()
        elif key.startswith('token2wav.'):
            voice_state['token2wav'][key] = tensor.cpu()
        else:
            # Other weights go to thinker by default
            voice_state['thinker'][key] = tensor.cpu()
    
    print(f"  Thinker weights: {len(voice_state['thinker'])}")
    print(f"  Talker weights: {len(voice_state['talker'])}")
    print(f"  Token2Wav weights: {len(voice_state['token2wav'])}")
    
    # Save tokenizer vocab
    print("\n[5] Saving tokenizer data...")
    try:
        voice_state['tokenizer']['vocab'] = tokenizer.get_vocab()
        print(f"  Vocab size: {len(voice_state['tokenizer']['vocab'])}")
    except:
        print("  ⚠ Could not save vocab")
    
    # Save
    print(f"\n[6] Saving to {OUTPUT_PATH}...")
    torch.save(voice_state, str(OUTPUT_PATH))
    
    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(f"  ✓ Saved: {OUTPUT_PATH.name} ({size_mb:.1f} MB)")
    
    # Verify
    print("\n[7] Verifying saved file...")
    loaded = torch.load(str(OUTPUT_PATH), map_location='cpu', weights_only=False)
    assert loaded['format'] == 'FLUX_VOICE'
    assert len(loaded['thinker']) > 0
    print(f"  ✓ Verification passed")
    
    return voice_state


def create_voice_state_from_existing():
    """
    Alternative: Create voice state from pre-downloaded model.
    Use this if you already have Qwen-Omni downloaded.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import BitsAndBytesConfig
    
    print("\n[Alternative] Loading from HuggingFace cache...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,  # Use cached files only
    )
    
    return model


def analyze_existing():
    """Analyze an existing quantized file."""
    if not OUTPUT_PATH.exists():
        print(f"  ✗ File not found: {OUTPUT_PATH}")
        return
    
    print(f"\n  Loading {OUTPUT_PATH}...")
    state = torch.load(str(OUTPUT_PATH), map_location='cpu', weights_only=False)
    
    print(f"\n  Format: {state.get('format', 'unknown')}")
    print(f"  Version: {state.get('version', 'unknown')}")
    print(f"  Base model: {state.get('base_model', 'unknown')}")
    print(f"  Quantization: {state.get('quantization', 'unknown')}")
    print(f"  Total params: {state.get('total_params', 'unknown'):,}")
    
    print(f"\n  Thinker weights: {len(state.get('thinker', {}))}")
    print(f"  Talker weights: {len(state.get('talker', {}))}")
    print(f"  Token2Wav weights: {len(state.get('token2wav', {}))}")
    
    if 'config' in state:
        print(f"\n  Config keys: {list(state['config'].keys())}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Quantize Qwen-Omni for FLUX')
    parser.add_argument('--analyze', action='store_true', help='Analyze existing file')
    parser.add_argument('--force', action='store_true', help='Force re-download')
    args = parser.parse_args()
    
    if args.analyze:
        analyze_existing()
    else:
        result = download_and_quantize()
        if result:
            log.success("Qwen-Omni quantized and saved!")
        else:
            log.warning("Quantization skipped or failed")
