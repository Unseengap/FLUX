"""
Phase Voice: Embed Qwen-Omni into Flux-Apex.

This script:
1. Loads Flux-Apex-V1.flx
2. Loads quantized Qwen-Omni (qwen_omni_4bit.pt)
3. Adds 'voice' component to .flx
4. Marks byte decoder as legacy
5. Removes external LLM reference
6. Updates version to 5.0-voice-embedded
7. Saves back to Flux-Apex-V1.flx

Usage:
    python phases/phase_voice/embed_voice_to_flx.py
    
    # Or with custom paths:
    python phases/phase_voice/embed_voice_to_flx.py \
        --flx checkpoints/Flux-Apex-V1.flx \
        --voice checkpoints/qwen_omni_4bit.pt \
        --output checkpoints/Flux-Apex-V1.flx
"""

import sys
from pathlib import Path
from datetime import datetime
import torch
import argparse

# Add project root
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from flux_utils import PhaseLogger

log = PhaseLogger(phase=99)


def embed_voice(
    flx_path: Path,
    voice_path: Path,
    output_path: Path,
    dry_run: bool = False,
):
    """
    Embed voice module into Flux-Apex.
    
    Args:
        flx_path: Path to Flux-Apex-V1.flx
        voice_path: Path to qwen_omni_4bit.pt
        output_path: Output path (can be same as flx_path)
        dry_run: If True, don't save, just show what would happen
    """
    
    log.separator("Phase Voice: Embed Qwen-Omni into FLUX")
    
    # ─────────────────────────────────────────────
    # Step 1: Load Flux-Apex
    # ─────────────────────────────────────────────
    
    print("\n[1] Loading Flux-Apex...")
    
    if not flx_path.exists():
        raise FileNotFoundError(f"Flux-Apex not found: {flx_path}")
    
    flx_size = flx_path.stat().st_size / 1e9
    print(f"    Path: {flx_path}")
    print(f"    Size: {flx_size:.2f} GB")
    
    flx = torch.load(str(flx_path), map_location='cpu', weights_only=False)
    
    print(f"    Version: {flx.get('version', 'unknown')}")
    print(f"    Phase: {flx.get('phase', 'unknown')}")
    print(f"    Top-level keys: {len(flx)}")
    
    # ─────────────────────────────────────────────
    # Step 2: Load Voice Module
    # ─────────────────────────────────────────────
    
    print("\n[2] Loading voice module...")
    
    if not voice_path.exists():
        raise FileNotFoundError(
            f"Voice module not found: {voice_path}\n"
            f"Run: python phases/phase_voice/quantize_qwen_omni.py"
        )
    
    voice_size = voice_path.stat().st_size / 1e9
    print(f"    Path: {voice_path}")
    print(f"    Size: {voice_size:.2f} GB")
    
    voice = torch.load(str(voice_path), map_location='cpu', weights_only=False)
    
    print(f"    Format: {voice.get('format', 'unknown')}")
    print(f"    Base model: {voice.get('base_model', 'unknown')}")
    print(f"    Quantization: {voice.get('quantization', 'unknown')}")
    print(f"    Thinker weights: {len(voice.get('thinker', {}))}")
    print(f"    Talker weights: {len(voice.get('talker', {}))}")
    print(f"    Token2Wav weights: {len(voice.get('token2wav', {}))}")
    
    # ─────────────────────────────────────────────
    # Step 3: Add voice component
    # ─────────────────────────────────────────────
    
    print("\n[3] Adding voice component...")
    
    flx['voice'] = {
        'format_version': '1.0',
        'base_model': voice.get('base_model', 'Qwen/Qwen2.5-Omni-7B'),
        'quantization': voice.get('quantization', '4bit'),
        'timestamp': datetime.now().isoformat(),
        
        'config': voice.get('config', {}),
        
        'tokenizer': voice.get('tokenizer', {}),
        
        'thinker': voice.get('thinker', {}),
        'talker': voice.get('talker', {}),
        'token2wav': voice.get('token2wav', {}),
        
        # Bridges will be initialized at runtime
        'bridges': {
            'wave_to_voice': {
                'wave_dim': 432,
                'voice_dim': 3584,
            },
            'voice_to_wave': {
                'voice_dim': 3584,
                'wave_dim': 432,
            },
        },
    }
    
    print(f"    ✓ Voice component added")
    
    # ─────────────────────────────────────────────
    # Step 4: Mark byte decoder as legacy
    # ─────────────────────────────────────────────
    
    print("\n[4] Marking byte decoder as legacy...")
    
    if 'decoder' in flx:
        flx['decoder']['legacy'] = True
        flx['decoder']['legacy_reason'] = 'Replaced by embedded voice module (Qwen2.5-Omni)'
        flx['decoder']['legacy_since'] = datetime.now().isoformat()
        flx['decoder']['removal_target'] = 'v6.0'
        print(f"    ✓ decoder marked as legacy")
    else:
        print(f"    ⚠ No decoder found in .flx")
    
    # ─────────────────────────────────────────────
    # Step 5: Mark LLM reference as legacy
    # ─────────────────────────────────────────────
    
    print("\n[5] Marking LLM reference as legacy...")
    
    if 'llm_reference' in flx:
        if isinstance(flx['llm_reference'], dict):
            flx['llm_reference']['legacy'] = True
            flx['llm_reference']['legacy_reason'] = 'Replaced by embedded voice module'
            flx['llm_reference']['legacy_since'] = datetime.now().isoformat()
        print(f"    ✓ llm_reference marked as legacy")
    
    if 'llm' in flx:
        if isinstance(flx['llm'], dict):
            flx['llm']['legacy'] = True
            flx['llm']['legacy_reason'] = 'Replaced by embedded voice module'
            flx['llm']['legacy_since'] = datetime.now().isoformat()
        print(f"    ✓ llm marked as legacy")
    
    # ─────────────────────────────────────────────
    # Step 6: Update components flag
    # ─────────────────────────────────────────────
    
    print("\n[6] Updating components...")
    
    if 'components' not in flx:
        flx['components'] = {}
    
    flx['components']['voice'] = True
    flx['components']['voice_thinker'] = True
    flx['components']['voice_talker'] = True
    flx['components']['voice_token2wav'] = True
    
    # Disable decoder by default
    flx['components']['decoder'] = False
    flx['components']['llm'] = False
    
    print(f"    ✓ Components updated")
    
    # ─────────────────────────────────────────────
    # Step 7: Update runtime_config
    # ─────────────────────────────────────────────
    
    print("\n[7] Updating runtime config...")
    
    if 'runtime_config' not in flx:
        flx['runtime_config'] = {}
    
    # Add voice config
    flx['runtime_config']['voice'] = {
        'enabled': True,
        'model_type': 'qwen_omni',
        'quantization': '4bit',
        'max_tokens': 512,
        'temperature': 0.7,
        'top_p': 0.9,
        'text_enabled': True,
        'audio_input_enabled': True,
        'audio_output_enabled': True,
        'vision_enabled': True,
        'use_flux_context': True,
        'flux_context_limit': 10,
        'store_to_field': True,
    }
    
    # Update generation config
    if 'generation' not in flx['runtime_config']:
        flx['runtime_config']['generation'] = {}
    
    flx['runtime_config']['generation']['voice_primary'] = True
    flx['runtime_config']['generation']['llm_primary'] = False
    flx['runtime_config']['generation']['byte_decoder_enabled'] = False
    flx['runtime_config']['generation']['generation_mode'] = 'voice'
    
    print(f"    ✓ Runtime config updated")
    
    # ─────────────────────────────────────────────
    # Step 8: Update metadata
    # ─────────────────────────────────────────────
    
    print("\n[8] Updating metadata...")
    
    old_version = flx.get('version', 'unknown')
    flx['version'] = '5.0-voice-embedded'
    flx['phase'] = 'phase_voice'
    
    if 'metadata' not in flx:
        flx['metadata'] = {}
    
    flx['metadata']['last_modified'] = datetime.now().isoformat()
    flx['metadata']['modified_components'] = ['voice', 'decoder', 'llm_reference']
    flx['metadata']['voice_embedded'] = True
    flx['metadata']['voice_base_model'] = voice.get('base_model', 'Qwen/Qwen2.5-Omni-7B')
    
    if 'capabilities' not in flx['metadata']:
        flx['metadata']['capabilities'] = []
    
    # Add voice capabilities
    for cap in ['voice', 'speech_synthesis', 'audio_understanding']:
        if cap not in flx['metadata']['capabilities']:
            flx['metadata']['capabilities'].append(cap)
    
    flx['modified'] = True
    
    print(f"    Version: {old_version} → {flx['version']}")
    print(f"    ✓ Metadata updated")
    
    # ─────────────────────────────────────────────
    # Step 9: Calculate sizes
    # ─────────────────────────────────────────────
    
    print("\n[9] Size analysis...")
    
    voice_params = 0
    if 'thinker' in flx['voice']:
        for v in flx['voice']['thinker'].values():
            if isinstance(v, torch.Tensor):
                voice_params += v.numel()
    
    print(f"    Voice parameters: {voice_params:,}")
    print(f"    Original .flx size: {flx_size:.2f} GB")
    print(f"    Voice component size: {voice_size:.2f} GB")
    print(f"    Estimated new size: {flx_size + voice_size:.2f} GB")
    
    # ─────────────────────────────────────────────
    # Step 10: Save
    # ─────────────────────────────────────────────
    
    print("\n[10] Saving...")
    
    if dry_run:
        print(f"    [DRY RUN] Would save to: {output_path}")
        print(f"    Skipping actual save.")
        return
    
    print(f"    Saving to: {output_path}")
    torch.save(flx, str(output_path))
    
    new_size = output_path.stat().st_size / 1e9
    print(f"    ✓ Saved: {output_path.name} ({new_size:.2f} GB)")
    
    # ─────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────
    
    print("\n" + "="*60)
    print("VOICE EMBEDDING COMPLETE")
    print("="*60)
    
    print(f"""
    Input:
        Flux-Apex: {flx_path} ({flx_size:.2f} GB)
        Voice:     {voice_path} ({voice_size:.2f} GB)
    
    Output:
        {output_path} ({new_size:.2f} GB)
    
    Changes:
        ✓ Added 'voice' component (Qwen2.5-Omni-7B, 4-bit)
        ✓ Marked 'decoder' as legacy
        ✓ Marked 'llm_reference' as legacy
        ✓ Updated version to {flx['version']}
        ✓ Updated runtime_config for voice generation
    
    The model is now self-contained with no external downloads needed.
    """)
    
    log.success("Voice embedding complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Embed Qwen-Omni voice module into Flux-Apex'
    )
    parser.add_argument(
        '--flx',
        type=Path,
        default=ROOT / 'checkpoints' / 'Flux-Apex-V1.flx',
        help='Path to Flux-Apex-V1.flx',
    )
    parser.add_argument(
        '--voice',
        type=Path,
        default=ROOT / 'checkpoints' / 'qwen_omni_4bit.pt',
        help='Path to quantized voice module',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output path (defaults to --flx path)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Don't save, just show what would happen",
    )
    
    args = parser.parse_args()
    
    output_path = args.output or args.flx
    
    embed_voice(
        flx_path=args.flx,
        voice_path=args.voice,
        output_path=output_path,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()
