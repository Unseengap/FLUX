#!/usr/bin/env python3
"""
Get Qwen2.5-Omni-7B metadata for FLUX integration analysis.
"""

from huggingface_hub import hf_hub_download, HfApi
import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / 'DOCS' / 'model_metadata'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_qwen_omni_metadata():
    print('='*70)
    print('QWEN2.5-OMNI-7B METADATA EXTRACTION')
    print('='*70)
    
    repo_id = 'Qwen/Qwen2.5-Omni-7B'
    
    # 1. Download config.json
    print('\n[1] Downloading config.json...')
    try:
        config_path = hf_hub_download(repo_id=repo_id, filename='config.json')
        with open(config_path) as f:
            config = json.load(f)
        
        # Save locally
        with open(OUTPUT_DIR / 'qwen_omni_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f'  ✓ Saved to {OUTPUT_DIR}/qwen_omni_config.json')
        print('\n  Key config values:')
        for key in ['architectures', 'model_type', 'hidden_size', 'num_hidden_layers', 
                    'num_attention_heads', 'vocab_size', 'intermediate_size']:
            if key in config:
                print(f'    {key}: {config[key]}')
        
        # Check for nested configs (audio_config, thinker_config, talker_config)
        for nested in ['audio_config', 'thinker_config', 'talker_config', 'vision_config']:
            if nested in config:
                print(f'\n  {nested}:')
                for k, v in list(config[nested].items())[:8]:
                    print(f'    {k}: {v}')
                if len(config[nested]) > 8:
                    print(f'    ... ({len(config[nested])} total keys)')
                    
    except Exception as e:
        print(f'  ✗ Error: {e}')
        config = None
    
    # 2. Download model index
    print('\n[2] Downloading model.safetensors.index.json...')
    try:
        index_path = hf_hub_download(repo_id=repo_id, filename='model.safetensors.index.json')
        with open(index_path) as f:
            index = json.load(f)
        
        # Save locally
        with open(OUTPUT_DIR / 'qwen_omni_index.json', 'w') as f:
            json.dump(index, f, indent=2)
        
        print(f'  ✓ Saved to {OUTPUT_DIR}/qwen_omni_index.json')
        
        metadata = index.get('metadata', {})
        total_size = metadata.get('total_size', 0)
        print(f'\n  Total size: {total_size:,} bytes ({total_size/1e9:.2f} GB)')
        
        # Count shards
        weight_map = index.get('weight_map', {})
        shards = sorted(set(weight_map.values()))
        print(f'  Number of shards: {len(shards)}')
        print(f'  Shard files: {shards}')
        
        # Analyze weight structure
        weights = list(weight_map.keys())
        print(f'\n  Total weight keys: {len(weights)}')
        
        # Group by module
        modules = {}
        for w in weights:
            parts = w.split('.')
            if len(parts) >= 2:
                module = parts[0]
                if module not in modules:
                    modules[module] = []
                modules[module].append(w)
        
        print('\n  Modules:')
        for module, keys in sorted(modules.items()):
            print(f'    {module}: {len(keys)} weights')
            
    except Exception as e:
        print(f'  ✗ Error: {e}')
        index = None
    
    # 3. Check for generation_config
    print('\n[3] Checking generation_config.json...')
    try:
        gen_path = hf_hub_download(repo_id=repo_id, filename='generation_config.json')
        with open(gen_path) as f:
            gen_config = json.load(f)
        
        with open(OUTPUT_DIR / 'qwen_omni_generation_config.json', 'w') as f:
            json.dump(gen_config, f, indent=2)
        
        print(f'  ✓ Saved generation config')
        print(f'  {gen_config}')
    except Exception as e:
        print(f'  ✗ Not found or error: {e}')
    
    # 4. Check for preprocessor_config
    print('\n[4] Checking preprocessor configs...')
    for filename in ['preprocessor_config.json', 'audio_preprocessor_config.json', 
                     'tokenizer_config.json', 'special_tokens_map.json']:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename)
            with open(path) as f:
                data = json.load(f)
            
            with open(OUTPUT_DIR / f'qwen_omni_{filename}', 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f'  ✓ {filename} ({len(data)} keys)')
        except:
            print(f'  ✗ {filename} not found')
    
    # 5. Size analysis for FLUX integration
    print('\n' + '='*70)
    print('SIZE ANALYSIS FOR FLUX INTEGRATION')
    print('='*70)
    
    if index:
        total_bytes = metadata.get('total_size', 0)
        total_gb = total_bytes / 1e9
        
        print(f'\n  Full model size: {total_gb:.2f} GB')
        print(f'  4-bit quantized estimate: {total_gb * 0.125:.2f} GB')
        print(f'  SVD 25% compressed estimate: {total_gb * 0.35:.2f} GB')
        print(f'  SVD 10% compressed estimate: {total_gb * 0.20:.2f} GB')
        
        print(f'\n  Current Flux-Apex-V1.flx: 5.79 GB')
        print(f'  With Qwen-Omni (4-bit): {5.79 + total_gb * 0.125:.2f} GB')
        print(f'  With Qwen-Omni (SVD 25%): {5.79 + total_gb * 0.35:.2f} GB')
    
    # 6. Component mapping for FLUX
    print('\n' + '='*70)
    print('COMPONENT MAPPING FOR FLUX')
    print('='*70)
    
    if config:
        print('\n  Qwen2.5-Omni Components → FLUX Mapping:')
        print('  ─' * 35)
        
        mappings = [
            ('audio_config', 'adapters.audio_to_wave', 'Audio encoder'),
            ('thinker_config', 'voice.thinker', 'Reasoning/planning LLM'),
            ('talker_config', 'voice.talker', 'Speech synthesis'),
            ('vision_config', 'adapters.vision_to_wave', 'Visual encoder (if present)'),
            ('text_config', 'voice.llm', 'Text generation LLM'),
        ]
        
        for qwen_key, flux_key, description in mappings:
            exists = '✓' if qwen_key in config else '✗'
            print(f'  {exists} {qwen_key:20} → {flux_key:25} ({description})')
    
    print('\n' + '='*70)
    print('DONE - Metadata saved to DOCS/model_metadata/')
    print('='*70)

if __name__ == '__main__':
    get_qwen_omni_metadata()
