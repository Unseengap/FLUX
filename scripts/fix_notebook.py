#!/usr/bin/env python3
"""Fix flux_embed_all_models.ipynb Cell 6 to use correct VLM model name"""

import json

notebook_path = '/Users/admin/Desktop/flux/notebooks/flux_embed_all_models.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

def set_cell_source(cell, source_code):
    """Set cell source from multiline string."""
    lines = source_code.split('\n')
    cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]

# Find and fix Cell 6 (VLM)
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    source = ''.join(cell['source'])
    
    if 'Cell 6' in source and 'VLM' in source:
        # Use Qwen2-VL-2B (NOT Qwen2.5-VL-2B which doesn't exist)
        new_code = '''"""Cell 6: Replace VLM with Fresh 2B Model"""

log.cell_start("Cell 6 - VLM")

# ALWAYS embed fresh 2B VLM (replacing any existing 3B)
# Delete old vlm section if present (it's the 3B from before)
if 'vlm' in flux_model:
    old_model = flux_model['vlm'].get('base_model', 'unknown')
    print(f"  Removing old VLM: {old_model}")
    del flux_model['vlm']
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Embed fresh Qwen2-VL-2B (NOTE: Qwen2, not Qwen2.5 - 2.5 has no 2B variant)
print("  Embedding fresh Qwen2-VL-2B-Instruct...")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Use Qwen2VLForConditionalGeneration (NOT Qwen2_5_VLForConditionalGeneration)
from transformers import Qwen2VLForConditionalGeneration

vlm_state = embed_hf_model(
    model_id='Qwen/Qwen2-VL-2B-Instruct',
    model_class=Qwen2VLForConditionalGeneration,
    quantization='fp16',
)

flux_model['models']['vision'] = {
    **vlm_state,
    'role': 'vision_language',
    'tasks': ['image_understanding', 'visual_qa', 'grid_analysis'],
    'lazy_load': True,
}

log.success(f"VLM embedded (Qwen2-VL-2B): {vlm_state['total_params']:,} params")
log.cell_end("Cell 6 - VLM", "PASS")'''
        set_cell_source(cell, new_code)
        print("Fixed Cell 6: Use Qwen2-VL-2B-Instruct (correct model name)")
        break

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook saved")
