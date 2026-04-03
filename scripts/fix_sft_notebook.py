#!/usr/bin/env python3
"""Fix the SFT notebook to use fast dataset loading instead of slow teacher generation."""

import json

NOTEBOOK_PATH = '/Users/admin/Desktop/flux/notebooks/flux_lm_sft_dpo.ipynb'

# New Cell 8 content - fast dataset loading
NEW_CELL_8 = '''# Cell 8: Load SFT Data (FAST - from HuggingFace datasets)

# ==== LOAD SFT DATA (OR USE CACHED) ====
SFT_DATA_PATH = CKPT_DIR / 'sft_data_v2.json'

if SFT_DATA_PATH.exists():
    print(f"Loading cached SFT data from {SFT_DATA_PATH}")
    with open(SFT_DATA_PATH, 'r') as f:
        sft_data = json.load(f)
    print(f"✓ Loaded {len(sft_data)} SFT samples")
else:
    print("="*60)
    print("Loading SFT data from HuggingFace (FAST - ~30 seconds)")
    print("="*60)
    
    # Load from pre-distilled datasets
    sft_data = load_sft_datasets()
    
    # Save for future runs
    with open(SFT_DATA_PATH, 'w') as f:
        json.dump(sft_data, f)
    print(f"\\n✓ Saved {len(sft_data)} SFT samples to {SFT_DATA_PATH}")
    
    # Copy to Drive
    if SAVE_TO_DRIVE and DRIVE_CKPT_DIR:
        shutil.copy2(SFT_DATA_PATH, DRIVE_CKPT_DIR / 'sft_data_v2.json')
        print(f"✓ Copied to Google Drive")

# Show distribution
print(f"\\nSFT Data Distribution:")
domain_counts = {}
for s in sft_data:
    d = s['domain']
    domain_counts[d] = domain_counts.get(d, 0) + 1
for domain, count in sorted(domain_counts.items()):
    print(f"  [{domain}] {count:,} samples")
print(f"  TOTAL: {len(sft_data):,} samples")

# Show samples
print(f"\\nSample data:")
for domain in ['code', 'reasoning', 'assistant']:
    samples = [s for s in sft_data if s['domain'] == domain]
    if samples:
        print(f"\\n[{domain.upper()}]")
        print(f"  Prompt: {samples[0]['prompt'][:60]}...")
        print(f"  Response: {samples[0]['response'][:60]}...")
'''

def main():
    # Load notebook
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)
    
    # Find cell 9 (old Cell 8 with teacher generation)
    cell_idx = 9
    
    # Update the cell source
    lines = NEW_CELL_8.split('\n')
    nb['cells'][cell_idx]['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]
    
    # Save notebook
    with open(NOTEBOOK_PATH, 'w') as f:
        json.dump(nb, f, indent=1)
    
    print("✓ Updated Cell 8 to use fast dataset loading")
    print(f"  Old: Slow teacher generation (~30s/sample)")
    print(f"  New: HuggingFace datasets (~30s total)")

if __name__ == '__main__':
    main()
