#!/usr/bin/env python3
"""Update SFT notebook to use streaming mode for all datasets."""

import json

NOTEBOOK_PATH = '/Users/admin/Desktop/flux/notebooks/flux_lm_sft_dpo.ipynb'

# New Cell 7 content - ALL streaming mode
NEW_CELL_7 = '''# Cell 7: Load SFT Data from HuggingFace Datasets (STREAMING - instant start)

def load_sft_datasets_streaming() -> List[dict]:
    """
    Load pre-distilled SFT data using STREAMING mode.
    INSTANT START: No waiting for full download.
    
    All datasets use streaming=True for immediate training start.
    """
    all_data = []
    
    # ==== CODE: CodeAlpaca-20k (GPT-4 distilled) - STREAMING ====
    print("\\nLoading CODE data (CodeAlpaca-20k) [STREAMING]...")
    try:
        code_ds = load_dataset('sahil2801/CodeAlpaca-20k', split='train', streaming=True)
        max_code = SFT_DATASETS['code']['max_samples']
        
        count = 0
        for sample in tqdm(code_ds, desc="Streaming code", total=max_code):
            if count >= max_code:
                break
            instruction = sample.get('instruction', '')
            output = sample.get('output', '')
            if instruction and output and len(output) > 20:
                all_data.append(format_code_sample(instruction, output))
                count += 1
        
        print(f"  ✓ Streamed {len([d for d in all_data if d['domain']=='code'])} code samples")
    except Exception as e:
        print(f"  ⚠ CodeAlpaca failed: {e}")
    
    # ==== REASONING: GSM8K (high-quality math CoT) - STREAMING ====
    print("\\nLoading REASONING data (GSM8K) [STREAMING]...")
    try:
        math_ds = load_dataset('gsm8k', 'main', split='train', streaming=True)
        max_math = SFT_DATASETS['reasoning']['max_samples']
        
        count = 0
        for sample in tqdm(math_ds, desc="Streaming reasoning", total=max_math):
            if count >= max_math:
                break
            question = sample.get('question', '')
            answer = sample.get('answer', '')
            if question and answer:
                all_data.append(format_reasoning_sample(question, answer))
                count += 1
        
        print(f"  ✓ Streamed {len([d for d in all_data if d['domain']=='reasoning'])} reasoning samples")
    except Exception as e:
        print(f"  ⚠ GSM8K failed: {e}")
    
    # ==== ASSISTANT: OpenHermes-2.5 (GPT-4/Claude distilled) - STREAMING ====
    print("\\nLoading ASSISTANT data (OpenHermes-2.5) [STREAMING]...")
    try:
        assistant_ds = load_dataset('teknium/OpenHermes-2.5', split='train', streaming=True)
        max_assistant = SFT_DATASETS['assistant']['max_samples']
        
        count = 0
        for sample in tqdm(assistant_ds, desc="Streaming assistant", total=max_assistant):
            if count >= max_assistant:
                break
            
            conversations = sample.get('conversations', [])
            if conversations and len(conversations) >= 2:
                formatted = format_assistant_sample(conversations)
                if formatted['response']:  # Has valid response
                    all_data.append(formatted)
                    count += 1
        
        print(f"  ✓ Streamed {len([d for d in all_data if d['domain']=='assistant'])} assistant samples")
    except Exception as e:
        print(f"  ⚠ OpenHermes failed: {e}, trying Dolly fallback...")
        # Fallback to Dolly (smaller, faster)
        try:
            dolly_ds = load_dataset('databricks/databricks-dolly-15k', split='train', streaming=True)
            count = 0
            for sample in tqdm(dolly_ds, desc="Streaming Dolly fallback", total=10000):
                if count >= 10000:
                    break
                instruction = sample.get('instruction', '')
                response = sample.get('response', '')
                if instruction and response:
                    formatted = f"<|user|>{instruction}<|end|>\\n<|assistant|>{response}<|end|>"
                    all_data.append({
                        'domain': 'assistant',
                        'prompt': instruction,
                        'response': response,
                        'formatted': formatted,
                    })
                    count += 1
            print(f"  ✓ Streamed {len([d for d in all_data if d['domain']=='assistant'])} assistant samples (Dolly)")
        except Exception as e2:
            print(f"  ⚠ Dolly fallback also failed: {e2}")
    
    # Shuffle
    random.shuffle(all_data)
    
    return all_data


# Alias for backward compatibility
load_sft_datasets = load_sft_datasets_streaming

print("✓ Streaming dataset functions defined")
print("\\nDatasets (ALL STREAMING):")
for domain, cfg in SFT_DATASETS.items():
    print(f"  [{domain}] {cfg['hf_id']} → up to {cfg['max_samples']:,} samples")
'''

def main():
    # Load notebook
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)
    
    # Find cell 8 (Cell 7 in notebook numbering)
    cell_idx = 8
    
    # Update the cell source
    lines = NEW_CELL_7.split('\n')
    nb['cells'][cell_idx]['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]
    
    # Save notebook
    with open(NOTEBOOK_PATH, 'w') as f:
        json.dump(nb, f, indent=1)
    
    print("✓ Updated Cell 7 to use streaming for ALL datasets")
    print("  - CodeAlpaca-20k: streaming=True")
    print("  - GSM8K: streaming=True")
    print("  - OpenHermes-2.5: streaming=True")
    print("  - Dolly fallback: streaming=True")

if __name__ == '__main__':
    main()
