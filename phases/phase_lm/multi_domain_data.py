"""
Multi-Domain Dataset for FLUX-LM Universal Training.

Handles loading and mixing data from multiple domains:
- TEXT: Conversation, reasoning, tool use
- CODE: Python, JavaScript, etc.
- DOCS: Markdown, HTML, LaTeX
- DATA: JSON, CSV, YAML
- MEDIA: MIDI (future)
- PROTOCOLS: HTTP, API examples

For 1B+ model training, uses 100% of available datasets.
Scaled for ~2B tokens needed for 1B parameter model (Chinchilla scaling).
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler


# ─────────────────────────────────────────────
# Special Tokens (extend byte vocabulary)
# ─────────────────────────────────────────────

SPECIAL_TOKENS = {
    # Conversation
    '<|system|>': 256,
    '<|user|>': 257,
    '<|assistant|>': 258,
    '<|end|>': 259,
    
    # Reasoning
    '<|reasoning|>': 260,
    '<|end_reasoning|>': 261,
    '<|step|>': 262,
    '<|problem|>': 263,
    '<|answer|>': 264,
    '<|end_answer|>': 265,
    
    # Tool use
    '<|tool_call|>': 266,
    '<|end_tool_call|>': 267,
    '<|tool_result|>': 268,
    '<|end_tool_result|>': 269,
    
    # MCP
    '<|mcp_request|>': 270,
    '<|mcp_response|>': 271,
    '<|end_mcp|>': 272,
    
    # Code
    '<|code|>': 273,
    '<|end_code|>': 274,
    '<|lang:python|>': 275,
    '<|lang:javascript|>': 276,
    '<|lang:typescript|>': 277,
    '<|lang:java|>': 278,
    '<|lang:rust|>': 279,
    '<|lang:go|>': 280,
    '<|lang:sql|>': 281,
    '<|lang:shell|>': 282,
    
    # Documents
    '<|doc:markdown|>': 283,
    '<|doc:html|>': 284,
    '<|doc:latex|>': 285,
    '<|doc:json|>': 286,
    '<|doc:yaml|>': 287,
    '<|doc:csv|>': 288,
    
    # Domain markers
    '<|domain:text|>': 300,
    '<|domain:code|>': 301,
    '<|domain:doc|>': 302,
    '<|domain:data|>': 303,
    '<|domain:media|>': 304,
    '<|domain:protocol|>': 305,
}

# Reverse mapping
TOKEN_TO_SPECIAL = {v: k for k, v in SPECIAL_TOKENS.items()}

# Extended vocabulary size
VOCAB_SIZE = 320  # 256 bytes + 64 special tokens


def encode_special(text: str) -> List[int]:
    """Encode text with special tokens to byte sequence."""
    result = []
    i = 0
    while i < len(text):
        # Check for special tokens
        found = False
        for token, token_id in SPECIAL_TOKENS.items():
            if text[i:].startswith(token):
                result.append(token_id)
                i += len(token)
                found = True
                break
        
        if not found:
            # Regular byte
            result.append(ord(text[i]))
            i += 1
    
    return result


def decode_special(byte_seq: List[int]) -> str:
    """Decode byte sequence with special tokens to text."""
    result = []
    for b in byte_seq:
        if b in TOKEN_TO_SPECIAL:
            result.append(TOKEN_TO_SPECIAL[b])
        elif 0 <= b < 256:
            result.append(chr(b))
        else:
            result.append(f'<UNK:{b}>')
    return ''.join(result)


# ─────────────────────────────────────────────
# Dataset Configurations
# ─────────────────────────────────────────────

@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    source: str              # HuggingFace dataset name or path
    subset: Optional[str]    # Dataset subset/config
    split: str               # Train/validation split
    max_samples: int         # Max samples to use (10% rule)
    domain: str              # Domain category
    format_fn: str           # Name of formatting function
    weight: float            # Sampling weight


# Dataset registry - easily accessible datasets
DATASET_CONFIGS = {
    # ─────────────────────────────────────────
    # TEXT - Reasoning (Priority 1)
    # ─────────────────────────────────────────
    'opus_reasoning': DatasetConfig(
        name='opus_reasoning',
        source='nohurry/Opus-4.6-Reasoning-3000x-filtered',
        subset=None,
        split='train',
        max_samples=3000,  # All of it (small dataset)
        domain='reasoning',
        format_fn='format_opus_reasoning',
        weight=0.15,
    ),
    'gsm8k': DatasetConfig(
        name='gsm8k',
        source='gsm8k',
        subset='main',
        split='train',
        max_samples=8500,  # All of it
        domain='reasoning',
        format_fn='format_gsm8k',
        weight=0.08,
    ),
    'arc_challenge': DatasetConfig(
        name='arc_challenge',
        source='ai2_arc',
        subset='ARC-Challenge',
        split='train',
        max_samples=2500,  # All of it
        domain='reasoning',
        format_fn='format_arc',
        weight=0.05,
    ),
    
    # ─────────────────────────────────────────
    # TEXT - Assistant (Priority 1)
    # ─────────────────────────────────────────
    'openassistant': DatasetConfig(
        name='openassistant',
        source='OpenAssistant/oasst1',
        subset=None,
        split='train',
        max_samples=160000,  # 100% for 1B model
        domain='assistant',
        format_fn='format_openassistant',
        weight=0.12,
    ),
    'dolly': DatasetConfig(
        name='dolly',
        source='databricks/databricks-dolly-15k',
        subset=None,
        split='train',
        max_samples=15000,  # All of it
        domain='assistant',
        format_fn='format_dolly',
        weight=0.08,
    ),
    'alpaca': DatasetConfig(
        name='alpaca',
        source='tatsu-lab/alpaca',
        subset=None,
        split='train',
        max_samples=52000,  # 100% for 1B model
        domain='assistant',
        format_fn='format_alpaca',
        weight=0.05,
    ),
    
    # ─────────────────────────────────────────
    # TEXT - Tool Use (Priority 2)
    # ─────────────────────────────────────────
    'gorilla': DatasetConfig(
        name='gorilla',
        source='gorilla-llm/APIBench',
        subset=None,
        split='train',
        max_samples=10000,  # All
        domain='tool_use',
        format_fn='format_gorilla',
        weight=0.08,
    ),
    
    # ─────────────────────────────────────────
    # CODE (Priority 1)
    # ─────────────────────────────────────────
    'humaneval': DatasetConfig(
        name='humaneval',
        source='openai_humaneval',
        subset=None,
        split='test',
        max_samples=164,  # All of it
        domain='code',
        format_fn='format_humaneval',
        weight=0.03,
    ),
    'mbpp': DatasetConfig(
        name='mbpp',
        source='mbpp',
        subset=None,
        split='train',
        max_samples=974,  # All
        domain='code',
        format_fn='format_mbpp',
        weight=0.03,
    ),
    'code_search_net': DatasetConfig(
        name='code_search_net',
        source='code_search_net',
        subset='python',
        split='train',
        max_samples=500000,  # 100% for 1B model (~500K Python functions)
        domain='code',
        format_fn='format_code_search_net',
        weight=0.10,
    ),
    
    # ─────────────────────────────────────────
    # DOCS & DATA (Priority 3)
    # ─────────────────────────────────────────
    'wikitext': DatasetConfig(
        name='wikitext',
        source='wikitext',
        subset='wikitext-103-raw-v1',
        split='train',
        max_samples=100000,  # 100% for 1B model
        domain='docs',
        format_fn='format_wikitext',
        weight=0.10,
    ),
    
    # ─────────────────────────────────────────
    # MULTILINGUAL (Priority 2)
    # ─────────────────────────────────────────
    'opus100_en_fr': DatasetConfig(
        name='opus100_en_fr',
        source='opus100',
        subset='en-fr',
        split='train',
        max_samples=100000,  # 100K for 1B model
        domain='multilingual',
        format_fn='format_opus100',
        weight=0.03,
    ),
    'opus100_en_de': DatasetConfig(
        name='opus100_en_de',
        source='opus100',
        subset='en-de',
        split='train',
        max_samples=100000,  # 100K for 1B model
        domain='multilingual',
        format_fn='format_opus100',
        weight=0.03,
    ),
    'opus100_en_zh': DatasetConfig(
        name='opus100_en_zh',
        source='opus100',
        subset='en-zh',
        split='train',
        max_samples=100000,  # 100K for 1B model
        domain='multilingual',
        format_fn='format_opus100',
        weight=0.03,
    ),
    'opus100_en_es': DatasetConfig(
        name='opus100_en_es',
        source='opus100',
        subset='en-es',
        split='train',
        max_samples=100000,  # 100K for 1B model
        domain='multilingual',
        format_fn='format_opus100',
        weight=0.02,
    ),
    'opus100_en_ja': DatasetConfig(
        name='opus100_en_ja',
        source='opus100',
        subset='en-ja',
        split='train',
        max_samples=100000,  # 100K for 1B model
        domain='multilingual',
        format_fn='format_opus100',
        weight=0.02,
    ),
}


# ─────────────────────────────────────────────
# Formatting Functions
# ─────────────────────────────────────────────

def format_opus_reasoning(example: dict) -> str:
    """Format Opus reasoning dataset."""
    problem = example.get('instruction', example.get('problem', ''))
    reasoning = example.get('output', example.get('response', ''))
    
    return f"""<|domain:text|><|problem|>
{problem}
<|end|>
<|reasoning|>
{reasoning}
<|end_reasoning|>"""


def format_gsm8k(example: dict) -> str:
    """Format GSM8K math problems."""
    question = example['question']
    answer = example['answer']
    
    return f"""<|domain:text|><|problem|>
{question}
<|end|>
<|reasoning|>
{answer}
<|end_reasoning|>"""


def format_arc(example: dict) -> str:
    """Format ARC science questions."""
    question = example['question']
    choices = example['choices']
    answer_key = example['answerKey']
    
    # Format choices
    choices_text = '\n'.join([
        f"{label}) {text}" 
        for label, text in zip(choices['label'], choices['text'])
    ])
    
    return f"""<|domain:text|><|problem|>
{question}

{choices_text}
<|end|>
<|answer|>
{answer_key}
<|end_answer|>"""


def format_openassistant(example: dict) -> str:
    """Format OpenAssistant conversations."""
    text = example.get('text', '')
    role = example.get('role', 'assistant')
    
    if role == 'prompter':
        return f"<|user|>{text}<|end|>"
    else:
        return f"<|assistant|>{text}<|end|>"


def format_dolly(example: dict) -> str:
    """Format Dolly instruction-response pairs."""
    instruction = example['instruction']
    context = example.get('context', '')
    response = example['response']
    
    if context:
        return f"""<|domain:text|><|system|>You are a helpful assistant.<|end|>
<|user|>{instruction}

Context: {context}<|end|>
<|assistant|>{response}<|end|>"""
    else:
        return f"""<|domain:text|><|system|>You are a helpful assistant.<|end|>
<|user|>{instruction}<|end|>
<|assistant|>{response}<|end|>"""


def format_alpaca(example: dict) -> str:
    """Format Alpaca instruction-response pairs."""
    instruction = example['instruction']
    input_text = example.get('input', '')
    output = example['output']
    
    if input_text:
        return f"""<|domain:text|><|user|>{instruction}

{input_text}<|end|>
<|assistant|>{output}<|end|>"""
    else:
        return f"""<|domain:text|><|user|>{instruction}<|end|>
<|assistant|>{output}<|end|>"""


def format_gorilla(example: dict) -> str:
    """Format Gorilla API examples."""
    # Handle different Gorilla formats
    if 'api_call' in example:
        instruction = example.get('instruction', '')
        api_call = example['api_call']
        return f"""<|domain:protocol|><|user|>{instruction}<|end|>
<|tool_call|>
{api_call}
<|end_tool_call|>"""
    else:
        return str(example)


def format_humaneval(example: dict) -> str:
    """Format HumanEval code problems."""
    prompt = example['prompt']
    canonical = example['canonical_solution']
    
    return f"""<|domain:code|><|lang:python|><|code|>
{prompt}{canonical}
<|end_code|>"""


def format_mbpp(example: dict) -> str:
    """Format MBPP code problems."""
    text = example['text']  # Problem description
    code = example['code']
    
    return f"""<|domain:code|><|problem|>
{text}
<|end|>
<|lang:python|><|code|>
{code}
<|end_code|>"""


def format_code_search_net(example: dict) -> str:
    """Format CodeSearchNet examples."""
    code = example.get('whole_func_string', example.get('func_code_string', ''))
    docstring = example.get('func_documentation_string', '')
    
    if docstring:
        return f"""<|domain:code|><|lang:python|>
# {docstring}
<|code|>
{code}
<|end_code|>"""
    else:
        return f"""<|domain:code|><|lang:python|><|code|>
{code}
<|end_code|>"""


def format_wikitext(example: dict) -> str:
    """Format WikiText articles."""
    text = example['text'].strip()
    if not text or text.startswith('='):
        return ""
    return f"<|domain:doc|><|doc:markdown|>{text}"


def format_opus100(example: dict) -> str:
    """Format OPUS-100 translation pairs."""
    translation = example['translation']
    # Get the language pair
    langs = list(translation.keys())
    if len(langs) != 2:
        return ""
    
    src_lang, tgt_lang = langs
    src_text = translation[src_lang]
    tgt_text = translation[tgt_lang]
    
    return f"""<|domain:text|><|user|>Translate from {src_lang} to {tgt_lang}:
{src_text}<|end|>
<|assistant|>{tgt_text}<|end|>"""


# Format function registry
FORMAT_FUNCTIONS = {
    'format_opus_reasoning': format_opus_reasoning,
    'format_gsm8k': format_gsm8k,
    'format_arc': format_arc,
    'format_openassistant': format_openassistant,
    'format_dolly': format_dolly,
    'format_alpaca': format_alpaca,
    'format_gorilla': format_gorilla,
    'format_humaneval': format_humaneval,
    'format_mbpp': format_mbpp,
    'format_code_search_net': format_code_search_net,
    'format_wikitext': format_wikitext,
    'format_opus100': format_opus100,
}


# ─────────────────────────────────────────────
# Dataset Classes
# ─────────────────────────────────────────────

class DomainDataset(Dataset):
    """Dataset for a single domain with special token support."""
    
    def __init__(
        self,
        texts: List[str],
        max_len: int = 512,
        domain: str = 'text',
    ):
        self.texts = [t for t in texts if t and len(t) > 10]
        self.max_len = max_len
        self.domain = domain
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Encode with special tokens
        byte_seq = encode_special(text)[:self.max_len + 1]
        
        # Ensure minimum length
        if len(byte_seq) < 10:
            byte_seq = byte_seq + [0] * (10 - len(byte_seq))
        
        # Input and target (shifted by 1)
        input_bytes = byte_seq[:-1]
        target_bytes = byte_seq[1:]
        
        # Pad to max_len
        pad_len = self.max_len - len(input_bytes)
        if pad_len > 0:
            input_bytes = input_bytes + [0] * pad_len
            target_bytes = target_bytes + [-100] * pad_len
        
        return {
            'input': torch.tensor(input_bytes, dtype=torch.long),
            'target': torch.tensor(target_bytes, dtype=torch.long),
            'domain': self.domain,
        }


class MultiDomainDataset(Dataset):
    """Combined dataset from multiple domains with weighted sampling."""
    
    def __init__(
        self,
        domain_datasets: Dict[str, DomainDataset],
        weights: Optional[Dict[str, float]] = None,
    ):
        self.domain_datasets = domain_datasets
        self.domains = list(domain_datasets.keys())
        
        # Calculate cumulative indices
        self.cumulative_sizes = []
        total = 0
        for domain in self.domains:
            total += len(domain_datasets[domain])
            self.cumulative_sizes.append(total)
        
        self.total_size = total
        
        # Weights for sampling
        if weights:
            self.weights = weights
        else:
            # Default: proportional to dataset size
            self.weights = {
                d: len(domain_datasets[d]) / total 
                for d in self.domains
            }
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        # Find which domain this index belongs to
        for i, (domain, cum_size) in enumerate(zip(self.domains, self.cumulative_sizes)):
            prev_size = self.cumulative_sizes[i-1] if i > 0 else 0
            if idx < cum_size:
                local_idx = idx - prev_size
                return self.domain_datasets[domain][local_idx]
        
        raise IndexError(f"Index {idx} out of range")
    
    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """Get sampler for weighted domain sampling."""
        sample_weights = []
        
        for i, (domain, cum_size) in enumerate(zip(self.domains, self.cumulative_sizes)):
            prev_size = self.cumulative_sizes[i-1] if i > 0 else 0
            domain_size = cum_size - prev_size
            weight = self.weights.get(domain, 1.0) / domain_size
            sample_weights.extend([weight] * domain_size)
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self),
            replacement=True,
        )


# ─────────────────────────────────────────────
# Data Loading Functions
# ─────────────────────────────────────────────

def load_dataset_safe(config: DatasetConfig) -> List[str]:
    """Safely load a dataset with error handling."""
    try:
        from datasets import load_dataset
        
        print(f"  Loading {config.name}...", end=' ')
        
        # Load dataset
        if config.subset:
            ds = load_dataset(config.source, config.subset, split=config.split)
        else:
            ds = load_dataset(config.source, split=config.split)
        
        # Get format function
        format_fn = FORMAT_FUNCTIONS.get(config.format_fn)
        if not format_fn:
            print(f"⚠ No format function for {config.format_fn}")
            return []
        
        # Sample and format
        n_samples = min(len(ds), config.max_samples)
        if n_samples < len(ds):
            indices = random.sample(range(len(ds)), n_samples)
            samples = [ds[i] for i in indices]
        else:
            samples = list(ds)
        
        texts = []
        for sample in samples:
            try:
                text = format_fn(sample)
                if text and len(text) > 10:
                    texts.append(text)
            except Exception as e:
                continue
        
        print(f"✓ {len(texts):,} samples")
        return texts
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return []


def load_all_datasets(
    dataset_names: Optional[List[str]] = None,
    max_len: int = 512,
) -> Tuple[MultiDomainDataset, Dict[str, int]]:
    """
    Load all configured datasets.
    
    Args:
        dataset_names: List of dataset names to load (None = all)
        max_len: Maximum sequence length
    
    Returns:
        MultiDomainDataset and stats dict
    """
    print("\n" + "=" * 60)
    print("Loading Multi-Domain Datasets")
    print("=" * 60)
    
    if dataset_names is None:
        dataset_names = list(DATASET_CONFIGS.keys())
    
    domain_texts: Dict[str, List[str]] = {}
    stats = {}
    
    for name in dataset_names:
        if name not in DATASET_CONFIGS:
            print(f"  ⚠ Unknown dataset: {name}")
            continue
        
        config = DATASET_CONFIGS[name]
        texts = load_dataset_safe(config)
        
        if texts:
            domain = config.domain
            if domain not in domain_texts:
                domain_texts[domain] = []
            domain_texts[domain].extend(texts)
            stats[name] = len(texts)
    
    # Create domain datasets
    domain_datasets = {}
    weights = {}
    
    print("\n" + "-" * 40)
    print("Domain Summary:")
    
    for domain, texts in domain_texts.items():
        if texts:
            domain_datasets[domain] = DomainDataset(texts, max_len, domain)
            # Get average weight from configs
            domain_weight = sum(
                DATASET_CONFIGS[n].weight 
                for n in dataset_names 
                if n in DATASET_CONFIGS and DATASET_CONFIGS[n].domain == domain
            )
            weights[domain] = domain_weight
            print(f"  {domain:15s}: {len(texts):>7,} samples (weight: {domain_weight:.2f})")
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    
    print("-" * 40)
    total_samples = sum(len(ds) for ds in domain_datasets.values())
    print(f"  TOTAL: {total_samples:,} samples across {len(domain_datasets)} domains")
    print("=" * 60 + "\n")
    
    return MultiDomainDataset(domain_datasets, weights), stats


def create_dataloaders(
    train_dataset: MultiDomainDataset,
    val_dataset: Optional[MultiDomainDataset],
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    weighted_sampling: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create DataLoaders with optional weighted sampling."""
    
    if weighted_sampling:
        sampler = train_dataset.get_weighted_sampler()
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    
    return train_loader, val_loader


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Testing Multi-Domain Data Loader")
    print("=" * 60)
    
    # Test special token encoding
    test_text = "<|user|>Hello world<|end|><|assistant|>Hi there!<|end|>"
    encoded = encode_special(test_text)
    decoded = decode_special(encoded)
    
    print(f"\nSpecial token test:")
    print(f"  Original: {test_text}")
    print(f"  Encoded:  {encoded[:20]}...")
    print(f"  Decoded:  {decoded}")
    
    # Test loading a small dataset
    print("\nLoading test datasets (small subset)...")
    
    # Temporarily reduce max samples for testing
    test_configs = ['dolly', 'humaneval', 'mbpp']
    
    # Override max_samples for quick test
    for name in test_configs:
        if name in DATASET_CONFIGS:
            DATASET_CONFIGS[name] = DatasetConfig(
                **{**DATASET_CONFIGS[name].__dict__, 'max_samples': 100}
            )
    
    dataset, stats = load_all_datasets(test_configs, max_len=256)
    
    print(f"\nDataset stats: {stats}")
    print(f"Total samples: {len(dataset)}")
    
    # Test a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample:")
        print(f"  Input shape: {sample['input'].shape}")
        print(f"  Target shape: {sample['target'].shape}")
        print(f"  Domain: {sample['domain']}")
        
        # Decode first 100 bytes
        decoded_input = decode_special(sample['input'][:100].tolist())
        print(f"  First 100 bytes: {decoded_input[:100]}...")
    
    print("\n✓ Multi-domain data loader ready!")
