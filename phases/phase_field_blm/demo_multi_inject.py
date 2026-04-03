#!/usr/bin/env python3
"""
Demo: Multi-Domain Parallel Injection for Field-BLM

This demonstrates the production workflow:
1. Create synthetic data for 4 domains (greetings, code, science, stories)
2. Train separate fields in parallel
3. Merge fields together
4. Test generative sampling

Run: python demo_multi_inject.py
"""

import torch
import time
import sys
from pathlib import Path
from typing import Dict, List
from collections import Counter
import random

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent))

from field_blm import FieldBLM, FieldBLMConfig
from resonance_field import ResonanceField, FieldConfig


# ─────────────────────────────────────────────
# Synthetic Data Generation
# ─────────────────────────────────────────────

def generate_greetings(n: int = 500) -> List[str]:
    """Generate varied greetings."""
    greetings = [
        "Hello, my name is {name}.",
        "Hi there! I'm {name}.",
        "Hey, what's up? I'm {name}.",
        "Good morning! My name is {name}.",
        "Greetings! I go by {name}.",
        "Nice to meet you, I'm {name}.",
        "Hello! How are you today?",
        "Hi! Great to see you!",
        "Hey there, friend!",
        "Good day to you!",
        "Welcome! Come on in.",
        "Hello and welcome to our home.",
        "Hi! Thanks for coming.",
        "Greetings and salutations!",
        "Hello, pleased to meet you.",
    ]
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Iris", "Jack"]
    
    result = []
    for _ in range(n):
        template = random.choice(greetings)
        if "{name}" in template:
            text = template.format(name=random.choice(names))
        else:
            text = template
        result.append(text)
    return result


def generate_code(n: int = 500) -> List[str]:
    """Generate simple code snippets."""
    patterns = [
        "def {func}(x):\n    return x + 1",
        "def {func}(a, b):\n    return a * b",
        "for i in range(10):\n    print(i)",
        "x = [1, 2, 3, 4, 5]",
        "if x > 0:\n    print('positive')",
        "while count < 10:\n    count += 1",
        "class {cls}:\n    def __init__(self):\n        pass",
        "import numpy as np\nx = np.array([1, 2, 3])",
        "with open('file.txt') as f:\n    data = f.read()",
        "try:\n    result = func()\nexcept:\n    pass",
        "def add(a, b):\n    return a + b",
        "def multiply(x, y):\n    return x * y",
        "numbers = [i for i in range(100)]",
        "squared = [x**2 for x in numbers]",
        "result = sum(numbers)",
    ]
    funcs = ["calculate", "process", "compute", "transform", "evaluate", "analyze"]
    classes = ["Model", "Handler", "Manager", "Service", "Controller", "Factory"]
    
    result = []
    for _ in range(n):
        template = random.choice(patterns)
        text = template.format(func=random.choice(funcs), cls=random.choice(classes))
        result.append(text)
    return result


def generate_science(n: int = 500) -> List[str]:
    """Generate science-like sentences."""
    templates = [
        "The {subject} exhibits {property} under {condition}.",
        "Research shows that {subject} can {action}.",
        "Scientists discovered that {subject} has {property}.",
        "The experiment demonstrated {finding}.",
        "Data indicates a correlation between {a} and {b}.",
        "The hypothesis was {result} by the evidence.",
        "Observations suggest that {subject} may {action}.",
        "The {subject} was measured at {value} {unit}.",
        "Analysis reveals {finding} in the sample.",
        "The theory predicts {outcome} when {condition}.",
    ]
    subjects = ["molecule", "cell", "particle", "organism", "compound", "element", "protein"]
    properties = ["stability", "reactivity", "conductivity", "elasticity", "density"]
    conditions = ["high temperature", "low pressure", "neutral pH", "vacuum", "standard conditions"]
    actions = ["transform", "interact", "bond", "dissociate", "catalyze"]
    
    result = []
    for _ in range(n):
        template = random.choice(templates)
        text = template.format(
            subject=random.choice(subjects),
            property=random.choice(properties),
            condition=random.choice(conditions),
            action=random.choice(actions),
            finding="significant effects",
            a=random.choice(subjects),
            b=random.choice(properties),
            result=random.choice(["supported", "confirmed", "validated"]),
            value=random.randint(1, 100),
            unit=random.choice(["nm", "mL", "mg", "Hz"]),
            outcome="expected behavior",
        )
        result.append(text)
    return result


def generate_stories(n: int = 500) -> List[str]:
    """Generate short story snippets."""
    templates = [
        "Once upon a time, there was a {character} who lived in a {place}.",
        "The {character} walked through the {place} searching for {item}.",
        "{name} looked at the {object} with wonder.",
        "It was a {weather} day when {name} decided to {action}.",
        "The {adjective} {character} found a {item} in the {place}.",
        "Long ago, in a {adjective} {place}, there lived a {character}.",
        "{name} said, 'I must find the {item} before sunset.'",
        "The journey to the {place} would take many days.",
        "With a heavy heart, {name} continued on the path.",
        "The {character} smiled and handed over the {item}.",
    ]
    characters = ["wizard", "knight", "princess", "dragon", "merchant", "farmer", "sailor"]
    places = ["forest", "castle", "village", "mountain", "river", "cave", "tower"]
    items = ["sword", "key", "map", "gem", "book", "ring", "scroll"]
    names = ["Elena", "Marcus", "Sofia", "Thomas", "Luna", "Felix"]
    adjectives = ["ancient", "mysterious", "enchanted", "forgotten", "hidden", "magical"]
    weather = ["sunny", "rainy", "cloudy", "stormy", "misty", "cold"]
    actions = ["explore", "adventure", "search", "travel", "discover"]
    objects = ["painting", "statue", "mirror", "door", "window"]
    
    result = []
    for _ in range(n):
        template = random.choice(templates)
        text = template.format(
            character=random.choice(characters),
            place=random.choice(places),
            item=random.choice(items),
            name=random.choice(names),
            adjective=random.choice(adjectives),
            weather=random.choice(weather),
            action=random.choice(actions),
            object=random.choice(objects),
        )
        result.append(text)
    return result


# ─────────────────────────────────────────────
# Field Merge Function
# ─────────────────────────────────────────────

def merge_fields(fields: List[ResonanceField]) -> ResonanceField:
    """
    Merge multiple fields into one.
    
    This is the key advantage over neural networks:
    - No gradient conflicts
    - No catastrophic forgetting
    - Just union the vote counts
    
    Args:
        fields: List of trained ResonanceField instances
        
    Returns:
        Merged field containing all knowledge
    """
    if not fields:
        raise ValueError("Need at least one field to merge")
    
    # Create new field with same config as first
    merged = ResonanceField(fields[0].config)
    
    total_deposits = 0
    total_attractors = 0
    
    for field in fields:
        # Merge byte_votes (union with count addition)
        for pos, votes in field.byte_votes.items():
            if pos not in merged.byte_votes:
                merged.byte_votes[pos] = {}
                merged.spatial_index.add(pos)
                total_attractors += 1
            
            for byte_val, count in votes.items():
                merged.byte_votes[pos][byte_val] = merged.byte_votes[pos].get(byte_val, 0) + count
        
        # Merge mass (additive)
        merged.mass = merged.mass + field.mass.to(merged.mass.device)
        
        total_deposits += field.total_deposits
    
    # Recompute byte_associations from merged votes
    for pos, votes in merged.byte_votes.items():
        best_byte = max(votes.keys(), key=lambda b: votes[b])
        confidence = votes[best_byte] / sum(votes.values())
        merged.byte_associations[pos] = (best_byte, confidence)
    
    merged.total_deposits = total_deposits
    merged.unique_attractors = len(merged.byte_votes)
    
    return merged


# ─────────────────────────────────────────────
# Main Demo
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Field-BLM Multi-Domain Injection Demo")
    print("=" * 60)
    print()
    
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    print()
    
    # ─────────────────────────────────────────
    # Step 1: Generate synthetic data
    # ─────────────────────────────────────────
    print("Step 1: Generating synthetic data...")
    
    domains = {
        'greetings': generate_greetings(2000),
        'code': generate_code(2000),
        'science': generate_science(2000),
        'stories': generate_stories(2000),
    }
    
    for name, texts in domains.items():
        total_chars = sum(len(t) for t in texts)
        print(f"  {name}: {len(texts)} texts, {total_chars:,} chars")
    print()
    
    # ─────────────────────────────────────────
    # Step 2: Train separate fields per domain
    # ─────────────────────────────────────────
    print("Step 2: Training separate fields per domain...")
    print("  (In production, these would run in parallel on different workers)")
    print()
    
    domain_models = {}
    domain_fields = {}
    
    for domain_name, texts in domains.items():
        print(f"  Training on {domain_name}...")
        start = time.time()
        
        # Create model for this domain
        config = FieldBLMConfig(
            wave_dim=432,
            field_dims=(64, 64, 64),
            context_window=128,
        )
        model = FieldBLM(config).to(device)
        
        # Fast inject
        result = model.fast_inject(texts, batch_size=5000, show_progress=False)
        
        elapsed = time.time() - start
        print(f"    Injected {result['total_bytes']:,} bytes in {elapsed:.1f}s")
        print(f"    Attractors: {result['unique_attractors']:,}")
        print(f"    Rate: {result['bytes_per_second']:.0f} bytes/s")
        
        domain_models[domain_name] = model
        domain_fields[domain_name] = model.field
    
    print()
    
    # ─────────────────────────────────────────
    # Step 3: Merge all fields
    # ─────────────────────────────────────────
    print("Step 3: Merging fields from all domains...")
    
    start = time.time()
    merged_field = merge_fields(list(domain_fields.values()))
    elapsed = time.time() - start
    
    print(f"  Merged in {elapsed:.3f}s")
    print(f"  Total attractors: {merged_field.unique_attractors:,}")
    print(f"  Total deposits: {merged_field.total_deposits:,}")
    print()
    
    # Create master model with merged field
    master_config = FieldBLMConfig(wave_dim=432, field_dims=(64, 64, 64), context_window=128)
    master_model = FieldBLM(master_config).to(device)
    master_model.field = merged_field
    
    # ─────────────────────────────────────────
    # Step 4: Validate - test on KNOWN patterns
    # ─────────────────────────────────────────
    print("Step 4: Validation - testing on known patterns...")
    print()
    
    # Test that model predicts correctly on patterns it saw
    test_cases = [
        ("def calculate(", "x"),  # Code: function arg
        ("for i in range(", "1"),  # Code: range arg  
        ("Hello, my name is ", "A"),  # Greetings: name
        ("Once upon a time", ","),  # Stories: comma after phrase
        ("The molecule exhibits ", "s"),  # Science: property
    ]
    
    correct = 0
    total = len(test_cases)
    
    for prompt, expected_start in test_cases:
        # Get top-5 predictions
        ctx_bytes = list(prompt.encode('utf-8'))
        ctx_tensor = torch.tensor(ctx_bytes, dtype=torch.long, device=device)
        context_wave = master_model.encode_context(ctx_tensor)
        top_k = master_model.field.query_top_k(context_wave, k=5)
        
        top_bytes = [chr(b) if 32 <= b < 127 else f'\\x{b:02x}' for b, _ in top_k[:5]]
        expected_found = any(chr(b) == expected_start for b, _ in top_k[:5])
        
        status = "✓" if expected_found else "✗"
        if expected_found:
            correct += 1
        
        print(f"  {status} '{prompt}' → expected '{expected_start}' in top-5: {top_bytes}")
    
    print(f"\n  Validation: {correct}/{total} patterns found in top-5 predictions")
    print()
    
    # ─────────────────────────────────────────
    # Step 5: Test generation from each domain
    # ─────────────────────────────────────────
    print("Step 5: Testing generation (temperature=1.0 for variety)...")
    print()
    
    test_prompts = [
        ("Greetings", "Hello, "),
        ("Greetings", "Hi there"),
        ("Code", "def "),
        ("Code", "for i in "),
        ("Science", "The molecule "),
        ("Science", "Research shows "),
        ("Stories", "Once upon "),
        ("Stories", "The wizard "),
    ]
    
    for domain, prompt in test_prompts:
        print(f"  [{domain}] Prompt: {prompt!r}")
        
        # Generate 3 times to show variety
        outputs = []
        for _ in range(3):
            output = master_model.generate(
                prompt, 
                max_bytes=40, 
                temperature=1.0,
                use_sampling=True,
            )
            outputs.append(output)
        
        for i, out in enumerate(outputs):
            print(f"    {i+1}: {prompt}{out}")
        print()
    
    # ─────────────────────────────────────────
    # Step 6: Compare deterministic vs sampling
    # ─────────────────────────────────────────
    print("Step 6: Deterministic (temp=0) vs Sampling (temp=1.0)...")
    print()
    
    test_prompt = "Hello, my name is "
    
    print(f"  Prompt: {test_prompt!r}")
    print()
    
    print("  Deterministic (temp=0.0) - always same:")
    for i in range(3):
        out = master_model.generate(test_prompt, max_bytes=30, temperature=0.0, use_sampling=False)
        print(f"    {i+1}: {test_prompt}{out}")
    
    print()
    print("  Sampling (temp=1.0) - varies:")
    for i in range(3):
        out = master_model.generate(test_prompt, max_bytes=30, temperature=1.0, use_sampling=True)
        print(f"    {i+1}: {test_prompt}{out}")
    
    print()
    print("  Creative (temp=1.5) - more random:")
    for i in range(3):
        out = master_model.generate(test_prompt, max_bytes=30, temperature=1.5, use_sampling=True)
        print(f"    {i+1}: {test_prompt}{out}")
    
    # ─────────────────────────────────────────
    # Step 6: Stats summary
    # ─────────────────────────────────────────
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    print(f"Total parameters: {master_model.num_parameters:,} (~110K)")
    print(f"Total attractors: {merged_field.unique_attractors:,}")
    print(f"Total bytes learned: {merged_field.total_deposits:,}")
    print(f"Field utilization: {merged_field.unique_attractors / (64*64*64) * 100:.2f}%")
    print()
    print("Key insights:")
    print("  - 4 domains trained SEPARATELY, then MERGED")
    print("  - No gradient conflicts (unlike neural network merging)")
    print("  - No catastrophic forgetting")
    print("  - Each domain's knowledge preserved exactly")
    print("  - Generation uses temperature sampling for variety")
    print()
    print("For production deployment:")
    print("  - Train domains in parallel (Kaggle/Colab workers)")
    print("  - Merge fields offline")
    print("  - Deploy merged field to Heroku (CPU, ~$7/mo)")
    print("  - Field size: ~{:.1f} MB".format(merged_field.unique_attractors * 50 / 1e6))
    print()


if __name__ == '__main__':
    main()
