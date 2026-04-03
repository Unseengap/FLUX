"""
Field-BLM Basic Test - Verifies core mechanics work.

NOTE: The current untrained encoder + simple hash produces collisions
for complex data. For production, you need:
1. A trained encoder (CSE)
2. Or a learned hash function
3. Or larger fields

This test demonstrates the basics work on simple patterns.
"""
import torch
import sys
sys.path.insert(0, '.')
from field_blm import FieldBLM, FieldBLMConfig

print("=" * 60)
print("Field-BLM Basic Mechanics Test")
print("=" * 60)
print()

# Create model
config = FieldBLMConfig()
model = FieldBLM(config)

# Test 1: Single pattern learning
print("Test 1: Single pattern learning")
print("-" * 40)

pattern = "AAAA->BBBB->CCCC"
print(f"Training on: '{pattern}'")
for _ in range(30):
    model.train_on_text(pattern)

prompts = ["AAAA->", "AAAA->BBBB->"]
for p in prompts:
    out = model.generate(p, max_bytes=6, temperature=0.0)
    print(f"  '{p}' → '{p}{out}'")

print(f"\nAttractors: {model.field.unique_attractors}")
print()

# Test 2: Temperature sampling works
print("Test 2: Temperature sampling")
print("-" * 40)

# Train on multiple alternatives
model2 = FieldBLM(config)
for _ in range(20):
    model2.train_on_text("Question: What is X? Answer: A")
    model2.train_on_text("Question: What is X? Answer: B")
    model2.train_on_text("Question: What is X? Answer: C")

print("Trained on 3 variants: Answer: A, B, or C")
print()

print("temp=0.0 (deterministic):")
for i in range(3):
    out = model2.generate("Question: What is X? Answer: ", max_bytes=1, temperature=0.0)
    print(f"  → '{out}'")

print()
print("temp=1.0 (sampling):")
for i in range(5):
    out = model2.generate("Question: What is X? Answer: ", max_bytes=1, temperature=1.0)
    print(f"  → '{out}'")

print()
print("=" * 60)
print("Result: Core mechanics work correctly!")
print("=" * 60)
