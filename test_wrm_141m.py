#!/usr/bin/env python3
"""
Test Wave Resonance Memory (WRM) Anti-Forgetting on Flux-LM-141M.

This demonstrates:
1. Capturing model's knowledge state before fine-tuning
2. Fine-tuning on new domain (code) WITHOUT WRM → catastrophic forgetting
3. Fine-tuning on new domain WITH WRM → knowledge preserved

The original model was trained on WikiText-103 (Wikipedia text).
We'll fine-tune on code snippets and measure Wikipedia generation quality.
"""

import sys
sys.path.insert(0, 'phases/phase_lm')

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from flux_lm import FluxLM, GenerationConfig

# ─────────────────────────────────────────────
# Simplified WRM for Original FluxLM
# ─────────────────────────────────────────────

@dataclass
class WRMConfig:
    """Configuration for Wave Resonance Memory."""
    wave_anchor_weight: float = 0.1
    byte_transition_weight: float = 0.05
    num_anchors: int = 50


class WaveResonanceMemorySimple(nn.Module):
    """
    Simplified WRM for the original FluxLM (no special tokens).
    
    Captures:
    1. Wave space anchors (semantic representations)
    2. Byte transition patterns (critical predictions)
    """
    
    def __init__(self, config: WRMConfig = None):
        super().__init__()
        self.config = config or WRMConfig()
        
        # Wave anchors: stored wave representations
        self.wave_anchors: List[Tensor] = []
        self.wave_centroid: Optional[Tensor] = None
        
        # Byte transition memory: P(next_byte | current_byte)
        self.transition_probs: Optional[Tensor] = None
        self.critical_transitions: Optional[Tensor] = None
        
        self.initialized = False
    
    @torch.no_grad()
    def initialize_from_model(
        self,
        model: FluxLM,
        calibration_texts: List[str],
        device: str = 'cpu',
    ):
        """Capture model's knowledge state before fine-tuning."""
        print("\n" + "=" * 60)
        print("Initializing Wave Resonance Memory")
        print("=" * 60)
        
        model.eval()
        model.to(device)
        
        # 1. Compute wave anchors from calibration texts
        print(f"\nComputing wave anchors from {len(calibration_texts)} samples...")
        all_waves = []
        
        for i, text in enumerate(calibration_texts[:self.config.num_anchors]):
            try:
                # Encode text to bytes
                byte_seq = list(text.encode('utf-8')[:256])
                if len(byte_seq) < 10:
                    continue
                
                bytes_tensor = torch.tensor(byte_seq, dtype=torch.long, device=device).unsqueeze(0)
                
                # Get wave representation
                semantic_wave = model.cse.encode_bytes(bytes_tensor)
                wave = semantic_wave.full  # [1, seq, 432]
                wave_mean = wave.mean(dim=(0, 1))  # [432]
                
                all_waves.append(wave_mean.cpu())
                
            except Exception as e:
                continue
        
        if all_waves:
            self.wave_anchors = all_waves
            self.wave_centroid = torch.stack(all_waves).mean(dim=0)
            print(f"  ✓ Captured {len(all_waves)} wave anchors")
        
        # 2. Compute byte transition statistics
        print("\nComputing byte transition memory...")
        transition_counts = torch.zeros(256, 256)
        
        for text in calibration_texts:
            byte_seq = list(text.encode('utf-8'))
            for i in range(len(byte_seq) - 1):
                curr, next_ = byte_seq[i], byte_seq[i + 1]
                if curr < 256 and next_ < 256:
                    transition_counts[curr, next_] += 1
        
        # Normalize to probabilities
        row_sums = transition_counts.sum(dim=1, keepdim=True).clamp(min=1)
        self.transition_probs = transition_counts / row_sums
        
        # Mark critical transitions (high confidence predictions)
        self.critical_transitions = self.transition_probs > 0.3
        num_critical = self.critical_transitions.sum().item()
        print(f"  ✓ Captured {num_critical:,} critical byte transitions")
        
        self.initialized = True
        print("\n✓ WRM initialized!")
        
        model.train()
    
    def compute_preservation_loss(
        self,
        model: FluxLM,
        input_bytes: Tensor,
        logits: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute preservation loss to prevent forgetting.
        
        Args:
            model: The model being fine-tuned
            input_bytes: [batch, seq] input byte values
            logits: [batch, seq, 256] model output logits
        
        Returns:
            total_loss: Combined preservation loss
            breakdown: Dict with loss components
        """
        if not self.initialized:
            zero = torch.tensor(0.0, device=input_bytes.device)
            return zero, {'wave_anchor': 0.0, 'byte_transition': 0.0}
        
        device = input_bytes.device
        
        # 1. Wave anchor loss: penalize drift from captured wave space
        wave_loss = torch.tensor(0.0, device=device)
        if self.wave_centroid is not None:
            centroid = self.wave_centroid.to(device)
            
            # Get current wave encoding
            semantic_wave = model.cse.encode_bytes(input_bytes)
            wave = semantic_wave.full  # [batch, seq, 432]
            wave_mean = wave.mean(dim=(0, 1))  # [432]
            
            # Cosine distance from centroid
            cos_sim = F.cosine_similarity(
                wave_mean.unsqueeze(0), 
                centroid.unsqueeze(0)
            )
            wave_loss = (1 - cos_sim).squeeze()
        
        # 2. Byte transition loss: preserve critical predictions
        byte_loss = torch.tensor(0.0, device=device)
        if self.transition_probs is not None and self.critical_transitions is not None:
            trans_probs = self.transition_probs.to(device)
            critical = self.critical_transitions.to(device)
            
            batch_size, seq_len, vocab_size = logits.shape
            
            # Get predicted probabilities
            pred_probs = F.softmax(logits, dim=-1)  # [batch, seq, 256]
            
            # For each position, compare to expected transitions
            total_kl = 0.0
            count = 0
            
            for b in range(min(batch_size, 2)):  # Sample for efficiency
                for t in range(min(seq_len - 1, 64)):
                    curr_byte = input_bytes[b, t].item()
                    if curr_byte < 256 and critical[curr_byte].any():
                        # KL divergence from stored distribution
                        expected = trans_probs[curr_byte]
                        predicted = pred_probs[b, t, :256]
                        
                        # KL(expected || predicted)
                        kl = F.kl_div(
                            predicted.log().clamp(min=-100),
                            expected,
                            reduction='sum'
                        )
                        total_kl += kl
                        count += 1
            
            if count > 0:
                byte_loss = total_kl / count
        
        # Combined loss
        total_loss = (
            self.config.wave_anchor_weight * wave_loss +
            self.config.byte_transition_weight * byte_loss
        )
        
        breakdown = {
            'wave_anchor': wave_loss.item() if isinstance(wave_loss, Tensor) else wave_loss,
            'byte_transition': byte_loss.item() if isinstance(byte_loss, Tensor) else byte_loss,
        }
        
        return total_loss, breakdown


# ─────────────────────────────────────────────
# Test Functions
# ─────────────────────────────────────────────

def evaluate_wikipedia_quality(model: FluxLM, device: str) -> Dict[str, float]:
    """Evaluate model's ability to generate Wikipedia-style text."""
    model.eval()
    model.to(device)
    
    wiki_prompts = [
        "The city of",
        "In the year",
        "The scientist discovered",
        "During the war",
        "The university was",
    ]
    
    scores = {
        'coherent': 0,
        'wiki_style': 0,
        'total': len(wiki_prompts),
    }
    
    for prompt in wiki_prompts:
        try:
            output = model.generate(prompt, GenerationConfig(
                max_new_bytes=60,
                temperature=0.7,
                top_k=50,
            ))
            
            generated = output[len(prompt):]
            
            # Check coherence: contains actual words
            words = generated.split()
            if len(words) >= 3:
                scores['coherent'] += 1
            
            # Check wiki style: contains dates, names, or facts
            wiki_markers = ['19', '20', 'the', 'was', 'is', 'by', 'in']
            if any(m in generated.lower() for m in wiki_markers):
                scores['wiki_style'] += 1
                
        except:
            continue
    
    return scores


def evaluate_code_quality(model: FluxLM, device: str) -> Dict[str, float]:
    """Evaluate model's ability to generate code."""
    model.eval()
    model.to(device)
    
    code_prompts = [
        "def fibonacci(",
        "class User:",
        "for i in range(",
        "import numpy",
        "if __name__",
    ]
    
    scores = {
        'coherent': 0,
        'code_style': 0,
        'total': len(code_prompts),
    }
    
    for prompt in code_prompts:
        try:
            output = model.generate(prompt, GenerationConfig(
                max_new_bytes=60,
                temperature=0.7,
                top_k=50,
            ))
            
            generated = output[len(prompt):]
            
            # Check coherence
            if len(generated) >= 5:
                scores['coherent'] += 1
            
            # Check code style
            code_markers = ['def', 'return', ':', '(', ')', 'self', 'import', '=']
            if any(m in generated for m in code_markers):
                scores['code_style'] += 1
                
        except:
            continue
    
    return scores


def fine_tune_on_code(
    model: FluxLM,
    wrm: Optional[WaveResonanceMemorySimple],
    steps: int = 50,
    device: str = 'cpu',
) -> List[float]:
    """Fine-tune model on code snippets."""
    model.train()
    model.to(device)
    
    # Simple code snippets for fine-tuning
    code_samples = [
        "def hello():\n    print('Hello World')\n",
        "def add(a, b):\n    return a + b\n",
        "class Cat:\n    def __init__(self, name):\n        self.name = name\n",
        "for i in range(10):\n    print(i)\n",
        "import os\nimport sys\n",
        "if x > 0:\n    return True\nelse:\n    return False\n",
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n",
        "list_comp = [x*2 for x in range(5)]\n",
        "with open('file.txt', 'r') as f:\n    data = f.read()\n",
        "try:\n    result = func()\nexcept Exception as e:\n    print(e)\n",
    ] * 10  # Repeat for more data
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses = []
    
    for step in range(steps):
        # Sample a batch
        sample = code_samples[step % len(code_samples)]
        byte_seq = list(sample.encode('utf-8')[:128])
        
        input_bytes = torch.tensor(byte_seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
        target_bytes = torch.tensor(byte_seq[1:], dtype=torch.long, device=device).unsqueeze(0)
        
        # Forward pass
        output = model(input_bytes, target_bytes)
        main_loss = output['loss']
        
        # Add WRM preservation loss if available
        total_loss = main_loss
        if wrm is not None:
            preservation_loss, breakdown = wrm.compute_preservation_loss(
                model, input_bytes, output['logits']
            )
            total_loss = main_loss + preservation_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        losses.append(main_loss.item())
        
        if (step + 1) % 10 == 0:
            wrm_info = f" | WRM: {preservation_loss.item():.4f}" if wrm else ""
            print(f"  Step {step+1}/{steps}: Loss={main_loss.item():.4f}{wrm_info}")
    
    return losses


# ─────────────────────────────────────────────
# Main Test
# ─────────────────────────────────────────────

def main():
    device = 'cpu'  # Use CPU for this test
    
    print("=" * 70)
    print("WRM Anti-Forgetting Test on Flux-LM-141M")
    print("=" * 70)
    
    # Load original model
    print("\n1. Loading original Flux-LM-141M model...")
    original_model = FluxLM.load('checkpoints/Flux-LM-141M.pt', device=device)
    
    # Evaluate baseline
    print("\n2. Evaluating baseline (Wikipedia-trained model)...")
    baseline_wiki = evaluate_wikipedia_quality(original_model, device)
    baseline_code = evaluate_code_quality(original_model, device)
    
    print(f"  Wikipedia quality: {baseline_wiki['wiki_style']}/{baseline_wiki['total']}")
    print(f"  Code quality: {baseline_code['code_style']}/{baseline_code['total']}")
    
    # ─────────────────────────────────────────────
    # Test A: Fine-tune WITHOUT WRM (catastrophic forgetting)
    # ─────────────────────────────────────────────
    
    print("\n" + "=" * 70)
    print("TEST A: Fine-tuning WITHOUT WRM (expect forgetting)")
    print("=" * 70)
    
    # Copy model
    model_no_wrm = copy.deepcopy(original_model)
    
    # Fine-tune on code
    print("\nFine-tuning on code...")
    fine_tune_on_code(model_no_wrm, wrm=None, steps=50, device=device)
    
    # Evaluate after fine-tuning
    print("\nEvaluating after fine-tuning WITHOUT WRM...")
    no_wrm_wiki = evaluate_wikipedia_quality(model_no_wrm, device)
    no_wrm_code = evaluate_code_quality(model_no_wrm, device)
    
    print(f"  Wikipedia quality: {no_wrm_wiki['wiki_style']}/{no_wrm_wiki['total']} (was {baseline_wiki['wiki_style']})")
    print(f"  Code quality: {no_wrm_code['code_style']}/{no_wrm_code['total']} (was {baseline_code['code_style']})")
    
    wiki_degradation = baseline_wiki['wiki_style'] - no_wrm_wiki['wiki_style']
    print(f"  → Wikipedia degradation: -{wiki_degradation}")
    
    # ─────────────────────────────────────────────
    # Test B: Fine-tune WITH WRM (preserve knowledge)
    # ─────────────────────────────────────────────
    
    print("\n" + "=" * 70)
    print("TEST B: Fine-tuning WITH WRM (expect preservation)")
    print("=" * 70)
    
    # Copy model
    model_with_wrm = copy.deepcopy(original_model)
    
    # Initialize WRM from original model
    wrm = WaveResonanceMemorySimple(WRMConfig(
        wave_anchor_weight=0.15,
        byte_transition_weight=0.1,
        num_anchors=50,
    ))
    
    # Use Wikipedia-style calibration texts
    calibration_texts = [
        "The city of London is the capital of England and the United Kingdom.",
        "In the year 1969, the Apollo 11 mission landed on the moon.",
        "The scientist Albert Einstein developed the theory of relativity.",
        "During World War II, many countries participated in the conflict.",
        "The university was founded in 1850 and has produced many notable alumni.",
        "The river flows through the valley and into the sea.",
        "The president gave a speech at the national convention.",
        "The museum contains artifacts from ancient civilizations.",
        "The book was published in 1984 and became a bestseller.",
        "The company was established by a group of entrepreneurs.",
    ] * 5
    
    wrm.initialize_from_model(model_with_wrm, calibration_texts, device)
    
    # Fine-tune on code WITH WRM
    print("\nFine-tuning on code WITH WRM preservation...")
    fine_tune_on_code(model_with_wrm, wrm=wrm, steps=50, device=device)
    
    # Evaluate after fine-tuning
    print("\nEvaluating after fine-tuning WITH WRM...")
    with_wrm_wiki = evaluate_wikipedia_quality(model_with_wrm, device)
    with_wrm_code = evaluate_code_quality(model_with_wrm, device)
    
    print(f"  Wikipedia quality: {with_wrm_wiki['wiki_style']}/{with_wrm_wiki['total']} (was {baseline_wiki['wiki_style']})")
    print(f"  Code quality: {with_wrm_code['code_style']}/{with_wrm_code['total']} (was {baseline_code['code_style']})")
    
    wiki_preservation = baseline_wiki['wiki_style'] - with_wrm_wiki['wiki_style']
    print(f"  → Wikipedia degradation: -{wiki_preservation}")
    
    # ─────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────
    
    print("\n" + "=" * 70)
    print("SUMMARY: WRM Anti-Forgetting Results")
    print("=" * 70)
    
    print(f"\n{'Metric':<25} {'Baseline':>10} {'No WRM':>10} {'With WRM':>10}")
    print("-" * 55)
    print(f"{'Wikipedia Style':.<25} {baseline_wiki['wiki_style']:>10} {no_wrm_wiki['wiki_style']:>10} {with_wrm_wiki['wiki_style']:>10}")
    print(f"{'Code Style':.<25} {baseline_code['code_style']:>10} {no_wrm_code['code_style']:>10} {with_wrm_code['code_style']:>10}")
    
    print("\n" + "-" * 55)
    
    improvement = wiki_degradation - wiki_preservation
    if improvement > 0:
        print(f"✓ WRM preserved {improvement} more Wikipedia capabilities!")
        print("  → Anti-forgetting mechanism is working")
    elif improvement == 0:
        print("  WRM showed no improvement (may need tuning)")
    else:
        print("✗ WRM did not help (check configuration)")
    
    # Show sample generations
    print("\n" + "=" * 70)
    print("Sample Generations Comparison")
    print("=" * 70)
    
    test_prompt = "The city of Paris"
    
    print(f"\nPrompt: '{test_prompt}'")
    
    original_model.eval()
    model_no_wrm.eval()
    model_with_wrm.eval()
    
    print(f"\n  Original:  {original_model.generate(test_prompt, GenerationConfig(max_new_bytes=60, temperature=0.7))}")
    print(f"\n  No WRM:    {model_no_wrm.generate(test_prompt, GenerationConfig(max_new_bytes=60, temperature=0.7))}")
    print(f"\n  With WRM:  {model_with_wrm.generate(test_prompt, GenerationConfig(max_new_bytes=60, temperature=0.7))}")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
