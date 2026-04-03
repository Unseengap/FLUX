"""
Wave Resonance Memory (WRM): Anti-Catastrophic Forgetting for Byte Language Models.

A novel approach unique to BLMs that prevents forgetting at TWO levels:

1. **Semantic Wave Anchors (SWA)** - Preserves wave-space representations
   - Stores "anchor waves" for representative samples
   - Regularizes during fine-tuning to keep wave encodings stable
   - Works in continuous 432D semantic space

2. **Byte Transition Memory (BTM)** - Preserves critical byte predictions  
   - Learns which byte transitions are "critical" (high confidence)
   - Regularizes to maintain these transition probabilities
   - Unique to BLMs: 320 bytes is tractable (vs 50K tokens)

Why this is BLM-specific:
- Token LLMs have 50K+ tokens → transition matrix too large
- Token LLMs don't have wave encodings → no wave anchors
- BLMs: 320 bytes + 432D waves = perfect for dual-level memory

Usage:
    # Before SFT
    wrm = WaveResonanceMemory(model, calibration_samples)
    
    # During SFT
    loss = main_loss + wrm.compute_preservation_loss(model, input_ids, logits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json


# Try relative import, fall back to absolute
try:
    from .multi_domain_data import encode_special, decode_special, VOCAB_SIZE
except ImportError:
    from multi_domain_data import encode_special, decode_special, VOCAB_SIZE


@dataclass
class WRMConfig:
    """Configuration for Wave Resonance Memory."""
    # Semantic Wave Anchors
    num_anchors_per_domain: int = 50      # Samples per domain to anchor
    wave_anchor_weight: float = 0.1       # Weight for wave anchor loss
    
    # Byte Transition Memory
    transition_threshold: float = 0.8     # Confidence threshold for critical transitions
    transition_weight: float = 0.05       # Weight for transition preservation loss
    
    # Resonance Detection
    resonance_percentile: float = 0.9     # Top N% variance dims are "resonant"


class SemanticWaveAnchors(nn.Module):
    """
    Prevents forgetting by anchoring key wave representations.
    
    Unlike EWC (parameter-space) or replay (data-space), this operates in
    semantic wave space - unique to byte language models.
    
    Physics intuition: Stored knowledge creates "standing waves" in semantic
    space. We anchor these waves so new learning doesn't destructively interfere.
    """
    
    def __init__(self, wave_dim: int = 432):
        super().__init__()
        self.wave_dim = wave_dim
        self.anchors: Dict[str, Dict] = {}
        self.domain_centroids: Dict[str, Tensor] = {}
        self.frozen = False
    
    @torch.no_grad()
    def compute_anchors(
        self, 
        model, 
        samples: List[str], 
        domains: List[str],
        device: str = 'cuda'
    ):
        """
        Compute and store anchor waves for representative samples.
        Call this BEFORE fine-tuning.
        """
        model.eval()
        
        # Group by domain
        domain_waves = {}
        
        for sample, domain in zip(samples, domains):
            try:
                # Get wave encoding
                wave = model.encode_text(sample)  # [seq, 432]
                wave_mean = wave.mean(dim=0)      # [432] - pooled representation
                
                # Store individual anchor
                sample_key = hash(sample) % (10**9)  # Compact key
                self.anchors[sample_key] = {
                    'wave': wave_mean.cpu().clone(),
                    'domain': domain,
                    'text_hash': hash(sample),
                }
                
                # Accumulate for domain centroid
                if domain not in domain_waves:
                    domain_waves[domain] = []
                domain_waves[domain].append(wave_mean.cpu())
                
            except Exception as e:
                print(f"  Skipping sample: {e}")
                continue
        
        # Compute domain centroids (average wave per domain)
        for domain, waves in domain_waves.items():
            if waves:
                centroid = torch.stack(waves).mean(dim=0)
                self.domain_centroids[domain] = centroid
        
        self.frozen = True
        print(f"✓ Anchored {len(self.anchors)} samples across {len(self.domain_centroids)} domains")
        
        model.train()
    
    def compute_anchor_loss(
        self, 
        model, 
        input_ids: Tensor,
        domain: Optional[str] = None,
    ) -> Tensor:
        """
        Compute loss based on drift from domain centroid.
        
        Args:
            model: The BLM being fine-tuned
            input_ids: Current batch [batch, seq]
            domain: Optional domain hint
        
        Returns:
            Loss penalizing drift from anchored wave space
        """
        if not self.frozen or not self.domain_centroids:
            return torch.tensor(0.0, device=input_ids.device)
        
        # Get current wave encoding for batch
        # We need to access the internal encoding
        batch_size, seq_len = input_ids.shape
        
        with torch.amp.autocast('cuda'):
            # Encode through CSE
            is_special = input_ids >= 256
            cse_input = input_ids.clone()
            cse_input[is_special] = 0
            
            semantic_wave = model.cse.encode_bytes(cse_input)
            wave = semantic_wave.full  # [batch, seq, 432]
            
            # Handle special tokens if model has them
            if hasattr(model, 'special_token_embed') and is_special.any():
                special_indices = (input_ids - 256).clamp(0, model.num_special_tokens - 1)
                special_embeds = model.special_token_embed(special_indices)
                wave = torch.where(
                    is_special.unsqueeze(-1).expand_as(wave),
                    special_embeds,
                    wave
                )
            
            # Pool to get batch representation
            wave_pooled = wave.mean(dim=(0, 1))  # [432]
        
        # Compute distance to nearest domain centroid
        min_distance = float('inf')
        for d, centroid in self.domain_centroids.items():
            centroid = centroid.to(wave_pooled.device)
            
            # Cosine distance in wave space
            cos_sim = F.cosine_similarity(wave_pooled.unsqueeze(0), centroid.unsqueeze(0))
            distance = 1 - cos_sim
            
            if domain and d == domain:
                # Use this domain's centroid
                min_distance = distance
                break
            elif distance < min_distance:
                min_distance = distance
        
        return min_distance if isinstance(min_distance, Tensor) else torch.tensor(0.0, device=input_ids.device)
    
    def get_resonant_dimensions(self) -> Tensor:
        """
        Identify which wave dimensions are most consistent (resonant) across anchors.
        These dimensions encode stable knowledge that should be protected.
        """
        if not self.anchors:
            return torch.ones(self.wave_dim)
        
        all_waves = torch.stack([a['wave'] for a in self.anchors.values()])
        
        # Low variance = consistent = resonant = important
        variance = all_waves.var(dim=0)
        
        # Inverse variance as importance (with smoothing)
        importance = 1.0 / (variance + 0.01)
        importance = importance / importance.max()  # Normalize to [0, 1]
        
        return importance


class ByteTransitionMemory(nn.Module):
    """
    Preserves critical byte transition probabilities during fine-tuning.
    
    UNIQUE TO BLMs: With only 320 bytes, we can track a 320x320 transition
    matrix. Token models with 50K+ tokens cannot do this efficiently.
    
    Key insight: Certain byte transitions are critical for language:
    - 'q' → 'u' (English)
    - '(' → follows identifier in code
    - '\n' → indentation patterns
    
    We identify high-confidence transitions and regularize to preserve them.
    """
    
    def __init__(self, vocab_size: int = 320, threshold: float = 0.8):
        super().__init__()
        self.vocab_size = vocab_size
        self.threshold = threshold
        
        # Transition statistics
        self.register_buffer('transition_probs', torch.zeros(vocab_size, vocab_size))
        self.register_buffer('transition_counts', torch.zeros(vocab_size, vocab_size))
        self.register_buffer('critical_mask', torch.zeros(vocab_size, vocab_size, dtype=torch.bool))
        
        self.frozen = False
        self.num_critical = 0
    
    @torch.no_grad()
    def compute_transitions(
        self,
        model,
        calibration_texts: List[str],
        device: str = 'cuda',
        max_samples: int = 1000,
    ):
        """
        Learn byte transition probabilities from the pre-trained model.
        Call this BEFORE fine-tuning.
        """
        model.eval()
        
        transition_sum = torch.zeros(self.vocab_size, self.vocab_size, device=device)
        transition_count = torch.zeros(self.vocab_size, self.vocab_size, device=device)
        
        for text in calibration_texts[:max_samples]:
            try:
                # Encode text
                byte_ids = encode_special(text)
                if len(byte_ids) < 2:
                    continue
                
                # Truncate to reasonable length
                byte_ids = byte_ids[:512]
                
                input_ids = torch.tensor(byte_ids[:-1], device=device).unsqueeze(0)
                target_ids = torch.tensor(byte_ids[1:], device=device)
                
                # Get model predictions
                with torch.amp.autocast('cuda'):
                    output = model(input_ids)
                    logits = output['logits'].squeeze(0)  # [seq, vocab]
                    probs = F.softmax(logits, dim=-1)
                
                # Accumulate transition probabilities
                for i, (inp, tgt) in enumerate(zip(byte_ids[:-1], byte_ids[1:])):
                    if inp < self.vocab_size and tgt < self.vocab_size:
                        prob = probs[i, tgt].item()
                        transition_sum[inp, tgt] += prob
                        transition_count[inp, tgt] += 1
                        
            except Exception as e:
                continue
        
        # Average probabilities
        mask = transition_count > 0
        self.transition_probs = torch.zeros_like(transition_sum)
        self.transition_probs[mask] = transition_sum[mask] / transition_count[mask]
        self.transition_counts = transition_count.cpu()
        
        # Identify critical transitions (high confidence)
        self.critical_mask = self.transition_probs > self.threshold
        self.num_critical = self.critical_mask.sum().item()
        
        # Move to CPU for storage
        self.transition_probs = self.transition_probs.cpu()
        self.critical_mask = self.critical_mask.cpu()
        
        self.frozen = True
        
        print(f"✓ Identified {self.num_critical:,} critical byte transitions (threshold={self.threshold})")
        
        # Show top transitions
        top_transitions = []
        for i in range(self.vocab_size):
            for j in range(self.vocab_size):
                if self.critical_mask[i, j] and self.transition_counts[i, j] > 10:
                    prob = self.transition_probs[i, j].item()
                    from_char = chr(i) if i < 128 and i >= 32 else f'<{i}>'
                    to_char = chr(j) if j < 128 and j >= 32 else f'<{j}>'
                    top_transitions.append((prob, from_char, to_char))
        
        top_transitions.sort(reverse=True)
        print(f"  Top critical transitions:")
        for prob, from_c, to_c in top_transitions[:10]:
            print(f"    '{from_c}' → '{to_c}': {prob:.3f}")
        
        model.train()
    
    def compute_preservation_loss(
        self,
        logits: Tensor,
        input_ids: Tensor,
    ) -> Tensor:
        """
        Compute loss that penalizes deviation from critical transitions.
        
        Args:
            logits: Model output [batch, seq, vocab]
            input_ids: Input bytes [batch, seq]
        
        Returns:
            Loss penalizing drift from critical byte transitions
        """
        if not self.frozen or self.num_critical == 0:
            return torch.tensor(0.0, device=logits.device)
        
        batch_size, seq_len, _ = logits.shape
        device = logits.device
        
        # Get current probabilities
        probs = F.softmax(logits, dim=-1)
        
        total_loss = 0.0
        count = 0
        
        # Move masks to device
        ref_probs = self.transition_probs.to(device)
        critical = self.critical_mask.to(device)
        
        for b in range(batch_size):
            for s in range(seq_len - 1):
                prev_byte = input_ids[b, s].item()
                
                if prev_byte >= self.vocab_size:
                    continue
                
                # Check all critical transitions from this byte
                critical_nexts = critical[prev_byte].nonzero(as_tuple=True)[0]
                
                for next_byte in critical_nexts:
                    next_byte = next_byte.item()
                    
                    ref_prob = ref_probs[prev_byte, next_byte]
                    curr_prob = probs[b, s, next_byte]
                    
                    # Penalize if current probability drops below reference
                    # This preserves learned byte patterns
                    loss = F.relu(ref_prob - curr_prob)
                    total_loss += loss
                    count += 1
        
        return total_loss / max(count, 1)


class WaveResonanceMemory(nn.Module):
    """
    Complete anti-forgetting system for Byte Language Models.
    
    Combines:
    - Semantic Wave Anchors (continuous wave space)
    - Byte Transition Memory (discrete byte predictions)
    
    This dual-level approach is unique to BLMs and cannot be applied
    to standard token-based language models.
    """
    
    def __init__(self, config: WRMConfig = None):
        super().__init__()
        self.config = config or WRMConfig()
        
        self.wave_anchors = SemanticWaveAnchors()
        self.byte_memory = ByteTransitionMemory(
            vocab_size=VOCAB_SIZE,
            threshold=self.config.transition_threshold,
        )
        
        self.initialized = False
    
    @torch.no_grad()
    def initialize_from_model(
        self,
        model,
        calibration_texts: List[str],
        domains: Optional[List[str]] = None,
        device: str = 'cuda',
    ):
        """
        Initialize memory from pre-trained model.
        Call this ONCE before fine-tuning begins.
        """
        print("\n" + "=" * 60)
        print("Initializing Wave Resonance Memory")
        print("=" * 60)
        
        # Default domains if not provided
        if domains is None:
            # Infer from text patterns
            domains = []
            for text in calibration_texts:
                if '<|code|>' in text or 'def ' in text:
                    domains.append('code')
                elif '<|reasoning|>' in text or '<|problem|>' in text:
                    domains.append('reasoning')
                else:
                    domains.append('assistant')
        
        # 1. Compute wave anchors
        print("\n1. Computing Semantic Wave Anchors...")
        anchor_samples = calibration_texts[:self.config.num_anchors_per_domain * 5]
        anchor_domains = domains[:len(anchor_samples)]
        self.wave_anchors.compute_anchors(model, anchor_samples, anchor_domains, device)
        
        # 2. Compute byte transitions
        print("\n2. Computing Byte Transition Memory...")
        self.byte_memory.compute_transitions(model, calibration_texts, device)
        
        self.initialized = True
        print("\n" + "=" * 60)
        print("✓ Wave Resonance Memory initialized!")
        print("=" * 60 + "\n")
    
    def compute_preservation_loss(
        self,
        model,
        input_ids: Tensor,
        logits: Tensor,
        domain: Optional[str] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute combined preservation loss.
        
        Args:
            model: The BLM being fine-tuned
            input_ids: Input byte IDs [batch, seq]
            logits: Model output logits [batch, seq, vocab]
            domain: Optional domain hint
        
        Returns:
            total_loss: Combined preservation loss
            breakdown: Dict with individual loss components
        """
        if not self.initialized:
            zero = torch.tensor(0.0, device=input_ids.device)
            return zero, {'wave_anchor': 0.0, 'byte_transition': 0.0}
        
        # Wave anchor loss
        wave_loss = self.wave_anchors.compute_anchor_loss(model, input_ids, domain)
        
        # Byte transition loss
        byte_loss = self.byte_memory.compute_preservation_loss(logits, input_ids)
        
        # Weighted combination
        total_loss = (
            self.config.wave_anchor_weight * wave_loss +
            self.config.transition_weight * byte_loss
        )
        
        breakdown = {
            'wave_anchor': wave_loss.item() if isinstance(wave_loss, Tensor) else wave_loss,
            'byte_transition': byte_loss.item() if isinstance(byte_loss, Tensor) else byte_loss,
        }
        
        return total_loss, breakdown
    
    def save(self, path: str):
        """Save WRM state to disk."""
        state = {
            'config': self.config,
            'wave_anchors': self.wave_anchors.anchors,
            'domain_centroids': {k: v.cpu() for k, v in self.wave_anchors.domain_centroids.items()},
            'transition_probs': self.byte_memory.transition_probs.cpu(),
            'transition_counts': self.byte_memory.transition_counts.cpu(),
            'critical_mask': self.byte_memory.critical_mask.cpu(),
            'initialized': self.initialized,
        }
        torch.save(state, path)
        print(f"✓ WRM saved to {path}")
    
    def load(self, path: str):
        """Load WRM state from disk."""
        state = torch.load(path, map_location='cpu')
        
        self.config = state['config']
        self.wave_anchors.anchors = state['wave_anchors']
        self.wave_anchors.domain_centroids = state['domain_centroids']
        self.wave_anchors.frozen = True
        
        self.byte_memory.transition_probs = state['transition_probs']
        self.byte_memory.transition_counts = state['transition_counts']
        self.byte_memory.critical_mask = state['critical_mask']
        self.byte_memory.num_critical = state['critical_mask'].sum().item()
        self.byte_memory.frozen = True
        
        self.initialized = state['initialized']
        print(f"✓ WRM loaded from {path}")


def create_wrm_for_sft(
    model,
    train_dataset,
    num_calibration_samples: int = 500,
    device: str = 'cuda',
) -> WaveResonanceMemory:
    """
    Convenience function to create and initialize WRM before SFT.
    
    Args:
        model: Pre-trained BLM
        train_dataset: Training dataset (to sample calibration data)
        num_calibration_samples: Number of samples for calibration
        device: Device to use
    
    Returns:
        Initialized WaveResonanceMemory
    """
    # Sample calibration data from training set
    indices = torch.randperm(len(train_dataset))[:num_calibration_samples]
    
    calibration_texts = []
    domains = []
    
    for idx in indices:
        sample = train_dataset[idx.item()]
        # Decode back to text
        input_ids = sample['input'].tolist()
        text = decode_special(input_ids)
        calibration_texts.append(text)
        
        # Get domain if available
        domain = sample.get('domain', 'unknown')
        domains.append(domain)
    
    # Create and initialize WRM
    wrm = WaveResonanceMemory()
    wrm.initialize_from_model(model, calibration_texts, domains, device)
    
    return wrm


# ─────────────────────────────────────────────
# Example Usage
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Wave Resonance Memory - Demo")
    print("=" * 50)
    
    # This would be used like:
    """
    # Before SFT
    wrm = WaveResonanceMemory()
    wrm.initialize_from_model(model, calibration_texts, domains)
    
    # During SFT training loop
    for batch in dataloader:
        input_ids = batch['input'].to(device)
        target_ids = batch['target'].to(device)
        
        # Forward pass
        output = model(input_ids, target_ids)
        main_loss = output['loss']
        
        # Add preservation loss
        preservation_loss, breakdown = wrm.compute_preservation_loss(
            model, input_ids, output['logits']
        )
        
        total_loss = main_loss + preservation_loss
        total_loss.backward()
        
        # Log
        print(f"Main: {main_loss:.4f} | Wave: {breakdown['wave_anchor']:.4f} | Byte: {breakdown['byte_transition']:.4f}")
    """
    
    print("\nWRM provides dual-level anti-forgetting:")
    print("  1. Semantic Wave Anchors - preserves wave space representations")
    print("  2. Byte Transition Memory - preserves critical byte predictions")
    print("\nThis is UNIQUE to Byte Language Models!")
