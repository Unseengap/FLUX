"""
Thermodynamic Settler: Energy-based inference and learning without backprop.

Instead of gradient descent, the system "settles" to minimum energy states.
Learning is just depositing into the field - no weight updates.

Key properties:
- NO BACKPROPAGATION
- Temperature-based exploration
- Energy = disagreement with field
- Settling = finding stable prediction
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class ThermodynamicConfig:
    """Configuration for thermodynamic settler."""
    initial_temperature: float = 1.0
    min_temperature: float = 0.1
    cooling_rate: float = 0.99
    energy_threshold: float = 0.1      # Stable when energy < threshold
    max_settling_steps: int = 10       # Max iterations to settle
    exploration_noise: float = 0.1     # Noise for exploration


class ThermodynamicSettler:
    """
    Thermodynamic Settler: inference through energy minimization.
    
    NO PARAMETERS. NO BACKPROP.
    
    How it works:
    1. Given context wave, query field for prediction
    2. Calculate "energy" = uncertainty/disagreement
    3. If high energy + high temperature → explore (add noise)
    4. If low energy or low temperature → settle (commit)
    5. Learning = deposit correct answer into field
    
    Temperature schedule:
    - Start high (explore)
    - Cool down (exploit)
    - Reset on new sequence
    """
    
    def __init__(self, field, config: ThermodynamicConfig = None):
        """
        Args:
            field: ResonanceField instance
            config: ThermodynamicConfig
        """
        self.field = field
        self.config = config or ThermodynamicConfig()
        
        # Current temperature
        self.temperature = self.config.initial_temperature
        
        # Session statistics
        self.total_predictions = 0
        self.total_energy = 0.0
        self.settled_predictions = 0
        self.explored_predictions = 0
    
    def reset_temperature(self):
        """Reset temperature for new sequence."""
        self.temperature = self.config.initial_temperature
    
    def cool(self):
        """Cool temperature by one step."""
        self.temperature = max(
            self.config.min_temperature,
            self.temperature * self.config.cooling_rate
        )
    
    def compute_energy(
        self, 
        context_wave: torch.Tensor,
        top_k_predictions: List[Tuple[int, float]]
    ) -> float:
        """
        Compute energy = uncertainty in prediction.
        
        Low energy = confident prediction (sharp distribution)
        High energy = uncertain (flat distribution)
        
        Args:
            context_wave: [wave_dim] context
            top_k_predictions: List of (byte, prob) from field
            
        Returns:
            Energy value (0 = certain, higher = uncertain)
        """
        if not top_k_predictions:
            return float('inf')  # Maximum uncertainty
        
        # Entropy-based energy
        probs = [p for _, p in top_k_predictions]
        total = sum(probs)
        if total < 1e-10:
            return float('inf')
        
        probs = [p / total for p in probs]
        
        # Shannon entropy (normalized)
        entropy = 0.0
        for p in probs:
            if p > 1e-10:
                entropy -= p * math.log2(p)
        
        # Normalize: max entropy for 256 bytes = 8 bits
        energy = entropy / 8.0
        
        return energy
    
    def settle(
        self, 
        context_wave: torch.Tensor,
        return_distribution: bool = False
    ) -> Tuple[int, float]:
        """
        Settle to prediction through energy minimization.
        
        Args:
            context_wave: [wave_dim] context wave
            return_distribution: If True, return full distribution
            
        Returns:
            (predicted_byte, confidence) or ((predicted_byte, confidence), distribution)
        """
        self.total_predictions += 1
        
        # Query field for predictions
        top_k = self.field.query_top_k(context_wave, k=256)
        
        if not top_k:
            # No knowledge - random prediction
            return (0, 0.0) if not return_distribution else ((0, 0.0), [(i, 1/256) for i in range(256)])
        
        # Compute energy
        energy = self.compute_energy(context_wave, top_k)
        self.total_energy += energy
        
        # Decision: explore or exploit?
        if energy > self.config.energy_threshold and self.temperature > self.config.min_temperature:
            # High energy + high temperature = explore
            self.explored_predictions += 1
            prediction = self._explore(top_k)
        else:
            # Low energy or cool = exploit
            self.settled_predictions += 1
            prediction = top_k[0]  # Take best
        
        # Cool for next step
        self.cool()
        
        if return_distribution:
            return prediction, top_k
        return prediction
    
    def _explore(
        self, 
        top_k: List[Tuple[int, float]]
    ) -> Tuple[int, float]:
        """
        Explore: sample from distribution with temperature scaling.
        
        Higher temperature = more random
        Lower temperature = more greedy
        """
        if not top_k:
            return (0, 0.0)
        
        # Temperature-scaled sampling
        bytes_list = [b for b, _ in top_k]
        probs = torch.tensor([p for _, p in top_k])
        
        # Apply temperature
        if self.temperature > 0:
            scaled_probs = probs / self.temperature
            scaled_probs = F.softmax(scaled_probs, dim=0)
        else:
            scaled_probs = probs / probs.sum()
        
        # Sample
        idx = torch.multinomial(scaled_probs, 1).item()
        sampled_byte = bytes_list[idx]
        confidence = probs[idx].item()
        
        return (sampled_byte, confidence)
    
    def learn(
        self, 
        context_wave: torch.Tensor, 
        correct_byte: int,
        evidence: float = 1.0
    ):
        """
        Learn = deposit correct answer into field.
        
        NO BACKPROP. Just storage.
        
        Args:
            context_wave: [wave_dim] context wave
            correct_byte: The actual next byte (ground truth)
            evidence: Strength of evidence
        """
        self.field.deposit(context_wave, correct_byte, evidence)
    
    def predict_and_learn(
        self, 
        context_wave: torch.Tensor,
        correct_byte: int,
        evidence: float = 1.0
    ) -> Tuple[int, float, bool]:
        """
        Combined predict and learn in one step.
        
        Args:
            context_wave: [wave_dim] context
            correct_byte: Ground truth byte
            evidence: Learning strength
            
        Returns:
            (predicted_byte, confidence, was_correct)
        """
        # Predict
        predicted, confidence = self.settle(context_wave)
        was_correct = (predicted == correct_byte)
        
        # Learn (always - reinforces correct or corrects errors)
        self.learn(context_wave, correct_byte, evidence)
        
        return predicted, confidence, was_correct
    
    def stats(self) -> Dict:
        """Return settler statistics."""
        return {
            'total_predictions': self.total_predictions,
            'settled_predictions': self.settled_predictions,
            'explored_predictions': self.explored_predictions,
            'settle_ratio': self.settled_predictions / max(1, self.total_predictions),
            'avg_energy': self.total_energy / max(1, self.total_predictions),
            'current_temperature': self.temperature,
            'field_stats': self.field.stats(),
        }


class ThermodynamicBatch:
    """
    Batch processing for thermodynamic learning.
    
    Processes sequences in parallel while maintaining
    the single-pass, no-epoch philosophy.
    """
    
    def __init__(self, settler: ThermodynamicSettler):
        self.settler = settler
    
    def process_sequence(
        self, 
        waves: torch.Tensor,  # [seq_len, wave_dim]
        target_bytes: torch.Tensor,  # [seq_len] target bytes
        reset_temperature: bool = True
    ) -> Dict:
        """
        Process entire sequence.
        
        Args:
            waves: [seq_len, wave_dim] context waves at each position
            target_bytes: [seq_len] ground truth bytes
            reset_temperature: Whether to reset temp at start
            
        Returns:
            Dict with accuracy, avg_confidence, etc.
        """
        if reset_temperature:
            self.settler.reset_temperature()
        
        seq_len = waves.size(0)
        correct = 0
        total_conf = 0.0
        
        for i in range(seq_len):
            pred, conf, was_correct = self.settler.predict_and_learn(
                waves[i],
                target_bytes[i].item()
            )
            if was_correct:
                correct += 1
            total_conf += conf
        
        return {
            'accuracy': correct / seq_len,
            'avg_confidence': total_conf / seq_len,
            'seq_len': seq_len,
        }


# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    from resonance_field import ResonanceField
    
    # Create field and settler
    field = ResonanceField()
    settler = ThermodynamicSettler(field)
    
    print("Thermodynamic Settler")
    print(f"  Initial temperature: {settler.temperature}")
    print(f"  Min temperature: {settler.config.min_temperature}")
    print(f"  Cooling rate: {settler.config.cooling_rate}")
    print()
    
    # Test learning
    print("Testing learn + predict cycle...")
    
    # Create a simple pattern: after wave -> 'a'
    wave = torch.randn(432)
    
    # Learn this association multiple times
    print("  Learning wave -> 'a' (10 times)...")
    for _ in range(10):
        settler.learn(wave, ord('a'))
    
    # Reset temperature for prediction
    settler.reset_temperature()
    
    # Predict
    pred, conf = settler.settle(wave)
    print(f"  Prediction: {pred} ({chr(pred)}), confidence: {conf:.3f}")
    
    # Stats
    print(f"\nSettler stats: {settler.stats()}")
    
    # Test predict_and_learn
    print("\nTesting predict_and_learn...")
    wave2 = torch.randn(432)
    pred, conf, correct = settler.predict_and_learn(wave2, ord('b'))
    print(f"  First prediction (no prior knowledge): byte={pred}, correct={correct}")
    
    # Repeat - should learn
    for _ in range(5):
        pred, conf, correct = settler.predict_and_learn(wave2, ord('b'))
    print(f"  After 5 iterations: byte={pred} ({chr(pred)}), correct={correct}")
