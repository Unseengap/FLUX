"""
Resonance Field: Parameter-free knowledge storage for BLM.

Instead of weight matrices, knowledge is stored as "attractors" in a
continuous 3D field. Similar patterns cluster together spatially.

Key properties:
- NO learnable parameters
- Dynamic storage (grows with knowledge)
- No catastrophic forgetting
- O(log n) query via spatial indexing
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class FieldConfig:
    """Configuration for Resonance Field."""
    dims: Tuple[int, int, int] = (64, 64, 64)  # Spatial dimensions
    features: int = 512                         # Features per point
    wave_dim: int = 432                         # Input wave dimension
    decay_radius: int = 3                       # Influence spread radius
    min_mass: float = 0.1                       # Minimum mass to be attractor
    temperature: float = 1.0                    # Initial temperature


class SpatialIndex:
    """
    Efficient spatial index for finding nearest attractors.
    Provides O(log n) lookup instead of O(n).
    """
    
    def __init__(self, dims: Tuple[int, int, int]):
        self.dims = dims
        self.attractor_positions: List[Tuple[int, int, int]] = []
        self.position_to_idx: Dict[Tuple[int, int, int], int] = {}
    
    def add(self, pos: Tuple[int, int, int]):
        """Add attractor position to index."""
        if pos not in self.position_to_idx:
            self.position_to_idx[pos] = len(self.attractor_positions)
            self.attractor_positions.append(pos)
    
    def find_nearest(self, pos: Tuple[int, int, int], k: int = 1) -> List[Tuple[int, int, int]]:
        """Find k nearest attractors to position."""
        if not self.attractor_positions:
            return [pos]
        
        # Calculate distances
        distances = []
        for attr_pos in self.attractor_positions:
            dist = sum((a - b) ** 2 for a, b in zip(pos, attr_pos))
            distances.append((dist, attr_pos))
        
        # Sort by distance
        distances.sort(key=lambda x: x[0])
        
        return [d[1] for d in distances[:k]]
    
    def find_in_radius(self, pos: Tuple[int, int, int], radius: int) -> List[Tuple[int, int, int]]:
        """Find all attractors within radius."""
        result = []
        for attr_pos in self.attractor_positions:
            dist_sq = sum((a - b) ** 2 for a, b in zip(pos, attr_pos))
            if dist_sq <= radius ** 2:
                result.append(attr_pos)
        return result


class ResonanceField:
    """
    Resonance Field: stores knowledge as attractors in continuous space.
    
    NO PARAMETERS - all storage is dynamic.
    
    How it works:
    1. Wave → Position: Hash wave to (x, y, z) coordinates
    2. Deposit: Create/strengthen attractor at position
    3. Query: Find nearest strong attractor, return stored info
    
    Like a landscape where:
    - Attractors are valleys
    - Mass is depth of valley
    - Queries roll down to nearest valley
    """
    
    def __init__(self, config: FieldConfig = None):
        config = config or FieldConfig()
        self.config = config
        self.dims = config.dims
        self.features = config.features
        self.wave_dim = config.wave_dim
        
        # Field state: NOT parameters, just storage
        # Shape: [D, H, W, features]
        self.state = torch.zeros(
            config.dims[0], config.dims[1], config.dims[2], config.features
        )
        
        # Mass at each point (evidence accumulator)
        # Higher mass = stronger attractor = more confident
        self.mass = torch.zeros(config.dims)
        
        # Byte association: position → most common next byte
        # Dict[(x,y,z)] → (byte_value, confidence)
        self.byte_associations: Dict[Tuple[int, int, int], Tuple[int, float]] = {}
        
        # Byte votes: track all seen bytes at each position
        # Dict[(x,y,z)] → Dict[byte_value → count]
        self.byte_votes: Dict[Tuple[int, int, int], Dict[int, int]] = {}
        
        # Spatial index for efficient lookup
        self.spatial_index = SpatialIndex(config.dims)
        
        # Wave projection: how to map wave to position
        # Uses first 192 dims split into 3 groups of 64
        self.projection_split = (64, 64, 64)  # = 192 dims used
        
        # Statistics
        self.total_deposits = 0
        self.unique_attractors = 0
    
    @property
    def device(self):
        return self.state.device
    
    def to(self, device):
        """Move field to device."""
        self.state = self.state.to(device)
        self.mass = self.mass.to(device)
        return self
    
    def wave_to_position(self, wave: torch.Tensor) -> Tuple[int, int, int]:
        """
        Map wave vector to field position.
        
        Uses a combination of wave statistics to compute stable
        position coordinates that spread well across the grid.
        
        Args:
            wave: [wave_dim] or [batch, wave_dim]
        Returns:
            (x, y, z) position tuple
        """
        if wave.dim() == 2:
            wave = wave[0]  # Take first in batch
        
        wave = wave.detach().cpu()
        
        # Normalize wave to zero mean to handle encoder bias
        wave_norm = wave - wave.mean()
        
        # Use different strategies for each coordinate:
        # x: use first 64 elements variance + specific indices
        # y: use middle 64 elements + different indices  
        # z: use last 64 elements + remaining indices
        
        # Get specific indices for additional entropy
        idx1 = int(abs(wave_norm[0].item() * 1000)) % 64
        idx2 = int(abs(wave_norm[100].item() * 1000)) % 64
        idx3 = int(abs(wave_norm[200].item() * 1000)) % 64
        
        # Compute coordinates using normalized wave sections + indexed elements
        w1 = wave_norm[:144]  # First third
        w2 = wave_norm[144:288]  # Middle third
        w3 = wave_norm[288:]  # Last third
        
        # Use combination of mean, std, and specific element
        x_raw = (w1.mean() + w1[idx1] * 0.5).item()
        y_raw = (w2.mean() + w2[idx2] * 0.5).item() 
        z_raw = (w3.mean() + w3[idx3] * 0.5).item()
        
        # Map to grid using sigmoid for better spread
        x = int(torch.sigmoid(torch.tensor(x_raw * 3)).item() * (self.dims[0] - 1))
        y = int(torch.sigmoid(torch.tensor(y_raw * 3)).item() * (self.dims[1] - 1))
        z = int(torch.sigmoid(torch.tensor(z_raw * 3)).item() * (self.dims[2] - 1))
        
        # Clamp to valid range
        x = max(0, min(self.dims[0] - 1, x))
        y = max(0, min(self.dims[1] - 1, y))
        z = max(0, min(self.dims[2] - 1, z))
        
        return (x, y, z)
    
    def waves_to_positions_batch(self, waves: torch.Tensor) -> torch.Tensor:
        """
        VECTORIZED: Map batch of waves to positions.
        
        Args:
            waves: [batch, wave_dim] wave vectors
        Returns:
            [batch, 3] position tensor (x, y, z for each)
        """
        waves = waves.detach()
        batch_size = waves.size(0)
        
        # Normalize waves to zero mean (per wave)
        wave_means = waves.mean(dim=1, keepdim=True)
        waves_norm = waves - wave_means
        
        # Split into thirds
        w1 = waves_norm[:, :144]  # First third
        w2 = waves_norm[:, 144:288]  # Middle third
        w3 = waves_norm[:, 288:]  # Last third
        
        # Get specific indices for additional entropy
        idx1 = (waves_norm[:, 0].abs() * 1000).long() % 64
        idx2 = (waves_norm[:, 100].abs() * 1000).long() % 64
        idx3 = (waves_norm[:, 200].abs() * 1000).long() % 64
        
        # Gather indexed elements
        elem1 = torch.gather(w1, 1, idx1.unsqueeze(1).clamp(0, w1.size(1)-1)).squeeze(1)
        elem2 = torch.gather(w2, 1, idx2.unsqueeze(1).clamp(0, w2.size(1)-1)).squeeze(1)
        elem3 = torch.gather(w3, 1, idx3.unsqueeze(1).clamp(0, w3.size(1)-1)).squeeze(1)
        
        # Combine mean and indexed element
        x_raw = w1.mean(dim=1) + elem1 * 0.5
        y_raw = w2.mean(dim=1) + elem2 * 0.5
        z_raw = w3.mean(dim=1) + elem3 * 0.5
        
        # Map to grid using sigmoid
        x = (torch.sigmoid(x_raw * 3) * (self.dims[0] - 1)).long()
        y = (torch.sigmoid(y_raw * 3) * (self.dims[1] - 1)).long()
        z = (torch.sigmoid(z_raw * 3) * (self.dims[2] - 1)).long()
        
        # Clamp to valid range
        x = x.clamp(0, self.dims[0] - 1)
        y = y.clamp(0, self.dims[1] - 1)
        z = z.clamp(0, self.dims[2] - 1)
        
        return torch.stack([x, y, z], dim=1).cpu()  # [batch, 3]
    
    def bulk_deposit(
        self,
        positions: torch.Tensor,  # [N, 3] positions
        next_bytes: torch.Tensor,  # [N] byte values
        evidence: float = 1.0,
    ) -> int:
        """
        BULK deposit many (position, byte) pairs at once.
        
        Much faster than individual deposits by avoiding Python loops.
        
        Args:
            positions: [N, 3] tensor of (x, y, z) positions
            next_bytes: [N] tensor of byte values (0-255)
            evidence: Evidence strength per deposit
            
        Returns:
            Number of new attractors created
        """
        positions = positions.cpu()
        next_bytes = next_bytes.cpu()
        N = positions.size(0)
        
        new_attractors = 0
        
        # Group by position for efficiency
        # Convert positions to tuple keys
        pos_to_bytes = {}
        for i in range(N):
            pos = (positions[i, 0].item(), positions[i, 1].item(), positions[i, 2].item())
            byte_val = next_bytes[i].item()
            
            if pos not in pos_to_bytes:
                pos_to_bytes[pos] = []
            pos_to_bytes[pos].append(byte_val)
        
        # Now deposit grouped by position (much fewer dict operations)
        for pos, bytes_list in pos_to_bytes.items():
            count = len(bytes_list)
            total_evidence = evidence * count
            
            # Update mass
            self.mass[pos] += total_evidence
            
            # Track bytes
            if pos not in self.byte_votes:
                self.byte_votes[pos] = {}
                self.unique_attractors += 1
                self.spatial_index.add(pos)
                new_attractors += 1
            
            votes = self.byte_votes[pos]
            for b in bytes_list:
                votes[b] = votes.get(b, 0) + 1
            
            # Update primary association
            best_byte = max(votes.keys(), key=lambda b: votes[b])
            confidence = votes[best_byte] / sum(votes.values())
            self.byte_associations[pos] = (best_byte, confidence)
            
            self.total_deposits += count
        
        return new_attractors
    
    def deposit(
        self, 
        context_wave: torch.Tensor, 
        next_byte: int, 
        evidence: float = 1.0
    ):
        """
        Deposit knowledge into field. NO BACKPROP.
        
        This creates or strengthens an attractor at the position
        corresponding to context_wave, associated with next_byte.
        
        Args:
            context_wave: [wave_dim] wave representing context
            next_byte: The byte that follows this context (0-255)
            evidence: Strength of evidence (default 1.0)
        """
        pos = self.wave_to_position(context_wave)
        
        # Increase mass (more evidence = stronger attractor)
        self.mass[pos] += evidence
        
        # Store wave features at this position
        # If wave_dim < features, pad with zeros; if wave_dim > features, truncate
        wave_detached = context_wave.detach().cpu()
        if wave_detached.size(0) < self.features:
            wave_features = torch.zeros(self.features)
            wave_features[:wave_detached.size(0)] = wave_detached
        else:
            wave_features = wave_detached[:self.features]
        
        # Running average of waves landing here
        current_mass = self.mass[pos].item()
        old_weight = (current_mass - evidence) / current_mass
        new_weight = evidence / current_mass
        self.state[pos] = old_weight * self.state[pos] + new_weight * wave_features
        
        # Track byte votes
        if pos not in self.byte_votes:
            self.byte_votes[pos] = {}
            self.unique_attractors += 1
            self.spatial_index.add(pos)
        
        votes = self.byte_votes[pos]
        votes[next_byte] = votes.get(next_byte, 0) + 1
        
        # Update primary byte association (majority vote)
        best_byte = max(votes.keys(), key=lambda b: votes[b])
        confidence = votes[best_byte] / sum(votes.values())
        self.byte_associations[pos] = (best_byte, confidence)
        
        # Spread influence to neighbors (creates smooth basin)
        self._spread_influence(pos, evidence)
        
        self.total_deposits += 1
    
    def _spread_influence(self, pos: Tuple[int, int, int], evidence: float):
        """Spread attractor influence to neighboring positions."""
        radius = self.config.decay_radius
        x, y, z = pos
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    
                    nx = x + dx
                    ny = y + dy
                    nz = z + dz
                    
                    # Check bounds
                    if not (0 <= nx < self.dims[0] and 
                            0 <= ny < self.dims[1] and 
                            0 <= nz < self.dims[2]):
                        continue
                    
                    # Distance-based decay
                    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                    decay = max(0, 1 - dist / (radius + 1))
                    
                    # Add small mass to neighbors
                    self.mass[nx, ny, nz] += evidence * decay * 0.1
    
    def query(
        self, 
        context_wave: torch.Tensor
    ) -> Tuple[Optional[int], float, Tuple[int, int, int]]:
        """
        Query field for predicted next byte.
        
        Args:
            context_wave: [wave_dim] wave representing context
            
        Returns:
            (predicted_byte, confidence, position)
            predicted_byte is None if no attractor found
        """
        pos = self.wave_to_position(context_wave)
        
        # Check if we have an attractor at or near this position
        if pos in self.byte_associations:
            byte_val, conf = self.byte_associations[pos]
            return byte_val, conf, pos
        
        # Find nearest attractor
        nearest_positions = self.spatial_index.find_nearest(pos, k=1)
        
        if nearest_positions and nearest_positions[0] in self.byte_associations:
            nearest = nearest_positions[0]
            byte_val, conf = self.byte_associations[nearest]
            
            # Reduce confidence based on distance
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos, nearest)))
            adjusted_conf = conf * math.exp(-dist / 10)
            
            return byte_val, adjusted_conf, nearest
        
        return None, 0.0, pos
    
    def query_sample(
        self, 
        context_wave: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[Optional[int], float, Tuple[int, int, int]]:
        """
        Query field and SAMPLE from distribution (generative).
        
        Unlike query() which returns argmax, this samples proportionally
        to the vote counts, allowing diverse outputs.
        
        Args:
            context_wave: [wave_dim] wave representing context
            temperature: Sampling temperature
                - 0.0 = argmax (deterministic)
                - 1.0 = sample proportional to counts
                - >1.0 = more random/creative
                - <1.0 = more focused/conservative
            
        Returns:
            (sampled_byte, confidence, position)
            sampled_byte is None if no attractor found
        """
        pos = self.wave_to_position(context_wave)
        
        # Get votes from this position or nearest
        votes = None
        actual_pos = pos
        
        if pos in self.byte_votes:
            votes = self.byte_votes[pos]
        else:
            # Find nearest attractor
            nearest_positions = self.spatial_index.find_nearest(pos, k=1)
            if nearest_positions and nearest_positions[0] in self.byte_votes:
                actual_pos = nearest_positions[0]
                votes = self.byte_votes[actual_pos]
        
        if not votes:
            return None, 0.0, pos
        
        # Convert to tensors for sampling
        bytes_list = list(votes.keys())
        counts = torch.tensor([votes[b] for b in bytes_list], dtype=torch.float)
        
        if temperature == 0.0:
            # Argmax (deterministic)
            idx = counts.argmax().item()
        else:
            # Convert counts to log-space (like logits), then temperature-scale
            # This gives proper sampling: temp=1 ~ proportional to counts
            log_counts = torch.log(counts + 1e-10)
            probs = F.softmax(log_counts / temperature, dim=0)
            idx = torch.multinomial(probs, 1).item()
        
        sampled_byte = bytes_list[idx]
        confidence = counts[idx].item() / counts.sum().item()
        
        return sampled_byte, confidence, actual_pos
    
    def query_top_k(
        self, 
        context_wave: torch.Tensor,
        k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Query field for top-k predicted bytes.
        
        Args:
            context_wave: [wave_dim] wave representing context
            k: Number of predictions to return
            
        Returns:
            List of (byte_value, confidence) tuples, sorted by confidence
        """
        pos = self.wave_to_position(context_wave)
        
        # Get all bytes from nearby attractors
        byte_scores: Dict[int, float] = {}
        
        # Check nearby positions
        radius = max(5, self.config.decay_radius * 2)
        nearby = self.spatial_index.find_in_radius(pos, radius)
        
        for attr_pos in nearby:
            if attr_pos in self.byte_votes:
                votes = self.byte_votes[attr_pos]
                total_votes = sum(votes.values())
                mass = self.mass[attr_pos].item()
                
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos, attr_pos)))
                dist_factor = math.exp(-dist / 5)
                
                for byte_val, count in votes.items():
                    score = (count / total_votes) * mass * dist_factor
                    byte_scores[byte_val] = byte_scores.get(byte_val, 0) + score
        
        # If no nearby attractors, return uniform
        if not byte_scores:
            return [(i, 1/256) for i in range(k)]
        
        # Sort by score
        sorted_bytes = sorted(byte_scores.items(), key=lambda x: -x[1])
        
        # Normalize to probabilities
        total = sum(s for _, s in sorted_bytes)
        result = [(b, s/total) for b, s in sorted_bytes[:k]]
        
        return result
    
    def get_logits(self, context_wave: torch.Tensor) -> torch.Tensor:
        """
        Get byte logits for compatibility with standard training.
        
        Args:
            context_wave: [wave_dim] or [batch, wave_dim]
            
        Returns:
            [batch, 256] log probabilities
        """
        if context_wave.dim() == 1:
            context_wave = context_wave.unsqueeze(0)
        
        batch_size = context_wave.size(0)
        logits = torch.zeros(batch_size, 256, device=context_wave.device)
        
        for i in range(batch_size):
            top_k = self.query_top_k(context_wave[i], k=256)
            for byte_val, prob in top_k:
                logits[i, byte_val] = math.log(prob + 1e-10)
        
        return logits
    
    def stats(self) -> Dict:
        """Return field statistics."""
        return {
            'total_deposits': self.total_deposits,
            'unique_attractors': self.unique_attractors,
            'total_mass': self.mass.sum().item(),
            'max_mass': self.mass.max().item(),
            'avg_mass': self.mass.sum().item() / max(1, self.unique_attractors),
            'field_utilization': self.unique_attractors / (self.dims[0] * self.dims[1] * self.dims[2]),
        }
    
    def save(self, path: str):
        """Save field state to file."""
        state_dict = {
            'config': self.config,
            'state': self.state,
            'mass': self.mass,
            'byte_associations': self.byte_associations,
            'byte_votes': self.byte_votes,
            'attractor_positions': self.spatial_index.attractor_positions,
            'total_deposits': self.total_deposits,
            'unique_attractors': self.unique_attractors,
        }
        torch.save(state_dict, path)
    
    @classmethod
    def load(cls, path: str) -> 'ResonanceField':
        """Load field state from file."""
        state_dict = torch.load(path)
        
        field = cls(state_dict['config'])
        field.state = state_dict['state']
        field.mass = state_dict['mass']
        field.byte_associations = state_dict['byte_associations']
        field.byte_votes = state_dict['byte_votes']
        field.total_deposits = state_dict['total_deposits']
        field.unique_attractors = state_dict['unique_attractors']
        
        # Rebuild spatial index
        for pos in state_dict['attractor_positions']:
            field.spatial_index.add(tuple(pos))
        
        return field


# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    # Create field
    field = ResonanceField()
    
    print("Resonance Field Created")
    print(f"  Dims: {field.dims}")
    print(f"  Features: {field.features}")
    print(f"  Field size: {field.state.numel():,} values")
    print(f"  Mass size: {field.mass.numel():,} values")
    print(f"  Total storage: {(field.state.numel() + field.mass.numel()) * 4 / 1e6:.2f} MB")
    print(f"")
    print(f"Note: This is STORAGE, not parameters!")
    print(f"      Parameters = 0")
    print(f"      Storage grows with knowledge")
    
    # Test deposit
    print(f"\nTesting deposit...")
    wave = torch.randn(432)
    field.deposit(wave, next_byte=ord('a'), evidence=1.0)
    
    print(f"  Deposited 'a' at wave position")
    print(f"  Stats: {field.stats()}")
    
    # Test query
    print(f"\nTesting query...")
    result, conf, pos = field.query(wave)
    print(f"  Query result: byte={result} ({chr(result) if result else '?'}), conf={conf:.3f}")
    
    # Test slightly different wave
    wave2 = wave + torch.randn_like(wave) * 0.1
    result2, conf2, pos2 = field.query(wave2)
    print(f"  Similar wave: byte={result2} ({chr(result2) if result2 else '?'}), conf={conf2:.3f}")
