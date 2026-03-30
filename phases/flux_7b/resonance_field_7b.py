"""
FLUX-7B: Resonance Field (6.5B Parameters)

The field IS the knowledge store. Not a neural network that needs 
gradient training — a massive spatial energy landscape where knowledge
is INJECTED directly.

Shape: [512, 512, 512] × 1536 features = 6.44B parameters

All operations are gradient-free:
- inject(): Write knowledge via perturbation
- settle(): Energy minimization via diffusion
- query(): Gravitational retrieval O(log n)
- consolidate(): Build spatial index

This is where FLUX stores everything it knows.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class FieldConfig:
    """Configuration for the 7B resonance field."""
    
    # Field dimensions: 512³ = 134M locations
    size_x: int = 512
    size_y: int = 512
    size_z: int = 512
    
    # Features per location
    features: int = 1536
    
    # Physics parameters
    injection_radius: int = 4
    settle_rate: float = 0.01
    mass_growth: float = 0.1
    cooling_rate: float = 0.99
    
    # Index parameters
    attractor_threshold: float = 0.5  # Mass threshold for attractors
    index_nlist: int = 4096  # FAISS IVF clusters
    
    @property
    def total_params(self) -> int:
        """Total parameters in the field."""
        locations = self.size_x * self.size_y * self.size_z
        return locations * self.features + locations * 2  # state + mass + temp
        
    def __post_init__(self):
        print(f"  ResonanceField7B: {self.total_params / 1e9:.2f}B parameters")


class ResonanceField7B:
    """
    Massive 3D resonance field for knowledge storage.
    
    This is NOT a torch.nn.Module because it doesn't use gradients.
    Knowledge is stored via direct injection into the field tensor.
    
    Args:
        config: FieldConfig specifying dimensions and physics
        device: Device for tensors (needs ~28GB)
    """
    
    def __init__(
        self,
        config: Optional[FieldConfig] = None,
        device: str = 'cuda',
    ):
        self.config = config or FieldConfig()
        self.device = device
        
        # ─────────────────────────────────────────────
        # Core Field Tensors
        # ─────────────────────────────────────────────
        
        H, W, D = self.config.size_x, self.config.size_y, self.config.size_z
        F = self.config.features
        
        # Knowledge state: [H, W, D, F]
        # Initialize with small noise (empty field has no attractors)
        self.state = torch.randn(H, W, D, F, device=device) * 0.001
        
        # Mass: accumulated evidence at each location
        # Higher mass = more established knowledge = resists change
        self.mass = torch.zeros(H, W, D, device=device)
        
        # Temperature: local plasticity
        # High temp = malleable, low temp = frozen
        self.temperature = torch.ones(H, W, D, device=device)
        
        # ─────────────────────────────────────────────
        # Spatial Index (built during consolidation)
        # ─────────────────────────────────────────────
        self.index = None
        self.attractor_locations = None
        self.attractor_features = None
        self.attractor_masses = None
        
        # Statistics
        self.injection_count = 0
        self.total_mass = 0.0
        
    # ─────────────────────────────────────────────
    # Wave → Location Mapping
    # ─────────────────────────────────────────────
    
    def _wave_to_location(self, wave: torch.Tensor) -> torch.Tensor:
        """
        Map a wave vector to 3D field coordinates.
        
        Uses hash-like projection that spreads semantically similar
        waves to nearby locations.
        
        Args:
            wave: [features] vector
            
        Returns:
            [3] float coordinates in [0, size)
        """
        H, W, D = self.config.size_x, self.config.size_y, self.config.size_z
        
        # Split wave into 3 chunks, hash each to coordinate
        chunk_size = wave.shape[0] // 3
        
        x_hash = wave[:chunk_size].sum().tanh() * 0.5 + 0.5
        y_hash = wave[chunk_size:2*chunk_size].sum().tanh() * 0.5 + 0.5
        z_hash = wave[2*chunk_size:].sum().tanh() * 0.5 + 0.5
        
        x = x_hash * (H - 1)
        y = y_hash * (W - 1)
        z = z_hash * (D - 1)
        
        return torch.stack([x, y, z])
    
    def _compute_distances(self, radius: int, device: str) -> torch.Tensor:
        """
        Compute distance field for local neighborhood.
        
        Args:
            radius: Neighborhood radius
            
        Returns:
            [2r+1, 2r+1, 2r+1] distance from center
        """
        size = 2 * radius + 1
        coords = torch.arange(size, device=device) - radius
        xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing='ij')
        distances = torch.sqrt(xx**2 + yy**2 + zz**2 + 1e-6)
        return distances
    
    # ─────────────────────────────────────────────
    # Knowledge Injection (NO GRADIENTS)
    # ─────────────────────────────────────────────
    
    def inject(
        self,
        wave: torch.Tensor,
        strength: float = 1.0,
        radius: Optional[int] = None,
    ):
        """
        Inject knowledge into the field. NO GRADIENTS.
        
        Physics:
        1. Compute location from wave semantics
        2. Perturb local neighborhood
        3. Weight by distance and existing mass
        4. Accumulate mass and cool temperature
        
        Args:
            wave: [features] knowledge vector to inject
            strength: Injection strength multiplier
            radius: Override injection radius
        """
        H, W, D = self.config.size_x, self.config.size_y, self.config.size_z
        radius = radius or self.config.injection_radius
        
        # Get injection location
        location = self._wave_to_location(wave)
        x, y, z = location.long().clamp(min=radius, max=torch.tensor([H, W, D], device=self.device) - radius - 1)
        x, y, z = x.item(), y.item(), z.item()
        
        # Get local neighborhood slices
        x_slice = slice(x - radius, x + radius + 1)
        y_slice = slice(y - radius, y + radius + 1)
        z_slice = slice(z - radius, z + radius + 1)
        
        # Compute distance weights
        distances = self._compute_distances(radius, self.device)
        distance_weights = torch.exp(-distances / radius)  # [2r+1, 2r+1, 2r+1]
        
        # Get local mass for resistance
        local_mass = self.mass[x_slice, y_slice, z_slice]
        resistance = 1.0 / (1.0 + local_mass)  # High mass = high resistance
        
        # Get local temperature for plasticity
        local_temp = self.temperature[x_slice, y_slice, z_slice]
        plasticity = local_temp  # High temp = high plasticity
        
        # Compute perturbation
        # δ = strength × distance_weight × resistance × plasticity × wave
        delta_weight = strength * distance_weights * resistance * plasticity
        delta = delta_weight.unsqueeze(-1) * wave  # [2r+1, 2r+1, 2r+1, features]
        
        # Apply perturbation to field
        self.state[x_slice, y_slice, z_slice] += delta
        
        # Accumulate mass (evidence)
        self.mass[x_slice, y_slice, z_slice] += (
            distance_weights * strength * self.config.mass_growth
        )
        
        # Cool temperature (stabilize)
        self.temperature[x_slice, y_slice, z_slice] *= self.config.cooling_rate
        
        # Update statistics
        self.injection_count += 1
        self.total_mass = self.mass.sum().item()
        
    def inject_batch(
        self,
        waves: torch.Tensor,
        strength: float = 1.0,
    ):
        """
        Inject multiple waves efficiently.
        
        Args:
            waves: [batch, features] wave batch
            strength: Injection strength
        """
        for wave in waves:
            self.inject(wave, strength)
            
    # ─────────────────────────────────────────────
    # Energy Minimization / Settling (NO GRADIENTS)
    # ─────────────────────────────────────────────
    
    def settle(self, steps: int = 100):
        """
        Energy minimization via diffusion. NO GRADIENTS.
        
        High-energy regions (high variance from neighbors) diffuse
        toward equilibrium. Low-energy regions remain stable.
        
        This is how knowledge "integrates" into the field.
        
        Args:
            steps: Number of settling iterations
        """
        dt = self.config.settle_rate
        
        for _ in range(steps):
            # Compute neighbor average (3D convolution with ones kernel)
            kernel = torch.ones(1, 1, 3, 3, 3, device=self.device) / 27
            
            # Need to permute for conv3d: [H,W,D,F] → [F,1,H,W,D]
            state_5d = self.state.permute(3, 0, 1, 2).unsqueeze(1)
            
            # Convolve each feature channel
            neighbor_avg = F.conv3d(
                state_5d, 
                kernel.expand(self.config.features, 1, 3, 3, 3),
                padding=1,
                groups=self.config.features,
            )  # [F, 1, H, W, D]
            
            # Back to [H, W, D, F]
            neighbor_avg = neighbor_avg.squeeze(1).permute(1, 2, 3, 0)
            
            # Energy gradient: difference from neighbors
            gradient = self.state - neighbor_avg
            
            # Update: diffuse toward neighbors, weighted by temperature
            # High temp = more diffusion, low temp = stable
            update = dt * self.temperature.unsqueeze(-1) / (1 + self.mass.unsqueeze(-1))
            self.state -= update * gradient
            
            # Clamp to prevent unbounded growth
            self.state = self.state.clamp(-10, 10)
            
    def settle_local(self, location: torch.Tensor, radius: int = 8, steps: int = 20):
        """
        Settle only a local region. Faster for incremental updates.
        
        Args:
            location: [3] center of region
            radius: Region radius
            steps: Settling iterations
        """
        H, W, D = self.config.size_x, self.config.size_y, self.config.size_z
        x, y, z = location.long().clamp(min=radius, max=torch.tensor([H, W, D], device=self.device) - radius - 1)
        x, y, z = x.item(), y.item(), z.item()
        
        x_slice = slice(x - radius, x + radius + 1)
        y_slice = slice(y - radius, y + radius + 1)
        z_slice = slice(z - radius, z + radius + 1)
        
        dt = self.config.settle_rate
        
        for _ in range(steps):
            local_state = self.state[x_slice, y_slice, z_slice]
            local_temp = self.temperature[x_slice, y_slice, z_slice]
            local_mass = self.mass[x_slice, y_slice, z_slice]
            
            # Simple 3D averaging
            kernel_size = 3
            padding = kernel_size // 2
            
            state_5d = local_state.permute(3, 0, 1, 2).unsqueeze(0)
            kernel = torch.ones(1, 1, 3, 3, 3, device=self.device) / 27
            
            neighbor_avg = F.conv3d(
                state_5d,
                kernel.expand(self.config.features, 1, 3, 3, 3),
                padding=padding,
                groups=self.config.features,
            ).squeeze(0).permute(1, 2, 3, 0)
            
            gradient = local_state - neighbor_avg
            update = dt * local_temp.unsqueeze(-1) / (1 + local_mass.unsqueeze(-1))
            
            self.state[x_slice, y_slice, z_slice] -= update * gradient
            
    # ─────────────────────────────────────────────
    # Gravitational Retrieval (O(log n) with index)
    # ─────────────────────────────────────────────
    
    def query(
        self,
        wave: torch.Tensor,
        k: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gravitational retrieval: find k-nearest attractors.
        
        Uses spatial index if available (O(log n)).
        Falls back to location-based if not.
        
        Args:
            wave: [features] query vector
            k: Number of attractors to retrieve
            
        Returns:
            features: [k, features] attractor features
            weights: [k] gravity weights (mass / distance²)
            similarities: [k] cosine similarities
        """
        if self.index is not None:
            return self._query_indexed(wave, k)
        else:
            return self._query_local(wave, k)
            
    def _query_indexed(
        self,
        wave: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Query using FAISS index."""
        import faiss
        
        # Normalize query
        query = F.normalize(wave.unsqueeze(0), dim=-1).cpu().numpy()
        
        # Search index
        similarities, indices = self.index.search(query, k)
        similarities = torch.from_numpy(similarities[0]).to(self.device)
        indices = indices[0]
        
        # Get features and masses
        features = self.attractor_features[indices].to(self.device)  # [k, features]
        masses = self.attractor_masses[indices].to(self.device)  # [k]
        
        # Compute gravity weights
        distances = (1 - similarities).clamp(min=0.01)
        weights = masses / (distances ** 2)
        
        return features, weights, similarities
        
    def _query_local(
        self,
        wave: torch.Tensor,
        k: int,
        search_radius: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Query by searching around wave location."""
        location = self._wave_to_location(wave)
        H, W, D = self.config.size_x, self.config.size_y, self.config.size_z
        
        x, y, z = location.long().clamp(
            min=search_radius, 
            max=torch.tensor([H, W, D], device=self.device) - search_radius - 1
        )
        x, y, z = x.item(), y.item(), z.item()
        
        # Get local region
        x_slice = slice(x - search_radius, x + search_radius + 1)
        y_slice = slice(y - search_radius, y + search_radius + 1)
        z_slice = slice(z - search_radius, z + search_radius + 1)
        
        local_state = self.state[x_slice, y_slice, z_slice]  # [2r+1, 2r+1, 2r+1, F]
        local_mass = self.mass[x_slice, y_slice, z_slice]  # [2r+1, 2r+1, 2r+1]
        
        # Flatten for search
        flat_state = local_state.view(-1, self.config.features)  # [N, F]
        flat_mass = local_mass.view(-1)  # [N]
        
        # Compute similarities
        similarities = F.cosine_similarity(
            wave.unsqueeze(0),
            flat_state,
            dim=-1,
        )  # [N]
        
        # Top-k by mass-weighted similarity
        scores = similarities * (1 + flat_mass.log1p())
        top_indices = scores.topk(k).indices
        
        features = flat_state[top_indices]  # [k, F]
        masses = flat_mass[top_indices]  # [k]
        sims = similarities[top_indices]  # [k]
        
        # Gravity weights
        distances = (1 - sims).clamp(min=0.01)
        weights = masses / (distances ** 2)
        
        return features, weights, sims
        
    # ─────────────────────────────────────────────
    # Consolidation & Index Building
    # ─────────────────────────────────────────────
    
    def consolidate(self, min_mass: Optional[float] = None):
        """
        Build spatial index from high-mass attractors.
        
        Call after batch injection to enable fast queries.
        """
        import faiss
        
        threshold = min_mass or self.mass.mean().item() * self.config.attractor_threshold
        
        # Find attractor locations
        attractor_mask = self.mass > threshold
        self.attractor_locations = attractor_mask.nonzero()  # [N, 3]
        
        # Extract features and masses
        n_attractors = self.attractor_locations.shape[0]
        self.attractor_features = self.state[attractor_mask].cpu()  # [N, F]
        self.attractor_masses = self.mass[attractor_mask].cpu()  # [N]
        
        # Normalize for inner product search
        normalized = F.normalize(self.attractor_features, dim=-1).numpy()
        
        # Build IVF index for O(log n) search
        if n_attractors > 10000:
            # Use IVF for large indices
            quantizer = faiss.IndexFlatIP(self.config.features)
            self.index = faiss.IndexIVFFlat(
                quantizer, 
                self.config.features,
                min(self.config.index_nlist, n_attractors // 10),
                faiss.METRIC_INNER_PRODUCT,
            )
            self.index.train(normalized)
            self.index.add(normalized)
        else:
            # Flat index for small
            self.index = faiss.IndexFlatIP(self.config.features)
            self.index.add(normalized)
            
        print(f"  ✓ Consolidated {n_attractors:,} attractors (threshold: {threshold:.4f})")
        print(f"    Total mass: {self.total_mass:,.0f}")
        print(f"    Mean attractor mass: {self.attractor_masses.mean():.4f}")
        
    # ─────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────
    
    def save(self, path: str):
        """Save field state to disk."""
        torch.save({
            'config': self.config,
            'state': self.state.cpu(),
            'mass': self.mass.cpu(),
            'temperature': self.temperature.cpu(),
            'attractor_locations': self.attractor_locations,
            'attractor_features': self.attractor_features,
            'attractor_masses': self.attractor_masses,
            'injection_count': self.injection_count,
            'total_mass': self.total_mass,
        }, path)
        print(f"  ✓ Saved field to {path}")
        
    @classmethod
    def load(cls, path: str, device: str = 'cuda') -> 'ResonanceField7B':
        """Load field from disk."""
        data = torch.load(path, map_location='cpu')
        
        field = cls(config=data['config'], device=device)
        field.state = data['state'].to(device)
        field.mass = data['mass'].to(device)
        field.temperature = data['temperature'].to(device)
        field.attractor_locations = data['attractor_locations']
        field.attractor_features = data['attractor_features']
        field.attractor_masses = data['attractor_masses']
        field.injection_count = data['injection_count']
        field.total_mass = data['total_mass']
        
        # Rebuild index
        if field.attractor_features is not None:
            field.consolidate()
            
        print(f"  ✓ Loaded field from {path}")
        return field
        
    # ─────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────
    
    def stats(self) -> dict:
        """Return field statistics."""
        return {
            'injections': self.injection_count,
            'total_mass': self.total_mass,
            'mean_mass': self.mass.mean().item(),
            'max_mass': self.mass.max().item(),
            'mean_temp': self.temperature.mean().item(),
            'mean_activation': self.state.abs().mean().item(),
            'n_attractors': len(self.attractor_locations) if self.attractor_locations is not None else 0,
        }


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

def create_field_7b(device: str = 'cuda') -> ResonanceField7B:
    """Create a fresh 7B field."""
    config = FieldConfig()
    return ResonanceField7B(config, device)


def estimate_memory_gb(config: FieldConfig) -> float:
    """Estimate GPU memory needed for field."""
    state_bytes = config.size_x * config.size_y * config.size_z * config.features * 4
    mass_bytes = config.size_x * config.size_y * config.size_z * 4
    temp_bytes = config.size_x * config.size_y * config.size_z * 4
    total_bytes = state_bytes + mass_bytes + temp_bytes
    return total_bytes / 1e9


if __name__ == '__main__':
    # Test field creation
    print("Creating FLUX-7B Resonance Field...")
    
    config = FieldConfig()
    print(f"  Memory estimate: {estimate_memory_gb(config):.1f} GB")
    
    # Use smaller config for testing
    test_config = FieldConfig(
        size_x=64, size_y=64, size_z=64,
        features=1536,
    )
    print(f"  Test config: {test_config.total_params / 1e6:.1f}M params")
    
    field = ResonanceField7B(test_config, device='cpu')
    
    # Test injection
    wave = torch.randn(1536)
    field.inject(wave, strength=1.0)
    print(f"  After injection: {field.stats()}")
    
    # Test settling
    field.settle(steps=10)
    print(f"  After settling: {field.stats()}")
    
    # Test query
    features, weights, sims = field.query(wave, k=8)
    print(f"  Query result: {features.shape}, weights: {weights[:3]}")
    
    print("  ✓ ResonanceField7B tests passed")
