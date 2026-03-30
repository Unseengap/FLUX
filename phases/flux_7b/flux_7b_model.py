"""
FLUX-7B: Complete System

Brings together all components:
- ResonanceField7B (6.5B, knowledge storage, no gradients)
- PhysicsStack (300M, frozen after Phase 7)
- EmissionHead (200M, the ONLY trainable component)

Total: 7B parameters
Trainable: 200M parameters

This is fundamentally different from LLMs:
- Knowledge is INJECTED, not trained
- Only spelling (emission) needs gradients
- Add new knowledge in seconds, not days
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

# Setup paths
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

for _phase in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6', 'phase7', 'phase8']:
    _p = str(_PHASES_DIR / _phase)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from resonance_field_7b import ResonanceField7B, FieldConfig
from emission_head import EmissionHead, EmissionConfig


@dataclass
class FLUX7BConfig:
    """Configuration for complete FLUX-7B system."""
    
    # Field config
    field_size: int = 512
    field_features: int = 1536
    
    # Emission config
    emission_d_model: int = 1024
    emission_layers: int = 12
    emission_heads: int = 16
    
    # Physics stack (loaded from checkpoint)
    wave_dim: int = 432
    
    # Retrieval
    k_attractors: int = 32
    
    def get_field_config(self) -> FieldConfig:
        return FieldConfig(
            size_x=self.field_size,
            size_y=self.field_size,
            size_z=self.field_size,
            features=self.field_features,
        )
        
    def get_emission_config(self) -> EmissionConfig:
        return EmissionConfig(
            field_features=self.field_features,
            wave_dim=self.wave_dim,
            d_model=self.emission_d_model,
            num_layers=self.emission_layers,
            num_heads=self.emission_heads,
            k_attractors=self.k_attractors,
        )


class PhysicsStack(nn.Module):
    """
    Physics components from Phases 1-7.
    Frozen after initial training.
    
    Components:
        CSE (10M): bytes → waves
        wave_to_field (10M): project waves to field space
        field_to_wave (10M): project field features to wave space
        
    Later can add: GR, CGN, TL, Memory
    For now, simplified version.
    """
    
    def __init__(self, wave_dim: int = 432, field_features: int = 1536):
        super().__init__()
        
        self.wave_dim = wave_dim
        self.field_features = field_features
        
        # Wave ↔ Field bridges
        self.wave_to_field = nn.Sequential(
            nn.Linear(wave_dim, field_features),
            nn.GELU(),
            nn.Linear(field_features, field_features),
            nn.LayerNorm(field_features),
        )
        
        self.field_to_wave = nn.Sequential(
            nn.Linear(field_features, wave_dim),
            nn.GELU(),
            nn.Linear(wave_dim, wave_dim),
            nn.LayerNorm(wave_dim),
        )
        
        # CSE will be loaded from checkpoint
        self.cse = None
        
    def load_cse(self, cse_module):
        """Load CSE from checkpoint."""
        self.cse = cse_module
        # Freeze CSE
        for p in self.cse.parameters():
            p.requires_grad = False
            
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to wave sequence."""
        if self.cse is None:
            raise RuntimeError("CSE not loaded. Call load_cse() first.")
        wave = self.cse.encode(text)
        return wave.full  # [seq, wave_dim]
        
    def freeze(self):
        """Freeze all parameters."""
        for p in self.parameters():
            p.requires_grad = False
            
    def project_to_field(self, waves: torch.Tensor) -> torch.Tensor:
        """Project waves to field space for injection."""
        return self.wave_to_field(waves)
        
    def project_to_waves(self, features: torch.Tensor) -> torch.Tensor:
        """Project field features to wave space."""
        return self.field_to_wave(features)


class FLUX7B:
    """
    Complete FLUX-7B System.
    
    NOT an nn.Module because the field doesn't use gradients.
    
    Components:
        field: ResonanceField7B (6.5B, no gradients)
        physics: PhysicsStack (300M, frozen)
        emission: EmissionHead (200M, trainable)
        
    Usage:
        # Create system
        flux = FLUX7B.create()
        
        # Inject knowledge (no gradients!)
        flux.inject_document("Paris is the capital of France...")
        
        # Generate (uses trained emission head)
        output = flux.generate("What is the capital of France?")
    """
    
    def __init__(
        self,
        field: ResonanceField7B,
        physics: PhysicsStack,
        emission: EmissionHead,
        config: FLUX7BConfig,
        device: str = 'cuda',
    ):
        self.field = field
        self.physics = physics
        self.emission = emission
        self.config = config
        self.device = device
        
        # Move emission to device
        self.emission = self.emission.to(device)
        
    @classmethod
    def create(
        cls,
        config: Optional[FLUX7BConfig] = None,
        device: str = 'cuda',
        load_checkpoint: Optional[str] = None,
    ) -> 'FLUX7B':
        """
        Create FLUX-7B system.
        
        Args:
            config: System configuration
            device: Device for tensors
            load_checkpoint: Optional checkpoint to load
        """
        config = config or FLUX7BConfig()
        
        print("Creating FLUX-7B...")
        
        # Field (6.5B)
        field_config = config.get_field_config()
        field = ResonanceField7B(field_config, device)
        
        # Physics stack
        physics = PhysicsStack(config.wave_dim, config.field_features)
        physics.freeze()
        
        # Emission head (200M)
        emission_config = config.get_emission_config()
        emission = EmissionHead(emission_config)
        
        system = cls(field, physics, emission, config, device)
        
        if load_checkpoint:
            system.load(load_checkpoint)
            
        # Print summary
        emission_params = sum(p.numel() for p in emission.parameters())
        physics_params = sum(p.numel() for p in physics.parameters())
        field_params = field.config.total_params
        
        print(f"\n  FLUX-7B Summary:")
        print(f"    Field: {field_params / 1e9:.2f}B params (no gradients)")
        print(f"    Physics: {physics_params / 1e6:.0f}M params (frozen)")
        print(f"    Emission: {emission_params / 1e6:.0f}M params (trainable)")
        print(f"    Total: {(field_params + physics_params + emission_params) / 1e9:.2f}B params")
        print(f"    Trainable: {emission_params / 1e6:.0f}M params")
        
        return system
        
    # ─────────────────────────────────────────────
    # Knowledge Injection (NO GRADIENTS)
    # ─────────────────────────────────────────────
    
    def inject_document(self, text: str, strength: float = 1.0):
        """
        Inject a document into the field.
        
        This is INSTANT knowledge acquisition — no training needed.
        
        Args:
            text: Document text
            strength: Injection strength
        """
        # Encode to waves
        waves = self.physics.encode(text)  # [seq, wave_dim]
        
        # Project to field space
        field_waves = self.physics.project_to_field(waves)  # [seq, field_features]
        
        # Inject each wave
        for wave in field_waves:
            self.field.inject(wave.to(self.device), strength)
            
    def inject_corpus(
        self,
        documents: List[str],
        strength: float = 1.0,
        settle_every: int = 100,
        settle_steps: int = 50,
    ):
        """
        Inject entire corpus into field.
        
        Args:
            documents: List of document texts
            strength: Injection strength
            settle_every: Settle field every N documents
            settle_steps: Settling iterations
        """
        print(f"  Injecting {len(documents)} documents...")
        
        for i, doc in enumerate(documents):
            self.inject_document(doc, strength)
            
            if (i + 1) % settle_every == 0:
                self.field.settle(settle_steps)
                print(f"    {i + 1}/{len(documents)} injected, settled")
                
        # Final settle and consolidate
        self.field.settle(settle_steps * 2)
        self.field.consolidate()
        print(f"  ✓ Injection complete: {self.field.stats()}")
        
    def add_knowledge(self, text: str):
        """
        Add new knowledge to the system. Takes seconds.
        
        Unlike LLMs which need fine-tuning ($$$) or RAG (retrieval only),
        FLUX directly integrates new knowledge into the field.
        """
        self.inject_document(text, strength=1.0)
        self.field.settle(steps=20)
        
    # ─────────────────────────────────────────────
    # Generation
    # ─────────────────────────────────────────────
    
    def generate(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text given a prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum output length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated text
        """
        # Encode prompt to waves
        waves = self.physics.encode(prompt)  # [seq, wave_dim]
        waves = waves.to(self.device)
        
        # Query field for relevant knowledge
        query_wave = self.physics.project_to_field(waves.mean(dim=0, keepdim=True).squeeze(0))
        field_features, gravity_weights, _ = self.field.query(
            query_wave.to(self.device), 
            k=self.config.k_attractors
        )
        
        # Generate via emission head
        output_bytes = self.emission.generate(
            field_features.unsqueeze(0),
            gravity_weights.unsqueeze(0),
            waves.unsqueeze(0),
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
        )
        
        return output_bytes.decode('utf-8', errors='replace')
        
    def forward_train(
        self,
        prompt_waves: torch.Tensor,
        field_features: torch.Tensor,
        gravity_weights: torch.Tensor,
        target_bytes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training forward pass (for emission head only).
        
        Args:
            prompt_waves: [batch, seq, wave_dim] encoded prompt
            field_features: [batch, k, field_features] from field.query()
            gravity_weights: [batch, k] gravity weights
            target_bytes: [batch, tgt_len] target byte sequence
            
        Returns:
            [batch, tgt_len, 256] logits
        """
        return self.emission(field_features, gravity_weights, prompt_waves, target_bytes)
        
    # ─────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────
    
    def save(self, path: str):
        """Save complete system."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save field
        self.field.save(str(save_dir / 'field.pt'))
        
        # Save physics stack
        torch.save(self.physics.state_dict(), str(save_dir / 'physics.pt'))
        
        # Save emission head
        torch.save(self.emission.state_dict(), str(save_dir / 'emission.pt'))
        
        # Save config
        torch.save(self.config, str(save_dir / 'config.pt'))
        
        print(f"  ✓ Saved FLUX-7B to {path}")
        
    def load(self, path: str):
        """Load complete system."""
        load_dir = Path(path)
        
        # Load field
        self.field = ResonanceField7B.load(str(load_dir / 'field.pt'), self.device)
        
        # Load physics
        self.physics.load_state_dict(torch.load(str(load_dir / 'physics.pt')))
        
        # Load emission
        self.emission.load_state_dict(torch.load(str(load_dir / 'emission.pt')))
        self.emission = self.emission.to(self.device)
        
        print(f"  ✓ Loaded FLUX-7B from {path}")
        
    # ─────────────────────────────────────────────
    # Training Utilities
    # ─────────────────────────────────────────────
    
    def get_trainable_parameters(self):
        """Get only trainable parameters (emission head)."""
        return self.emission.parameters()
        
    def train_mode(self):
        """Set to training mode."""
        self.emission.train()
        
    def eval_mode(self):
        """Set to evaluation mode."""
        self.emission.eval()


# ─────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────

def create_flux_7b_full(device: str = 'cuda') -> FLUX7B:
    """Create full 7B system (requires ~28GB VRAM)."""
    config = FLUX7BConfig(
        field_size=512,
        field_features=1536,
        emission_d_model=1024,
        emission_layers=12,
    )
    return FLUX7B.create(config, device)


def create_flux_7b_test(device: str = 'cuda') -> FLUX7B:
    """Create smaller test system (~1GB VRAM)."""
    config = FLUX7BConfig(
        field_size=64,  # 64³ instead of 512³
        field_features=512,
        emission_d_model=256,
        emission_layers=2,
    )
    return FLUX7B.create(config, device)


if __name__ == '__main__':
    print("Testing FLUX-7B System...")
    
    # Create test system
    system = create_flux_7b_test(device='cpu')
    
    # Test injection
    system.inject_document("The capital of France is Paris.")
    system.inject_document("Python is a programming language.")
    system.field.settle(steps=10)
    
    print(f"\n  Field stats: {system.field.stats()}")
    
    # Test query
    waves = torch.randn(10, 432)  # Fake waves
    system.physics.wave_to_field = system.physics.wave_to_field.to('cpu')
    query = system.physics.project_to_field(waves.mean(dim=0))
    features, weights, sims = system.field.query(query, k=8)
    print(f"  Query result: {features.shape}, top weights: {weights[:3]}")
    
    # Test emission forward
    batch = 1
    field_features = torch.randn(batch, 32, 512)
    gravity_weights = torch.rand(batch, 32)
    wave_context = torch.randn(batch, 64, 432)
    target_bytes = torch.randint(0, 256, (batch, 50))
    
    logits = system.emission(field_features, gravity_weights, wave_context, target_bytes)
    print(f"  Emission forward: {logits.shape}")
    
    print("\n  ✓ FLUX-7B tests passed")
