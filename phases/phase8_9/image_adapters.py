"""
Phase 8.9: Image ↔ Wave Adapters

ImageToWave: Encode images into FLUX wave space using patch-based encoding.
WaveToImage_Universal: Decode waves to images using 3 physics-based renderers.

The 3 physics engines:
1. Gravity Renderer: Mass attractors pull colors (smooth gradients)
2. Interference Renderer: Wave superposition (ripples, patterns)
3. Thermodynamic Renderer: Energy minimization (organic textures)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, Union
from dataclasses import dataclass
import math
import sys
from pathlib import Path

# Add project paths
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PHASES_DIR / 'phase8_8') not in sys.path:
    sys.path.insert(0, str(_PHASES_DIR / 'phase8_8'))

from wave_to_x import XToWave, WaveToX, register_input_adapter, register_output_adapter


# ─────────────────────────────────────────────
# Style Presets
# ─────────────────────────────────────────────

@dataclass
class StylePreset:
    """Blending weights for 3 physics engines."""
    name: str
    gravity: float
    interference: float
    thermodynamic: float
    
    def weights(self) -> Tuple[float, float, float]:
        return (self.gravity, self.interference, self.thermodynamic)


STYLE_PRESETS = {
    'photorealistic': StylePreset('photorealistic', 0.7, 0.2, 0.1),
    'abstract': StylePreset('abstract', 0.2, 0.6, 0.2),
    'crystalline': StylePreset('crystalline', 0.1, 0.3, 0.6),
    'organic': StylePreset('organic', 0.4, 0.1, 0.5),
    'dream': StylePreset('dream', 0.33, 0.33, 0.34),
}


# ─────────────────────────────────────────────
# Image → Wave
# ─────────────────────────────────────────────

@register_input_adapter('image')
class ImageToWave(XToWave):
    """
    Image → Wave adapter.
    
    Uses patch-based encoding:
    1. Split image into patches (e.g., 16×16)
    2. Embed each patch to wave dimension
    3. Add positional encoding
    4. Output: [num_patches, 432] waves
    """
    
    def __init__(
        self,
        wave_dim: int = 432,
        patch_size: int = 16,
        max_patches: int = 256,  # 16×16 grid = 256×256 image
        channels: int = 3,
        device: str = 'cpu',
    ):
        super().__init__('image', wave_dim)
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.channels = channels
        self.device = device
        
        # Patch embedding: flatten patch → wave
        patch_dim = patch_size * patch_size * channels
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, wave_dim * 2),
            nn.GELU(),
            nn.Linear(wave_dim * 2, wave_dim),
        )
        
        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(max_patches, wave_dim) * 0.02)
        
        # Global pooling for single-vector mode
        self.global_pool = nn.Sequential(
            nn.Linear(wave_dim, wave_dim),
            nn.Tanh(),
        )
        
        self.to(device)
    
    def encode(
        self,
        image: Tensor,
        mode: str = 'patches',
    ) -> Tensor:
        """
        Encode image to wave space.
        
        Args:
            image: [H, W, 3] or [3, H, W] or [B, 3, H, W] tensor (values 0-1)
            mode: 'patches' → [num_patches, 432]
                  'pooled' → [432] single vector
        
        Returns:
            Wave tensor
        """
        # Normalize input shape to [B, C, H, W]
        if image.dim() == 3:
            if image.shape[-1] == 3:  # [H, W, 3]
                image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)  # [1, C, H, W]
        
        image = image.to(self.device)
        B, C, H, W = image.shape
        
        # Calculate patch grid
        pH = H // self.patch_size
        pW = W // self.patch_size
        num_patches = pH * pW
        
        # Extract patches: [B, C, pH, patch_size, pW, patch_size]
        patches = image.unfold(2, self.patch_size, self.patch_size)
        patches = patches.unfold(3, self.patch_size, self.patch_size)
        # → [B, C, pH, pW, patch_size, patch_size]
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        # → [B, pH, pW, C, patch_size, patch_size]
        patches = patches.reshape(B, num_patches, -1)
        # → [B, num_patches, patch_dim]
        
        # Embed patches
        waves = self.patch_embed(patches)  # [B, num_patches, wave_dim]
        
        # Add positional encoding
        waves = waves + self.pos_embed[:num_patches]
        
        # Remove batch dim if single image
        if B == 1:
            waves = waves.squeeze(0)  # [num_patches, wave_dim]
        
        if mode == 'pooled':
            pooled = waves.mean(dim=-2)  # Mean over patches
            return self.global_pool(pooled)
        
        return waves
    
    def forward(self, image: Tensor, mode: str = 'patches') -> Tensor:
        return self.encode(image, mode)


# ─────────────────────────────────────────────
# Wave → Image (3 Physics Engines)
# ─────────────────────────────────────────────

class GravityRenderer(nn.Module):
    """
    Physics Engine 1: Gravity-based rendering.
    
    Concept: Wave features define "mass attractors" in image space.
    Colors flow toward attractors based on gravitational pull.
    Result: Smooth gradients with focal points.
    """
    
    def __init__(self, wave_dim: int = 432, num_attractors: int = 16):
        super().__init__()
        self.num_attractors = num_attractors
        
        # Wave → attractor positions and masses
        self.to_positions = nn.Linear(wave_dim, num_attractors * 2)  # x, y
        self.to_masses = nn.Linear(wave_dim, num_attractors)
        self.to_colors = nn.Linear(wave_dim, num_attractors * 3)  # RGB per attractor
        
        # Background color
        self.to_background = nn.Linear(wave_dim, 3)
    
    def forward(self, wave: Tensor, H: int, W: int) -> Tensor:
        """
        Render image using gravity simulation.
        
        Args:
            wave: [432] wave tensor
            H, W: Output image size
        
        Returns:
            [H, W, 3] image tensor (0-1 range)
        """
        device = wave.device
        
        # Get attractor properties
        positions = torch.sigmoid(self.to_positions(wave))  # [num_attractors * 2]
        positions = positions.view(self.num_attractors, 2)  # [N, 2] in [0, 1]
        
        masses = F.softplus(self.to_masses(wave))  # [num_attractors]
        
        colors = torch.sigmoid(self.to_colors(wave))  # [num_attractors * 3]
        colors = colors.view(self.num_attractors, 3)  # [N, 3]
        
        background = torch.sigmoid(self.to_background(wave))  # [3]
        
        # Create coordinate grid
        y_coords = torch.linspace(0, 1, H, device=device)
        x_coords = torch.linspace(0, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        
        # Calculate gravitational influence at each pixel
        image = background.view(1, 1, 3).expand(H, W, 3).clone()
        
        for i in range(self.num_attractors):
            pos = positions[i]  # [2]
            mass = masses[i]
            color = colors[i]  # [3]
            
            # Distance from attractor
            dist = ((grid - pos) ** 2).sum(dim=-1).sqrt()  # [H, W]
            
            # Gravitational pull (inverse square, clamped)
            pull = mass / (dist + 0.1) ** 2  # [H, W]
            pull = pull / (pull.max() + 1e-6)  # Normalize
            
            # Blend color based on pull
            image = image + pull.unsqueeze(-1) * color.view(1, 1, 3)
        
        # Normalize to [0, 1]
        image = image / (image.max() + 1e-6)
        return image.clamp(0, 1)


class InterferenceRenderer(nn.Module):
    """
    Physics Engine 2: Wave interference rendering.
    
    Concept: Wave features define wave sources in image space.
    Multiple waves interfere constructively/destructively.
    Result: Ripple patterns, moiré effects.
    """
    
    def __init__(self, wave_dim: int = 432, num_sources: int = 8):
        super().__init__()
        self.num_sources = num_sources
        
        # Wave → source properties
        self.to_positions = nn.Linear(wave_dim, num_sources * 2)
        self.to_frequencies = nn.Linear(wave_dim, num_sources)
        self.to_phases = nn.Linear(wave_dim, num_sources)
        self.to_amplitudes = nn.Linear(wave_dim, num_sources * 3)  # RGB
    
    def forward(self, wave: Tensor, H: int, W: int) -> Tensor:
        """
        Render image using wave interference.
        
        Args:
            wave: [432] wave tensor
            H, W: Output image size
        
        Returns:
            [H, W, 3] image tensor (0-1 range)
        """
        device = wave.device
        
        # Get source properties
        positions = torch.sigmoid(self.to_positions(wave)).view(self.num_sources, 2)
        frequencies = F.softplus(self.to_frequencies(wave)) * 20 + 1  # [1, 21]
        phases = self.to_phases(wave) * 2 * math.pi  # [0, 2π]
        amplitudes = torch.sigmoid(self.to_amplitudes(wave)).view(self.num_sources, 3)
        
        # Create coordinate grid
        y_coords = torch.linspace(0, 1, H, device=device)
        x_coords = torch.linspace(0, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        
        # Superpose waves from all sources
        image = torch.zeros(H, W, 3, device=device)
        
        for i in range(self.num_sources):
            pos = positions[i]
            freq = frequencies[i]
            phase = phases[i]
            amp = amplitudes[i]
            
            # Distance from source
            dist = ((grid - pos) ** 2).sum(dim=-1).sqrt()  # [H, W]
            
            # Circular wave: cos(freq * dist + phase)
            wave_val = torch.cos(freq * dist * 2 * math.pi + phase)  # [-1, 1]
            wave_val = (wave_val + 1) / 2  # [0, 1]
            
            # Add colored wave
            image = image + wave_val.unsqueeze(-1) * amp.view(1, 1, 3)
        
        # Normalize
        image = image / (image.max() + 1e-6)
        return image.clamp(0, 1)


class ThermodynamicRenderer(nn.Module):
    """
    Physics Engine 3: Thermodynamic rendering.
    
    Concept: Image starts as high-energy noise, settles to equilibrium.
    Wave features define the energy landscape.
    Result: Organic textures, crystalline structures.
    """
    
    def __init__(self, wave_dim: int = 432, settling_steps: int = 10):
        super().__init__()
        self.settling_steps = settling_steps
        
        # Wave → initial energy field
        self.to_energy = nn.Linear(wave_dim, 64)  # Compressed field
        self.energy_upsample = nn.Sequential(
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Linear(256, 3),  # RGB
        )
        
        # Temperature schedule
        self.to_temperature = nn.Linear(wave_dim, 1)
    
    def forward(self, wave: Tensor, H: int, W: int) -> Tensor:
        """
        Render image using thermodynamic settling.
        
        Args:
            wave: [432] wave tensor
            H, W: Output image size
        
        Returns:
            [H, W, 3] image tensor (0-1 range)
        """
        device = wave.device
        
        # Get initial energy field
        energy = self.to_energy(wave)  # [64]
        temp = torch.sigmoid(self.to_temperature(wave))  # [1]
        
        # Initialize with structured noise based on energy
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Create initial image influenced by energy
        # Use energy as coefficients for Fourier-like basis
        image = torch.zeros(H, W, 3, device=device)
        
        for i in range(min(16, len(energy) // 4)):
            freq_x = (i % 4 + 1)
            freq_y = (i // 4 + 1)
            
            basis = torch.sin(freq_x * grid_x * math.pi) * torch.sin(freq_y * grid_y * math.pi)
            
            # Map energy to RGB
            r_coeff = energy[i * 4].item() if i * 4 < len(energy) else 0
            g_coeff = energy[i * 4 + 1].item() if i * 4 + 1 < len(energy) else 0
            b_coeff = energy[i * 4 + 2].item() if i * 4 + 2 < len(energy) else 0
            
            image[:, :, 0] += basis * r_coeff
            image[:, :, 1] += basis * g_coeff
            image[:, :, 2] += basis * b_coeff
        
        # Thermodynamic settling: blur + sharpen iterations
        for step in range(self.settling_steps):
            # Cooling: reduce noise
            cooling = 1 - (step / self.settling_steps) * (1 - temp)
            
            # Simple blur (average pooling simulation)
            image_padded = F.pad(image.permute(2, 0, 1).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
            kernel = torch.ones(1, 1, 3, 3, device=device) / 9
            blurred = []
            for c in range(3):
                blurred.append(F.conv2d(image_padded[:, c:c+1], kernel, padding=0))
            image_blurred = torch.cat(blurred, dim=1).squeeze(0).permute(1, 2, 0)
            
            # Blend based on temperature
            image = cooling * image + (1 - cooling) * image_blurred
        
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        return image.clamp(0, 1)


@register_output_adapter('image')
class WaveToImage_Universal(WaveToX):
    """
    Wave → Image adapter with 3 physics-based renderers.
    
    Blends outputs from:
    - GravityRenderer: Smooth gradients
    - InterferenceRenderer: Wave patterns
    - ThermodynamicRenderer: Organic textures
    
    Style presets control the blend.
    """
    
    def __init__(
        self,
        wave_dim: int = 432,
        default_size: Tuple[int, int] = (256, 256),
        device: str = 'cpu',
    ):
        super().__init__('image', wave_dim)
        self.default_size = default_size
        self.device = device
        
        # Three physics renderers
        self.gravity = GravityRenderer(wave_dim)
        self.interference = InterferenceRenderer(wave_dim)
        self.thermodynamic = ThermodynamicRenderer(wave_dim)
        
        # Optional: learned blending from wave
        self.auto_blend = nn.Sequential(
            nn.Linear(wave_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1),
        )
        
        self.to(device)
    
    def decode(
        self,
        wave: Tensor,
        size: Optional[Tuple[int, int]] = None,
        style: str = 'dream',
        custom_weights: Optional[Tuple[float, float, float]] = None,
        auto_blend: bool = False,
    ) -> Tensor:
        """
        Decode wave to image.
        
        Args:
            wave: [432] wave tensor
            size: (H, W) output size, defaults to (256, 256)
            style: Preset name ('photorealistic', 'abstract', etc.)
            custom_weights: (gravity, interference, thermo) weights, sum to 1
            auto_blend: If True, derive blend from wave content
        
        Returns:
            [H, W, 3] image tensor (0-1 range)
        """
        wave = wave.to(self.device)
        
        # Handle sequence input: pool to single vector
        if wave.dim() == 2:
            wave = wave.mean(dim=0)
        
        H, W = size or self.default_size
        
        # Get blend weights
        if auto_blend:
            weights = self.auto_blend(wave)  # [3]
            w_g, w_i, w_t = weights[0], weights[1], weights[2]
        elif custom_weights:
            w_g, w_i, w_t = custom_weights
        else:
            preset = STYLE_PRESETS.get(style, STYLE_PRESETS['dream'])
            w_g, w_i, w_t = preset.weights()
        
        # Render with each engine
        img_gravity = self.gravity(wave, H, W)
        img_interference = self.interference(wave, H, W)
        img_thermo = self.thermodynamic(wave, H, W)
        
        # Blend
        image = w_g * img_gravity + w_i * img_interference + w_t * img_thermo
        
        return image.clamp(0, 1)
    
    def forward(self, wave: Tensor, **kwargs) -> Tensor:
        return self.decode(wave, **kwargs)
    
    def render_all_styles(
        self,
        wave: Tensor,
        size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Tensor]:
        """Render image with all style presets."""
        results = {}
        for style_name in STYLE_PRESETS:
            results[style_name] = self.decode(wave, size=size, style=style_name)
        return results


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

def image_to_wave(
    image: Tensor,
    mode: str = 'patches',
    device: str = 'cpu',
) -> Tensor:
    """Quick function to encode image."""
    adapter = ImageToWave(device=device)
    return adapter.encode(image, mode=mode)


def wave_to_image(
    wave: Tensor,
    size: Tuple[int, int] = (256, 256),
    style: str = 'dream',
    device: str = 'cpu',
) -> Tensor:
    """Quick function to decode wave to image."""
    adapter = WaveToImage_Universal(device=device)
    return adapter.decode(wave, size=size, style=style)
