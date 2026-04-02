"""
Phase 8.8: Grid ↔ Wave Adapters

The key to ARC Prize. Encodes 2D grids (up to 30×30, 10 colors)
into FLUX's wave space, enabling spatial reasoning.

Two encoding modes:
1. Holistic: Entire grid → single [432] vector (for rule matching)
2. Spatial: Grid → [H×W, 432] per-cell waves (for detailed reasoning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Union

try:
    from phases.phase8_8.wave_to_x import XToWave, WaveToX, register_input_adapter, register_output_adapter
except ImportError:
    from wave_to_x import XToWave, WaveToX, register_input_adapter, register_output_adapter


@register_input_adapter('grid')
class GridToWave(XToWave):
    """
    Grid → Wave adapter.
    
    Encodes ARC-style grids (H×W matrix with integer colors 0-9)
    into FLUX's 432-dimensional wave space.
    
    Architecture:
        1. Color embedding: each color → learned 32-dim vector
        2. Position encoding: (x, y) → 64-dim spatial signal
        3. Local convolution: capture 3×3 neighborhood patterns
        4. Global pooling: compress to single wave (optional)
    
    The encoding captures:
        - What colors are present
        - Where they are located
        - What patterns they form (lines, blocks, symmetry)
    """
    
    def __init__(
        self,
        wave_dim: int = 432,
        num_colors: int = 10,
        max_size: int = 30,
        color_dim: int = 32,
        pattern_dim: int = 64,
        device: str = 'cpu',
    ):
        super().__init__('grid', wave_dim)
        self.num_colors = num_colors
        self.max_size = max_size
        self.color_dim = color_dim
        self.pattern_dim = pattern_dim
        self.device = device
        
        # Color embedding (0-9 → 32-dim)
        self.color_embed = nn.Embedding(num_colors + 1, color_dim)  # +1 for padding
        
        # Position encoding (learnable, max 30×30)
        self.pos_embed_h = nn.Embedding(max_size, pattern_dim // 2)
        self.pos_embed_w = nn.Embedding(max_size, pattern_dim // 2)
        
        # Local pattern convolutions
        self.conv1 = nn.Conv2d(color_dim, pattern_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(pattern_dim, pattern_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(pattern_dim, pattern_dim, 3, padding=1)
        
        # Global pooling → wave projection
        # color_dim + pattern_dim (position) + pattern_dim (conv) = feature_dim
        feature_dim = color_dim + pattern_dim + pattern_dim
        self.to_wave = nn.Linear(feature_dim, wave_dim)
        
        # For holistic mode: compress entire grid to one vector
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(pattern_dim, wave_dim),
        )
        
        self.to(device)
    
    def encode(
        self,
        grid: Union[Tensor, list],
        mode: str = 'holistic',
    ) -> Tensor:
        """
        Encode grid to wave space.
        
        Args:
            grid: [H, W] integer tensor (values 0-9) or list of lists
            mode: 'holistic' (single vector) or 'spatial' (per-cell)
        
        Returns:
            'holistic': [432] wave tensor
            'spatial': [H*W, 432] wave tensor
        """
        # Convert to tensor if needed
        if isinstance(grid, list):
            # Handle nested list structures
            if len(grid) == 0:
                grid = [[0]]
            if not isinstance(grid[0], list):
                grid = [grid]  # Wrap single row
            grid = torch.tensor(grid, dtype=torch.long, device=self.device)
        else:
            grid = grid.to(self.device)
        
        # Ensure 2D
        if grid.dim() == 1:
            grid = grid.unsqueeze(0)
        elif grid.dim() > 2:
            # Flatten extra dimensions
            grid = grid.view(-1, grid.shape[-1])
        
        H, W = grid.shape
        
        # Color embedding: [H, W] → [H, W, color_dim]
        color_feats = self.color_embed(grid)
        
        # Position encoding
        h_pos = self.pos_embed_h(torch.arange(H, device=self.device))  # [H, 32]
        w_pos = self.pos_embed_w(torch.arange(W, device=self.device))  # [W, 32]
        
        # Expand to grid: [H, W, 64]
        pos_feats = torch.cat([
            h_pos.unsqueeze(1).expand(-1, W, -1),
            w_pos.unsqueeze(0).expand(H, -1, -1),
        ], dim=-1)
        
        # Convolution for local patterns
        # [H, W, color_dim] → [1, color_dim, H, W]
        x = color_feats.permute(2, 0, 1).unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # → [1, pattern_dim, H, W] → [H, W, pattern_dim]
        pattern_feats = x.squeeze(0).permute(1, 2, 0)
        
        if mode == 'holistic':
            # Global pooling over entire grid
            global_feat = self.global_pool(x)  # [1, wave_dim]
            return global_feat.squeeze(0)
        
        else:  # spatial
            # Concatenate features per cell
            cell_feats = torch.cat([color_feats, pos_feats, pattern_feats], dim=-1)
            # [H, W, feature_dim] → [H*W, feature_dim]
            cell_feats = cell_feats.view(H * W, -1)
            # Project to wave space
            waves = self.to_wave(cell_feats)  # [H*W, 432]
            return waves
    
    def encode_pair(
        self,
        input_grid: Union[Tensor, list],
        output_grid: Union[Tensor, list],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Encode input/output pair and compute delta.
        
        This is the key for ARC: the delta represents the transformation rule.
        
        Returns:
            (input_wave, output_wave, delta_wave)
        """
        input_wave = self.encode(input_grid, mode='holistic')
        output_wave = self.encode(output_grid, mode='holistic')
        delta_wave = output_wave - input_wave
        return input_wave, output_wave, delta_wave


@register_output_adapter('grid')
class WaveToGrid(WaveToX):
    """
    Wave → Grid adapter.
    
    Decodes FLUX waves back to ARC-style grids.
    
    Two modes:
    1. From delta: Apply transformation wave to input grid
    2. Direct: Decode wave directly to grid (for generation)
    
    Architecture:
        wave [432] → hidden [256] → [H, W, num_colors] logits → argmax
    """
    
    def __init__(
        self,
        wave_dim: int = 432,
        num_colors: int = 10,
        max_size: int = 30,
        hidden_dim: int = 256,
        device: str = 'cpu',
    ):
        super().__init__('grid', wave_dim)
        self.num_colors = num_colors
        self.max_size = max_size
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Wave → spatial features
        self.wave_proj = nn.Linear(wave_dim, hidden_dim)
        
        # Size predictor (predicts H, W)
        self.size_head = nn.Sequential(
            nn.Linear(wave_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [H, W] as continuous, then round
        )
        
        # Spatial decoder
        self.spatial_expand = nn.Linear(hidden_dim, max_size * max_size * hidden_dim // 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim // 2, num_colors, 3, padding=1),
        )
        
        # For transform mode: combine input + delta
        self.transform_combine = nn.Sequential(
            nn.Linear(wave_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, wave_dim),
        )
        
        self.to(device)
    
    def decode(
        self,
        wave: Tensor,
        grid_size: Optional[Tuple[int, int]] = None,
        input_grid: Optional[Tensor] = None,
        temperature: float = 0.0,
    ) -> Tensor:
        """
        Decode wave to grid.
        
        Args:
            wave: [432] wave tensor (the transformation or direct encoding)
            grid_size: (H, W) if known, else predicted
            input_grid: For transform mode, the input to transform
            temperature: >0 for sampling, 0 for argmax
        
        Returns:
            [H, W] integer tensor (values 0-9)
        """
        wave = wave.to(self.device)
        if wave.dim() == 1:
            wave = wave.unsqueeze(0)
        
        # If input_grid provided, this is transform mode
        if input_grid is not None:
            # Encode input and combine with delta wave
            encoder = GridToWave(device=self.device)
            input_wave = encoder.encode(input_grid, mode='holistic')
            
            combined = torch.cat([input_wave.unsqueeze(0), wave], dim=-1)
            wave = self.transform_combine(combined)
        
        # Predict or use given size
        if grid_size is None:
            size_pred = self.size_head(wave)
            H = int(torch.clamp(size_pred[0, 0], 1, self.max_size).item())
            W = int(torch.clamp(size_pred[0, 1], 1, self.max_size).item())
        else:
            H, W = grid_size
        
        # Decode to grid
        hidden = self.wave_proj(wave)  # [1, hidden]
        
        # Expand to spatial
        spatial = self.spatial_expand(hidden)  # [1, max*max*hidden//4]
        spatial = spatial.view(1, self.hidden_dim // 4, self.max_size, self.max_size)
        
        # Decode
        logits = self.decoder(spatial)  # [1, num_colors, max, max]
        
        # Crop to actual size
        logits = logits[:, :, :H, :W]  # [1, num_colors, H, W]
        
        # Sample or argmax
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=1)
            # Sample per cell
            probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, self.num_colors)
            samples = torch.multinomial(probs_flat, 1).squeeze(-1)
            grid = samples.view(H, W)
        else:
            grid = logits.argmax(dim=1).squeeze(0)  # [H, W]
        
        return grid
    
    def apply_transform(
        self,
        delta_wave: Tensor,
        input_grid: Union[Tensor, list],
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """
        Apply a transformation wave to an input grid.
        
        This is the key ARC operation:
            output = apply(delta, input)
        
        Where delta was extracted from training examples.
        """
        return self.decode(
            wave=delta_wave,
            grid_size=grid_size,
            input_grid=input_grid,
            temperature=0.0,
        )


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Grid Adapters — Module Check")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test GridToWave
    encoder = GridToWave(device=device)
    test_grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    
    wave_holistic = encoder.encode(test_grid, mode='holistic')
    print(f"  GridToWave holistic: {wave_holistic.shape}")  # [432]
    
    wave_spatial = encoder.encode(test_grid, mode='spatial')
    print(f"  GridToWave spatial: {wave_spatial.shape}")  # [9, 432]
    
    # Test WaveToGrid
    decoder = WaveToGrid(device=device)
    grid_out = decoder.decode(wave_holistic, grid_size=(3, 3))
    print(f"  WaveToGrid: {grid_out.shape}")  # [3, 3]
    
    print("  ✓ Grid adapters ready")
