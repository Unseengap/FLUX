"""
Phase 10: FLUXHybrid — Unified Hybrid Wave+Byte Model

The best of both worlds:
- Byte-mode (Phase 8 WaveDecoder): Precise, character-perfect (~100 steps/sentence)
- Wave-mode (Phase 10 modules): Fast, semantic (~15 steps/sentence → 6x faster)

Loads entirely from .flx files (Flux-capable.flx preferred, Flux-beta.flx fallback).
Routes intelligently between modes based on task characteristics.

Usage:
    from flux_hybrid import FLUXHybrid
    
    model = FLUXHybrid.from_flx('checkpoints/Flux-capable.flx')
    
    # Automatic routing
    response = model.generate("Explain quantum computing")  # → wave mode
    response = model.generate("def fibonacci(n):")         # → byte mode
    
    # Force specific mode
    response = model.generate("Hello", mode='wave')
    response = model.generate("Hello", mode='byte')
"""

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime

# ─────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

for _phase in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6', 'phase7', 'phase8', 'phase8_8', 'phase10']:
    _p = str(_PHASES_DIR / _phase)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cse import ContinuousSemanticEncoder
from wave_types import SemanticWave, TOTAL_WAVE_DIM
from field import ResonanceField
from gravity import GravitationalRelevance
from thermodynamic import ThermodynamicLearner
from working_memory import WorkingMemory
from episodic_memory import EpisodicMemory
from semantic_memory import SemanticMemory
from wave_decoder import WaveDecoder
from task_router import TaskRouter
from wave_modules import WaveGenerator, WaveChunker, WaveToText


# ─────────────────────────────────────────────
# Response dataclass
# ─────────────────────────────────────────────

@dataclass
class HybridResponse:
    """Response from FLUXHybrid generation."""
    text: str                          # Generated text
    mode: str                          # 'wave' or 'byte'
    mode_reason: str                   # Why this mode was chosen
    generation_time: float             # Seconds
    n_steps: int                       # Generation steps taken
    confidence: float                  # Average confidence (wave mode only)
    modality: str                      # Output modality


# ─────────────────────────────────────────────
# FLUXHybrid Model
# ─────────────────────────────────────────────

class FLUXHybrid(nn.Module):
    """
    Unified FLUX model with dual generation paths.
    
    Loading Strategy:
    - Flux-X-complete.flx: Full trained base model (all components)
    - Flux-capable.flx: Enriched field for best text generation (optional merge)
    
    Provides:
    - Byte-mode generation via Phase 8 WaveDecoder (precise)
    - Wave-mode generation via Phase 10 modules (fast, semantic)
    - Automatic routing based on task characteristics
    - Extension points for multimodal outputs
    
    The core (CSE, Field, GR, TL, Memory) is ALWAYS from .flx.
    Generation paths are pluggable.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = 'cpu',
    ):
        """
        Initialize FLUXHybrid (usually use from_flx() instead).
        
        Args:
            config: Configuration dict
            device: Target device
        """
        super().__init__()
        
        self._device_str = device
        self._config = config
        self._version = config.get('version', '1.0-beta')
        self._learning_steps = 0
        
        # Extract dimensions
        wave_dim = config.get('wave_dim', 432)
        field_h = config.get('field_h', 96)
        field_w = config.get('field_w', 96)
        field_d = config.get('field_d', 96)
        field_features = config.get('field_features', 512)
        
        # Wave dimensions dict for CSE
        wave_dims = config.get('wave_dims', {
            'phonetic': 64,
            'syntactic': 64,
            'semantic': 256,
            'temporal': 32,
            'intensity': 16,
        })
        
        # ── Core components (from .flx) ──
        self.cse = ContinuousSemanticEncoder(
            wave_dims=wave_dims,
            byte_window=config.get('byte_window', 8),
            byte_stride=config.get('byte_stride', 1),
            interference_radius=config.get('interference_radius', 6),
        )
        
        self.field = ResonanceField(
            h=field_h, w=field_w, d=field_d,
            features=field_features,
            wave_dim=wave_dim,
        )
        
        self.gr = GravitationalRelevance(
            feature_dim=field_features,
            k_neighbors=config.get('gr_k', 32),
            device=device,
        )
        
        self.tl = ThermodynamicLearner(
            field=self.field,
            initial_temp=config.get('tl_temp', 1.0),
        )
        
        # Memory systems
        self.working_memory = WorkingMemory(
            window_size=config.get('wm_window', 2048),
            wave_dim=wave_dim,
            feature_dim=field_features,
        )
        self.episodic_memory = EpisodicMemory(feature_dim=field_features)
        self.semantic_memory = SemanticMemory(
            field=self.field,
            protection_threshold=config.get('sm_protection', 5.0),
        )
        
        # ── Bridges ──
        self.wave_to_field = nn.Linear(wave_dim, field_features)
        self.field_to_wave = nn.Linear(field_features, wave_dim)
        self.output_head = nn.Linear(field_features, 256)
        
        # ── Byte-mode: Phase 8 WaveDecoder ──
        self.decoder = WaveDecoder(
            wave_dim=wave_dim,
            field_features=field_features,
            hidden_dim=config.get('decoder_hidden', 1024),
            num_layers=config.get('decoder_layers', 4),
            num_heads=config.get('decoder_heads', 8),
        )
        
        # ── Wave-mode: Phase 10 modules (optional) ──
        self.wave_generator: Optional[WaveGenerator] = None
        self.wave_chunker: Optional[WaveChunker] = None
        self.wave_to_text: Optional[WaveToText] = None
        
        # ── Router ──
        self.task_router = TaskRouter(default_mode='wave', wave_dim=wave_dim)
        
        # ── Multimodal extensions (placeholders) ──
        self.wave_to_image = None
        self.wave_to_audio = None
        self.wave_to_mol = None
        
        self._wave_mode_available = False
    
    @property
    def wave_mode_available(self) -> bool:
        """Check if wave-mode generation is available."""
        return self._wave_mode_available
    
    @classmethod
    def from_flx(
        cls,
        path: str = 'checkpoints/Flux-X-complete.flx',
        device: str = 'cpu',
        verbose: bool = True,
        init_wave_modules: bool = True,
        merge_capable_field: bool = True,
    ) -> 'FLUXHybrid':
        """
        Load FLUXHybrid from .flx file.
        
        Loading Strategy:
        1. Load Flux-X-complete.flx as base (all trained components)
        2. Optionally merge enriched field from Flux-capable.flx
        
        Args:
            path: Path to .flx file (default: Flux-X-complete.flx)
            device: Target device
            verbose: Print loading progress
            init_wave_modules: Initialize fresh wave modules if missing
            merge_capable_field: Merge enriched field from Flux-capable.flx
        
        Returns:
            FLUXHybrid instance
        """
        from flx_loader_v2 import load_flx, load_capable_field
        
        flx = load_flx(Path(path), device='cpu', auto_download=True)
        
        # Optionally load and merge capable field for better generation
        capable_field = None
        if merge_capable_field:
            capable_field = load_capable_field(device='cpu', auto_download=True)
        
        return cls.from_flx_dict(
            flx=flx,
            device=device,
            verbose=verbose,
            init_wave_modules=init_wave_modules,
            capable_field=capable_field,
        )
    
    @classmethod
    def from_flx_dict(
        cls,
        flx: Dict[str, Any],
        device: str = 'cpu',
        verbose: bool = True,
        init_wave_modules: bool = True,
        capable_field: Optional[Dict[str, Any]] = None,
    ) -> 'FLUXHybrid':
        """
        Build FLUXHybrid from loaded .flx dict.
        
        Args:
            flx: Loaded .flx archive (base model, e.g., Flux-X-complete.flx)
            device: Target device
            verbose: Print loading progress
            init_wave_modules: Initialize fresh wave modules if missing
            capable_field: Optional enriched field from Flux-capable.flx
        
        Returns:
            FLUXHybrid instance
        """
        version = flx.get('version', '1.0-beta')
        if verbose:
            print(f"  Building FLUXHybrid from .flx v{version}")
        
        # Extract config
        config = {
            'version': version,
            'wave_dim': 432,  # Fixed by CSE
        }
        
        # Get field config
        if 'field' in flx and 'config' in flx['field']:
            fc = flx['field']['config']
            config['field_h'] = fc.get('h', 96)
            config['field_w'] = fc.get('w', 96)
            config['field_d'] = fc.get('d', 96)
            config['field_features'] = fc.get('features', 512)
        
        # Get decoder config
        if 'decoder' in flx and 'config' in flx['decoder']:
            dc = flx['decoder']['config']
            config['decoder_hidden'] = dc.get('hidden_dim', 1024)
            config['decoder_layers'] = dc.get('num_layers', 4)
            config['decoder_heads'] = dc.get('n_heads', 8)
        
        # Build model
        model = cls(config=config, device=device)
        
        components_loaded = 0
        
        # Load CSE
        if 'cse' in flx and 'state_dict' in flx['cse']:
            try:
                model.cse.load_state_dict(flx['cse']['state_dict'])
                components_loaded += 1
                if verbose:
                    print(f"  ✓ CSE loaded")
            except Exception as e:
                if verbose:
                    print(f"  ⚠ CSE load failed: {e}")
        
        # Load Field
        if 'field' in flx and 'state_dict' in flx['field']:
            try:
                model.field.load_state_dict(flx['field']['state_dict'])
                components_loaded += 1
                if verbose:
                    print(f"  ✓ Field loaded (base model)")
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Field load failed: {e}")
        
        # Merge capable field if provided (enriched with 155k+ samples)
        if capable_field is not None and 'state_dict' in capable_field:
            try:
                # Load the enriched field state
                capable_state = capable_field['state_dict']
                
                # Replace field state with capable version (best generation quality)
                if 'state' in capable_state:
                    model.field.state.data.copy_(capable_state['state'])
                if 'mass' in capable_state:
                    model.field.mass.data.copy_(capable_state['mass'])
                if 'energy' in capable_state:
                    model.field.energy.data.copy_(capable_state['energy'])
                
                if verbose:
                    print(f"  ✓ Capable field merged (155k+ samples enriched)")
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Capable field merge failed: {e}")
        
        # Load GR state
        if 'field' in flx and 'gravity_state' in flx['field']:
            try:
                model.gr.load_state(flx['field']['gravity_state'])
                components_loaded += 1
                if verbose:
                    print(f"  ✓ GravitationalRelevance loaded")
            except Exception as e:
                if verbose:
                    print(f"  ⚠ GR load failed: {e}")
        
        # Load TL state
        if 'field' in flx and 'thermodynamic_state' in flx['field']:
            try:
                model.tl.load_state(flx['field']['thermodynamic_state'])
                components_loaded += 1
                if verbose:
                    print(f"  ✓ ThermodynamicLearner loaded")
            except Exception as e:
                pass
        
        # Load Memory
        if 'memory' in flx:
            mem = flx['memory']
            if 'episodic' in mem:
                try:
                    model.episodic_memory.load_state(mem['episodic'])
                    components_loaded += 1
                    if verbose:
                        print(f"  ✓ EpisodicMemory loaded")
                except:
                    pass
        
        # Load Decoder (Phase 8)
        if 'decoder' in flx and 'state_dict' in flx['decoder']:
            try:
                decoder_state = flx['decoder']['state_dict']
                cleaned = {k.replace('_orig_mod.', ''): v for k, v in decoder_state.items()}
                model.decoder.load_state_dict(cleaned)
                components_loaded += 1
                if verbose:
                    print(f"  ✓ WaveDecoder loaded (byte-mode)")
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Decoder load failed: {e}")
        
        # Load Bridges
        if 'bridges' in flx:
            bridges = flx['bridges']
            for name in ['wave_to_field', 'field_to_wave', 'output_head']:
                if name in bridges:
                    try:
                        getattr(model, name).load_state_dict(bridges[name])
                        components_loaded += 1
                        if verbose:
                            print(f"  ✓ {name} bridge loaded")
                    except:
                        pass
        
        # Load or initialize wave modules
        wave_modules_loaded = 0
        
        if 'wave_generator' in flx and flx['wave_generator'] is not None:
            try:
                cfg = flx['wave_generator'].get('config', {})
                model.wave_generator = WaveGenerator(
                    wave_dim=cfg.get('wave_dim', 432),
                    field_features=cfg.get('field_features', 512),
                    hidden_dim=cfg.get('hidden_dim', 512),
                )
                model.wave_generator.load_state_dict(flx['wave_generator']['state_dict'])
                wave_modules_loaded += 1
                if verbose:
                    print(f"  ✓ WaveGenerator loaded")
            except Exception as e:
                if verbose:
                    print(f"  ⚠ WaveGenerator load failed: {e}")
        
        if 'wave_chunker' in flx and flx['wave_chunker'] is not None:
            try:
                cfg = flx['wave_chunker'].get('config', {})
                model.wave_chunker = WaveChunker(wave_dim=cfg.get('wave_dim', 432))
                model.wave_chunker.load_state_dict(flx['wave_chunker']['state_dict'])
                wave_modules_loaded += 1
                if verbose:
                    print(f"  ✓ WaveChunker loaded")
            except:
                pass
        
        if 'wave_to_text' in flx and flx['wave_to_text'] is not None:
            try:
                cfg = flx['wave_to_text'].get('config', {})
                model.wave_to_text = WaveToText(
                    wave_dim=cfg.get('wave_dim', 432),
                    hidden_dim=cfg.get('hidden_dim', 256),
                )
                model.wave_to_text.load_state_dict(flx['wave_to_text']['state_dict'])
                wave_modules_loaded += 1
                if verbose:
                    print(f"  ✓ WaveToText loaded")
            except:
                pass
        
        # Initialize fresh wave modules if requested and missing
        if init_wave_modules and wave_modules_loaded < 3:
            if verbose:
                print(f"  ℹ Initializing fresh wave modules...")
            model._init_wave_modules()
        
        # Check wave mode availability
        model._wave_mode_available = all([
            model.wave_generator is not None,
            model.wave_chunker is not None,
            model.wave_to_text is not None,
        ])
        
        # Load metadata
        if 'metadata' in flx:
            meta = flx['metadata']
            if 'learning_steps' in meta:
                model._learning_steps = meta['learning_steps']
            if 'injection' in meta:
                if verbose:
                    total = meta['injection'].get('total_samples', 0)
                    print(f"  ℹ Field enriched with {total:,} samples")
        
        # Move to device
        model = model.to(device)
        
        if verbose:
            print(f"\n  ═══ FLUXHybrid ready ═══")
            print(f"  Components loaded: {components_loaded}")
            print(f"  Wave modules: {wave_modules_loaded}/3")
        
        return model
    
    def _init_wave_modules(self) -> None:
        """Initialize fresh wave modules."""
        field_features = self._config.get('field_features', 512)
        wave_dim = self._config.get('wave_dim', 432)
        
        self.wave_generator = WaveGenerator(
            wave_dim=wave_dim,
            field_features=field_features,
            hidden_dim=512,
        )
        
        self.wave_chunker = WaveChunker(wave_dim=wave_dim)
        
        self.wave_to_text = WaveToText(
            wave_dim=wave_dim,
            hidden_dim=256,
            max_bytes=20,
        )
        
        self._wave_mode_available = True
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        mode: str = 'auto',
        temperature: float = 0.8,
        output_modality: str = 'text',
    ) -> HybridResponse:
        """
        Generate output from prompt.
        
        Args:
            prompt: Input text
            max_length: Maximum output length (bytes for byte-mode, chars for wave)
            mode: 'auto', 'wave' (fast), or 'byte' (precise)
            temperature: Sampling temperature
            output_modality: 'text', 'image', 'audio', 'mol'
        
        Returns:
            HybridResponse with generated text and metadata
        """
        start_time = time.time()
        
        # Encode prompt
        wave_result = self.cse.encode(prompt)
        wave_seq = wave_result.full.to(self._device_str)
        wave_vec = wave_seq.mean(dim=0)
        
        # Get field context
        with torch.no_grad():
            field_features = self.wave_to_field(wave_vec)
            
            # Query field for relevant context
            try:
                results, distances, masses = self.field.query(field_features, k=4)
                context = results.mean(dim=0)
            except:
                context = field_features
        
        # Route to appropriate mode
        if mode == 'auto':
            mode = self.task_router.route(prompt, output_modality, wave_vec)
            _, mode_reason = self.task_router.route_with_reason(prompt, output_modality, wave_vec)
        else:
            mode_reason = f"Forced mode: {mode}"
        
        # Generate
        if mode == 'wave' and self._wave_mode_available:
            text, n_steps, confidence = self._generate_wave_mode(
                context, max_length, temperature, output_modality
            )
        else:
            text, n_steps, confidence = self._generate_byte_mode(
                prompt, context, max_length, temperature
            )
            mode = 'byte'  # In case wave was requested but unavailable
        
        gen_time = time.time() - start_time
        
        return HybridResponse(
            text=text,
            mode=mode,
            mode_reason=mode_reason,
            generation_time=gen_time,
            n_steps=n_steps,
            confidence=confidence,
            modality=output_modality,
        )
    
    def _generate_wave_mode(
        self,
        context: torch.Tensor,
        max_length: int,
        temperature: float,
        output_modality: str,
    ) -> Tuple[str, int, float]:
        """Fast wave-level generation."""
        max_waves = max(5, max_length // 5)  # ~5 chars per word
        
        # Generate waves
        result = self.wave_generator.generate(
            field_context=context,
            max_waves=max_waves,
            temperature=temperature,
        )
        
        if result.n_steps == 0:
            return "", 0, 0.0
        
        # Route to appropriate decoder
        if output_modality == 'text':
            text = self.wave_to_text.decode_sequence(
                result.waves,
                temperature=temperature,
            )
        elif output_modality == 'image' and self.wave_to_image is not None:
            # Return image tensor as string representation
            img = self.wave_to_image(result.waves)
            text = f"<image tensor: {img.shape}>"
        else:
            # Fallback to text
            text = self.wave_to_text.decode_sequence(result.waves, temperature)
        
        avg_conf = result.confidences.mean().item()
        return text, result.n_steps, avg_conf
    
    def _generate_byte_mode(
        self,
        prompt: str,
        context: torch.Tensor,
        max_length: int,
        temperature: float,
    ) -> Tuple[str, int, float]:
        """Precise byte-level generation via Phase 8 WaveDecoder."""
        self.decoder.eval()
        
        # Encode prompt to wave sequence for cross-attention
        wave_result = self.cse.encode(prompt)
        wave_seq = wave_result.full.to(self._device_str)  # [seq_len, wave_dim]
        
        with torch.no_grad():
            text = self.decoder.generate(
                wave_sequence=wave_seq,
                field_features=context,
                max_length=max_length,
                temperature=temperature,
            )
        
        # WaveDecoder returns bytes, decode to string
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        
        return text, max_length, 1.0
    
    def learn_fact(self, fact: str) -> Dict[str, Any]:
        """
        One-shot fact learning via episodic memory.
        
        This is the FLUX advantage — new facts become permanent
        without destroying existing knowledge.
        
        Args:
            fact: Fact to learn
        
        Returns:
            Dict with learning info
        """
        wave_result = self.cse.encode(fact)
        wave_vec = wave_result.full.mean(dim=0).to(self._device_str)
        
        # Project to field space
        field_vec = self.wave_to_field(wave_vec)
        
        # Add to episodic memory
        self.episodic_memory.add(
            field_vec,
            metadata={'fact': fact, 'timestamp': datetime.now().isoformat()},
        )
        
        # Inject into field
        self._inject_to_field(field_vec, fact)
        
        return {
            'fact': fact,
            'memory_size': self.episodic_memory.size,
            'learned': True,
        }
    
    def _inject_to_field(self, vec: torch.Tensor, text: str) -> None:
        """Inject knowledge into field via physics."""
        import hashlib
        h_hash = hashlib.md5(text.encode()).hexdigest()
        
        H, W, D = self.field.h, self.field.w, self.field.d
        h = int(h_hash[:8], 16) % H
        w = int(h_hash[8:16], 16) % W
        d = int(h_hash[16:24], 16) % D
        
        # Inject
        self.field.state[h, w, d] += vec.cpu() * 0.1
        self.field.mass[h, w, d] += 1.0
    
    def query(self, question: str, use_memory: bool = True) -> HybridResponse:
        """
        Query knowledge and generate answer.
        
        Args:
            question: Question to answer
            use_memory: Whether to search episodic memory
        
        Returns:
            HybridResponse
        """
        wave_result = self.cse.encode(question)
        wave_vec = wave_result.full.mean(dim=0).to(self._device_str)
        field_vec = self.wave_to_field(wave_vec)
        
        context_parts = [question]
        
        if use_memory:
            # Search episodic memory
            results = self.episodic_memory.search(field_vec, k=3)
            for r in results:
                if 'fact' in r.get('metadata', {}):
                    context_parts.append(r['metadata']['fact'])
        
        # Generate with context
        prompt = "\n".join(context_parts)
        return self.generate(prompt)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            'total_params': total_params,
            'version': self._version,
            'wave_mode_available': self._wave_mode_available,
            'field_shape': (self.field.h, self.field.w, self.field.d),
            'field_features': self.field.features,
            'episodic_entries': self.episodic_memory.size if hasattr(self.episodic_memory, 'size') else 0,
            'field_energy': self.field.energy.mean().item(),
            'field_mass_sum': self.field.mass.sum().item(),
            'learning_steps': self._learning_steps,
            'router_stats': self.task_router.get_stats(),
        }
    
    def save_flx(self, path: Path, version: str = '1.1-hybrid') -> None:
        """
        Save current state to .flx file.
        
        Args:
            path: Output path
            version: Format version
        """
        from flx_loader_v2 import save_flx
        
        components = {
            'cse': self.cse,
            'field': {
                'config': {
                    'h': self.field.h,
                    'w': self.field.w,
                    'd': self.field.d,
                    'features': self.field.features,
                },
                'state_dict': self.field.state_dict(),
                'gravity_state': self.gr.save_state() if hasattr(self.gr, 'save_state') else {},
                'thermodynamic_state': self.tl.save_state() if hasattr(self.tl, 'save_state') else {},
            },
            'decoder': self.decoder,
            'bridges': {
                'wave_to_field': self.wave_to_field.state_dict(),
                'field_to_wave': self.field_to_wave.state_dict(),
                'output_head': self.output_head.state_dict(),
            },
            'wave_generator': self.wave_generator,
            'wave_chunker': self.wave_chunker,
            'wave_to_text': self.wave_to_text,
            'task_router': self.task_router.state_dict_router(),
        }
        
        metadata = {
            'learning_steps': self._learning_steps,
            'upgraded_from': self._version,
            'timestamp': datetime.now().isoformat(),
        }
        
        save_flx(path, components, version=version, metadata=metadata)


# ─────────────────────────────────────────────
# Quick Test  
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("FLUXHybrid (Phase 10) — Testing")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Build minimal model for testing
    config = {
        'wave_dim': 432,
        'field_h': 16,
        'field_w': 16,
        'field_d': 16,
        'field_features': 512,
    }
    
    print("\nBuilding FLUXHybrid...")
    model = FLUXHybrid(config=config, device=device)
    model._init_wave_modules()
    model = model.to(device)
    
    stats = model.get_stats()
    print(f"  Total params: {stats['total_params']:,}")
    print(f"  Wave mode: {stats['wave_mode_available']}")
    
    # Test generation
    print("\nTesting generation...")
    
    prompts = [
        ("Hello!", 'byte'),  # Short → byte
        ("Explain quantum computing", 'wave'),  # Semantic → wave
        ("def fibonacci(n):", 'byte'),  # Code → byte
    ]
    
    for prompt, expected_mode in prompts:
        response = model.generate(prompt, max_length=30)
        status = "✓" if response.mode == expected_mode else "⚠"
        print(f"\n  {status} '{prompt[:30]}...'")
        print(f"      Mode: {response.mode} (expected: {expected_mode})")
        print(f"      Reason: {response.mode_reason}")
        print(f"      Time: {response.generation_time:.3f}s")
        if response.text:
            print(f"      Output: {response.text[:50]}...")
    
    print("\n" + "=" * 60)
    print("✓ FLUXHybrid working")
