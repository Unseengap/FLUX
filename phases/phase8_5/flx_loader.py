"""
Phase 8.5: FLX Loader — Load FLUXLarge from .flx Archive

Loads the complete FLUX model from a .flx file, preserving:
- All 14 trained components
- Thermodynamic state (temperature, energy history)
- Episodic/semantic memory contents
- Field attractor state

The .flx format is self-describing and version-aware.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

for _p in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6', 'phase7', 'phase8']:
    _pp = str(_PHASES_DIR / _p)
    if _pp not in sys.path:
        sys.path.insert(0, _pp)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ─────────────────────────────────────────────
# FLX Format Constants
# ─────────────────────────────────────────────

SUPPORTED_VERSIONS = ['1.0-beta', '1.0', '1.1', '1.1-hybrid', '2.0']

DEFAULT_FLX_PATH = Path('checkpoints/Flux-beta.flx')


# ─────────────────────────────────────────────
# FLX Loader Functions
# ─────────────────────────────────────────────

def load_flx(path: Path, device: str = 'cpu') -> Dict[str, Any]:
    """
    Load raw .flx archive contents.
    
    Args:
        path: Path to .flx file
        device: Target device for tensors
    
    Returns:
        Dict with all .flx sections
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is invalid
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f".flx file not found: {path}")
    
    flx = torch.load(str(path), map_location=device)
    
    # Validate format
    fmt = flx.get('format', '')
    if fmt != 'FLUX':
        raise ValueError(f"Invalid .flx format: expected 'FLUX', got '{fmt}'")
    
    version = flx.get('version', 'unknown')
    if not any(version.startswith(v.split('-')[0]) for v in SUPPORTED_VERSIONS):
        print(f"  ⚠ Unknown .flx version: {version}")
    
    return flx


def get_flx_info(path: Path) -> Dict[str, Any]:
    """
    Get .flx file metadata without full load.
    
    Args:
        path: Path to .flx file
    
    Returns:
        Dict with version, metadata, component list
    """
    flx = load_flx(path, device='cpu')
    
    components = []
    for key in ['cse', 'field', 'memory', 'decoder', 'causal', 'bridges']:
        if key in flx:
            components.append(key)
    
    return {
        'format': flx.get('format'),
        'version': flx.get('version'),
        'metadata': flx.get('metadata', {}),
        'components': components,
        'file_size_mb': path.stat().st_size / 1e6,
    }


def load_flux_from_flx(
    path: Path = DEFAULT_FLX_PATH,
    device: str = 'cpu',
    verbose: bool = True,
) -> 'FLUXLarge':
    """
    Load complete FLUXLarge model from .flx archive.
    
    This reconstructs the full model with all trained weights,
    thermodynamic state, and memory contents.
    
    Args:
        path: Path to .flx file
        device: Target device
        verbose: Print loading progress
    
    Returns:
        FLUXLarge instance ready for inference/continued training
    """
    from flux_large import FLUXLarge, FLUX_LARGE_CONFIG
    
    path = Path(path)
    if verbose:
        print(f"\n  Loading from {path.name}...")
    
    # Load raw archive
    flx = load_flx(path, device='cpu')
    version = flx.get('version', '1.0-beta')
    
    if verbose:
        print(f"  ✓ Format: FLUX v{version}")
    
    # Extract config
    config = FLUX_LARGE_CONFIG.copy()
    
    # Override from .flx metadata/sections
    if 'cse' in flx and 'config' in flx['cse']:
        cse_cfg = flx['cse']['config']
        if 'wave_dim' in cse_cfg:
            config['wave_dim'] = cse_cfg['wave_dim']
    
    if 'field' in flx and 'config' in flx['field']:
        field_cfg = flx['field']['config']
        for key in ['h', 'w', 'd', 'features']:
            full_key = f"field_{key}" if key != 'features' else 'field_features'
            if key in field_cfg:
                config[full_key] = field_cfg[key]
    
    if 'decoder' in flx and 'config' in flx['decoder']:
        # Decoder config for reference (WaveDecoder built separately)
        pass
    
    # Build fresh model shell
    model = FLUXLarge(config=config, device=device)
    
    components_loaded = 0
    
    # ── Load CSE (Continuous Semantic Encoder) ──
    if 'cse' in flx and 'state_dict' in flx['cse']:
        try:
            model.cse.load_state_dict(flx['cse']['state_dict'])
            components_loaded += 1
            if verbose:
                print(f"  ✓ CSE loaded (wave_dim={config['wave_dim']})")
        except Exception as e:
            print(f"  ⚠ CSE load failed: {e}")
    
    # ── Load Field (Resonance Field) ──
    if 'field' in flx and 'state_dict' in flx['field']:
        try:
            model.field.load_state_dict(flx['field']['state_dict'])
            components_loaded += 1
            if verbose:
                h, w, d = config['field_h'], config['field_w'], config['field_d']
                print(f"  ✓ Field loaded ({h}×{w}×{d} × {config['field_features']})")
        except Exception as e:
            print(f"  ⚠ Field load failed: {e}")
    
    # ── Load Gravitational Relevance ──
    if 'field' in flx and 'gravity_state' in flx['field']:
        try:
            model.gr.load_state(flx['field']['gravity_state'])
            components_loaded += 1
            if verbose:
                print(f"  ✓ GravitationalRelevance loaded")
        except Exception as e:
            print(f"  ⚠ GR load failed: {e}")
    
    # ── Load Thermodynamic Learner ──
    if 'field' in flx and 'thermodynamic_state' in flx['field']:
        try:
            model.tl.load_state(flx['field']['thermodynamic_state'])
            components_loaded += 1
            if verbose:
                temp = model.tl.temp_manager.temperature
                print(f"  ✓ ThermodynamicLearner loaded (temp={temp:.4f})")
        except Exception as e:
            print(f"  ⚠ TL load failed: {e}")
    
    # ── Load Memory Systems ──
    if 'memory' in flx:
        mem = flx['memory']
        
        # Working Memory
        if 'working' in mem:
            try:
                model.working_memory.load_state_dict_memory(mem['working'])
                components_loaded += 1
                if verbose:
                    print(f"  ✓ WorkingMemory loaded")
            except Exception as e:
                print(f"  ⚠ WorkingMemory load failed: {e}")
        
        # Episodic Memory
        if 'episodic' in mem:
            try:
                model.episodic_memory.load_state(mem['episodic'])
                components_loaded += 1
                entries = model.episodic_memory.size
                if verbose:
                    print(f"  ✓ EpisodicMemory loaded ({entries} entries)")
            except Exception as e:
                print(f"  ⚠ EpisodicMemory load failed: {e}")
        
        # Semantic Memory
        if 'semantic' in mem:
            try:
                model.semantic_memory.load_state(mem['semantic'])
                components_loaded += 1
                if verbose:
                    print(f"  ✓ SemanticMemory loaded")
            except Exception as e:
                print(f"  ⚠ SemanticMemory load failed: {e}")
    
    # ── Load Decoder ──
    if 'decoder' in flx and 'state_dict' in flx['decoder']:
        try:
            decoder_state = flx['decoder']['state_dict']
            
            # Clean _orig_mod. prefix if present (from torch.compile)
            cleaned = {}
            for k, v in decoder_state.items():
                clean_k = k.replace('_orig_mod.', '')
                cleaned[clean_k] = v
            
            model.decoder.load_state_dict(cleaned)
            components_loaded += 1
            if verbose:
                decoder_cfg = flx['decoder'].get('config', {})
                print(f"  ✓ WaveDecoder loaded (hidden={decoder_cfg.get('hidden_dim', '?')})")
        except Exception as e:
            print(f"  ⚠ Decoder load failed: {e}")
    
    # ── Load Causal Components ──
    if 'causal' in flx:
        causal = flx['causal']
        
        if 'cgn_state' in causal:
            try:
                model.cgn.load_state(causal['cgn_state'])
                components_loaded += 1
                if verbose:
                    print(f"  ✓ CausalGeometryNode loaded")
            except Exception as e:
                print(f"  ⚠ CGN load failed: {e}")
        
        if 'graph_state' in causal:
            try:
                model.causal_graph.load_state(causal['graph_state'])
                components_loaded += 1
                if verbose:
                    print(f"  ✓ CausalGraph loaded")
            except Exception as e:
                print(f"  ⚠ CausalGraph load failed: {e}")
    
    # ── Load Bridges ──
    if 'bridges' in flx:
        bridges = flx['bridges']
        
        if 'wave_to_field' in bridges:
            try:
                model.wave_to_field.load_state_dict(bridges['wave_to_field'])
                components_loaded += 1
                if verbose:
                    print(f"  ✓ wave_to_field bridge loaded")
            except Exception as e:
                print(f"  ⚠ wave_to_field load failed: {e}")
        
        if 'field_to_wave' in bridges:
            try:
                model.field_to_wave.load_state_dict(bridges['field_to_wave'])
                components_loaded += 1
                if verbose:
                    print(f"  ✓ field_to_wave bridge loaded")
            except Exception as e:
                print(f"  ⚠ field_to_wave load failed: {e}")
        
        if 'router' in bridges:
            try:
                model.memory_router.load_state(bridges['router'])
                components_loaded += 1
                if verbose:
                    print(f"  ✓ MemoryRouter loaded")
            except Exception as e:
                print(f"  ⚠ MemoryRouter load failed: {e}")
        
        if 'output_head' in bridges:
            try:
                model.output_head.load_state_dict(bridges['output_head'])
                components_loaded += 1
                if verbose:
                    print(f"  ✓ OutputHead loaded")
            except Exception as e:
                print(f"  ⚠ OutputHead load failed: {e}")
    
    # ── Load Metadata ──
    if 'metadata' in flx:
        meta = flx['metadata']
        if 'learning_steps' in meta:
            model._learning_steps = meta['learning_steps']
    
    # Move to target device
    model = model.to(device)
    
    if verbose:
        stats = model.get_stats()
        print(f"\n  ═══ FLUXLarge loaded from .flx: {stats.total_params:,} params ═══")
        print(f"  Components: {components_loaded}/14")
        print(f"  Learning steps: {model._learning_steps}")
        print(f"  Episodic entries: {stats.episodic_entries}")
        print(f"  Field energy: {stats.field_energy:.4f}")
    
    return model


def verify_flx(path: Path = DEFAULT_FLX_PATH) -> Tuple[bool, str]:
    """
    Verify a .flx file is valid and loadable.
    
    Args:
        path: Path to .flx file
    
    Returns:
        (success, message)
    """
    try:
        flx = load_flx(path, device='cpu')
        
        required = ['format', 'version', 'cse', 'field', 'decoder']
        missing = [k for k in required if k not in flx]
        
        if missing:
            return False, f"Missing required sections: {missing}"
        
        # Check cse has state_dict
        if 'state_dict' not in flx.get('cse', {}):
            return False, "CSE section missing state_dict"
        
        # Check field has state_dict
        if 'state_dict' not in flx.get('field', {}):
            return False, "Field section missing state_dict"
        
        # Check decoder has state_dict
        if 'state_dict' not in flx.get('decoder', {}):
            return False, "Decoder section missing state_dict"
        
        return True, f"Valid .flx v{flx['version']}"
        
    except FileNotFoundError:
        return False, f"File not found: {path}"
    except Exception as e:
        return False, f"Load error: {e}"


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("FLX Loader — Testing")
    
    # Check for local .flx
    flx_path = Path('checkpoints/Flux-beta.flx')
    
    if flx_path.exists():
        print(f"\n  Found: {flx_path}")
        
        # Verify
        success, msg = verify_flx(flx_path)
        print(f"  Verify: {msg}")
        
        if success:
            # Get info
            info = get_flx_info(flx_path)
            print(f"  Version: {info['version']}")
            print(f"  Size: {info['file_size_mb']:.1f} MB")
            print(f"  Components: {info['components']}")
            
            # Full load
            print("\n  Loading full model...")
            model = load_flux_from_flx(flx_path, device='cpu', verbose=True)
            
            print("\n  ✓ FLX Loader test complete")
    else:
        print(f"\n  ⚠ No .flx file found at {flx_path}")
        print("    Run Phase 8 training or download from HuggingFace")
