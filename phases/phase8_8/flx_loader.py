"""
Phase 8.8: FLX Loader — Load FLUXLarge from .flx Archive

Loads the complete FLUX model from a .flx file (local or HuggingFace Hub).
This is the foundation for Phase 8.8 — everything builds on top of Flux-beta.flx.
"""

import sys
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

# Add paths
for _p in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6', 'phase7', 'phase8']:
    _pp = str(_PHASES_DIR / _p)
    if _pp not in sys.path:
        sys.path.insert(0, _pp)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ─────────────────────────────────────────────
# HuggingFace Config
# ─────────────────────────────────────────────

HF_REPO_ID = "UnseenGAP/FLUX"
HF_FLX_PATH = "checkpoints/Flux-beta.flx"

SUPPORTED_VERSIONS = ['1.0-beta', '1.0', '1.1', '1.1-hybrid', '2.0']

DEFAULT_FLX_PATH = _PROJECT_ROOT / 'checkpoints' / 'Flux-beta.flx'


# ─────────────────────────────────────────────
# Token Resolution
# ─────────────────────────────────────────────

def get_hf_token() -> Optional[str]:
    """
    Resolve HuggingFace token from multiple sources.
    Priority: HF_TOKEN env var > Kaggle secrets > Colab secrets > cached login
    """
    import os
    
    # Environment variable
    token = os.environ.get('HF_TOKEN')
    if token:
        return token.strip()
    
    # Kaggle secrets
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            return token.strip()
    except:
        pass
    
    # Colab secrets
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
        if token:
            return token.strip()
    except:
        pass
    
    # huggingface-cli login token (cached)
    try:
        from huggingface_hub import HfFolder
        cached = HfFolder.get_token()
        if cached:
            return cached.strip()
    except:
        pass
    
    return None


# ─────────────────────────────────────────────
# Download from HuggingFace
# ─────────────────────────────────────────────

def download_flx_from_hf(
    dest_path: Path = DEFAULT_FLX_PATH,
    hf_token: Optional[str] = None,
    repo_id: str = HF_REPO_ID,
    verbose: bool = True,
) -> bool:
    """
    Download Flux-beta.flx from HuggingFace Hub.
    
    Args:
        dest_path: Local destination path
        hf_token: HuggingFace API token (optional)
        repo_id: HuggingFace repo ID
        verbose: Print progress messages
    
    Returns:
        True if download succeeded
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    token = hf_token or get_hf_token()
    
    try:
        from huggingface_hub import hf_hub_download
        
        if verbose:
            print(f"  ↓ Downloading {HF_FLX_PATH} from {repo_id}...")
        
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=HF_FLX_PATH,
            local_dir=str(_PROJECT_ROOT),
            token=token,
        )
        
        if verbose:
            size_mb = Path(downloaded_path).stat().st_size / 1e6
            print(f"  ✓ Downloaded Flux-beta.flx ({size_mb:.1f} MB)")
        
        return True
        
    except ImportError:
        if verbose:
            print("  ⚠ huggingface_hub not installed — run: pip install huggingface_hub")
        return False
    except Exception as e:
        if verbose:
            print(f"  ✗ Download failed: {e}")
        return False


# ─────────────────────────────────────────────
# Load FLX Archive
# ─────────────────────────────────────────────

def load_flx(
    path: Path = DEFAULT_FLX_PATH,
    device: str = 'cpu',
    auto_download: bool = True,
) -> Dict[str, Any]:
    """
    Load raw .flx archive contents.
    
    If file doesn't exist locally and auto_download is True,
    attempts to download from HuggingFace Hub first.
    
    Args:
        path: Path to .flx file
        device: Target device for tensors
        auto_download: Auto-download from HF if missing
    
    Returns:
        Dict with all .flx sections
    """
    path = Path(path)
    
    if not path.exists():
        if auto_download:
            print(f"  ℹ {path.name} not found locally, checking HuggingFace Hub...")
            if download_flx_from_hf(path):
                pass  # Download succeeded
            else:
                raise FileNotFoundError(
                    f".flx file not found: {path}\n"
                    f"    Download from: https://huggingface.co/{HF_REPO_ID}/tree/main/checkpoints"
                )
        else:
            raise FileNotFoundError(f".flx file not found: {path}")
    
    # Load checkpoint (weights_only=False for numpy arrays)
    flx = torch.load(str(path), map_location=device, weights_only=False)
    
    # Validate format
    fmt = flx.get('format', '')
    if fmt != 'FLUX':
        raise ValueError(f"Invalid .flx format: expected 'FLUX', got '{fmt}'")
    
    version = flx.get('version', 'unknown')
    if not any(version.startswith(v.split('-')[0]) for v in SUPPORTED_VERSIONS):
        print(f"  ⚠ Unknown .flx version: {version}")
    
    return flx


def get_flx_info(path: Path = DEFAULT_FLX_PATH) -> Dict[str, Any]:
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
        
        if 'state_dict' not in flx.get('cse', {}):
            return False, "CSE section missing state_dict"
        
        if 'state_dict' not in flx.get('field', {}):
            return False, "Field section missing state_dict"
        
        if 'state_dict' not in flx.get('decoder', {}):
            return False, "Decoder section missing state_dict"
        
        return True, f"Valid .flx v{flx['version']}"
        
    except FileNotFoundError:
        return False, f"File not found: {path}"
    except Exception as e:
        return False, f"Load error: {e}"


# ─────────────────────────────────────────────
# Load Full FLUXLarge Model
# ─────────────────────────────────────────────

def load_flux_from_flx(
    path: Path = DEFAULT_FLX_PATH,
    device: str = 'cpu',
    verbose: bool = True,
    auto_download: bool = True,
):
    """
    Load complete FLUXLarge model from .flx archive.
    
    This reconstructs the full model with all trained weights,
    thermodynamic state, and memory contents.
    
    If .flx file doesn't exist locally, auto-downloads from HuggingFace Hub.
    
    Args:
        path: Path to .flx file
        device: Target device
        verbose: Print loading progress
        auto_download: Download from HF Hub if missing
    
    Returns:
        FLUXLarge instance ready for inference/continued training
    """
    from flux_large import FLUXLarge, FLUX_LARGE_CONFIG
    
    path = Path(path)
    if verbose:
        print(f"\n  Loading from {path.name}...")
    
    # Load raw archive (auto-downloads if missing)
    flx = load_flx(path, device='cpu', auto_download=auto_download)
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
    
    # Build fresh model shell
    model = FLUXLarge(config=config, device=device)
    
    components_loaded = 0
    
    # Load CSE
    if 'cse' in flx and 'state_dict' in flx['cse']:
        try:
            model.cse.load_state_dict(flx['cse']['state_dict'])
            components_loaded += 1
            if verbose:
                print(f"  ✓ CSE loaded (wave_dim={config['wave_dim']})")
        except Exception as e:
            print(f"  ⚠ CSE load failed: {e}")
    
    # Load Field
    if 'field' in flx and 'state_dict' in flx['field']:
        try:
            model.field.load_state_dict(flx['field']['state_dict'])
            components_loaded += 1
            if verbose:
                h, w, d = config['field_h'], config['field_w'], config['field_d']
                print(f"  ✓ Field loaded ({h}×{w}×{d} × {config['field_features']})")
        except Exception as e:
            print(f"  ⚠ Field load failed: {e}")
    
    # Load Gravitational Relevance
    if 'field' in flx and 'gravity_state' in flx['field']:
        try:
            model.gr.load_state(flx['field']['gravity_state'])
            components_loaded += 1
            if verbose:
                print(f"  ✓ GravitationalRelevance loaded")
        except Exception as e:
            print(f"  ⚠ GR load failed: {e}")
    
    # Load Thermodynamic Learner
    if 'field' in flx and 'thermodynamic_state' in flx['field']:
        try:
            model.tl.load_state(flx['field']['thermodynamic_state'])
            components_loaded += 1
            if verbose:
                temp = model.tl.temp_manager.temperature
                print(f"  ✓ ThermodynamicLearner loaded (temp={temp:.4f})")
        except Exception as e:
            print(f"  ⚠ TL load failed: {e}")
    
    # Load Memory Systems
    if 'memory' in flx:
        mem = flx['memory']
        
        if 'working' in mem:
            try:
                model.working_memory.load_state_dict_memory(mem['working'])
                components_loaded += 1
                if verbose:
                    print(f"  ✓ WorkingMemory loaded")
            except Exception as e:
                print(f"  ⚠ WorkingMemory load failed: {e}")
        
        if 'episodic' in mem:
            try:
                model.episodic_memory.load_state(mem['episodic'])
                components_loaded += 1
                entries = model.episodic_memory.size
                if verbose:
                    print(f"  ✓ EpisodicMemory loaded ({entries} entries)")
            except Exception as e:
                print(f"  ⚠ EpisodicMemory load failed: {e}")
        
        if 'semantic' in mem:
            try:
                model.semantic_memory.load_state(mem['semantic'])
                components_loaded += 1
                if verbose:
                    print(f"  ✓ SemanticMemory loaded")
            except Exception as e:
                print(f"  ⚠ SemanticMemory load failed: {e}")
    
    # Load Decoder (clean _orig_mod. prefix from torch.compile)
    if 'decoder' in flx and 'state_dict' in flx['decoder']:
        try:
            decoder_state = flx['decoder']['state_dict']
            cleaned = {k.replace('_orig_mod.', ''): v for k, v in decoder_state.items()}
            model.decoder.load_state_dict(cleaned)
            components_loaded += 1
            if verbose:
                decoder_cfg = flx['decoder'].get('config', {})
                print(f"  ✓ WaveDecoder loaded (hidden={decoder_cfg.get('hidden_dim', '?')})")
        except Exception as e:
            print(f"  ⚠ Decoder load failed: {e}")
    
    # Load Causal Components
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
    
    # Load Bridges
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
    
    # Load Metadata
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


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("FLX Loader (Phase 8.8) — Testing")
    
    flx_path = DEFAULT_FLX_PATH
    
    if flx_path.exists():
        print(f"\n  Found: {flx_path}")
        
        success, msg = verify_flx(flx_path)
        print(f"  Verify: {msg}")
        
        if success:
            info = get_flx_info(flx_path)
            print(f"  Version: {info['version']}")
            print(f"  Size: {info['file_size_mb']:.1f} MB")
            print(f"  Components: {info['components']}")
    else:
        print(f"\n  ℹ No .flx file found at {flx_path}")
        print("    Will auto-download from HuggingFace Hub when needed")
