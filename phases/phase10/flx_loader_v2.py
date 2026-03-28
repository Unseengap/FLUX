"""
Phase 10: FLX Loader v2 — Hybrid Wave+Byte Model Support

Extends Phase 8.8 flx_loader to support:
- Flux-X-complete.flx: Full trained base model (all components)
- Flux-capable.flx: Knowledge-enriched field (155k+ samples, best generation)
- v1.0-beta: Original Flux-beta.flx format (fallback)
- v1.1-hybrid: Phase 10 with wave modules

Loading Strategy:
1. Load Flux-X-complete.flx as base (all trained components)
2. Optionally merge enriched field from Flux-capable.flx (best text generation)

Gracefully handles missing wave modules (fresh init if needed).
"""

import sys
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

# Add paths
for _p in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6', 'phase7', 'phase8', 'phase8_8', 'phase10']:
    _pp = str(_PHASES_DIR / _p)
    if _pp not in sys.path:
        sys.path.insert(0, _pp)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ─────────────────────────────────────────────
# HuggingFace Config
# ─────────────────────────────────────────────

HF_REPO_ID = "UnseenGAP/FLUX"

# Primary: Full trained base model
HF_X_COMPLETE_PATH = "checkpoints/Flux-X-complete.flx"
# Secondary: Knowledge-enriched field (best generation quality)
HF_CAPABLE_PATH = "checkpoints/Flux-capable.flx"
# Legacy fallback
HF_BETA_PATH = "checkpoints/Flux-beta.flx"

SUPPORTED_VERSIONS = ['1.0-beta', '1.0', '1.1', '1.1-hybrid', '1.5-capable', '2.0']

# Flux-X-complete.flx is the PRIMARY model (full trained components)
DEFAULT_FLX_PATH = _PROJECT_ROOT / 'checkpoints' / 'Flux-X-complete.flx'
# Flux-capable.flx for enriched field/generation
CAPABLE_FLX_PATH = _PROJECT_ROOT / 'checkpoints' / 'Flux-capable.flx'
# Legacy fallback
FALLBACK_FLX_PATH = _PROJECT_ROOT / 'checkpoints' / 'Flux-beta.flx'


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
    hf_path: str = HF_CAPABLE_PATH,
    verbose: bool = True,
) -> bool:
    """
    Download .flx from HuggingFace Hub.
    
    Args:
        dest_path: Local destination path
        hf_token: HuggingFace API token (optional)
        repo_id: HuggingFace repo ID
        hf_path: Path within the repo
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
            print(f"  ↓ Downloading {hf_path} from {repo_id}...")
        
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=hf_path,
            local_dir=str(_PROJECT_ROOT),
            token=token,
        )
        
        if verbose:
            size_mb = Path(downloaded_path).stat().st_size / 1e6
            print(f"  ✓ Downloaded {dest_path.name} ({size_mb:.1f} MB)")
        
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
    
    Loading priority:
    1. Flux-X-complete.flx (full trained base model)
    2. Flux-capable.flx (if X-complete not available)
    3. Flux-beta.flx (legacy fallback)
    
    Args:
        path: Path to .flx file
        device: Target device for tensors
        auto_download: Auto-download from HF if missing
    
    Returns:
        Dict with all .flx sections
    """
    path = Path(path)
    
    # Try primary path first
    if not path.exists():
        if auto_download:
            print(f"  ℹ {path.name} not found locally, checking HuggingFace Hub...")
            
            # Determine which file to download based on path name
            if 'X-complete' in path.name or 'x-complete' in path.name.lower():
                hf_path = HF_X_COMPLETE_PATH
            elif 'capable' in path.name.lower():
                hf_path = HF_CAPABLE_PATH
            else:
                hf_path = HF_X_COMPLETE_PATH  # Default to X-complete
            
            if download_flx_from_hf(path, hf_path=hf_path):
                pass  # Download succeeded
            else:
                # Try fallbacks in order: X-complete → capable → beta
                fallback_order = [
                    (DEFAULT_FLX_PATH, HF_X_COMPLETE_PATH),
                    (CAPABLE_FLX_PATH, HF_CAPABLE_PATH),
                    (FALLBACK_FLX_PATH, HF_BETA_PATH),
                ]
                
                loaded = False
                for fb_path, fb_hf in fallback_order:
                    if fb_path.exists():
                        print(f"  ℹ Falling back to {fb_path.name}")
                        path = fb_path
                        loaded = True
                        break
                    elif auto_download and download_flx_from_hf(fb_path, hf_path=fb_hf):
                        path = fb_path
                        loaded = True
                        break
                
                if not loaded:
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
    print(f"  ✓ Loaded {path.name} (FLUX v{version})")
    
    return flx


def load_capable_field(
    device: str = 'cpu',
    auto_download: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Load the enriched field state from Flux-capable.flx.
    
    This loads ONLY the field component (not the full model) for
    merging into a base model. The capable field has 155k+ samples
    injected for better text generation.
    
    Args:
        device: Target device
        auto_download: Download from HF if missing
    
    Returns:
        Dict with field state, or None if not available
    """
    capable_path = CAPABLE_FLX_PATH
    
    if not capable_path.exists():
        if auto_download:
            print(f"  ℹ Downloading Flux-capable.flx for enriched field...")
            if not download_flx_from_hf(capable_path, hf_path=HF_CAPABLE_PATH):
                print(f"  ⚠ Could not download Flux-capable.flx")
                return None
        else:
            return None
    
    try:
        capable_flx = torch.load(str(capable_path), map_location=device, weights_only=False)
        
        if 'field' in capable_flx:
            print(f"  ✓ Loaded capable field (enriched with 155k+ samples)")
            return capable_flx['field']
        else:
            print(f"  ⚠ Flux-capable.flx missing field section")
            return None
            
    except Exception as e:
        print(f"  ⚠ Error loading capable field: {e}")
        return None


def get_flx_info(path: Path = DEFAULT_FLX_PATH) -> Dict[str, Any]:
    """
    Get .flx file metadata without full load.
    
    Args:
        path: Path to .flx file
    
    Returns:
        Dict with version, metadata, component list
    """
    path = Path(path)
    if not path.exists():
        return {'error': f'File not found: {path}'}

    
    flx = load_flx(path, device='cpu', auto_download=False)
    
    components = []
    for key in ['cse', 'field', 'memory', 'decoder', 'causal', 'bridges', 
                'wave_generator', 'wave_chunker', 'wave_to_text', 'task_router']:
        if key in flx and flx[key] is not None:
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
        path = Path(path)
        if not path.exists():
            return False, f"File not found: {path}"
        
        flx = torch.load(str(path), map_location='cpu', weights_only=False)
        
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
# Save FLX Archive (v1.1-hybrid format)
# ─────────────────────────────────────────────

def save_flx(
    path: Path,
    components: Dict[str, Any],
    version: str = '1.1-hybrid',
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save components to .flx format.
    
    Preserves modular structure for component-level upgrades.
    
    Args:
        path: Output path
        components: Dict with model components (cse, field, decoder, etc.)
        version: Format version string
        metadata: Optional extra metadata
    """
    flx = {
        'format': 'FLUX',
        'version': version,
        'metadata': metadata or {},
        'timestamp': datetime.now().isoformat(),
    }
    
    # Helper to move tensors to CPU
    def safe_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        elif isinstance(obj, dict):
            return {k: safe_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [safe_cpu(v) for v in obj]
        else:
            return obj
    
    # Serialize each component
    for name, module in components.items():
        if module is None:
            flx[name] = None
        elif hasattr(module, 'state_dict'):
            config = {}
            for attr in ['config', 'wave_dim', 'hidden_dim', 'h', 'w', 'd', 'features']:
                if hasattr(module, attr):
                    val = getattr(module, attr)
                    if not callable(val):
                        config[attr] = val
            
            flx[name] = {
                'config': config,
                'state_dict': safe_cpu(module.state_dict()),
            }
        elif isinstance(module, dict):
            flx[name] = safe_cpu(module)
        else:
            flx[name] = module
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(flx, str(path))
    
    size_mb = path.stat().st_size / 1e6
    print(f"  ✓ Saved {path.name} ({size_mb:.1f} MB) — FLUX v{version}")


# ─────────────────────────────────────────────
# Load Full FLUX Hybrid Model
# ─────────────────────────────────────────────

def load_flux_hybrid(
    path: Path = DEFAULT_FLX_PATH,
    device: str = 'cpu',
    verbose: bool = True,
    auto_download: bool = True,
    init_wave_modules: bool = True,
):
    """
    Load complete FLUXHybrid model from .flx archive.
    
    This reconstructs the full model with all trained weights.
    If wave modules are missing (v1.0-beta), optionally initializes fresh.
    
    Args:
        path: Path to .flx file
        device: Target device
        verbose: Print loading progress
        auto_download: Download from HF Hub if missing
        init_wave_modules: Initialize fresh wave modules if missing
    
    Returns:
        FLUXHybrid instance ready for inference
    """
    from flux_hybrid import FLUXHybrid
    
    path = Path(path)
    if verbose:
        print(f"\n  Loading FLUXHybrid from {path.name}...")
    
    # Load raw archive (auto-downloads if missing)
    flx = load_flx(path, device='cpu', auto_download=auto_download)
    version = flx.get('version', '1.0-beta')
    
    if verbose:
        print(f"  ✓ Format: FLUX v{version}")
    
    # Build FLUXHybrid 
    model = FLUXHybrid.from_flx_dict(
        flx=flx,
        device=device,
        verbose=verbose,
        init_wave_modules=init_wave_modules,
    )
    
    if verbose:
        stats = model.get_stats()
        print(f"\n  ═══ FLUXHybrid loaded: {stats['total_params']:,} params ═══")
        print(f"  Byte-mode: ✓ (WaveDecoder)")
        print(f"  Wave-mode: {'✓' if model.wave_mode_available else '✗ (fresh init)'}")
        print(f"  Task Router: ✓")
    
    return model


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("FLX Loader v2 (Phase 10) — Testing")
    
    for flx_path in [DEFAULT_FLX_PATH, FALLBACK_FLX_PATH]:
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
            print(f"\n  ℹ Not found: {flx_path}")
    
    print("\n  Note: Will auto-download from HuggingFace Hub when needed")
