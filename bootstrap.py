"""
FLUX Bootstrap — Self-extractor for wake-from-.flx capability.

This module enables FLUX to "wake up" from a .flx file by extracting
and loading the embedded runtime code without needing the original
source files on disk.

Usage:
    from bootstrap import wake_up
    result = wake_up('checkpoints/Flux-Apex-V1.flx')
"""

from __future__ import annotations

import sys
import gzip
import base64
import types
from pathlib import Path
from typing import Dict, Any, Optional


def decompress_source(compressed: str) -> str:
    """Decompress base64+gzip encoded source code."""
    decoded = base64.b64decode(compressed.encode('ascii'))
    return gzip.decompress(decoded).decode('utf-8')


def load_module_from_source(name: str, source: str, package: Optional[str] = None) -> types.ModuleType:
    """Create a module from source code string."""
    module = types.ModuleType(name)
    module.__file__ = f"<embedded:{name}>"
    module.__loader__ = None
    module.__package__ = package
    
    try:
        code = compile(source, f"<embedded:{name}>", 'exec')
        exec(code, module.__dict__)
    except Exception as e:
        # Store error but don't fail - allow partial loading
        module.__load_error__ = str(e)
    
    return module


def extract_runtime(flx: Dict[str, Any], verbose: bool = False) -> Dict[str, types.ModuleType]:
    """
    Extract and load all embedded modules from a .flx file.
    
    Args:
        flx: Loaded .flx dictionary (from torch.load)
        verbose: Print loading progress
        
    Returns:
        Dictionary mapping module paths to loaded module objects
    """
    runtime = flx.get('runtime', {})
    code_bundle = runtime.get('code', {})
    
    if not code_bundle:
        if verbose:
            print("⚠ No embedded runtime code found")
        return {}
    
    modules = {}
    errors = []
    
    # Create package hierarchy first
    packages = set()
    for path in code_bundle.keys():
        parts = path.replace('.py', '').split('/')
        for i in range(1, len(parts)):
            pkg = '.'.join(parts[:i])
            packages.add(pkg)
    
    # Register packages in sys.modules
    for pkg in sorted(packages):
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__file__ = f"<embedded:{pkg}>"
            mod.__path__ = []
            mod.__package__ = pkg
            sys.modules[pkg] = mod
    
    # Load each module
    for path, compressed in code_bundle.items():
        # Convert path to module name: phases/phase1/cse.py -> phases.phase1.cse
        module_name = path.replace('/', '.').replace('.py', '')
        
        # Determine package
        parts = module_name.rsplit('.', 1)
        package = parts[0] if len(parts) > 1 else None
        
        try:
            source = decompress_source(compressed)
            module = load_module_from_source(module_name, source, package)
            
            # Register in sys.modules
            sys.modules[module_name] = module
            modules[path] = module
            
            # Also register in parent package
            if package and package in sys.modules:
                attr_name = parts[1] if len(parts) > 1 else module_name
                setattr(sys.modules[package], attr_name, module)
            
            if verbose:
                if hasattr(module, '__load_error__'):
                    print(f"⚠ {path} (loaded with error: {module.__load_error__})")
                else:
                    print(f"✓ {path}")
                    
        except Exception as e:
            errors.append((path, str(e)))
            if verbose:
                print(f"✗ {path}: {e}")
    
    if verbose:
        print(f"\nLoaded {len(modules)}/{len(code_bundle)} modules")
        if errors:
            print(f"Errors: {len(errors)}")
    
    return modules


def wake_up(
    flx_path: str,
    device: str = 'cpu',
    extract_code: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Wake up FLUX from a .flx file.
    
    This is the main entry point for bootstrapping FLUX from its
    self-contained .flx archive.
    
    Args:
        flx_path: Path to the .flx file
        device: Device to load tensors to ('cpu', 'cuda', 'mps')
        extract_code: Whether to extract and load embedded code
        verbose: Print loading progress
        
    Returns:
        Dictionary with:
            'flx': The loaded model dictionary
            'modules': Dict of extracted modules (if extract_code=True)
            'version': Model version string
            'components': Dict of component enable flags
    """
    import torch
    
    path = Path(flx_path)
    if not path.exists():
        raise FileNotFoundError(f"FLX file not found: {flx_path}")
    
    if verbose:
        print(f"Loading {path.name}...")
    
    # Load the archive
    flx = torch.load(str(path), map_location=device, weights_only=False)
    
    result = {
        'flx': flx,
        'version': flx.get('version', 'unknown'),
        'components': flx.get('components', {}),
        'modules': {},
    }
    
    if verbose:
        print(f"Version: {result['version']}")
        print(f"Components: {len(result['components'])}")
    
    # Extract embedded code
    if extract_code:
        if verbose:
            print("\nExtracting embedded runtime...")
        result['modules'] = extract_runtime(flx, verbose=verbose)
    
    return result


def get_component(flx: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    """Get a component from the loaded FLX dictionary."""
    return flx.get(name)


def list_components(flx: Dict[str, Any]) -> list:
    """List all top-level components in the FLX."""
    skip = {'format', 'version', 'phase', 'timestamp', 'metadata', 
            'runtime_config', 'components', 'can_continue_learning'}
    return [k for k in flx.keys() if k not in skip]


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='FLUX Bootstrap')
    parser.add_argument('flx_path', help='Path to .flx file')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda/mps)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--no-extract', action='store_true', help='Skip code extraction')
    
    args = parser.parse_args()
    
    result = wake_up(
        args.flx_path,
        device=args.device,
        extract_code=not args.no_extract,
        verbose=args.verbose,
    )
    
    print(f"\n{'='*60}")
    print("FLUX BOOTSTRAP COMPLETE")
    print(f"{'='*60}")
    print(f"Version: {result['version']}")
    print(f"Components: {list_components(result['flx'])}")
    print(f"Modules loaded: {len(result['modules'])}")
