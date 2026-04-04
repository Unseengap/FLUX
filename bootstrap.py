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
import importlib
import importlib.abc
import importlib.machinery
from pathlib import Path
from typing import Dict, Any, Optional


# ─────────────────────────────────────────────
# Embedded Module Import Hook
# ─────────────────────────────────────────────

class EmbeddedModuleFinder(importlib.abc.MetaPathFinder):
    """Custom finder for embedded modules."""
    
    def __init__(self, code_bundle: Dict[str, str]):
        # Map module names to compressed source
        self.modules = {}
        self.packages = set()
        
        for path, compressed in code_bundle.items():
            # Convert path to module name
            module_name = path.replace('/', '.').replace('.py', '')
            
            # Handle __init__.py specially - it defines the package
            if module_name.endswith('.__init__'):
                pkg_name = module_name[:-9]  # Remove '.__init__'
                self.modules[pkg_name] = compressed
                self.packages.add(pkg_name)
            else:
                self.modules[module_name] = compressed
            
            # Also register parent packages
            parts = module_name.split('.')
            for i in range(1, len(parts)):
                pkg = '.'.join(parts[:i])
                self.packages.add(pkg)
    
    def find_spec(self, fullname, path, target=None):
        if fullname in self.modules:
            is_package = fullname in self.packages
            return importlib.machinery.ModuleSpec(
                fullname,
                EmbeddedModuleLoader(self.modules[fullname], is_package),
                is_package=is_package,
            )
        # Check if it's a package without __init__.py (namespace package)
        if fullname in self.packages:
            return importlib.machinery.ModuleSpec(
                fullname,
                EmbeddedPackageLoader(),
                is_package=True,
            )
        return None


class EmbeddedModuleLoader(importlib.abc.Loader):
    """Custom loader for embedded modules."""
    
    def __init__(self, compressed_source: str, is_package: bool = False):
        self.compressed_source = compressed_source
        self.is_package = is_package
    
    def create_module(self, spec):
        # Create module with proper attributes pre-set
        module = types.ModuleType(spec.name)
        if self.is_package:
            module.__file__ = f"<embedded:{spec.name}/__init__.py>"
            module.__path__ = [f"<embedded:{spec.name}>"]
        else:
            module.__file__ = f"<embedded:{spec.name.replace('.', '/')}.py>"
        module.__loader__ = self
        module.__package__ = spec.name if self.is_package else spec.name.rsplit('.', 1)[0] if '.' in spec.name else ''
        module.__spec__ = spec
        return module
    
    def exec_module(self, module):
        source = decompress_source(self.compressed_source)
        code = compile(source, module.__file__, 'exec')
        exec(code, module.__dict__)


class EmbeddedPackageLoader(importlib.abc.Loader):
    """Loader for embedded packages (directories)."""
    
    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__file__ = f"<embedded:{spec.name}/__init__.py>"
        module.__loader__ = self
        module.__package__ = spec.name
        module.__path__ = [f"<embedded:{spec.name}>"]
        module.__spec__ = spec
        return module
    
    def exec_module(self, module):
        pass  # Packages don't need code execution


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
    
    Uses a custom import hook to handle inter-module dependencies.
    
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
    
    # Install custom import hook
    finder = EmbeddedModuleFinder(code_bundle)
    sys.meta_path.insert(0, finder)
    
    modules = {}
    
    # Load modules in dependency order (leaf modules first)
    # Sort by depth (fewer dots = higher level = load last)
    sorted_paths = sorted(code_bundle.keys(), key=lambda p: p.count('/'))
    
    for path in sorted_paths:
        module_name = path.replace('/', '.').replace('.py', '')
        
        try:
            # Use the standard import machinery (now with our hook)
            module = importlib.import_module(module_name)
            modules[path] = module
            
            if verbose:
                print(f"✓ {path}")
                
        except Exception as e:
            # Create placeholder module with error
            module = types.ModuleType(module_name)
            module.__load_error__ = str(e)
            sys.modules[module_name] = module
            modules[path] = module
            
            if verbose:
                print(f"⚠ {path} (error: {e})")
    
    if verbose:
        success = sum(1 for m in modules.values() if not hasattr(m, '__load_error__'))
        print(f"\nLoaded {success}/{len(code_bundle)} modules successfully")
    
    return modules


# Import helper for embedded modules to use
def import_embedded(module_name: str):
    """Import a module that may be embedded or on disk."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    return importlib.import_module(module_name)


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
