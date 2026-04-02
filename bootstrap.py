"""
FLUX Bootstrap — Wake up from .flx file alone.

This module enables FLUX to bootstrap its entire runtime from a single .flx file,
making it truly self-contained and autonomous.

Usage (command line):
    python bootstrap.py checkpoints/Flux-Apex-V1.flx
    
Usage (embedded in .flx):
    python -c "import torch; exec(torch.load('Flux-Apex-V1.flx')['runtime']['bootstrap'])"
    
Usage (Python):
    from bootstrap import wake_up, extract_runtime, register_modules
    
    flux = wake_up('checkpoints/Flux-Apex-V1.flx')
    # flux['orchestrator'] — ready to use orchestrator
    # flux['agent'] — ready to use unified agent
    # flux['flx'] — raw .flx state
    # flux['modules'] — all registered modules
"""

import sys
import gc
import types
import importlib.util
import gzip
import base64
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple


# ─────────────────────────────────────────────
# Module Registration
# ─────────────────────────────────────────────

def create_module_from_source(name: str, source: str, package: Optional[str] = None) -> types.ModuleType:
    """
    Create a Python module from source code string.
    
    Args:
        name: Full module name (e.g., 'phases.phase1.cse')
        source: Python source code
        package: Parent package name (e.g., 'phases.phase1')
        
    Returns:
        Compiled module object
    """
    # Create module spec
    spec = importlib.util.spec_from_loader(name, loader=None, origin=f"<flx:{name}>")
    module = types.ModuleType(name)
    module.__spec__ = spec
    module.__file__ = f"<flx:{name}>"
    module.__loader__ = None
    
    if package:
        module.__package__ = package
        
    # Compile and execute
    try:
        code = compile(source, f"<flx:{name}>", 'exec')
        exec(code, module.__dict__)
    except Exception as e:
        print(f"  ⚠ Error compiling {name}: {e}")
        return None
        
    return module


def register_modules(code_bundle: Dict[str, str], verbose: bool = True) -> Dict[str, types.ModuleType]:
    """
    Register all embedded Python modules in sys.modules.
    
    Args:
        code_bundle: Dict mapping file paths to source code
        verbose: Print progress
        
    Returns:
        Dict of registered module objects
    """
    modules = {}
    
    # Sort files by dependency order
    # Root modules first, then phases in order
    def sort_key(path: str) -> Tuple[int, int, str]:
        if not path.startswith('phases/'):
            return (0, 0, path)
            
        # Extract phase number
        parts = path.split('/')
        if len(parts) >= 2:
            phase_name = parts[1]
            # phase1 → 1, phase1_5 → 1.5, phase_unified → 100
            if phase_name.startswith('phase'):
                phase_part = phase_name.replace('phase', '').replace('_', '.')
                if phase_part == 'unified':
                    phase_num = 100
                elif phase_part == 'orchestrator':
                    phase_num = 99
                elif phase_part == 'vlm_native':
                    phase_num = 101
                elif phase_part == 'voice':
                    phase_num = 102
                else:
                    try:
                        phase_num = float(phase_part.replace('.', '', 1) if phase_part.count('.') > 1 else phase_part)
                    except ValueError:
                        phase_num = 50
            else:
                phase_num = 50
        else:
            phase_num = 50
            
        # __init__.py should come first within each phase
        is_init = 1 if '__init__' in path else 2
        
        return (1, phase_num, path)
    
    file_order = sorted(code_bundle.keys(), key=sort_key)
    
    # First pass: Create package structures
    packages_created = set()
    for file_path in file_order:
        # Skip non-Python files
        if not file_path.endswith('.py'):
            continue
            
        # Get module name from file path
        # e.g., 'phases/phase1/cse.py' → 'phases.phase1.cse'
        module_name = file_path.replace('/', '.').replace('.py', '')
        
        # Create parent packages
        parts = module_name.split('.')
        for i in range(1, len(parts)):
            package_name = '.'.join(parts[:i])
            if package_name not in packages_created and package_name not in sys.modules:
                pkg = types.ModuleType(package_name)
                pkg.__path__ = []
                pkg.__package__ = package_name
                pkg.__file__ = f"<flx:{package_name}>"
                sys.modules[package_name] = pkg
                packages_created.add(package_name)
                if verbose:
                    print(f"  📦 Created package: {package_name}")
    
    # Second pass: Register actual modules
    for file_path in file_order:
        if not file_path.endswith('.py'):
            continue
            
        source = code_bundle[file_path]
        module_name = file_path.replace('/', '.').replace('.py', '')
        
        # Determine package
        parts = module_name.split('.')
        package = '.'.join(parts[:-1]) if len(parts) > 1 else None
        
        # Create and register module
        module = create_module_from_source(module_name, source, package)
        if module:
            sys.modules[module_name] = module
            modules[module_name] = module
            
            # Also add to parent package
            if package and package in sys.modules:
                setattr(sys.modules[package], parts[-1], module)
                
            if verbose:
                print(f"  ✓ Registered: {module_name}")
                
    return modules


def decompress_code(compressed: str) -> str:
    """Decompress gzip+base64 encoded source code."""
    try:
        return gzip.decompress(base64.b64decode(compressed)).decode('utf-8')
    except Exception:
        # Return as-is if not compressed
        return compressed


# ─────────────────────────────────────────────
# Runtime Extraction
# ─────────────────────────────────────────────

def extract_runtime(flx: Dict[str, Any], verbose: bool = True) -> Dict[str, str]:
    """
    Extract the embedded runtime code from a .flx state.
    
    Args:
        flx: Loaded .flx file state
        verbose: Print progress
        
    Returns:
        Dict mapping file paths to source code
    """
    if 'runtime' not in flx:
        raise ValueError(
            "This .flx file has no embedded runtime. Cannot bootstrap.\n"
            "To add runtime, use: notebooks/flux_codebase_embed.ipynb"
        )
    
    runtime = flx['runtime']
    code_bundle = runtime.get('code', {})
    
    if verbose:
        print(f"  Version: {runtime.get('version', 'unknown')}")
        print(f"  Embedded at: {runtime.get('embedded_at', 'unknown')}")
        print(f"  Files: {len(code_bundle)}")
    
    # Decompress if needed
    decompressed = {}
    for path, source in code_bundle.items():
        if isinstance(source, bytes) or (isinstance(source, str) and source.startswith('H4sI')):
            decompressed[path] = decompress_code(source)
        else:
            decompressed[path] = source
            
    return decompressed


# ─────────────────────────────────────────────
# Main Wake-Up Function
# ─────────────────────────────────────────────

def wake_up(
    flx_path: str = 'checkpoints/Flux-Apex-V1.flx',
    device: str = 'auto',
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Bootstrap FLUX from a .flx file.
    
    This function:
    1. Loads the .flx archive
    2. Extracts all embedded Python modules
    3. Registers them in sys.modules
    4. Initializes the orchestrator and agent
    5. Returns a ready-to-use FLUX instance
    
    Args:
        flx_path: Path to the .flx file
        device: Device to use ('auto', 'cuda', 'mps', 'cpu')
        verbose: Print progress
        
    Returns:
        Dict with keys: flx, orchestrator, agent, modules, path, device
    """
    import torch
    
    if verbose:
        print(f"\n🌊 FLUX Bootstrap")
        print(f"{'─' * 50}")
        print(f"Loading {flx_path}...")
    
    # Auto-detect device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    if verbose:
        print(f"  Device: {device}")
    
    # Load archive
    path = Path(flx_path)
    if not path.exists():
        raise FileNotFoundError(f"FLUX archive not found: {flx_path}")
        
    flx = torch.load(str(path), map_location='cpu', weights_only=False)
    
    # Verify format
    if flx.get('format') != 'FLUX':
        raise ValueError(f"Invalid format: {flx.get('format')}. Expected 'FLUX'.")
    
    if verbose:
        print(f"  Format: {flx.get('format')}")
        print(f"  Model Version: {flx.get('version', 'unknown')}")
    
    # Extract and register runtime
    try:
        code_bundle = extract_runtime(flx, verbose)
        modules = register_modules(code_bundle, verbose)
        print(f"  ✓ Loaded {len(modules)} modules")
    except ValueError as e:
        if verbose:
            print(f"  ⚠ No embedded runtime: {e}")
        modules = {}
    
    # Initialize orchestrator
    orchestrator = None
    try:
        from phases.phase_orchestrator import FLUXOrchestrator
        orchestrator = FLUXOrchestrator(flx, device=device)
        if verbose:
            print("  ✓ Orchestrator initialized")
    except ImportError as e:
        if verbose:
            print(f"  ⚠ Orchestrator not available: {e}")
    except Exception as e:
        if verbose:
            print(f"  ⚠ Orchestrator init failed: {e}")
    
    # Initialize agent
    agent = None
    try:
        from phases.phase_unified import FLUXUnifiedAgent, create_unified_agent
        agent = create_unified_agent(flx, device=device)
        if verbose:
            print("  ✓ Agent initialized")
    except ImportError as e:
        if verbose:
            print(f"  ⚠ Agent not available: {e}")
    except Exception as e:
        if verbose:
            print(f"  ⚠ Agent init failed: {e}")
    
    # Cleanup
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    if verbose:
        print(f"{'─' * 50}")
        print("✓ FLUX is awake.\n")
    
    return {
        'flx': flx,
        'orchestrator': orchestrator,
        'agent': agent,
        'modules': modules,
        'path': str(path),
        'device': device,
    }


# ─────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────

def list_embedded_files(flx_path: str) -> List[str]:
    """List all files embedded in a .flx archive."""
    import torch
    flx = torch.load(flx_path, map_location='cpu', weights_only=False)
    
    if 'runtime' not in flx:
        return []
        
    return list(flx['runtime'].get('code', {}).keys())


def get_runtime_info(flx_path: str) -> Dict[str, Any]:
    """Get metadata about the embedded runtime."""
    import torch
    flx = torch.load(flx_path, map_location='cpu', weights_only=False)
    
    if 'runtime' not in flx:
        return {'embedded': False}
        
    runtime = flx['runtime']
    code = runtime.get('code', {})
    
    return {
        'embedded': True,
        'version': runtime.get('version', 'unknown'),
        'embedded_at': runtime.get('embedded_at', 'unknown'),
        'total_files': len(code),
        'total_lines': sum(len(s.split('\n')) for s in code.values() if isinstance(s, str)),
        'has_bootstrap': 'bootstrap' in runtime,
        'files': list(code.keys()),
    }


def verify_bootstrap(flx_path: str) -> bool:
    """Verify that a .flx file can bootstrap successfully."""
    try:
        flux = wake_up(flx_path, verbose=False)
        has_runtime = len(flux['modules']) > 0
        return has_runtime
    except Exception as e:
        print(f"Bootstrap verification failed: {e}")
        return False


# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────

def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Bootstrap FLUX from a .flx file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python bootstrap.py checkpoints/Flux-Apex-V1.flx
    python bootstrap.py Flux-Apex-V1.flx --device cuda
    python bootstrap.py Flux-Apex-V1.flx --list  # List embedded files
    python bootstrap.py Flux-Apex-V1.flx --info  # Show runtime info
        """
    )
    
    parser.add_argument('flx_path', nargs='?', default='checkpoints/Flux-Apex-V1.flx',
                        help='Path to .flx file')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use')
    parser.add_argument('--list', action='store_true',
                        help='List embedded files and exit')
    parser.add_argument('--info', action='store_true',
                        help='Show runtime info and exit')
    parser.add_argument('--verify', action='store_true',
                        help='Verify bootstrap capability and exit')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    if args.list:
        files = list_embedded_files(args.flx_path)
        if files:
            print(f"Embedded files ({len(files)}):")
            for f in sorted(files):
                print(f"  {f}")
        else:
            print("No embedded runtime found.")
        return
        
    if args.info:
        info = get_runtime_info(args.flx_path)
        print("Runtime Info:")
        for k, v in info.items():
            if k != 'files':
                print(f"  {k}: {v}")
        return
        
    if args.verify:
        success = verify_bootstrap(args.flx_path)
        print(f"Bootstrap verification: {'✓ PASS' if success else '✗ FAIL'}")
        sys.exit(0 if success else 1)
    
    # Full wake-up
    flux = wake_up(args.flx_path, device=args.device, verbose=not args.quiet)
    
    # Interactive mode hint
    print("FLUX components available:")
    print(f"  flux['flx']          — Raw .flx state")
    print(f"  flux['orchestrator'] — {'Ready' if flux['orchestrator'] else 'Not available'}")
    print(f"  flux['agent']        — {'Ready' if flux['agent'] else 'Not available'}")
    print(f"  flux['modules']      — {len(flux['modules'])} registered modules")
    print(f"  flux['device']       — {flux['device']}")
    
    return flux


if __name__ == '__main__':
    main()
