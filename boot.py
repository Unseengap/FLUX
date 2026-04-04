#!/usr/bin/env python3
"""
FLUX Hub Boot Script

Main entry point for booting FLUX on Memory Fabric hardware.
Uses CustomTkinter UI with brand sound and real model loading.

Usage:
    python boot.py [--model PATH] [--device DEVICE]
"""

import sys
import os
import argparse
import threading
from pathlib import Path

# Add FLUX to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description="FLUX Hub Boot")
    parser.add_argument('--model', type=str, 
                       default='checkpoints/Flux-Apex-V1.flx',
                       help='Path to .flx model file')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device for inference')
    parser.add_argument('--no-ui', action='store_true',
                       help='Skip UI, boot in terminal only')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo sequence without real model')
    args = parser.parse_args()
    
    if args.demo:
        from flux_hub.boot_ui import demo_boot_sequence
        demo_boot_sequence()
        return
    
    if args.no_ui:
        boot_terminal(args.model, args.device)
    else:
        boot_with_ui(args.model, args.device)


def get_device(preference: str = 'auto') -> str:
    """Determine best available device."""
    import torch
    
    if preference != 'auto':
        return preference
        
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def get_system_info() -> dict:
    """Gather system information."""
    import torch
    
    info = {
        'gpu': 'Not detected',
        'gpu_vram': '0 GB',
        'ram': '0 GB',
        'storage': '0 GB',
    }
    
    # GPU
    if torch.cuda.is_available():
        info['gpu'] = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        info['gpu_vram'] = f"{vram:.1f} GB"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info['gpu'] = "Apple Metal"
        info['gpu_vram'] = "Shared"
    
    # RAM (approximate)
    try:
        import psutil
        ram = psutil.virtual_memory().total / 1e9
        info['ram'] = f"{ram:.1f} GB"
    except ImportError:
        pass
    
    # Storage
    try:
        import shutil
        total, used, free = shutil.disk_usage(ROOT)
        info['storage'] = f"{free / 1e9:.1f} GB free"
    except:
        pass
    
    return info


def boot_terminal(model_path: str, device: str):
    """Boot FLUX in terminal mode (no UI)."""
    import torch
    from bootstrap import wake_up
    
    print("=" * 60)
    print("FLUX MEMORY FABRIC — Terminal Boot")
    print("=" * 60)
    
    # Device
    device = get_device(device)
    print(f"\n✓ Device: {device}")
    
    # System info
    info = get_system_info()
    print(f"✓ GPU: {info['gpu']} ({info['gpu_vram']})")
    print(f"✓ RAM: {info['ram']}")
    print(f"✓ Storage: {info['storage']}")
    
    # Model
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"\n✗ Model not found: {model_path}")
        print("  Download from HuggingFace:")
        print("  huggingface-cli download UnseenGAP/FLUX checkpoints/Flux-Apex-V1.flx --local-dir .")
        sys.exit(1)
    
    print(f"\n● Loading model: {model_path.name}...")
    print(f"  Size: {model_path.stat().st_size / 1e9:.2f} GB")
    
    # Bootstrap
    print("\n● Bootstrapping...")
    result = wake_up(str(model_path), device=device, verbose=True)
    
    print(f"\n✓ FLUX Ready!")
    print(f"  Version: {result['version']}")
    print(f"  Modules: {len(result['modules'])}")
    print("=" * 60)
    
    return result


def boot_with_ui(model_path: str, device: str):
    """Boot FLUX with CustomTkinter UI."""
    import torch
    from flux_hub.boot_ui import FluxBootUI, BootConfig
    
    model_path = Path(model_path)
    device = get_device(device)
    boot_result = {}
    
    # Create UI
    ui = FluxBootUI()
    
    def run_boot():
        """Run boot sequence in background thread."""
        try:
            # Phase 0: Hardware init
            ui.root.after(0, ui.phase_hardware_init)
            
            # Phase 1: System check
            info = get_system_info()
            ui.root.after(1000, lambda: ui.phase_system_check(
                gpu_info=f"{info['gpu'][:20]}... ({info['gpu_vram']})" if len(info['gpu']) > 20 else f"{info['gpu']} ({info['gpu_vram']})",
                ram_info=info['ram'],
                storage_info=info['storage']
            ))
            
            # Wait for system check animation
            import time
            time.sleep(3)
            
            # Phase 2: Check model
            if not model_path.exists():
                ui.root.after(0, lambda: ui.show_error(
                    "Model not found",
                    f"Expected: {model_path}"
                ))
                return
            
            ui.root.after(0, lambda: ui.phase_model_loading(str(model_path)))
            time.sleep(2)
            
            # Phase 3: Bootstrap
            ui.root.after(0, ui.phase_bootstrap)
            time.sleep(1)
            
            # Actually load the model
            from bootstrap import wake_up
            
            def on_module(name, idx, total):
                ui.root.after(0, lambda: ui.update_module(name, idx, total))
            
            result = wake_up(
                str(model_path),
                device=device,
                verbose=False,
            )
            
            # Update module count
            total = len(result.get('modules', {}))
            ui.root.after(0, lambda: ui.update_module("complete", total, total))
            
            boot_result['result'] = result
            
            # Phase 4: Ready
            ui.root.after(500, ui.phase_ready)
            
        except Exception as e:
            ui.root.after(0, lambda: ui.show_error(str(e)))
            import traceback
            traceback.print_exc()
    
    # Start boot in background
    boot_thread = threading.Thread(target=run_boot, daemon=True)
    
    def on_ready():
        """Called when boot is complete."""
        print("\n✓ FLUX Boot complete!")
        result = boot_result.get('result', {})
        print(f"  Version: {result.get('version', 'unknown')}")
        print(f"  Modules: {len(result.get('modules', {}))}")
        
        # Launch dashboard after brief delay
        def launch():
            ui.close()
            from flux_hub.dashboard_ui import FluxDashboard
            dashboard = FluxDashboard()
            dashboard.run()
        
        ui.root.after(2000, launch)
    
    ui.on_ready(on_ready)
    
    # Start boot after window shows
    ui.root.after(500, boot_thread.start)
    
    # Run UI
    ui.run()
    
    return boot_result.get('result')


if __name__ == "__main__":
    main()
