"""
Embed Orchestration into .flx — Make FLUX self-describing.

This script adds the orchestration capabilities (tool definitions,
system prompt) directly into the .flx file so the model knows
what it can do.

Usage:
    python embed_orchestration.py
    
This updates Flux-Apex-V1.flx from v5.0-vlm-embedded to v5.1-orchestrated.
"""

import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import torch

try:
    from .tool_registry import FLUX_TOOLS, ToolCategory
    from .system_prompt import FLUX_SYSTEM_PROMPT, FLUX_SYSTEM_PROMPT_SHORT
except ImportError:
    from tool_registry import FLUX_TOOLS, ToolCategory
    from system_prompt import FLUX_SYSTEM_PROMPT, FLUX_SYSTEM_PROMPT_SHORT


def serialize_tool_definitions() -> dict:
    """Convert tool definitions to serializable format for .flx."""
    tools = {}
    
    for name, tool in FLUX_TOOLS.items():
        tools[name] = {
            'name': tool.name,
            'description': tool.description,
            'category': tool.category.value,
            'component_path': tool.component_path,
            'method_name': tool.method_name,
            'params': tool.params,
            'returns': tool.returns,
            'example': tool.example,
            'requires_wave_dim': tool.requires_wave_dim,
        }
    
    return tools


def build_orchestration_config() -> dict:
    """Build the orchestration section for .flx."""
    return {
        # Version info
        'version': '1.0',
        'enabled': True,
        
        # Tool definitions (serialized)
        'tools': serialize_tool_definitions(),
        
        # Tool categories
        'tool_categories': {
            cat.value: [t.name for t in FLUX_TOOLS.values() if t.category == cat]
            for cat in ToolCategory
        },
        
        # System prompts
        'system_prompt': FLUX_SYSTEM_PROMPT,
        'system_prompt_short': FLUX_SYSTEM_PROMPT_SHORT,
        
        # Runtime settings
        'settings': {
            'max_iterations': 5,
            'tool_timeout_ms': 5000,
            'context_injection': True,
            'verbose_logging': False,
        },
        
        # Tool call format
        'format': {
            'tool_tag': '<tool>',
            'params_tag': '<params>',
            'variables': ['$LAST_WAVE', '$LAST_GRID', '$INPUT_GRID', '$INPUT_IMAGE', '$CONTEXT'],
        },
        
        # Capabilities summary (for quick discovery)
        'capabilities': {
            'tool_count': len(FLUX_TOOLS),
            'can_encode_text': True,
            'can_encode_grid': True,
            'can_query_field': True,
            'can_reason_causally': True,
            'can_explore': True,
            'can_remember': True,
            'can_decode_grid': True,
        },
    }


def embed_orchestration(flx_path: str, output_path: str = None):
    """
    Embed orchestration capabilities into .flx file.
    
    Args:
        flx_path: Path to input .flx file
        output_path: Path for output (defaults to overwrite input)
    """
    if output_path is None:
        output_path = flx_path
    
    print(f"\n  Loading {flx_path}...")
    model = torch.load(flx_path, map_location='cpu', weights_only=False)
    
    current_version = model.get('version', 'unknown')
    print(f"  Current version: {current_version}")
    
    # Add orchestration section
    print(f"\n  Building orchestration config...")
    model['orchestration'] = build_orchestration_config()
    
    # Update version
    model['version'] = '5.1-orchestrated'
    model['phase'] = 'phase_orchestrator'
    
    # Update components flags
    if 'components' not in model:
        model['components'] = {}
    model['components']['orchestration'] = True
    model['components']['tool_use'] = True
    
    # Update runtime config
    if 'runtime_config' not in model:
        model['runtime_config'] = {}
    model['runtime_config']['orchestration'] = {
        'enabled': True,
        'max_tool_iterations': 5,
        'context_injection': True,
    }
    
    # Update metadata
    if 'metadata' not in model:
        model['metadata'] = {}
    model['metadata']['last_modified'] = datetime.now().isoformat()
    model['metadata']['modified_components'] = ['orchestration']
    
    if 'capabilities' not in model['metadata']:
        model['metadata']['capabilities'] = []
    if 'tool_use' not in model['metadata']['capabilities']:
        model['metadata']['capabilities'].append('tool_use')
    if 'multi_step_reasoning' not in model['metadata']['capabilities']:
        model['metadata']['capabilities'].append('multi_step_reasoning')
    if 'self_describing' not in model['metadata']['capabilities']:
        model['metadata']['capabilities'].append('self_describing')
    
    # Save
    print(f"\n  Saving to {output_path}...")
    torch.save(model, output_path, pickle_protocol=4)
    
    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"  ✓ Saved: {size_mb:.1f} MB")
    
    # Verify
    print(f"\n  Verifying embedded orchestration...")
    verify = torch.load(output_path, map_location='cpu', weights_only=False)
    
    assert 'orchestration' in verify, "Orchestration section missing!"
    assert verify['version'] == '5.1-orchestrated', "Version not updated!"
    
    orch = verify['orchestration']
    print(f"    ✓ Tools: {len(orch['tools'])}")
    print(f"    ✓ System prompt: {len(orch['system_prompt'])} chars")
    print(f"    ✓ Categories: {list(orch['tool_categories'].keys())}")
    print(f"    ✓ Capabilities: {orch['capabilities']}")
    
    print(f"\n  ═══ EMBEDDING COMPLETE ═══")
    print(f"  Version: {current_version} → 5.1-orchestrated")
    print(f"  The model now knows its own cognitive tools.")
    
    return verify


def show_embedded_tools(flx_path: str):
    """Show the tools embedded in a .flx file."""
    model = torch.load(flx_path, map_location='cpu', weights_only=False)
    
    if 'orchestration' not in model:
        print(f"  ✗ No orchestration section in {flx_path}")
        return
    
    orch = model['orchestration']
    
    print(f"\n  ═══ EMBEDDED TOOLS in {Path(flx_path).name} ═══")
    print(f"  Version: {model.get('version')}")
    print(f"  Orchestration version: {orch.get('version')}")
    print(f"\n  Tools ({len(orch['tools'])}):\n")
    
    for cat, tools in orch['tool_categories'].items():
        print(f"    {cat.upper()}:")
        for tool_name in tools:
            tool = orch['tools'][tool_name]
            print(f"      - {tool_name}: {tool['description'][:50]}...")
        print()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Embed orchestration into .flx')
    parser.add_argument('--input', '-i', type=str, 
                        default='checkpoints/Flux-Apex-V1.flx',
                        help='Input .flx file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output .flx file (default: overwrite input)')
    parser.add_argument('--show', '-s', action='store_true',
                        help='Show embedded tools instead of embedding')
    
    args = parser.parse_args()
    
    flx_path = ROOT / args.input
    
    if args.show:
        show_embedded_tools(str(flx_path))
    else:
        output = str(ROOT / args.output) if args.output else str(flx_path)
        embed_orchestration(str(flx_path), output)
