"""
FLUX Model Inspector — Dynamic structure analysis for any .flx file.

Schema-free introspection that discovers components without assumptions.
Handles all .flx versions: v4.0, v5.0-voice, v6.0-autonomous, v7.x-fabric.

Usage:
    from flux_inspector import inspect_flx, generate_report
    
    # Full inspection
    report = inspect_flx('checkpoints/Flux-Apex-V1.flx')
    print(report['summary'])
    
    # Or via CLI
    python flux_inspector.py checkpoints/Flux-Apex-V1.flx
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
import json


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class ComponentInfo:
    """Information about a single component."""
    name: str
    component_type: str  # 'native', 'embedded_model', 'detection', 'onnx', 'config'
    parameters: int
    tensors: int
    size_bytes: int
    has_weights: bool
    has_config: bool
    lazy_load: Optional[bool] = None
    base_model: Optional[str] = None
    quantization: Optional[str] = None
    legacy: bool = False
    legacy_reason: Optional[str] = None
    extra: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


@dataclass
class InspectionReport:
    """Complete inspection report for a .flx file."""
    path: str
    file_size_bytes: int
    format: str
    version: str
    phase: str
    timestamp: Optional[str]
    can_continue_learning: bool
    
    top_level_keys: List[str]
    max_depth: int
    
    native_components: Dict[str, ComponentInfo]
    embedded_models: Dict[str, ComponentInfo]
    detection_models: Dict[str, ComponentInfo]
    
    total_parameters: int
    total_tensors: int
    
    component_flags: Dict[str, bool]
    runtime_config: Dict[str, Any]
    metadata: Dict[str, Any]
    
    legacy_components: List[str]
    removed_components: List[str]
    
    warnings: List[str]
    errors: List[str]


# ─────────────────────────────────────────────
# Core Inspection Functions
# ─────────────────────────────────────────────

def count_parameters(obj: Any) -> Tuple[int, int]:
    """
    Count parameters and tensors in any nested structure.
    
    Returns:
        (total_params, tensor_count)
    """
    if isinstance(obj, torch.Tensor):
        return obj.numel(), 1
    elif isinstance(obj, np.ndarray):
        return obj.size, 1
    elif isinstance(obj, dict):
        total_params = 0
        total_tensors = 0
        for v in obj.values():
            p, t = count_parameters(v)
            total_params += p
            total_tensors += t
        return total_params, total_tensors
    elif isinstance(obj, (list, tuple)):
        total_params = 0
        total_tensors = 0
        for item in obj:
            p, t = count_parameters(item)
            total_params += p
            total_tensors += t
        return total_params, total_tensors
    else:
        return 0, 0


def estimate_size_bytes(obj: Any) -> int:
    """Estimate memory size of nested structure in bytes."""
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    elif isinstance(obj, np.ndarray):
        return obj.nbytes
    elif isinstance(obj, dict):
        return sum(estimate_size_bytes(v) for v in obj.values())
    elif isinstance(obj, (list, tuple)):
        return sum(estimate_size_bytes(item) for item in obj)
    elif isinstance(obj, str):
        return len(obj.encode('utf-8'))
    elif isinstance(obj, (int, float, bool)):
        return 8
    else:
        return 0


def get_max_depth(obj: Any, current_depth: int = 0) -> int:
    """Get maximum nesting depth of a structure."""
    if isinstance(obj, dict):
        if not obj:
            return current_depth
        return max(get_max_depth(v, current_depth + 1) for v in obj.values())
    elif isinstance(obj, (list, tuple)):
        if not obj:
            return current_depth
        return max(get_max_depth(item, current_depth + 1) for item in obj)
    else:
        return current_depth


def detect_component_type(name: str, data: Any) -> str:
    """
    Detect the type of a component based on its structure.
    
    Returns one of: 'native', 'embedded_model', 'detection', 'onnx', 'config', 'metadata'
    """
    if not isinstance(data, dict):
        return 'config'
    
    # Check for ONNX models (InsightFace style)
    if 'onnx_models' in data or any(k.endswith('.onnx') for k in data.keys()):
        return 'onnx'
    
    # Check for embedded HuggingFace models
    if 'base_model' in data or 'weights' in data:
        # Detection models
        detection_keywords = ['face', 'object', 'depth', 'pose', 'speaker', 'diarization']
        if any(kw in name.lower() for kw in detection_keywords):
            return 'detection'
        return 'embedded_model'
    
    # Check for voice module (Qwen-Omni structure)
    if name == 'voice' and ('thinker' in data or 'talker' in data):
        return 'embedded_model'
    
    # Native FLUX components
    native_names = {
        'cse', 'field', 'memory', 'bridges', 'adapters', 'causal',
        'decoder', 'grid_to_wave', 'spatial_memory', 'causal_tracker',
        'rule_inducer', 'goal_planner', 'learned_rules'
    }
    if name in native_names:
        return 'native'
    
    # Check structure for native pattern (state_dict + config)
    if 'state_dict' in data and ('config' in data or len(data) <= 5):
        return 'native'
    
    # Config/metadata
    if name in {'runtime_config', 'metadata', 'components', 'orchestration'}:
        return 'config'
    
    # Default based on content
    if 'state_dict' in data:
        return 'native'
    
    return 'config'


def analyze_component(name: str, data: Any) -> ComponentInfo:
    """Analyze a single component and return info."""
    component_type = detect_component_type(name, data)
    
    params, tensors = count_parameters(data)
    size_bytes = estimate_size_bytes(data)
    
    has_weights = False
    has_config = False
    lazy_load = None
    base_model = None
    quantization = None
    legacy = False
    legacy_reason = None
    extra = {}
    
    if isinstance(data, dict):
        # Check for weights
        has_weights = bool(
            'state_dict' in data or 
            'weights' in data or 
            'thinker' in data or
            'onnx_models' in data or
            any(isinstance(v, (torch.Tensor, np.ndarray)) for v in data.values())
        )
        
        # Check for config
        has_config = 'config' in data
        
        # Embedded model specifics
        if component_type in ('embedded_model', 'detection', 'onnx'):
            lazy_load = data.get('lazy_load', True)
            base_model = data.get('base_model')
            quantization = data.get('quantization', 'unknown')
        
        # Legacy status
        legacy = data.get('legacy', False)
        legacy_reason = data.get('legacy_reason')
        
        # Voice module details
        if name == 'voice':
            extra['thinker_tensors'] = len(data.get('thinker', {}))
            extra['talker_tensors'] = len(data.get('talker', {}))
            extra['token2wav_tensors'] = len(data.get('token2wav', {}))
        
        # Field details
        if name == 'field' and 'state_dict' in data:
            sd = data['state_dict']
            if 'state' in sd and isinstance(sd['state'], torch.Tensor):
                extra['field_shape'] = list(sd['state'].shape)
        
        # Memory details
        if name == 'memory':
            for tier in ['working', 'episodic', 'semantic']:
                if tier in data:
                    extra[f'{tier}_present'] = True
        
        # ONNX model details
        if 'onnx_models' in data:
            extra['onnx_model_count'] = len(data['onnx_models'])
            extra['onnx_model_names'] = list(data['onnx_models'].keys())
    
    return ComponentInfo(
        name=name,
        component_type=component_type,
        parameters=params,
        tensors=tensors,
        size_bytes=size_bytes,
        has_weights=has_weights,
        has_config=has_config,
        lazy_load=lazy_load,
        base_model=base_model,
        quantization=quantization,
        legacy=legacy,
        legacy_reason=legacy_reason,
        extra=extra,
    )


def inspect_flx(path: Union[str, Path]) -> InspectionReport:
    """
    Perform complete inspection of a .flx file.
    
    Args:
        path: Path to .flx file
    
    Returns:
        InspectionReport with all discovered information
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    file_size = path.stat().st_size
    
    # Load with weights_only=False to get full structure
    raw = torch.load(str(path), map_location='cpu', weights_only=False)
    
    warnings = []
    errors = []
    
    # Basic metadata
    fmt = raw.get('format', 'UNKNOWN')
    if fmt != 'FLUX':
        warnings.append(f"Unexpected format: {fmt} (expected 'FLUX')")
    
    version = raw.get('version', 'unknown')
    phase = raw.get('phase', 'unknown')
    timestamp = raw.get('timestamp')
    can_continue = raw.get('can_continue_learning', True)
    
    top_level_keys = list(raw.keys())
    max_depth = get_max_depth(raw)
    
    # Component flags
    component_flags = raw.get('components', {})
    
    # Runtime config
    runtime_config = raw.get('runtime_config', {})
    
    # Metadata
    metadata = raw.get('metadata', {})
    
    # Removed components
    removed_components = raw.get('removed_components', [])
    
    # Analyze all components
    native_components = {}
    embedded_models = {}
    detection_models = {}
    legacy_list = []
    
    # Skip these keys (not components)
    skip_keys = {
        'format', 'version', 'phase', 'timestamp', 'can_continue_learning',
        'components', 'runtime_config', 'metadata', 'modified', 
        'modified_components', 'removed_components', 'state'
    }
    
    # Check for 'models' dict (v6.0+ schema)
    if 'models' in raw and isinstance(raw['models'], dict):
        for model_name, model_data in raw['models'].items():
            info = analyze_component(model_name, model_data)
            if info.component_type == 'detection' or info.component_type == 'onnx':
                detection_models[model_name] = info
            else:
                embedded_models[model_name] = info
            if info.legacy:
                legacy_list.append(model_name)
    
    # Analyze top-level components
    for key in top_level_keys:
        if key in skip_keys or key == 'models':
            continue
        
        data = raw[key]
        info = analyze_component(key, data)
        
        if info.component_type == 'native':
            native_components[key] = info
        elif info.component_type == 'embedded_model':
            embedded_models[key] = info
        elif info.component_type in ('detection', 'onnx'):
            detection_models[key] = info
        # Skip config types - they're not components
        
        if info.legacy:
            legacy_list.append(key)
    
    # Calculate totals
    total_params = sum(c.parameters for c in native_components.values())
    total_params += sum(c.parameters for c in embedded_models.values())
    total_params += sum(c.parameters for c in detection_models.values())
    
    total_tensors = sum(c.tensors for c in native_components.values())
    total_tensors += sum(c.tensors for c in embedded_models.values())
    total_tensors += sum(c.tensors for c in detection_models.values())
    
    return InspectionReport(
        path=str(path),
        file_size_bytes=file_size,
        format=fmt,
        version=version,
        phase=phase,
        timestamp=timestamp,
        can_continue_learning=can_continue,
        top_level_keys=top_level_keys,
        max_depth=max_depth,
        native_components=native_components,
        embedded_models=embedded_models,
        detection_models=detection_models,
        total_parameters=total_params,
        total_tensors=total_tensors,
        component_flags=component_flags,
        runtime_config=runtime_config,
        metadata=metadata,
        legacy_components=legacy_list,
        removed_components=removed_components,
        warnings=warnings,
        errors=errors,
    )


# ─────────────────────────────────────────────
# Report Generation
# ─────────────────────────────────────────────

def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.2f} GB"


def format_params(params: int) -> str:
    """Format parameter count as human-readable string."""
    if params < 1000:
        return str(params)
    elif params < 1_000_000:
        return f"{params / 1000:.1f}K"
    elif params < 1_000_000_000:
        return f"{params / 1_000_000:.1f}M"
    else:
        return f"{params / 1_000_000_000:.2f}B"


def generate_report(report: InspectionReport, format: str = 'text') -> str:
    """
    Generate a formatted report from inspection results.
    
    Args:
        report: InspectionReport from inspect_flx()
        format: 'text' or 'markdown'
    
    Returns:
        Formatted string
    """
    lines = []
    
    if format == 'markdown':
        lines.append(f"# FLUX Model Inspection Report")
        lines.append("")
        lines.append(f"**File:** `{report.path}`")
        lines.append(f"**Size:** {format_size(report.file_size_bytes)}")
        lines.append(f"**Version:** {report.version}")
        lines.append(f"**Phase:** {report.phase}")
        lines.append(f"**Format:** {report.format}")
        lines.append(f"**Can Continue Learning:** {report.can_continue_learning}")
        lines.append("")
        
        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Parameters | {format_params(report.total_parameters)} |")
        lines.append(f"| Total Tensors | {report.total_tensors:,} |")
        lines.append(f"| Native Components | {len(report.native_components)} |")
        lines.append(f"| Embedded Models | {len(report.embedded_models)} |")
        lines.append(f"| Detection Models | {len(report.detection_models)} |")
        lines.append(f"| Top-Level Keys | {len(report.top_level_keys)} |")
        lines.append(f"| Max Nesting Depth | {report.max_depth} |")
        lines.append("")
        
        # Native components
        if report.native_components:
            lines.append("## Native FLUX Components")
            lines.append("")
            lines.append("| Component | Parameters | Tensors | Has Weights | Legacy |")
            lines.append("|-----------|------------|---------|-------------|--------|")
            for name, info in sorted(report.native_components.items(), 
                                     key=lambda x: -x[1].parameters):
                legacy = "⚠️ Yes" if info.legacy else "No"
                weights = "✓" if info.has_weights else "✗"
                lines.append(f"| {name} | {format_params(info.parameters)} | {info.tensors:,} | {weights} | {legacy} |")
            lines.append("")
        
        # Embedded models
        if report.embedded_models:
            lines.append("## Embedded Models")
            lines.append("")
            lines.append("| Model | Base | Parameters | Lazy Load | Quantization |")
            lines.append("|-------|------|------------|-----------|--------------|")
            for name, info in sorted(report.embedded_models.items(),
                                     key=lambda x: -x[1].parameters):
                base = info.base_model or "N/A"
                if len(base) > 35:
                    base = "..." + base[-32:]
                lazy = "Yes" if info.lazy_load else "No"
                quant = info.quantization or "N/A"
                lines.append(f"| {name} | {base} | {format_params(info.parameters)} | {lazy} | {quant} |")
            lines.append("")
        
        # Detection models
        if report.detection_models:
            lines.append("## Detection Models")
            lines.append("")
            lines.append("| Model | Type | Parameters | Has Weights |")
            lines.append("|-------|------|------------|-------------|")
            for name, info in sorted(report.detection_models.items(),
                                     key=lambda x: -x[1].parameters):
                weights = "✓" if info.has_weights else "✗"
                lines.append(f"| {name} | {info.component_type} | {format_params(info.parameters)} | {weights} |")
            lines.append("")
        
        # Legacy components
        if report.legacy_components:
            lines.append("## Legacy Components")
            lines.append("")
            for name in report.legacy_components:
                # Find the component info
                info = (report.native_components.get(name) or 
                       report.embedded_models.get(name) or
                       report.detection_models.get(name))
                if info and info.legacy_reason:
                    lines.append(f"- **{name}**: {info.legacy_reason}")
                else:
                    lines.append(f"- **{name}**")
            lines.append("")
        
        # Warnings
        if report.warnings:
            lines.append("## Warnings")
            lines.append("")
            for w in report.warnings:
                lines.append(f"- ⚠️ {w}")
            lines.append("")
    
    else:  # text format
        lines.append("=" * 60)
        lines.append("FLUX Model Inspection Report")
        lines.append("=" * 60)
        lines.append(f"File: {report.path}")
        lines.append(f"Size: {format_size(report.file_size_bytes)}")
        lines.append(f"Version: {report.version}")
        lines.append(f"Phase: {report.phase}")
        lines.append(f"Format: {report.format}")
        lines.append(f"Can Continue Learning: {report.can_continue_learning}")
        lines.append("")
        
        lines.append(f"Total Parameters: {format_params(report.total_parameters)}")
        lines.append(f"Total Tensors: {report.total_tensors:,}")
        lines.append(f"Top-Level Keys: {len(report.top_level_keys)}")
        lines.append(f"Max Nesting Depth: {report.max_depth}")
        lines.append("")
        
        if report.native_components:
            lines.append("Native Components:")
            lines.append("-" * 40)
            for name, info in sorted(report.native_components.items(),
                                     key=lambda x: -x[1].parameters):
                weights = "✓" if info.has_weights else "✗"
                legacy = " [LEGACY]" if info.legacy else ""
                lines.append(f"  {name}: {format_params(info.parameters)} params, {info.tensors} tensors {weights}{legacy}")
                if info.extra:
                    for k, v in info.extra.items():
                        lines.append(f"    • {k}: {v}")
            lines.append("")
        
        if report.embedded_models:
            lines.append("Embedded Models:")
            lines.append("-" * 40)
            for name, info in sorted(report.embedded_models.items(),
                                     key=lambda x: -x[1].parameters):
                lazy = "lazy" if info.lazy_load else "eager"
                base = info.base_model or "N/A"
                lines.append(f"  {name}: {format_params(info.parameters)} params [{lazy}]")
                lines.append(f"    Base: {base}")
                if info.extra:
                    for k, v in info.extra.items():
                        lines.append(f"    • {k}: {v}")
            lines.append("")
        
        if report.detection_models:
            lines.append("Detection Models:")
            lines.append("-" * 40)
            for name, info in sorted(report.detection_models.items(),
                                     key=lambda x: -x[1].parameters):
                weights = "✓" if info.has_weights else "○"
                lines.append(f"  {name}: {format_params(info.parameters)} params {weights}")
                if info.extra:
                    for k, v in info.extra.items():
                        lines.append(f"    • {k}: {v}")
            lines.append("")
        
        if report.legacy_components:
            lines.append("Legacy Components:")
            lines.append("-" * 40)
            for name in report.legacy_components:
                lines.append(f"  ⚠ {name}")
            lines.append("")
        
        if report.warnings:
            lines.append("Warnings:")
            for w in report.warnings:
                lines.append(f"  ⚠ {w}")
            lines.append("")
        
        lines.append("=" * 60)
    
    return "\n".join(lines)


def get_component_tree(path: Union[str, Path], max_depth: int = 3) -> Dict[str, Any]:
    """
    Get a tree view of component structure (keys and types, not values).
    
    Args:
        path: Path to .flx file
        max_depth: Maximum depth to traverse
    
    Returns:
        Nested dict with structure info
    """
    path = Path(path)
    raw = torch.load(str(path), map_location='cpu', weights_only=False)
    
    def walk(obj, depth=0):
        if depth > max_depth:
            return "..."
        
        if isinstance(obj, torch.Tensor):
            return f"Tensor{list(obj.shape)}"
        elif isinstance(obj, np.ndarray):
            return f"ndarray{list(obj.shape)}"
        elif isinstance(obj, dict):
            return {k: walk(v, depth + 1) for k, v in obj.items()}
        elif isinstance(obj, list):
            if len(obj) == 0:
                return "[]"
            elif len(obj) <= 3:
                return [walk(item, depth + 1) for item in obj]
            else:
                return [walk(obj[0], depth + 1), f"... ({len(obj)} items)"]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return type(obj).__name__
        else:
            return type(obj).__name__
    
    return walk(raw)


def print_tree(tree: Dict[str, Any], indent: int = 0):
    """Pretty print a component tree."""
    for key, value in tree.items():
        prefix = "  " * indent
        if isinstance(value, dict):
            print(f"{prefix}{key}/")
            print_tree(value, indent + 1)
        elif isinstance(value, list):
            print(f"{prefix}{key}: [{len(value)} items]")
        else:
            print(f"{prefix}{key}: {value}")


# ─────────────────────────────────────────────
# CLI Interface
# ─────────────────────────────────────────────

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='FLUX Model Inspector')
    parser.add_argument('path', help='Path to .flx file')
    parser.add_argument('--format', choices=['text', 'markdown'], default='text',
                       help='Output format')
    parser.add_argument('--tree', action='store_true',
                       help='Show component tree instead of report')
    parser.add_argument('--depth', type=int, default=3,
                       help='Max depth for tree view')
    parser.add_argument('--json', action='store_true',
                       help='Output as JSON (for scripting)')
    
    args = parser.parse_args()
    
    path = Path(args.path)
    if not path.exists():
        print(f"Error: File not found: {path}")
        return 1
    
    if args.tree:
        print(f"Component tree for: {path}")
        print("=" * 60)
        tree = get_component_tree(path, max_depth=args.depth)
        print_tree(tree)
        return 0
    
    report = inspect_flx(path)
    
    if args.json:
        # Convert to JSON-serializable dict
        output = {
            'path': report.path,
            'file_size': report.file_size_bytes,
            'version': report.version,
            'phase': report.phase,
            'total_parameters': report.total_parameters,
            'total_tensors': report.total_tensors,
            'native_components': [c.name for c in report.native_components.values()],
            'embedded_models': [c.name for c in report.embedded_models.values()],
            'detection_models': [c.name for c in report.detection_models.values()],
            'legacy_components': report.legacy_components,
            'warnings': report.warnings,
        }
        print(json.dumps(output, indent=2))
    else:
        print(generate_report(report, format=args.format))
    
    return 0


if __name__ == '__main__':
    exit(main())
