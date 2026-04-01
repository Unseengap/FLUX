"""
FLUX Model Loader & Saver — Root-level utility for notebooks.

Easy-to-use interface for loading, modifying, and saving .flx models.
Supports component upgrades, additions, and runtime configuration.

Usage in notebooks:
    from flux_model import FLUXModel
    
    # Load existing model
    model = FLUXModel.load('checkpoints/Flux-Base.flx')
    
    # Upgrade a component
    model.upgrade_component('grid_to_wave', trained_encoder)
    
    # Add new component
    model.add_component('causal_tracker', tracker_state)
    
    # Change runtime config
    model.config.memory.episodic_memory_enabled = False
    model.config.learning.realtime_learning = True
    
    # Save
    model.save('checkpoints/Flux-MULTI.flx')
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict
import sys

# Add phases to path for imports
FLUX_ROOT = Path(__file__).parent
sys.path.insert(0, str(FLUX_ROOT / 'phases' / 'phase2'))

from flux_format import (
    FLUXRuntimeConfig,
    PerceptionConfig,
    MemoryConfig,
    GenerationConfig,
    ReasoningConfig,
    LearningConfig,
    FieldConfig,
    LLMConfig,
    DEFAULT_CONFIG,
    COMPONENT_REGISTRY,
    create_config_preset,
    print_config,
)

# Import lazy loader (optional - graceful fallback if not available)
try:
    from flux_lazy_loader import LazyModelManager, list_embedded_models as _list_embedded
    LAZY_LOADER_AVAILABLE = True
except ImportError:
    LAZY_LOADER_AVAILABLE = False
    LazyModelManager = None


# ─────────────────────────────────────────────
# Component Definitions
# ─────────────────────────────────────────────

COMPONENT_CATEGORIES = {
    'perception': ['cse', 'grid_to_wave', 'spatial_memory', 'perception_field'],
    'knowledge': ['field', 'working_memory', 'episodic_memory', 'semantic_memory'],
    'generation': ['decoder', 'llm', 'voice'],
    'reasoning': ['causal_tracker', 'rule_inducer', 'goal_planner', 'causal_graph'],
    'bridges': ['bridges', 'wave_to_field', 'field_to_wave'],
    # Embedded models (v6.0+ schema)
    'models': [
        'instruct', 'vlm', 'coder', 'whisper', 'tts', 'embedding', 'clip',
    ],
    # Detection models (v7.0+ schema)
    'detection': [
        'face', 'object_detect', 'depth', 'pose', 'speaker_detect',
    ],
}


# ─────────────────────────────────────────────
# FLUX Model Class
# ─────────────────────────────────────────────

class FLUXModel:
    """
    High-level interface for FLUX model files.
    
    Handles loading, saving, component upgrades, and configuration.
    """
    
    def __init__(self):
        """Create empty model — use FLUXModel.load() or FLUXModel.create() instead."""
        self.path: Optional[Path] = None
        self.version: str = '2.0-unified'
        self.phase: str = 'unified'
        self.config: FLUXRuntimeConfig = DEFAULT_CONFIG.copy()
        self.components: Dict[str, bool] = {k: False for k in COMPONENT_REGISTRY}
        self.state: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self._modified: bool = False
    
    # ─────────────────────────────────────────
    # Factory Methods
    # ─────────────────────────────────────────
    
    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        config_override: Optional[Dict[str, Any]] = None,
    ) -> 'FLUXModel':
        """
        Load a FLUX model from .flx file.
        
        Args:
            path: Path to .flx file
            config_override: Optional dict to override runtime config
        
        Returns:
            FLUXModel instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"FLUX model not found: {path}")
        
        raw = torch.load(str(path), map_location='cpu', weights_only=False)
        
        if raw.get('format') != 'FLUX':
            raise ValueError(f"Not a FLUX file: {path}")
        
        model = cls()
        model.path = path
        model.version = raw.get('version', '1.0')
        model.phase = raw.get('phase', 'unknown')
        model.metadata = raw.get('metadata', {})
        
        # Load runtime config
        if 'runtime_config' in raw:
            model.config = FLUXRuntimeConfig.from_dict(raw['runtime_config'])
        
        # Apply overrides
        if config_override:
            model.config.update(config_override)
        
        # Load component flags
        if 'components' in raw:
            model.components.update(raw['components'])
        
        # Store raw state for component access
        model.state = raw
        
        print(f"✓ Loaded: {path}")
        print(f"  Version: {model.version}")
        print(f"  Phase: {model.phase}")
        print(f"  Components: {model.active_component_count}")
        
        return model
    
    @classmethod
    def create(
        cls,
        preset: str = 'arc_full',
    ) -> 'FLUXModel':
        """
        Create a new empty FLUX model with preset config.
        
        Args:
            preset: Config preset name (arc_full, arc_lite, chat, etc.)
        
        Returns:
            Empty FLUXModel ready for component injection
        """
        model = cls()
        model.config = create_config_preset(preset)
        model.metadata['created'] = datetime.now().isoformat()
        model.metadata['preset'] = preset
        return model
    
    @classmethod
    def from_checkpoints(
        cls,
        base_path: Union[str, Path],
        inject: Optional[Dict[str, Union[str, Path]]] = None,
        preset: str = 'arc_full',
    ) -> 'FLUXModel':
        """
        Create FLUX model by combining multiple checkpoints.
        
        Args:
            base_path: Base .flx file path
            inject: Dict of {component_name: checkpoint_path} to inject
            preset: Config preset to use
        
        Returns:
            Combined FLUXModel
        
        Example:
            model = FLUXModel.from_checkpoints(
                base_path='checkpoints/Flux-X-complete.flx',
                inject={
                    'grid_to_wave': 'checkpoints/gridtowave_contrastive.pt',
                    'field': 'checkpoints/Flux-capable.flx',
                },
            )
        """
        # Load base
        model = cls.load(base_path)
        model.config = create_config_preset(preset)
        
        # Inject components
        if inject:
            for comp_name, ckpt_path in inject.items():
                model.inject_from_checkpoint(comp_name, ckpt_path)
        
        model.metadata['base'] = str(base_path)
        model.metadata['injected'] = list(inject.keys()) if inject else []
        
        return model
    
    # ─────────────────────────────────────────
    # Component Management
    # ─────────────────────────────────────────
    
    @property
    def active_component_count(self) -> int:
        """Count of active components."""
        return sum(1 for v in self.components.values() if v)
    
    @property
    def active_components(self) -> List[str]:
        """List of active component names."""
        return [k for k, v in self.components.items() if v]
    
    def has_component(self, name: str) -> bool:
        """Check if component exists and is active."""
        return self.components.get(name, False)
    
    def get_component(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get component state dict.
        
        Args:
            name: Component name
        
        Returns:
            Component state dict or None
        """
        if not self.has_component(name):
            return None
        
        # Map common names to state keys
        key_map = {
            'cse': 'cse',
            'field': 'field',
            'grid_to_wave': 'grid_to_wave',
            'spatial_memory': 'spatial_memory',
            'perception_field': 'perception_field',
            'decoder': 'decoder',
            'llm': 'llm',
            'memory': 'memory',
            'working_memory': 'memory',
            'episodic_memory': 'memory',
            'semantic_memory': 'memory',
            'causal_tracker': 'causal_tracker',
            'rule_inducer': 'rule_inducer',
            'goal_planner': 'goal_planner',
            'causal_graph': 'causal',
            'bridges': 'bridges',
        }
        
        key = key_map.get(name, name)
        return self.state.get(key)
    
    def add_component(
        self,
        name: str,
        state_dict: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a new component to the model.
        
        Args:
            name: Component name (e.g., 'causal_tracker')
            state_dict: Component state dictionary
            config: Optional component configuration
        """
        component_data = {
            'state_dict': state_dict,
            'config': config or {},
            'added': datetime.now().isoformat(),
        }
        
        self.state[name] = component_data
        self.components[name] = True
        self._modified = True
        
        print(f"  ✓ Added component: {name}")
    
    def upgrade_component(
        self,
        name: str,
        state_dict: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Upgrade/replace an existing component.
        
        Args:
            name: Component name
            state_dict: New state dictionary
            config: Optional new configuration
        """
        old_exists = self.has_component(name)
        
        component_data = {
            'state_dict': state_dict,
            'config': config or self.state.get(name, {}).get('config', {}),
            'upgraded': datetime.now().isoformat(),
        }
        
        self.state[name] = component_data
        self.components[name] = True
        self._modified = True
        
        action = "Upgraded" if old_exists else "Added"
        print(f"  ✓ {action} component: {name}")
    
    def remove_component(self, name: str):
        """
        Remove a component from the model.
        
        Args:
            name: Component name to remove
        """
        if name in self.state:
            del self.state[name]
        self.components[name] = False
        self._modified = True
        print(f"  ✓ Removed component: {name}")
    
    def mark_as_legacy(
        self,
        name: str,
        reason: str,
        removal_target: str = 'v5.0',
    ):
        """
        Mark a component as legacy (deprecated).
        
        Legacy components are flagged for future removal. AI agents should
        check for legacy status before using components.
        
        Args:
            name: Component name to mark as legacy
            reason: Explanation of what replaced this component
            removal_target: Version when this component will be removed
        """
        if name not in self.state:
            raise KeyError(f"Component not found: {name}")
        
        if not isinstance(self.state[name], dict):
            self.state[name] = {'state_dict': self.state[name]}
        
        self.state[name]['legacy'] = True
        self.state[name]['legacy_reason'] = reason
        self.state[name]['legacy_since'] = datetime.now().isoformat()
        self.state[name]['removal_target'] = removal_target
        self._modified = True
        
        print(f"  ⚠ Marked as legacy: {name}")
        print(f"    Reason: {reason}")
        print(f"    Removal target: {removal_target}")
    
    def is_legacy(self, name: str) -> bool:
        """
        Check if a component is marked as legacy.
        
        Args:
            name: Component name to check
        
        Returns:
            True if component is legacy, False otherwise
        """
        comp = self.state.get(name, {})
        if isinstance(comp, dict):
            return comp.get('legacy', False)
        return False
    
    def get_legacy_components(self) -> List[str]:
        """
        Get list of all legacy components.
        
        Returns:
            List of component names marked as legacy
        """
        return [
            name for name, data in self.state.items()
            if isinstance(data, dict) and data.get('legacy', False)
        ]
    
    def get_legacy_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get legacy information for a component.
        
        Args:
            name: Component name
        
        Returns:
            Dict with legacy info or None if not legacy
        """
        if not self.is_legacy(name):
            return None
        
        comp = self.state[name]
        return {
            'legacy': True,
            'reason': comp.get('legacy_reason', 'No reason provided'),
            'since': comp.get('legacy_since', 'Unknown'),
            'removal_target': comp.get('removal_target', 'Unknown'),
        }
    
    def inject_from_checkpoint(
        self,
        component_name: str,
        checkpoint_path: Union[str, Path],
        key_in_checkpoint: Optional[str] = None,
    ):
        """
        Inject a component from an external checkpoint file.
        
        Args:
            component_name: Target component name in this model
            checkpoint_path: Path to .pt or .flx file
            key_in_checkpoint: Key to extract (auto-detected if None)
        """
        checkpoint_path = Path(checkpoint_path)
        ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
        
        # Auto-detect key
        if key_in_checkpoint is None:
            # Try common patterns
            possible_keys = [
                f'{component_name}_state_dict',
                f'encoder_state_dict',
                component_name,
                'state_dict',
            ]
            
            for key in possible_keys:
                if key in ckpt:
                    key_in_checkpoint = key
                    break
            
            # For .flx files, look in nested structure
            if checkpoint_path.suffix == '.flx':
                if component_name in ckpt:
                    key_in_checkpoint = component_name
        
        if key_in_checkpoint is None:
            raise KeyError(
                f"Could not find component data in checkpoint. "
                f"Available keys: {list(ckpt.keys())}"
            )
        
        # Extract state
        state = ckpt[key_in_checkpoint]
        if isinstance(state, dict) and 'state_dict' in state:
            state_dict = state['state_dict']
            config = state.get('config', {})
        else:
            state_dict = state
            config = ckpt.get('config', {})
        
        self.upgrade_component(component_name, state_dict, config)
        
        # Track injection source
        if 'injections' not in self.metadata:
            self.metadata['injections'] = {}
        self.metadata['injections'][component_name] = str(checkpoint_path)
    
    # ─────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────
    
    def set_config_preset(self, preset: str):
        """
        Apply a configuration preset.
        
        Args:
            preset: Preset name (arc_full, arc_lite, chat, exploration, learning, inference)
        """
        self.config = create_config_preset(preset)
        self._modified = True
        print(f"  ✓ Applied preset: {preset}")
    
    def print_config(self):
        """Print current runtime configuration."""
        print_config(self.config)
    
    def enable_component(self, name: str):
        """Enable a component in runtime config."""
        # Map to config attribute
        config_map = {
            'cse': ('perception', 'cse_enabled'),
            'grid_to_wave': ('perception', 'grid_encoder_enabled'),
            'spatial_memory': ('perception', 'spatial_memory_enabled'),
            'perception_field': ('perception', 'perception_field_enabled'),
            'working_memory': ('memory', 'working_memory_enabled'),
            'episodic_memory': ('memory', 'episodic_memory_enabled'),
            'semantic_memory': ('memory', 'semantic_memory_enabled'),
            'llm': ('generation', 'llm_primary'),
            'decoder': ('generation', 'byte_decoder_enabled'),
            'causal_tracker': ('reasoning', 'causal_tracker_enabled'),
            'rule_inducer': ('reasoning', 'rule_inducer_enabled'),
            'goal_planner': ('reasoning', 'goal_planner_enabled'),
        }
        
        if name in config_map:
            section, attr = config_map[name]
            setattr(getattr(self.config, section), attr, True)
            self._modified = True
    
    def disable_component(self, name: str):
        """Disable a component in runtime config."""
        config_map = {
            'cse': ('perception', 'cse_enabled'),
            'grid_to_wave': ('perception', 'grid_encoder_enabled'),
            'spatial_memory': ('perception', 'spatial_memory_enabled'),
            'perception_field': ('perception', 'perception_field_enabled'),
            'working_memory': ('memory', 'working_memory_enabled'),
            'episodic_memory': ('memory', 'episodic_memory_enabled'),
            'semantic_memory': ('memory', 'semantic_memory_enabled'),
            'llm': ('generation', 'llm_primary'),
            'decoder': ('generation', 'byte_decoder_enabled'),
            'causal_tracker': ('reasoning', 'causal_tracker_enabled'),
            'rule_inducer': ('reasoning', 'rule_inducer_enabled'),
            'goal_planner': ('reasoning', 'goal_planner_enabled'),
        }
        
        if name in config_map:
            section, attr = config_map[name]
            setattr(getattr(self.config, section), attr, False)
            self._modified = True
    
    # ─────────────────────────────────────────
    # Save / Export
    # ─────────────────────────────────────────
    
    def save(
        self,
        path: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
    ):
        """
        Save the model to .flx file.
        
        Args:
            path: Output path (uses original path if None)
            overwrite: Allow overwriting existing file
        """
        if path is None:
            if self.path is None:
                raise ValueError("No path specified and model has no original path")
            path = self.path
        else:
            path = Path(path)
        
        if path.exists() and not overwrite and path != self.path:
            raise FileExistsError(f"File exists: {path}. Use overwrite=True to replace.")
        
        # Build save state
        save_state = {
            'format': 'FLUX',
            'version': self.version,
            'phase': self.phase,
            'runtime_config': self.config.to_dict(),
            'components': self.components,
            'timestamp': datetime.now().isoformat(),
            'can_continue_learning': True,
            'metadata': self.metadata,
        }
        
        # Add component states
        for comp_name, comp_data in self.state.items():
            if comp_name not in ['format', 'version', 'phase', 'runtime_config', 
                                 'components', 'timestamp', 'metadata']:
                save_state[comp_name] = comp_data
        
        # Save
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_state, str(path))
        
        size_mb = path.stat().st_size / 1e6
        
        print(f"\n✓ Saved: {path}")
        print(f"  Version: {self.version}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Components: {self.active_component_count}")
        
        self.path = path
        self._modified = False
    
    def export_component(
        self,
        name: str,
        path: Union[str, Path],
    ):
        """
        Export a single component to .pt file.
        
        Args:
            name: Component name
            path: Output path
        """
        if not self.has_component(name):
            raise KeyError(f"Component not found: {name}")
        
        comp_data = self.get_component(name)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(comp_data, str(path))
        print(f"  ✓ Exported {name} to {path}")
    
    # ─────────────────────────────────────────
    # Info / Debug
    # ─────────────────────────────────────────
    
    # ─────────────────────────────────────────
    # Embedded Models (v6.0+ schema)
    # ─────────────────────────────────────────
    
    def list_embedded_models(self) -> List[Dict[str, Any]]:
        """
        List all embedded models in the .flx file.
        
        Returns:
            List of dicts with model info (name, base_model, lazy_load, has_weights)
        """
        models = []
        
        # Check 'models' dict (v6.0+ schema)
        if 'models' in self.state and isinstance(self.state['models'], dict):
            for name, config in self.state['models'].items():
                if isinstance(config, dict):
                    models.append({
                        'name': name,
                        'base_model': config.get('base_model', 'unknown'),
                        'lazy_load': config.get('lazy_load', True),
                        'quantization': config.get('quantization', 'unknown'),
                        'has_weights': bool(
                            config.get('weights') or 
                            config.get('state_dict') or
                            config.get('onnx_models')
                        ),
                    })
        
        # Check top-level 'voice' (v5.0 schema)
        if 'voice' in self.state and 'voice' not in [m['name'] for m in models]:
            voice = self.state['voice']
            if isinstance(voice, dict) and ('thinker' in voice or 'weights' in voice):
                models.append({
                    'name': 'voice',
                    'base_model': voice.get('base_model', 'Qwen/Qwen2.5-Omni-7B'),
                    'lazy_load': False,
                    'quantization': voice.get('quantization', '4bit'),
                    'has_weights': True,
                })
        
        # Check top-level 'vlm' (backward compat)
        if 'vlm' in self.state and 'vlm' not in [m['name'] for m in models]:
            vlm = self.state['vlm']
            if isinstance(vlm, dict) and 'weights' in vlm:
                models.append({
                    'name': 'vlm',
                    'base_model': vlm.get('base_model', 'unknown'),
                    'lazy_load': vlm.get('lazy_load', True),
                    'quantization': vlm.get('quantization', 'fp16'),
                    'has_weights': True,
                })
        
        return models
    
    def get_embedded_model(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get an embedded model's config and weights.
        
        Args:
            name: Model name (e.g., 'instruct', 'vlm', 'face')
        
        Returns:
            Dict with model config and weights, or None
        """
        # Check 'models' dict first
        if 'models' in self.state and name in self.state['models']:
            return self.state['models'][name]
        
        # Check top-level
        if name in self.state:
            return self.state[name]
        
        return None
    
    def get_model_manager(self, device: str = 'auto'):
        """
        Get a LazyModelManager for all embedded models.
        
        Args:
            device: Device to load models to ('cuda', 'cpu', 'auto')
        
        Returns:
            LazyModelManager instance
        
        Raises:
            ImportError: If flux_lazy_loader is not available
        """
        if not LAZY_LOADER_AVAILABLE:
            raise ImportError(
                "flux_lazy_loader not available. "
                "Make sure flux_lazy_loader.py is in the project root."
            )
        
        return LazyModelManager(self.state, default_device=device, auto_load_eager=False)
    
    def has_embedded_model(self, name: str) -> bool:
        """
        Check if an embedded model exists.
        
        Args:
            name: Model name
        
        Returns:
            True if model exists with weights
        """
        model = self.get_embedded_model(name)
        if model is None:
            return False
        
        return bool(
            model.get('weights') or 
            model.get('state_dict') or
            model.get('onnx_models') or
            model.get('thinker')  # voice module
        )

    def summary(self):
        """Print model summary."""
        print("=" * 60)
        print("FLUX Model Summary")
        print("=" * 60)
        
        if self.path:
            print(f"Path: {self.path}")
        print(f"Version: {self.version}")
        print(f"Phase: {self.phase}")
        print(f"Modified: {self._modified}")
        
        print(f"\nComponents ({self.active_component_count} active):")
        for category, comps in COMPONENT_CATEGORIES.items():
            # Skip models/detection categories - handled separately
            if category in ('models', 'detection'):
                continue
            active = [c for c in comps if self.has_component(c)]
            if active:
                print(f"  [{category}]")
                for c in active:
                    legacy_marker = " ⚠ LEGACY" if self.is_legacy(c) else ""
                    print(f"    ✓ {c}{legacy_marker}")
        
        # Show embedded models
        embedded = self.list_embedded_models()
        if embedded:
            print(f"\n  [embedded models] ({len(embedded)})")
            for m in embedded:
                lazy = "lazy" if m['lazy_load'] else "eager"
                weights = "✓" if m['has_weights'] else "○"
                print(f"    {weights} {m['name']}: {m['base_model'][:40]} [{lazy}]")
        
        # Show legacy components
        legacy = self.get_legacy_components()
        if legacy:
            print(f"\n  [legacy - marked for removal]")
            for c in legacy:
                info = self.get_legacy_info(c)
                print(f"    ⚠ {c}")
                if info:
                    print(f"      Reason: {info['reason']}")
                    print(f"      Removal: {info['removal_target']}")
        
        inactive = [c for c in self.components if not self.components[c]]
        if inactive:
            print(f"\n  [inactive]")
            for c in inactive[:5]:
                print(f"    ✗ {c}")
            if len(inactive) > 5:
                print(f"    ... and {len(inactive)-5} more")
        
        if self.metadata:
            print(f"\nMetadata:")
            for k, v in list(self.metadata.items())[:5]:
                print(f"  {k}: {v}")
        
        print("=" * 60)
    
    def __repr__(self):
        return f"FLUXModel(version={self.version}, components={self.active_component_count})"


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

def load_flux(path: Union[str, Path], **kwargs) -> FLUXModel:
    """Load a FLUX model. Shorthand for FLUXModel.load()"""
    return FLUXModel.load(path, **kwargs)


def create_unified_model(
    base_path: str = 'checkpoints/Flux-X-complete.flx',
    grid_encoder_path: Optional[str] = 'checkpoints/gridtowave_contrastive.pt',
    enriched_field_path: Optional[str] = 'checkpoints/Flux-capable.flx',
    preset: str = 'arc_full',
) -> FLUXModel:
    """
    Create the unified FLUX model by combining checkpoints.
    
    Args:
        base_path: Path to Flux-X-complete.flx
        grid_encoder_path: Path to trained GridToWave
        enriched_field_path: Path to Flux-capable.flx with enriched field
        preset: Config preset to apply
    
    Returns:
        Combined FLUXModel
    """
    inject = {}
    
    if grid_encoder_path and Path(grid_encoder_path).exists():
        inject['grid_to_wave'] = grid_encoder_path
    
    if enriched_field_path and Path(enriched_field_path).exists():
        inject['field'] = enriched_field_path
    
    return FLUXModel.from_checkpoints(
        base_path=base_path,
        inject=inject if inject else None,
        preset=preset,
    )


def quick_save(
    state_dict: Dict[str, Any],
    path: Union[str, Path],
    component_name: str = 'custom',
    **metadata,
):
    """
    Quick save a state dict as minimal FLUX file.
    
    Args:
        state_dict: PyTorch state dict
        path: Output path
        component_name: Name for the component
        **metadata: Additional metadata
    """
    model = FLUXModel.create()
    model.add_component(component_name, state_dict)
    model.metadata.update(metadata)
    model.save(path)


# ─────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────

__all__ = [
    'FLUXModel',
    'load_flux',
    'create_unified_model',
    'quick_save',
    'FLUXRuntimeConfig',
    'DEFAULT_CONFIG',
    'create_config_preset',
    'print_config',
    'COMPONENT_CATEGORIES',
]


# ─────────────────────────────────────────────
# CLI / Test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='FLUX Model Utility')
    parser.add_argument('command', choices=['info', 'config', 'list'], 
                       help='Command to run')
    parser.add_argument('path', nargs='?', help='Path to .flx file')
    parser.add_argument('--preset', default='arc_full', help='Config preset')
    
    args = parser.parse_args()
    
    if args.command == 'info':
        if not args.path:
            print("Usage: python flux_model.py info <path.flx>")
        else:
            model = FLUXModel.load(args.path)
            model.summary()
    
    elif args.command == 'config':
        config = create_config_preset(args.preset)
        print(f"\nPreset: {args.preset}")
        print_config(config)
    
    elif args.command == 'list':
        print("\nAvailable config presets:")
        for preset in ['arc_full', 'arc_lite', 'chat', 'exploration', 'learning', 'inference']:
            print(f"  - {preset}")
        print("\nComponent categories:")
        for cat, comps in COMPONENT_CATEGORIES.items():
            print(f"  [{cat}]: {', '.join(comps)}")
