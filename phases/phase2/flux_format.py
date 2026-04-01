"""
FLUX model file format (.flx).

Unlike .pt files which are just weight dicts,
.flx files capture the full living state of the model:
- Field tensor (the "knowledge landscape")
- Attractor catalog (what stable things are known)
- Causal graph (why things are known) — placeholder until Phase 5
- Learning state (how many steps, temperature, etc.)
- Runtime config (what components are active at inference)

A .flx file can be loaded and the model continues from EXACTLY
where it left off — including mid-learning. There is no distinction
between "trained model" and "model in training" in FLUX.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict
from copy import deepcopy


# ─────────────────────────────────────────────
# Runtime Configuration
# ─────────────────────────────────────────────

@dataclass
class PerceptionConfig:
    """Controls perception components."""
    cse_enabled: bool = True              # Text encoding via CSE
    grid_encoder_enabled: bool = True     # ARC grid encoding (GridToWave)
    spatial_memory_enabled: bool = True   # Ice & Water fields
    perception_field_enabled: bool = True # Active vision system


@dataclass
class MemoryConfig:
    """Controls memory components."""
    working_memory_enabled: bool = True   # Session context
    episodic_memory_enabled: bool = True  # Fact storage
    semantic_memory_enabled: bool = True  # Deep field queries
    memory_write_enabled: bool = True     # Allow new memories


@dataclass  
class GenerationConfig:
    """Controls text generation."""
    voice_primary: bool = True            # Embedded voice module leads (NEW)
    vlm_primary: bool = False             # VLM generation (backward compat)
    llm_primary: bool = False             # External LLM (DEPRECATED)
    byte_decoder_enabled: bool = False    # Byte decoder (LEGACY)
    byte_decoder_learns_from_llm: bool = True  # Distillation mode
    generation_mode: str = 'voice'        # 'voice' | 'llm' | 'byte' | 'hybrid'


@dataclass
class ReasoningConfig:
    """Controls reasoning components."""
    causal_tracker_enabled: bool = True   # Track action→effect
    rule_inducer_enabled: bool = True     # Learn rules from patterns
    goal_planner_enabled: bool = True     # Decompose objectives
    hypothesis_testing: bool = True       # Test inferred rules


@dataclass
class LearningConfig:
    """Controls learning behavior."""
    realtime_learning: bool = True        # Learn during inference
    field_update_enabled: bool = True     # Modify field attractors
    causal_graph_update: bool = True      # Update causal links
    temperature: float = 0.3              # Thermodynamic temperature
    learning_rate: float = 0.01           # Field update rate


@dataclass
class FieldConfig:
    """Controls field behavior."""
    gravity_enabled: bool = True          # Gravitational relevance
    interference_enabled: bool = True     # Wave interference
    thermodynamic_enabled: bool = True    # Energy settling
    field_source: str = 'capable'         # 'complete' | 'capable' | 'custom'


@dataclass
class LLMConfig:
    """Controls LLM bridge (DEPRECATED - use VoiceConfig)."""
    model_name: str = 'Qwen/Qwen2.5-3B-Instruct'
    quantization: str = '4bit'
    max_tokens: int = 512
    temperature: float = 0.7
    use_flux_context: bool = True         # Inject FLUX memories into prompt
    flux_context_limit: int = 10          # Max memories to inject


@dataclass
class VoiceConfig:
    """
    Controls embedded voice module (Qwen2.5-Omni).
    
    Replaces external LLM with self-contained multimodal generation.
    Supports text, audio input/output, and vision.
    """
    enabled: bool = True                  # Use embedded voice module
    model_type: str = 'qwen_omni'         # 'qwen_omni' | 'custom'
    quantization: str = '4bit'            # 'none' | '4bit' | '8bit' | 'svd'
    
    # Generation settings
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    # Modality settings
    text_enabled: bool = True             # Text generation
    audio_input_enabled: bool = True      # Audio understanding
    audio_output_enabled: bool = True     # Speech synthesis
    vision_enabled: bool = True           # Image/video understanding
    
    # FLUX integration
    use_flux_context: bool = True         # Inject field context
    flux_context_limit: int = 10          # Max field retrievals
    store_to_field: bool = True           # Store outputs to field


@dataclass
class FLUXRuntimeConfig:
    """
    Master configuration for FLUX runtime behavior.
    
    Controls which components are active during inference,
    how generation works, and learning parameters.
    
    v5.0: Added voice config for embedded Qwen-Omni module.
    """
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    field_config: FieldConfig = field(default_factory=FieldConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)  # NEW: Embedded voice
    llm: LLMConfig = field(default_factory=LLMConfig)  # DEPRECATED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested dict for serialization."""
        return {
            'perception': asdict(self.perception),
            'memory': asdict(self.memory),
            'generation': asdict(self.generation),
            'reasoning': asdict(self.reasoning),
            'learning': asdict(self.learning),
            'field': asdict(self.field_config),
            'voice': asdict(self.voice),
            'llm': asdict(self.llm),
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FLUXRuntimeConfig':
        """Reconstruct from dict. Filters unknown fields for backward compat."""
        def filter_fields(dataclass_type, data: dict) -> dict:
            """Keep only fields that exist in the dataclass."""
            if not data:
                return {}
            valid_fields = {f.name for f in dataclass_type.__dataclass_fields__.values()}
            return {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(
            perception=PerceptionConfig(**filter_fields(PerceptionConfig, d.get('perception', {}))),
            memory=MemoryConfig(**filter_fields(MemoryConfig, d.get('memory', {}))),
            generation=GenerationConfig(**filter_fields(GenerationConfig, d.get('generation', {}))),
            reasoning=ReasoningConfig(**filter_fields(ReasoningConfig, d.get('reasoning', {}))),
            learning=LearningConfig(**filter_fields(LearningConfig, d.get('learning', {}))),
            field_config=FieldConfig(**filter_fields(FieldConfig, d.get('field', {}))),
            voice=VoiceConfig(**filter_fields(VoiceConfig, d.get('voice', {}))),
            llm=LLMConfig(**filter_fields(LLMConfig, d.get('llm', {}))),
        )
    
    def update(self, overrides: Dict[str, Any]):
        """Update config with nested dict of overrides."""
        for section, values in overrides.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
    
    def copy(self) -> 'FLUXRuntimeConfig':
        """Create a deep copy of this config."""
        return FLUXRuntimeConfig.from_dict(self.to_dict())


# Default configuration instance
DEFAULT_CONFIG = FLUXRuntimeConfig()


# ─────────────────────────────────────────────
# Component Registry
# ─────────────────────────────────────────────

COMPONENT_REGISTRY = {
    # Perception
    'cse': {'required': True, 'phase': 1},
    'grid_to_wave': {'required': False, 'phase': 8.8},
    'spatial_memory': {'required': False, 'phase': 9},
    'perception_field': {'required': False, 'phase': 'unified'},
    
    # Knowledge
    'field': {'required': True, 'phase': 2},
    'working_memory': {'required': False, 'phase': 6},
    'episodic_memory': {'required': False, 'phase': 6},
    'semantic_memory': {'required': False, 'phase': 6},
    
    # Generation - Voice (NEW in v5.0)
    'voice': {'required': False, 'phase': 'voice'},
    'voice_thinker': {'required': False, 'phase': 'voice'},
    'voice_talker': {'required': False, 'phase': 'voice'},
    'voice_token2wav': {'required': False, 'phase': 'voice'},
    
    # Generation - Legacy
    'decoder': {'required': False, 'phase': 8, 'legacy': True},
    'llm': {'required': False, 'phase': 11, 'legacy': True},
    
    # Reasoning
    'causal_tracker': {'required': False, 'phase': 'unified'},
    'rule_inducer': {'required': False, 'phase': 'unified'},
    'goal_planner': {'required': False, 'phase': 'unified'},
    'causal_graph': {'required': False, 'phase': 5},
    
    # Bridges
    'bridges': {'required': False, 'phase': 8},
}


def save_flux(
    field: 'ResonanceField',
    cse: 'ContinuousSemanticEncoder',
    path: str,
    attractor_catalog: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
    runtime_config: Optional[FLUXRuntimeConfig] = None,
    components: Optional[Dict[str, Any]] = None,
):
    """
    Save full FLUX state to .flx file.

    The .flx format captures the ENTIRE model state so that
    learning can continue seamlessly from where it left off.

    Args:
        field: the ResonanceField
        cse: the ContinuousSemanticEncoder (from Phase 1)
        path: output file path (should end in .flx)
        attractor_catalog: optional AttractorCatalog to save
        metadata: optional dict of extra metadata
        runtime_config: optional FLUXRuntimeConfig for inference settings
        components: optional dict of additional component states
    """
    if metadata is None:
        metadata = {}
    
    if runtime_config is None:
        runtime_config = DEFAULT_CONFIG
    
    if components is None:
        components = {}

    # Build component availability flags
    component_flags = {name: False for name in COMPONENT_REGISTRY}
    component_flags['cse'] = True
    component_flags['field'] = True
    for name in components:
        if name in component_flags:
            component_flags[name] = True

    state = {
        # Format identification
        'format': 'FLUX',
        'version': '2.0',
        'phase': metadata.get('phase', 2),

        # Component states
        'field_state': field.state_dict(),
        'cse_state': cse.state_dict(),

        # Field knowledge (not in state_dict because they're register_buffers
        # which ARE in state_dict, but we also save mass separately for clarity)
        'field_mass': field.mass.cpu(),
        'field_step_count': field.step_count,

        # Attractor catalog
        'attractor_catalog': (
            attractor_catalog.to_dict() if attractor_catalog else []
        ),

        # Causal graph placeholder (Phase 5)
        'causal_graph': components.get('causal_graph', {}),

        # Learning metadata
        'learning_steps': metadata.get('steps', field.step_count),
        'can_continue_learning': True,  # Always true in FLUX
        'timestamp': datetime.now().isoformat(),

        # Configuration (enough to reconstruct)
        'field_config': {
            'h': field.h,
            'w': field.w,
            'd': field.d,
            'features': field.features,
            'wave_dim': field.wave_dim,
        },
        'cse_config': {
            'wave_dims': cse.wave_dims,
            'byte_window': cse.byte_window,
            'byte_stride': cse.byte_stride,
            'interference_radius': cse.interference_radius,
        },

        # Runtime configuration (NEW in v2.0)
        'runtime_config': runtime_config.to_dict(),
        
        # Component availability flags (NEW in v2.0)
        'components': component_flags,
        
        # Additional component states (NEW in v2.0)
        'additional_components': components,

        # Extra metadata
        'metadata': metadata,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))

    size_mb = path.stat().st_size / 1e6
    print(f"✓ FLUX model saved: {path}")
    print(f"  Format: FLUX v{state['version']}")
    print(f"  File size: {size_mb:.1f} MB")
    print(f"  Learning steps: {state['learning_steps']}")
    print(f"  Can continue learning: {state['can_continue_learning']}")
    print(f"  Components: {sum(component_flags.values())} active")


def load_flux(
    path: str,
    config_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load FLUX state from .flx file.
    Returns dict with all components ready to initialize.

    Use the returned config dicts to reconstruct models:
        cse = ContinuousSemanticEncoder(**state['cse_config'])
        cse.load_state_dict(state['cse_state'])
        field = ResonanceField(**state['field_config'])
        field.load_state_dict(state['field_state'])

    Args:
        path: path to .flx file
        config_override: optional dict to override runtime config settings
    Returns:
        Dict with all saved state including runtime_config
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"FLUX model not found: {path}\n"
            f"Run training first to create the model file."
        )

    state = torch.load(str(path), map_location='cpu', weights_only=False)

    assert state.get('format') == 'FLUX', (
        f"Not a FLUX model file — got format: {state.get('format')}"
    )

    # Parse runtime config
    if 'runtime_config' in state:
        runtime_config = FLUXRuntimeConfig.from_dict(state['runtime_config'])
    else:
        runtime_config = DEFAULT_CONFIG.copy()
    
    # Apply overrides if provided
    if config_override:
        runtime_config.update(config_override)
    
    state['runtime_config'] = runtime_config

    print(f"✓ FLUX model loaded: {path}")
    print(f"  Version: {state['version']}")
    print(f"  Phase: {state.get('phase', 'unknown')}")
    print(f"  Learning steps: {state.get('learning_steps', 0)}")
    print(f"  Can continue: {state.get('can_continue_learning', True)}")
    
    # Print component status
    if 'components' in state:
        active = [k for k, v in state['components'].items() if v]
        print(f"  Active components: {', '.join(active)}")

    size_mb = path.stat().st_size / 1e6
    print(f"  File size: {size_mb:.1f} MB")

    return state


def get_runtime_config(state: Dict[str, Any]) -> FLUXRuntimeConfig:
    """
    Extract runtime config from loaded state.
    
    Args:
        state: dict returned by load_flux()
    Returns:
        FLUXRuntimeConfig instance
    """
    if isinstance(state.get('runtime_config'), FLUXRuntimeConfig):
        return state['runtime_config']
    elif isinstance(state.get('runtime_config'), dict):
        return FLUXRuntimeConfig.from_dict(state['runtime_config'])
    else:
        return DEFAULT_CONFIG.copy()


def is_component_enabled(
    state: Dict[str, Any],
    component_name: str,
) -> bool:
    """
    Check if a component is enabled in runtime config.
    
    Args:
        state: dict returned by load_flux()
        component_name: name of component to check
    Returns:
        True if component is enabled
    """
    config = get_runtime_config(state)
    
    # Map component names to config paths
    component_map = {
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
    
    if component_name in component_map:
        section, key = component_map[component_name]
        section_obj = getattr(config, section, None)
        if section_obj:
            return getattr(section_obj, key, True)
    
    return True  # Default to enabled


def verify_flux_integrity(path: str) -> bool:
    """
    Verify that a .flx file is complete and uncorrupted.

    Checks:
    - File loads without error
    - Format field is 'FLUX'
    - All required keys present
    - State dicts are non-empty

    Args:
        path: path to .flx file
    Returns:
        True if file passes all checks
    """
    required_keys = [
        'format', 'version',
        'field_state', 'cse_state',
        'field_config', 'cse_config',
    ]

    try:
        state = torch.load(str(path), map_location='cpu', weights_only=False)

        if state.get('format') != 'FLUX':
            print(f"  ✗ Wrong format: {state.get('format')}")
            return False

        for key in required_keys:
            if key not in state:
                print(f"  ✗ Missing key: {key}")
                return False

        if len(state['field_state']) == 0:
            print("  ✗ Empty field state dict")
            return False

        if len(state['cse_state']) == 0:
            print("  ✗ Empty CSE state dict")
            return False

        print(f"  ✓ FLUX file integrity verified: {path}")
        return True

    except Exception as e:
        print(f"  ✗ FLUX file corrupt or unreadable: {e}")
        return False


# ─────────────────────────────────────────────
# Unified Model Format (v2.0)
# ─────────────────────────────────────────────

def save_unified_flux(
    path: str,
    cse_state: Dict[str, Any],
    field_state: Dict[str, Any],
    runtime_config: Optional[FLUXRuntimeConfig] = None,
    grid_to_wave_state: Optional[Dict[str, Any]] = None,
    spatial_memory_state: Optional[Dict[str, Any]] = None,
    perception_field_state: Optional[Dict[str, Any]] = None,
    memory_state: Optional[Dict[str, Any]] = None,
    decoder_state: Optional[Dict[str, Any]] = None,
    causal_tracker_state: Optional[Dict[str, Any]] = None,
    rule_inducer_state: Optional[Dict[str, Any]] = None,
    goal_planner_state: Optional[Dict[str, Any]] = None,
    causal_graph: Optional[Dict[str, Any]] = None,
    bridges_state: Optional[Dict[str, Any]] = None,
    llm_config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Save the complete FLUX-UNIFIED model.
    
    This is the full stack combining all checkpoints:
    - Flux-X-complete.flx base
    - Trained GridToWave
    - Enriched field from Flux-capable
    - LLM bridge
    - New cognitive components
    
    Args:
        path: output path (should be FLUX-UNIFIED.flx)
        cse_state: CSE state dict and config
        field_state: Field state dict and config (enriched from capable)
        runtime_config: inference configuration
        grid_to_wave_state: trained GridToWave encoder
        spatial_memory_state: spatial memory state
        perception_field_state: active vision system state
        memory_state: working/episodic/semantic memory
        decoder_state: byte decoder state
        causal_tracker_state: action→effect tracker
        rule_inducer_state: rule learning state
        goal_planner_state: goal decomposition state
        causal_graph: causal geometry graph
        bridges_state: wave↔field bridges
        llm_config: LLM configuration (model name, quantization, etc.)
        metadata: additional metadata
    """
    if runtime_config is None:
        runtime_config = DEFAULT_CONFIG
    
    if metadata is None:
        metadata = {}
    
    # Build component flags
    components = {
        'cse': True,
        'field': True,
        'grid_to_wave': grid_to_wave_state is not None,
        'spatial_memory': spatial_memory_state is not None,
        'perception_field': perception_field_state is not None,
        'working_memory': memory_state is not None and 'working' in memory_state,
        'episodic_memory': memory_state is not None and 'episodic' in memory_state,
        'semantic_memory': memory_state is not None and 'semantic' in memory_state,
        'decoder': decoder_state is not None,
        'llm': llm_config is not None,
        'causal_tracker': causal_tracker_state is not None,
        'rule_inducer': rule_inducer_state is not None,
        'goal_planner': goal_planner_state is not None,
        'causal_graph': causal_graph is not None,
        'bridges': bridges_state is not None,
    }
    
    state = {
        # Format
        'format': 'FLUX',
        'version': '2.0-unified',
        'phase': 'unified',
        
        # Core components (required)
        'cse': cse_state,
        'field': field_state,
        
        # Perception
        'grid_to_wave': grid_to_wave_state,
        'spatial_memory': spatial_memory_state,
        'perception_field': perception_field_state,
        
        # Memory
        'memory': memory_state,
        
        # Generation
        'decoder': decoder_state,
        'llm': llm_config,
        
        # Reasoning (cognitive layer)
        'causal_tracker': causal_tracker_state,
        'rule_inducer': rule_inducer_state,
        'goal_planner': goal_planner_state,
        
        # Causal structure
        'causal': causal_graph,
        'bridges': bridges_state,
        
        # Configuration
        'runtime_config': runtime_config.to_dict(),
        'components': components,
        
        # Metadata
        'timestamp': datetime.now().isoformat(),
        'can_continue_learning': True,
        'metadata': metadata,
    }
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))
    
    size_mb = path.stat().st_size / 1e6
    active_components = [k for k, v in components.items() if v]
    
    print(f"✓ FLUX-UNIFIED model saved: {path}")
    print(f"  Version: {state['version']}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Components: {len(active_components)}")
    for comp in active_components:
        print(f"    ✓ {comp}")


def load_unified_flux(
    path: str,
    config_override: Optional[Dict[str, Any]] = None,
    load_components: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """
    Load FLUX-UNIFIED model with selective component loading.
    
    Args:
        path: path to FLUX-UNIFIED.flx
        config_override: runtime config overrides
        load_components: dict of {component_name: bool} to selectively load
    Returns:
        Dict with loaded state
    """
    state = load_flux(path, config_override)
    
    # Apply selective loading
    if load_components:
        for comp_name, should_load in load_components.items():
            if not should_load and comp_name in state:
                state[comp_name] = None
                if 'components' in state:
                    state['components'][comp_name] = False
    
    return state


def create_config_preset(preset_name: str) -> FLUXRuntimeConfig:
    """
    Create preset configurations for common use cases.
    
    Presets:
    - 'arc_full': All components for ARC-AGI-3
    - 'arc_lite': Minimal for fast inference
    - 'chat': LLM-focused for conversation
    - 'exploration': Spatial memory focused
    - 'learning': Maximum learning enabled
    - 'inference': Minimal learning, fast inference
    
    Args:
        preset_name: name of preset
    Returns:
        Configured FLUXRuntimeConfig
    """
    config = FLUXRuntimeConfig()
    
    if preset_name == 'arc_full':
        # All components enabled for ARC tasks
        pass  # Default is all enabled
    
    elif preset_name == 'arc_lite':
        # Minimal for fast inference
        config.reasoning.rule_inducer_enabled = False
        config.reasoning.hypothesis_testing = False
        config.learning.realtime_learning = False
        config.perception.perception_field_enabled = False
    
    elif preset_name == 'chat':
        # LLM-focused for conversation
        config.perception.grid_encoder_enabled = False
        config.perception.spatial_memory_enabled = False
        config.perception.perception_field_enabled = False
        config.reasoning.causal_tracker_enabled = False
        config.reasoning.rule_inducer_enabled = False
        config.reasoning.goal_planner_enabled = False
        config.generation.llm_primary = True
        config.generation.generation_mode = 'llm'
    
    elif preset_name == 'exploration':
        # Spatial memory focused
        config.perception.spatial_memory_enabled = True
        config.perception.perception_field_enabled = True
        config.reasoning.causal_tracker_enabled = True
        config.learning.realtime_learning = True
    
    elif preset_name == 'learning':
        # Maximum learning
        config.learning.realtime_learning = True
        config.learning.field_update_enabled = True
        config.learning.causal_graph_update = True
        config.generation.byte_decoder_learns_from_llm = True
    
    elif preset_name == 'inference':
        # Fast inference, no learning
        config.learning.realtime_learning = False
        config.learning.field_update_enabled = False
        config.learning.causal_graph_update = False
        config.memory.memory_write_enabled = False
    
    else:
        raise ValueError(f"Unknown preset: {preset_name}")
    
    return config


def print_config(config: FLUXRuntimeConfig):
    """Pretty print a runtime configuration."""
    print("FLUX Runtime Configuration")
    print("=" * 50)
    
    d = config.to_dict()
    for section, values in d.items():
        print(f"\n[{section}]")
        for key, value in values.items():
            status = "✓" if value is True else "✗" if value is False else str(value)
            print(f"  {key}: {status}")


# ─────────────────────────────────────────────
# Embedded VLM Loading (v5.0+)
# ─────────────────────────────────────────────

def load_embedded_vlm(
    model_state: Dict[str, Any],
    device: str = 'auto',
    torch_dtype: Any = None,
) -> tuple:
    """
    Load the embedded VLM from a .flx file's weights.
    
    CRITICAL: For trust_remote_code models like Qwen2.5-VL, you MUST use
    AutoModel.from_pretrained() to download the model architecture, then
    replace weights with load_state_dict(). Using from_config() will FAIL.
    
    The model architecture is cached by HuggingFace after first download.
    Subsequent loads use cached architecture + embedded .flx weights.
    
    Args:
        model_state: Dict from torch.load() of a .flx file
        device: Device to load model on ('auto', 'cuda', 'cpu', 'mps')
        torch_dtype: Data type (default: torch.float16)
    
    Returns:
        Tuple of (vlm_model, processor) ready for inference
    
    Example:
        >>> model = torch.load('Flux-Apex-V1.flx', map_location='cpu')
        >>> vlm_model, processor = load_embedded_vlm(model)
        >>> # Now use vlm_model.generate() with embedded weights
    
    Raises:
        ImportError: If transformers not installed
        KeyError: If 'vlm' not in model_state
        RuntimeError: If VLM loading fails
    """
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    except ImportError:
        raise ImportError(
            "transformers not installed. Run: pip install transformers"
        )
    
    if 'vlm' not in model_state:
        raise KeyError(
            "No embedded VLM in this .flx file. "
            "Run phase_voice_kaggle.ipynb to embed VLM weights."
        )
    
    if torch_dtype is None:
        torch_dtype = torch.float16
    
    vlm_state = model_state['vlm']
    embedded_weights = vlm_state.get('weights', {})
    base_model = vlm_state.get('base_model', 'Qwen/Qwen2.5-VL-3B-Instruct')
    
    if not embedded_weights:
        raise RuntimeError(
            "VLM section exists but weights dict is empty. "
            "The .flx file may be corrupted or incomplete."
        )
    
    print(f"Loading embedded VLM from .flx...")
    print(f"  Base model: {base_model}")
    print(f"  Embedded weights: {len(embedded_weights)} tensors")
    print(f"  Total params: {vlm_state.get('total_params', 0):,}")
    
    # Step 1: Load processor (small download - just tokenizer config)
    print(f"  [1/3] Loading processor...")
    processor = AutoProcessor.from_pretrained(
        base_model,
        trust_remote_code=True,
    )
    
    # Step 2: Load model architecture from HuggingFace (CACHED after first run)
    # CRITICAL: Use Qwen2_5_VLForConditionalGeneration for Qwen2.5-VL models!
    # Qwen2VLForConditionalGeneration is for Qwen2-VL (different MLP architecture).
    print(f"  [2/3] Loading model architecture (cached after first run)...")
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Step 3: Replace HuggingFace weights with our embedded .flx weights
    print(f"  [3/3] Loading embedded weights from .flx...")
    missing, unexpected = vlm_model.load_state_dict(embedded_weights, strict=False)
    
    if missing:
        print(f"    ⚠ Missing keys: {len(missing)} (may be ok for some tied weights)")
    if unexpected:
        print(f"    ⚠ Unexpected keys: {len(unexpected)}")
    
    vlm_model.eval()
    device_info = next(vlm_model.parameters()).device
    print(f"  ✓ VLM ready on {device_info} (weights from .flx!)")
    
    return vlm_model, processor


def check_vlm_embedded(model_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if a .flx file has embedded VLM and return info.
    
    Args:
        model_state: Dict from torch.load() of a .flx file
    
    Returns:
        Dict with VLM info or {'embedded': False} if not present
    
    Example:
        >>> model = torch.load('Flux-Apex-V1.flx', map_location='cpu')
        >>> info = check_vlm_embedded(model)
        >>> if info['embedded']:
        ...     print(f"VLM: {info['base_model']} ({info['total_params']:,} params)")
    """
    if 'vlm' not in model_state:
        return {'embedded': False}
    
    vlm = model_state['vlm']
    return {
        'embedded': True,
        'base_model': vlm.get('base_model', 'unknown'),
        'quantization': vlm.get('quantization', 'unknown'),
        'total_params': vlm.get('total_params', 0),
        'num_weights': len(vlm.get('weights', {})),
        'has_bridges': bool(vlm.get('bridges', {})),
        'text_enabled': vlm.get('config', {}).get('text_enabled', True),
        'vision_enabled': vlm.get('config', {}).get('vision_enabled', True),
    }


# ─────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────

__all__ = [
    # Config classes
    'FLUXRuntimeConfig',
    'PerceptionConfig',
    'MemoryConfig',
    'GenerationConfig',
    'ReasoningConfig',
    'LearningConfig',
    'FieldConfig',
    'VoiceConfig',  # NEW: Embedded voice module config
    'LLMConfig',    # DEPRECATED: Use VoiceConfig
    
    # Config instance
    'DEFAULT_CONFIG',
    
    # Registry
    'COMPONENT_REGISTRY',
    
    # Core functions
    'save_flux',
    'load_flux',
    'verify_flux_integrity',
    
    # Unified model functions
    'save_unified_flux',
    'load_unified_flux',
    
    # VLM utilities (v5.0+)
    'load_embedded_vlm',
    'check_vlm_embedded',
    
    # Utilities
    'get_runtime_config',
    'is_component_enabled',
    'create_config_preset',
    'print_config',
]


# ─────────────────────────────────────────────
# Usage Examples
# ─────────────────────────────────────────────

if __name__ == '__main__':
    # Example: Create and print a config
    print("Default config:")
    print_config(DEFAULT_CONFIG)
    
    print("\n" + "=" * 50)
    print("\nARC-lite preset:")
    arc_lite = create_config_preset('arc_lite')
    print_config(arc_lite)
    
    print("\n" + "=" * 50)
    print("\nChat preset:")
    chat = create_config_preset('chat')
    print_config(chat)
    
    # Example: Override config at load time
    print("\n" + "=" * 50)
    print("\nConfig override example:")
    override = {
        'memory': {'episodic_memory_enabled': False},
        'learning': {'realtime_learning': False},
    }
    config = DEFAULT_CONFIG.copy()
    config.update(override)
    print("After override (episodic=False, realtime_learning=False):")
    print(f"  episodic_memory_enabled: {config.memory.episodic_memory_enabled}")
    print(f"  realtime_learning: {config.learning.realtime_learning}")
