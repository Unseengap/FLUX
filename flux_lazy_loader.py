"""
FLUX Lazy Loader — On-demand loading of embedded models from .flx files.

Supports:
- HuggingFace Transformers models (AutoModel, AutoModelForCausalLM, etc.)
- ONNX models (InsightFace, custom detection)
- Timm models (HRNet, ViT variants)
- Sentence Transformers (embedding models)

Usage:
    from flux_lazy_loader import LazyModelManager
    
    # Initialize with loaded .flx state
    manager = LazyModelManager(flx_state)
    
    # Check what's available
    print(manager.list_models())
    
    # Load on demand
    instruct_model = manager.load('instruct')
    
    # Unload to free memory
    manager.unload('instruct')
"""

import gc
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass
import warnings


# ─────────────────────────────────────────────
# Model Type Detection
# ─────────────────────────────────────────────

@dataclass
class ModelSpec:
    """Specification for an embedded model."""
    name: str
    base_model: str
    model_type: str  # 'transformers', 'onnx', 'timm', 'sentence_transformers', 'custom'
    weights_key: str  # Key in the model dict containing weights
    lazy_load: bool
    quantization: str
    processor_class: Optional[str] = None
    model_class: Optional[str] = None
    extra_kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_kwargs is None:
            self.extra_kwargs = {}


def detect_model_type(name: str, config: Dict[str, Any]) -> str:
    """
    Detect the type of model from its configuration.
    
    Returns: 'transformers', 'onnx', 'timm', 'sentence_transformers', 'custom'
    """
    base_model = config.get('base_model', '').lower()
    
    # ONNX models (InsightFace style)
    if 'onnx_models' in config or 'onnx' in base_model:
        return 'onnx'
    
    # Sentence Transformers
    if 'sentence-transformers' in base_model or 'minilm' in base_model:
        return 'sentence_transformers'
    
    # Timm models (HRNet, ViT variants for detection)
    timm_indicators = ['hrnet', 'efficientnet', 'resnet', 'vit-base', 'swin']
    if any(ind in base_model for ind in timm_indicators) and 'huggingface' not in base_model:
        return 'timm'
    
    # HuggingFace Transformers (default for most)
    hf_indicators = ['qwen', 'llama', 'mistral', 'whisper', 'clip', 'owl', 'midas', 'dino', 'suno', 'bark']
    if any(ind in base_model for ind in hf_indicators):
        return 'transformers'
    
    # Voice module (custom Qwen-Omni)
    if name == 'voice' and ('thinker' in config or 'talker' in config):
        return 'custom'
    
    # Default to transformers
    return 'transformers'


def get_model_spec(name: str, config: Dict[str, Any]) -> ModelSpec:
    """Create a ModelSpec from model config."""
    model_type = detect_model_type(name, config)
    base_model = config.get('base_model', '')
    
    # Determine weights key
    weights_key = 'weights'
    if 'state_dict' in config:
        weights_key = 'state_dict'
    elif 'thinker' in config:
        weights_key = 'thinker'  # Voice module
    elif 'onnx_models' in config:
        weights_key = 'onnx_models'
    
    # Determine model class based on type and name
    model_class = None
    processor_class = None
    extra_kwargs = {}
    
    if model_type == 'transformers':
        base_lower = base_model.lower()
        
        if 'qwen2-vl' in base_lower or 'qwen2.5-vl' in base_lower:
            model_class = 'Qwen2_5_VLForConditionalGeneration'
            processor_class = 'AutoProcessor'
            extra_kwargs['trust_remote_code'] = True
        elif 'whisper' in base_lower:
            model_class = 'WhisperForConditionalGeneration'
            processor_class = 'WhisperProcessor'
        elif 'clip' in base_lower:
            model_class = 'CLIPModel'
            processor_class = 'CLIPProcessor'
        elif 'owl' in base_lower:
            model_class = 'Owlv2ForObjectDetection'
            processor_class = 'Owlv2Processor'
        elif 'bark' in base_lower or 'suno' in base_lower:
            model_class = 'BarkModel'
            processor_class = 'AutoProcessor'
        elif 'midas' in base_lower:
            model_class = 'AutoModel'  # MiDaS uses custom architecture
            extra_kwargs['trust_remote_code'] = True
        else:
            # Default: Causal LM for instruct/coder models
            model_class = 'AutoModelForCausalLM'
            processor_class = 'AutoTokenizer'
            extra_kwargs['trust_remote_code'] = True
    
    return ModelSpec(
        name=name,
        base_model=base_model,
        model_type=model_type,
        weights_key=weights_key,
        lazy_load=config.get('lazy_load', True),
        quantization=config.get('quantization', 'fp16'),
        processor_class=processor_class,
        model_class=model_class,
        extra_kwargs=extra_kwargs,
    )


# ─────────────────────────────────────────────
# Lazy Model Wrapper
# ─────────────────────────────────────────────

class LazyModel:
    """
    Lazy wrapper for an embedded model.
    
    Loads weights only when first accessed.
    """
    
    def __init__(
        self,
        spec: ModelSpec,
        weights_dict: Dict[str, Any],
        device: str = 'auto',
    ):
        self.spec = spec
        self.weights_dict = weights_dict
        self.device = device
        
        self._model = None
        self._processor = None
        self._loaded = False
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    @property
    def model(self):
        """Get the model, loading if necessary."""
        if not self._loaded:
            self.load()
        return self._model
    
    @property
    def processor(self):
        """Get the processor/tokenizer, loading if necessary."""
        if not self._loaded:
            self.load()
        return self._processor
    
    def load(self, device: Optional[str] = None) -> 'LazyModel':
        """
        Load the model into memory.
        
        Args:
            device: Override device ('cuda', 'cpu', 'auto')
        
        Returns:
            self for chaining
        """
        if self._loaded:
            return self
        
        device = device or self.device
        
        if self.spec.model_type == 'transformers':
            self._load_transformers(device)
        elif self.spec.model_type == 'onnx':
            self._load_onnx(device)
        elif self.spec.model_type == 'timm':
            self._load_timm(device)
        elif self.spec.model_type == 'sentence_transformers':
            self._load_sentence_transformers(device)
        elif self.spec.model_type == 'custom':
            self._load_custom(device)
        else:
            raise ValueError(f"Unknown model type: {self.spec.model_type}")
        
        self._loaded = True
        return self
    
    def _load_transformers(self, device: str):
        """Load a HuggingFace Transformers model."""
        try:
            from transformers import (
                AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoProcessor,
                AutoConfig,
            )
        except ImportError:
            raise ImportError("transformers not installed. Run: pip install transformers")
        
        # Dynamic import of specific model class
        model_class = AutoModelForCausalLM  # default
        processor_class = AutoTokenizer  # default
        
        if self.spec.model_class:
            if self.spec.model_class == 'Qwen2_5_VLForConditionalGeneration':
                from transformers import Qwen2_5_VLForConditionalGeneration
                model_class = Qwen2_5_VLForConditionalGeneration
            elif self.spec.model_class == 'WhisperForConditionalGeneration':
                from transformers import WhisperForConditionalGeneration
                model_class = WhisperForConditionalGeneration
            elif self.spec.model_class == 'CLIPModel':
                from transformers import CLIPModel
                model_class = CLIPModel
            elif self.spec.model_class == 'Owlv2ForObjectDetection':
                from transformers import Owlv2ForObjectDetection
                model_class = Owlv2ForObjectDetection
            elif self.spec.model_class == 'BarkModel':
                from transformers import BarkModel
                model_class = BarkModel
            elif self.spec.model_class == 'AutoModel':
                model_class = AutoModel
        
        if self.spec.processor_class:
            if self.spec.processor_class == 'AutoProcessor':
                processor_class = AutoProcessor
            elif self.spec.processor_class == 'WhisperProcessor':
                from transformers import WhisperProcessor
                processor_class = WhisperProcessor
            elif self.spec.processor_class == 'CLIPProcessor':
                from transformers import CLIPProcessor
                processor_class = CLIPProcessor
            elif self.spec.processor_class == 'Owlv2Processor':
                from transformers import Owlv2Processor
                processor_class = Owlv2Processor
        
        # Get dtype
        dtype = torch.float16 if self.spec.quantization in ('fp16', '4bit', '8bit') else torch.float32
        
        # Load processor (small download, just config/vocab)
        try:
            self._processor = processor_class.from_pretrained(
                self.spec.base_model,
                trust_remote_code=self.spec.extra_kwargs.get('trust_remote_code', True),
            )
        except Exception as e:
            warnings.warn(f"Could not load processor for {self.spec.name}: {e}")
            self._processor = None
        
        # Load model architecture (downloads config, not weights)
        device_map = 'auto' if device == 'auto' else None
        target_device = None if device == 'auto' else device
        
        self._model = model_class.from_pretrained(
            self.spec.base_model,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=self.spec.extra_kwargs.get('trust_remote_code', True),
        )
        
        # Load embedded weights
        embedded_weights = self.weights_dict.get(self.spec.weights_key, {})
        if embedded_weights:
            missing, unexpected = self._model.load_state_dict(embedded_weights, strict=False)
            if missing:
                warnings.warn(f"{self.spec.name}: {len(missing)} missing keys (may be tied weights)")
        
        if target_device:
            self._model = self._model.to(target_device)
        
        self._model.eval()
    
    def _load_onnx(self, device: str):
        """Load an ONNX model."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime not installed. Run: pip install onnxruntime-gpu")
        
        # ONNX models are stored as bytes
        onnx_models = self.weights_dict.get('onnx_models', {})
        
        # Set up providers
        if device == 'cuda' or (device == 'auto' and torch.cuda.is_available()):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        # Create sessions for each ONNX model
        self._model = {}
        for name, model_bytes in onnx_models.items():
            if isinstance(model_bytes, bytes):
                self._model[name] = ort.InferenceSession(model_bytes, providers=providers)
            elif isinstance(model_bytes, dict) and 'bytes' in model_bytes:
                self._model[name] = ort.InferenceSession(model_bytes['bytes'], providers=providers)
        
        self._processor = None  # ONNX models typically don't have processors
    
    def _load_timm(self, device: str):
        """Load a timm model."""
        try:
            import timm
        except ImportError:
            raise ImportError("timm not installed. Run: pip install timm")
        
        # Create model from timm
        model_name = self.spec.base_model
        if '/' in model_name:
            # HuggingFace format: 'timm/hrnet_w32'
            model_name = model_name.split('/')[-1]
        
        self._model = timm.create_model(model_name, pretrained=False)
        
        # Load embedded weights
        embedded_weights = self.weights_dict.get(self.spec.weights_key, {})
        if embedded_weights:
            self._model.load_state_dict(embedded_weights, strict=False)
        
        # Move to device
        target_device = 'cuda' if (device == 'auto' and torch.cuda.is_available()) else 'cpu' if device == 'auto' else device
        self._model = self._model.to(target_device)
        self._model.eval()
        
        self._processor = None
    
    def _load_sentence_transformers(self, device: str):
        """Load a Sentence Transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        
        # Load architecture
        self._model = SentenceTransformer(self.spec.base_model)
        
        # Load embedded weights if available
        embedded_weights = self.weights_dict.get(self.spec.weights_key, {})
        if embedded_weights:
            # Sentence transformers have nested structure
            for module_name, module in self._model._modules.items():
                if hasattr(module, 'load_state_dict'):
                    module_key = f'{module_name}'
                    if module_key in embedded_weights:
                        module.load_state_dict(embedded_weights[module_key], strict=False)
        
        target_device = 'cuda' if (device == 'auto' and torch.cuda.is_available()) else 'cpu' if device == 'auto' else device
        self._model = self._model.to(target_device)
        
        self._processor = None  # SentenceTransformer handles its own encoding
    
    def _load_custom(self, device: str):
        """Load a custom model (e.g., voice module)."""
        # For voice module, we need special handling
        if self.spec.name == 'voice':
            # Voice module is loaded via FLUXVoiceOmni class
            # Just store the weights dict for now
            self._model = self.weights_dict
            self._processor = None
            warnings.warn(
                f"Voice module loaded as raw dict. "
                f"Use FLUXVoiceOmni class for full functionality."
            )
        else:
            # Generic custom model - just store weights
            self._model = self.weights_dict
            self._processor = None
    
    def unload(self):
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._loaded = False
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __call__(self, *args, **kwargs):
        """Forward pass through the model."""
        return self.model(*args, **kwargs)
    
    def __repr__(self):
        status = "loaded" if self._loaded else "not loaded"
        return f"LazyModel({self.spec.name}, type={self.spec.model_type}, {status})"


# ─────────────────────────────────────────────
# Model Manager
# ─────────────────────────────────────────────

class LazyModelManager:
    """
    Manager for all embedded models in a .flx file.
    
    Handles lazy loading, unloading, and device management.
    """
    
    def __init__(
        self,
        flx_state: Dict[str, Any],
        default_device: str = 'auto',
        auto_load_eager: bool = True,
    ):
        """
        Initialize manager with loaded .flx state.
        
        Args:
            flx_state: Dict loaded from .flx file
            default_device: Default device for loading ('cuda', 'cpu', 'auto')
            auto_load_eager: Automatically load models with lazy_load=False
        """
        self.flx_state = flx_state
        self.default_device = default_device
        self.models: Dict[str, LazyModel] = {}
        
        self._discover_models()
        
        if auto_load_eager:
            self._load_eager_models()
    
    def _discover_models(self):
        """Discover all embedded models in the .flx state."""
        # Check 'models' dict (v6.0+ schema)
        if 'models' in self.flx_state:
            for name, config in self.flx_state['models'].items():
                if isinstance(config, dict):
                    spec = get_model_spec(name, config)
                    self.models[name] = LazyModel(spec, config, self.default_device)
        
        # Check top-level for voice module (v5.0 schema)
        if 'voice' in self.flx_state and 'voice' not in self.models:
            config = self.flx_state['voice']
            if isinstance(config, dict) and ('thinker' in config or 'weights' in config):
                spec = get_model_spec('voice', config)
                self.models['voice'] = LazyModel(spec, config, self.default_device)
        
        # Check for VLM (backward compat)
        if 'vlm' in self.flx_state and 'vlm' not in self.models:
            config = self.flx_state['vlm']
            if isinstance(config, dict) and 'weights' in config:
                spec = get_model_spec('vlm', config)
                self.models['vlm'] = LazyModel(spec, config, self.default_device)
    
    def _load_eager_models(self):
        """Load models that have lazy_load=False."""
        for name, model in self.models.items():
            if not model.spec.lazy_load:
                try:
                    model.load()
                    print(f"  ✓ Auto-loaded {name}")
                except Exception as e:
                    warnings.warn(f"Failed to auto-load {name}: {e}")
    
    def list_models(self) -> List[str]:
        """List all available model names."""
        return list(self.models.keys())
    
    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get info about a model without loading it."""
        if name not in self.models:
            return None
        
        model = self.models[name]
        return {
            'name': model.spec.name,
            'base_model': model.spec.base_model,
            'model_type': model.spec.model_type,
            'lazy_load': model.spec.lazy_load,
            'quantization': model.spec.quantization,
            'is_loaded': model.is_loaded,
        }
    
    def load(self, name: str, device: Optional[str] = None) -> LazyModel:
        """
        Load a model by name.
        
        Args:
            name: Model name
            device: Override device
        
        Returns:
            LazyModel instance (loaded)
        """
        if name not in self.models:
            raise KeyError(f"Model not found: {name}. Available: {self.list_models()}")
        
        return self.models[name].load(device)
    
    def unload(self, name: str):
        """Unload a model by name."""
        if name in self.models:
            self.models[name].unload()
    
    def unload_all(self):
        """Unload all models."""
        for model in self.models.values():
            model.unload()
    
    def get(self, name: str) -> Optional[LazyModel]:
        """Get a model by name (does not auto-load)."""
        return self.models.get(name)
    
    def __getitem__(self, name: str) -> LazyModel:
        """Get and load a model by name."""
        return self.load(name)
    
    def __contains__(self, name: str) -> bool:
        return name in self.models
    
    def __repr__(self):
        loaded = sum(1 for m in self.models.values() if m.is_loaded)
        return f"LazyModelManager({len(self.models)} models, {loaded} loaded)"


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

def load_model_from_flx(
    flx_path: Union[str, Path],
    model_name: str,
    device: str = 'auto',
) -> Tuple[Any, Any]:
    """
    Load a single model from a .flx file.
    
    Args:
        flx_path: Path to .flx file
        model_name: Name of model to load
        device: Device to load to
    
    Returns:
        (model, processor) tuple
    """
    flx_state = torch.load(str(flx_path), map_location='cpu', weights_only=False)
    manager = LazyModelManager(flx_state, default_device=device, auto_load_eager=False)
    
    if model_name not in manager:
        raise KeyError(f"Model '{model_name}' not found. Available: {manager.list_models()}")
    
    lazy_model = manager.load(model_name, device)
    return lazy_model.model, lazy_model.processor


def list_embedded_models(flx_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    List all embedded models in a .flx file without loading weights.
    
    Args:
        flx_path: Path to .flx file
    
    Returns:
        List of model info dicts
    """
    flx_state = torch.load(str(flx_path), map_location='cpu', weights_only=False)
    manager = LazyModelManager(flx_state, auto_load_eager=False)
    
    return [manager.get_model_info(name) for name in manager.list_models()]


# ─────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────

__all__ = [
    'LazyModel',
    'LazyModelManager',
    'ModelSpec',
    'get_model_spec',
    'detect_model_type',
    'load_model_from_flx',
    'list_embedded_models',
]
