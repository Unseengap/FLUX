"""
qwen_omni_bridge.py — Phase 12: Bridge for Qwen2.5-Omni Vision+Text

Unified multi-modal model interface for:
- Vision (image understanding)
- Text (language reasoning)
- Audio (optional, for games with sounds)

Qwen2.5-Omni is ONE model that handles all modalities, saving VRAM.
"""

import torch
from torch import Tensor
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import sys

# Try to import PIL for image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class QwenOmniConfig:
    """Configuration for Qwen-Omni bridge."""
    
    # Model selection (Qwen2-VL is more stable than Omni for now)
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    fallback_model: str = "Qwen/Qwen2.5-3B-Instruct"  # Text-only fallback
    
    # Quantization
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    
    # Capabilities
    enable_vision: bool = True
    enable_audio: bool = False
    
    # Generation settings
    max_new_tokens: int = 300
    temperature: float = 0.3
    top_p: float = 0.9
    do_sample: bool = True
    
    # Device
    device: str = "auto"


# ─────────────────────────────────────────────
# Qwen-Omni Bridge
# ─────────────────────────────────────────────

class QwenOmniBridge:
    """
    Bridge for Qwen2.5-Omni multi-modal model.
    
    Features:
    - Native vision input (images processed directly)
    - Text reasoning
    - Graceful fallback to text-only model
    - Memory-efficient 4-bit quantization
    
    Usage:
        bridge = QwenOmniBridge()
        
        # Text-only query
        response = bridge.chat("What is 2+2?")
        
        # Vision + text query
        response = bridge.chat_with_image(
            image=pil_image,
            prompt="Describe what you see"
        )
    """
    
    def __init__(
        self,
        config: Optional[QwenOmniConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize Qwen-Omni bridge.
        
        Args:
            config: Configuration options
            device: Override device (auto-detect if None)
        """
        self.config = config or QwenOmniConfig()
        
        # Resolve device
        if device:
            self._device = device
        elif self.config.device == "auto":
            self._device = self._detect_device()
        else:
            self._device = self.config.device
        
        # Model state
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._model_loaded = False
        self._vision_available = False
        
        # Try to load model
        self._load_model()
    
    def _detect_device(self) -> str:
        """Detect best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _load_model(self):
        """Load Qwen model with appropriate settings."""
        try:
            # First try Omni model with vision
            if self.config.enable_vision:
                self._try_load_omni()
            
            # Fallback to text-only
            if not self._model_loaded:
                self._try_load_text_only()
                
        except Exception as e:
            print(f"  ⚠ Model loading failed: {e}")
            print("  ⚠ Using mock model for testing")
            self._setup_mock()
    
    def _try_load_omni(self):
        """Try to load Qwen-Omni or Qwen2-VL with vision support."""
        try:
            print(f"  Loading {self.config.model_name}...")
            
            # Quantization config
            quantization_config = None
            if self.config.load_in_4bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            
            # Try Qwen2-VL specific class first (most reliable)
            try:
                from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
                
                # Use Qwen2-VL-7B as fallback (more stable than Omni)
                model_name = "Qwen/Qwen2-VL-7B-Instruct"
                print(f"  Using Qwen2-VL-7B-Instruct (vision model)...")
                
                self.processor = Qwen2VLProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                )
                
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto" if self._device == "cuda" else None,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                )
                
                self._model_loaded = True
                self._vision_available = True
                print(f"  ✓ Qwen2-VL-7B loaded with vision")
                return
                
            except ImportError:
                print(f"  ⚠ Qwen2VL classes not available, trying AutoModel...")
            
            # Fallback to AutoModel with trust_remote_code
            from transformers import AutoProcessor, AutoModel
            
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
            
            # Use AutoModel (not AutoModelForCausalLM) for vision models
            self.model = AutoModel.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map="auto" if self._device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            
            if self._device != "cuda":
                self.model = self.model.to(self._device)
            
            self._model_loaded = True
            self._vision_available = True
            print(f"  ✓ {self.config.model_name} loaded with vision")
            
        except ImportError as e:
            print(f"  ⚠ Omni model requires additional packages: {e}")
        except Exception as e:
            print(f"  ⚠ Omni model failed: {e}")
    
    def _try_load_text_only(self):
        """Load text-only fallback model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model_name = self.config.fallback_model
            print(f"  Loading text-only fallback: {model_name}...")
            
            # Quantization config
            quantization_config = None
            if self.config.load_in_4bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto" if self._device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            
            if self._device != "cuda":
                self.model = self.model.to(self._device)
            
            self._model_loaded = True
            self._vision_available = False
            print(f"  ✓ {model_name} loaded (text-only)")
            
        except Exception as e:
            print(f"  ⚠ Text-only model failed: {e}")
    
    def _setup_mock(self):
        """Setup mock model for testing without real model."""
        self._model_loaded = True
        self._vision_available = False
        self._mock_mode = True
    
    # ─────────────────────────────────────────────
    # Core API
    # ─────────────────────────────────────────────
    
    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Text-only chat.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            max_new_tokens: Override config max tokens
            temperature: Override config temperature
            
        Returns:
            Model response text
        """
        if hasattr(self, '_mock_mode') and self._mock_mode:
            return self._mock_response(prompt)
        
        max_tokens = max_new_tokens or self.config.max_new_tokens
        temp = temperature or self.config.temperature
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Generate
        if self.tokenizer:
            return self._generate_text(messages, max_tokens, temp)
        elif self.processor:
            return self._generate_with_processor(messages, None, max_tokens, temp)
        
        return "Model not loaded."
    
    def chat_with_image(
        self,
        image: Union['Image.Image', str, Path],
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Vision + text chat.
        
        Args:
            image: PIL Image, or path to image file
            prompt: User prompt about the image
            system_prompt: Optional system instruction
            max_new_tokens: Override config max tokens
            temperature: Override config temperature
            
        Returns:
            Model response text
        """
        if not self._vision_available:
            # Fallback: describe that we can't see the image
            return self.chat(
                f"[Note: I cannot see images directly. Please describe the image.]\n\n{prompt}",
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        
        if hasattr(self, '_mock_mode') and self._mock_mode:
            return self._mock_vision_response(prompt)
        
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        max_tokens = max_new_tokens or self.config.max_new_tokens
        temp = temperature or self.config.temperature
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]
        })
        
        return self._generate_with_processor(messages, image, max_tokens, temp)
    
    def _generate_text(
        self,
        messages: List[Dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate text with tokenizer-based model."""
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self._device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if self.config.do_sample else None,
                top_p=self.config.top_p if self.config.do_sample else None,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        return response.strip()
    
    def _generate_with_processor(
        self,
        messages: List[Dict],
        image: Optional['Image.Image'],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate with processor (for vision models like Qwen2-VL)."""
        try:
            # Try using qwen_vl_utils if available
            from qwen_vl_utils import process_vision_info
            has_qwen_utils = True
        except ImportError:
            has_qwen_utils = False
        
        # Format messages for Qwen2-VL
        if image is not None:
            # Qwen2-VL expects messages in specific format
            formatted_messages = []
            for msg in messages:
                if msg['role'] == 'system':
                    formatted_messages.append({
                        "role": "system",
                        "content": msg['content'] if isinstance(msg['content'], str) else msg['content']
                    })
                elif msg['role'] == 'user':
                    content = msg['content']
                    if isinstance(content, list):
                        # Already has image/text structure
                        formatted_content = []
                        for item in content:
                            if item.get('type') == 'image':
                                formatted_content.append({"type": "image", "image": image})
                            elif item.get('type') == 'text':
                                formatted_content.append({"type": "text", "text": item['text']})
                        formatted_messages.append({"role": "user", "content": formatted_content})
                    else:
                        # Text only - add image
                        formatted_messages.append({
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": content}
                            ]
                        })
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process inputs with image
            if has_qwen_utils:
                image_inputs, video_inputs = process_vision_info(formatted_messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self._device)
            else:
                # Simple fallback without qwen_vl_utils
                inputs = self.processor(
                    text=[text],
                    images=[image],
                    padding=True,
                    return_tensors="pt",
                ).to(self._device)
        else:
            # Text only
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            ).to(self._device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if self.config.do_sample else None,
                top_p=self.config.top_p if self.config.do_sample else None,
                do_sample=self.config.do_sample,
            )
        
        # Decode only the generated part
        generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return response.strip()
    
    def _format_messages(self, messages: List[Dict]) -> str:
        """Format messages for processor."""
        parts = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if isinstance(content, str):
                parts.append(f"{role}: {content}")
            elif isinstance(content, list):
                text_parts = [c['text'] for c in content if c.get('type') == 'text']
                parts.append(f"{role}: {' '.join(text_parts)}")
        
        return "\n".join(parts)
    
    def _mock_response(self, prompt: str) -> str:
        """Generate mock response for testing."""
        import random
        
        prompt_lower = prompt.lower()
        
        # Check for direction keywords
        if any(w in prompt_lower for w in ['up', 'north', 'above']):
            return "Based on the situation, I should go UP. ACTION: UP"
        elif any(w in prompt_lower for w in ['down', 'south', 'below']):
            return "I'll go DOWN to explore. ACTION: DOWN"
        elif any(w in prompt_lower for w in ['left', 'west']):
            return "Moving LEFT seems best. ACTION: LEFT"
        elif any(w in prompt_lower for w in ['right', 'east']):
            return "I'll try going RIGHT. ACTION: RIGHT"
        
        # Default random direction
        direction = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        return f"Analyzing the puzzle... I'll try {direction}. ACTION: {direction}"
    
    def _mock_vision_response(self, prompt: str) -> str:
        """Generate mock vision response."""
        return self._mock_response(prompt)
    
    # ─────────────────────────────────────────────
    # Status & Info
    # ─────────────────────────────────────────────
    
    @property
    def device(self) -> str:
        """Get current device."""
        return self._device
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded
    
    @property
    def has_vision(self) -> bool:
        """Check if vision is available."""
        return self._vision_available
    
    def get_info(self) -> Dict[str, Any]:
        """Get bridge information."""
        return {
            'model_name': self.config.model_name if self._vision_available else self.config.fallback_model,
            'device': self._device,
            'loaded': self._model_loaded,
            'vision': self._vision_available,
            'quantization': '4-bit' if self.config.load_in_4bit else 'full',
            'mock_mode': getattr(self, '_mock_mode', False),
        }
    
    def get_vram_usage(self) -> Optional[float]:
        """Get current VRAM usage in GB."""
        if self._device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 3)
        return None


# ─────────────────────────────────────────────
# Factory Function
# ─────────────────────────────────────────────

def create_qwen_bridge(
    enable_vision: bool = True,
    load_in_4bit: bool = True,
    device: Optional[str] = None,
) -> QwenOmniBridge:
    """
    Create Qwen bridge with sensible defaults.
    
    Args:
        enable_vision: Try to load vision-capable model
        load_in_4bit: Use 4-bit quantization
        device: Override device
        
    Returns:
        Configured QwenOmniBridge
    """
    config = QwenOmniConfig(
        enable_vision=enable_vision,
        load_in_4bit=load_in_4bit,
    )
    return QwenOmniBridge(config=config, device=device)


# ─────────────────────────────────────────────
# Module Exports
# ─────────────────────────────────────────────

__all__ = [
    'QwenOmniBridge',
    'QwenOmniConfig',
    'create_qwen_bridge',
]
