# Phase Voice: Embedded Multimodal Voice Module
## Replace External LLM with Self-Contained Qwen2.5-Omni

> Prerequisites:
> - `Flux-Apex-V1.flx` (v4.0-multi-modal-enhanced)
> - Qwen2.5-Omni-7B from HuggingFace
>
> Copilot: Open FLUX_FILE_FORMAT.md + QWEN_OMNI_REFERENCE.md + this file.
>
> **This phase embeds Qwen2.5-Omni directly into the .flx format,
> eliminating the need for external LLM downloads at runtime.**

---

## The Vision

Currently, Flux-Apex requires downloading an external LLM (Qwen2.5-3B) at runtime.
This phase embeds a multimodal voice model directly into the .flx file:

```
BEFORE (v4.0):
┌─────────────────────────────────────────────────────────────┐
│ Flux-Apex-V1.flx (5.79 GB)                                  │
│ ├── cse, field, memory, decoder, causal, adapters...       │
│ └── llm_reference: "Qwen/Qwen2.5-3B-Instruct"  ← EXTERNAL  │
│                           ↓                                 │
│                  [Download at runtime]                      │
│                           ↓                                 │
│           HuggingFace: Qwen2.5-3B (~2 GB download)         │
└─────────────────────────────────────────────────────────────┘

AFTER (v5.0):
┌─────────────────────────────────────────────────────────────┐
│ Flux-Apex-V1.flx (8.6 GB)                                   │
│ ├── cse, field, memory, decoder (legacy), causal, adapters │
│ └── voice: {                           ← EMBEDDED          │
│       thinker: {...},  # Text + Audio + Vision reasoning   │
│       talker: {...},   # Speech synthesis                  │
│       token2wav: {...} # Vocoder                           │
│     }                                                       │
│     NO EXTERNAL DOWNLOADS NEEDED                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Why Qwen2.5-Omni?

| Capability | Qwen2.5-3B (current) | Qwen2.5-Omni-7B |
|------------|---------------------|-----------------|
| Text generation | ✓ | ✓ |
| Audio input | ✗ | ✓ |
| Audio output (speech) | ✗ | ✓ |
| Vision input | ✗ | ✓ |
| Video input | ✗ | ✓ |
| Size (4-bit) | ~1.5 GB | ~2.8 GB |
| Any-to-any | ✗ | ✓ |

**Qwen2.5-Omni gives FLUX true multimodal voice in a single embedded module.**

---

## Module Architecture

```
                        USER INPUT
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
       ┌─────────┐    ┌──────────┐    ┌─────────┐
       │  Text   │    │  Audio   │    │  Image  │
       │ (UTF-8) │    │ (Mel)    │    │ (RGB)   │
       └────┬────┘    └────┬─────┘    └────┬────┘
            │              │               │
            ▼              ▼               ▼
    ┌────────────────────────────────────────────┐
    │               THINKER                      │
    │  ┌──────────┐  ┌───────────┐  ┌─────────┐ │
    │  │ text_emb │  │audio_tower│  │ visual  │ │
    │  └────┬─────┘  └─────┬─────┘  └────┬────┘ │
    │       │              │              │      │
    │       └──────────────┼──────────────┘      │
    │                      ▼                     │
    │         ┌────────────────────────┐         │
    │         │   model (28 layers)    │         │
    │         │   Qwen2.5 Transformer  │         │
    │         └───────────┬────────────┘         │
    │                     │                      │
    │                     ▼                      │
    │              ┌──────────┐                  │
    │              │ lm_head  │                  │
    │              └────┬─────┘                  │
    └───────────────────┼────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
   ┌─────────┐    ┌──────────┐    ┌──────────┐
   │  Text   │    │  TALKER  │    │  FLUX    │
   │ Output  │    │  (TTS)   │    │  Field   │
   └─────────┘    └────┬─────┘    └──────────┘
                       │
                       ▼
               ┌───────────────┐
               │   TOKEN2WAV   │
               │   (Vocoder)   │
               └───────┬───────┘
                       │
                       ▼
                 Audio Output
```

---

## .flx v5.0 Voice Format

```python
'voice': {
    'format_version': '1.0',
    'base_model': 'Qwen/Qwen2.5-Omni-7B',
    'quantization': '4bit',  # 'none' | '4bit' | '8bit' | 'svd'
    
    'config': {
        # Thinker config
        'thinker': {
            'text_hidden_size': 3584,
            'text_layers': 28,
            'text_heads': 28,
            'text_kv_heads': 4,
            'vocab_size': 152064,
            'audio_d_model': 1280,
            'audio_layers': 32,
            'audio_output_dim': 3584,
        },
        # Talker config
        'talker': {
            'hidden_size': 896,
            'num_layers': 24,
            'num_heads': 12,
            'vocab_size': 8448,  # TTS codec tokens
        },
        # Token2Wav config
        'token2wav': {
            'bigvgan_channels': 512,
            'dit_layers': 12,
        },
    },
    
    # Weights (4-bit quantized)
    'thinker': {
        'audio_tower': {...},  # 489 weight tensors
        'visual': {...},       # 518 weight tensors
        'model': {...},        # 338 weight tensors
        'lm_head': {...},      # 1 weight tensor
    },
    
    'talker': {
        'model': {...},                  # 290 weights
        'codec_head': {...},             # 1 weight
        'thinker_to_talker_proj': {...}, # 2 weights
    },
    
    'token2wav': {
        'code2wav_bigvgan_model': {...}, # 449 weights
        'code2wav_dit_model': {...},     # 360 weights
    },
    
    # Tokenizer data (embedded)
    'tokenizer': {
        'vocab': {...},        # Full vocabulary
        'merges': {...},       # BPE merges
        'special_tokens': {
            'audio_start': 151647,
            'audio_end': 151648,
            'audio_token': 151646,
            'image_token': 151655,
            'video_token': 151656,
            'bos': 151644,
            'eos': 151645,
            'pad': 151643,
        },
    },
}
```

---

## Bridge to FLUX 432-dim Waves

The voice module connects to FLUX's wave space via projection layers:

```python
'bridges': {
    # Existing bridges...
    'wave_to_field': {...},
    'field_to_wave': {...},
    
    # NEW: Voice bridges
    'wave_to_voice': {
        # Project FLUX 432-dim wave → Thinker 3584-dim hidden
        'projection': Linear(432, 3584),
        'norm': LayerNorm(3584),
    },
    'voice_to_wave': {
        # Project Thinker 3584-dim hidden → FLUX 432-dim wave
        'projection': Linear(3584, 432),
        'norm': LayerNorm(432),
    },
}
```

---

## Runtime Config Updates

```python
@dataclass
class VoiceConfig:
    """Controls embedded voice module."""
    enabled: bool = True
    model_type: str = 'qwen_omni'          # 'qwen_omni' | 'custom'
    quantization: str = '4bit'              # 'none' | '4bit' | '8bit'
    
    # Generation settings
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Modality settings
    text_enabled: bool = True
    audio_input_enabled: bool = True
    audio_output_enabled: bool = True
    vision_enabled: bool = True
    
    # FLUX integration
    use_flux_context: bool = True           # Inject field context
    flux_context_limit: int = 10            # Max field retrievals
    store_to_field: bool = True             # Store outputs to field


@dataclass
class GenerationConfig:
    """Updated generation config."""
    voice_primary: bool = True               # Use embedded voice (NEW)
    llm_primary: bool = False                # DEPRECATED - use voice
    byte_decoder_enabled: bool = False       # LEGACY - disabled by default
    byte_decoder_learns_from_voice: bool = True  # Distill from voice
    generation_mode: str = 'voice'           # 'voice' | 'byte' | 'hybrid'
```

---

## Component Status Changes

| Component | v4.0 Status | v5.0 Status | Notes |
|-----------|-------------|-------------|-------|
| decoder | Active | Legacy | Replaced by voice.thinker |
| llm | Reference only | Removed | Replaced by embedded voice |
| llm_reference | Active | Legacy | No longer needed |
| voice | N/A | Active | NEW: Embedded Qwen-Omni |
| voice.thinker | N/A | Active | Text/Audio/Vision reasoning |
| voice.talker | N/A | Active | Speech synthesis |
| voice.token2wav | N/A | Active | Vocoder |

---

## File Structure

```
phases/phase_voice/
├── PHASE_VOICE_SPEC.md           ← This file
├── voice_module.py               ← FLUXVoiceOmni class
├── voice_loader.py               ← Load/quantize Qwen-Omni
├── voice_bridges.py              ← Wave ↔ Voice projections
├── voice_tokenizer.py            ← Embedded tokenizer
├── quantize_qwen_omni.py         ← 4-bit quantization script
├── embed_voice_to_flx.py         ← Inject into Flux-Apex
├── test_phase_voice_test1.py     ← Test: Voice generation
├── test_phase_voice_test2.py     ← Test: Audio I/O
├── test_phase_voice_test3.py     ← Test: Vision input
├── demo_phase_voice_demo1.py     ← Demo: Multimodal chat
├── demo_phase_voice_demo2.py     ← Demo: Speech synthesis
└── RESULTS_PHASE_VOICE.md        ← Auto-generated
```

---

## Implementation Steps

### Step 1: Download & Quantize Qwen-Omni

```bash
python phases/phase_voice/quantize_qwen_omni.py
# Downloads Qwen2.5-Omni-7B
# Quantizes to 4-bit
# Saves to checkpoints/qwen_omni_4bit.pt
```

### Step 2: Create Voice Module Wrapper

```python
class FLUXVoiceOmni(nn.Module):
    """
    Embedded multimodal voice module for FLUX.
    Wraps Qwen2.5-Omni for text/audio/vision generation.
    """
    
    def __init__(self, state_dict: Dict[str, Any], device: str = 'cuda'):
        super().__init__()
        self.config = state_dict['config']
        self.load_thinker(state_dict['thinker'])
        self.load_talker(state_dict['talker'])
        self.load_token2wav(state_dict['token2wav'])
        self.load_tokenizer(state_dict['tokenizer'])
    
    def generate_text(
        self,
        prompt: str,
        flux_context: Optional[Tensor] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from prompt, optionally with FLUX field context."""
        ...
    
    def generate_speech(
        self,
        text: str,
        voice_style: str = 'default',
    ) -> Tensor:
        """Generate speech audio from text."""
        ...
    
    def understand_audio(
        self,
        audio: Tensor,
        sample_rate: int = 16000,
    ) -> Tensor:
        """Encode audio to hidden representations."""
        ...
    
    def understand_image(
        self,
        image: Tensor,
    ) -> Tensor:
        """Encode image to hidden representations."""
        ...
```

### Step 3: Embed into Flux-Apex

```python
from flux_model import FLUXModel
from phases.phase_voice.voice_loader import load_qwen_omni_4bit

# Load current model
model = FLUXModel.load('checkpoints/Flux-Apex-V1.flx')

# Load quantized Qwen-Omni
voice_state = load_qwen_omni_4bit('checkpoints/qwen_omni_4bit.pt')

# Add voice module
model.add_component('voice', voice_state, config={
    'type': 'qwen_omni',
    'quantization': '4bit',
})

# Mark byte decoder as legacy
model.state['decoder']['legacy'] = True
model.state['decoder']['legacy_reason'] = 'Replaced by embedded voice module'
model.state['decoder']['legacy_since'] = '2026-03-30'

# Remove external LLM reference
model.state['llm'] = None
model.state['llm_reference']['legacy'] = True

# Update version
model.metadata['version'] = '5.0-voice-embedded'

# Save
model.save('checkpoints/Flux-Apex-V1.flx', overwrite=True)
```

### Step 4: Update Generation Pipeline

```python
class FLUXModel:
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        mode: str = 'auto',  # 'voice' | 'byte' | 'auto'
    ) -> GenerationResult:
        """
        Generate response using embedded voice module.
        
        1. Encode prompt via CSE → wave
        2. Query field for relevant context
        3. Project wave → voice hidden dim
        4. Generate via voice.thinker
        5. Store response in field
        6. Return text (+ optional audio)
        """
        # Get FLUX context
        wave = self.cse.encode(prompt)
        field_context = self.query_field(wave)
        
        # Project to voice space
        voice_context = self.bridges.wave_to_voice(field_context)
        
        # Generate
        if mode == 'voice' or (mode == 'auto' and self.voice is not None):
            response = self.voice.generate_text(
                prompt,
                flux_context=voice_context,
                max_tokens=max_length,
                temperature=temperature,
            )
        else:
            # Fallback to byte decoder (legacy)
            response = self.decoder.generate(...)
        
        # Store in field
        response_wave = self.cse.encode(response)
        self.field.inject(response_wave)
        
        return GenerationResult(text=response)
```

---

## Size Analysis

| Component | Current (v4.0) | After Voice (v5.0) |
|-----------|----------------|-------------------|
| Core FLUX | 5.79 GB | 5.79 GB |
| External LLM (runtime) | +1.5 GB download | — |
| Embedded Voice | — | +2.8 GB |
| **Total in .flx** | 5.79 GB | 8.6 GB |
| **Runtime downloads** | 1.5 GB | 0 |

**Net result:** File is larger, but no runtime downloads. Fully self-contained.

---

## Acceptance Criteria

1. [ ] Qwen-Omni quantized to 4-bit and embedded in .flx
2. [ ] Voice generates coherent text from prompts
3. [ ] Voice understands audio input
4. [ ] Voice understands image input
5. [ ] Voice generates speech output
6. [ ] Wave ↔ Voice bridges functional
7. [ ] Field context injection works
8. [ ] Byte decoder marked as legacy
9. [ ] No external downloads required at runtime
10. [ ] Generation quality ≥ external LLM baseline

---

*Phase Voice: Making FLUX truly self-contained with embedded multimodal voice.*
