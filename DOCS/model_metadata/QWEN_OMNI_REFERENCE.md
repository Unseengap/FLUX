# Qwen2.5-Omni-7B Model Reference

**For FLUX Integration Planning**
*Extracted: March 30, 2026*

---

## Overview

| Property | Value |
|----------|-------|
| Model | Qwen2.5-Omni-7B |
| Architecture | Qwen2_5OmniModel |
| Total Size | 22.37 GB (full) |
| 4-bit Quantized | ~2.8 GB |
| Capabilities | Text + Audio + Vision (any-to-any) |

---

## Module Structure

### 1. THINKER (Main LLM - 1346 weights)

The reasoning/thinking component based on Qwen2.5.

**Text Model (28 layers):**
- hidden_size: 3584
- num_attention_heads: 28
- num_key_value_heads: 4 (GQA)
- intermediate_size: 18944
- vocab_size: 152,064
- max_position_embeddings: 32,768

**Audio Encoder (32 layers):**
- d_model: 1280
- encoder_attention_heads: 20
- encoder_ffn_dim: 5120
- output_dim: 3584 (projects to text hidden)
- num_mel_bins: 128

**Vision Encoder (NaViT-based):**
- Embedded in thinker architecture
- Supports images and video

**Thinker Submodules:**
- `audio_tower`: 489 weights
- `lm_head`: 1 weights
- `model`: 338 weights
- `visual`: 518 weights


### 2. TALKER (Speech Synthesis - 293 weights)

Generates speech from text/tokens.

**Architecture:**
- hidden_size: 896
- num_hidden_layers: 24
- num_attention_heads: 12
- num_key_value_heads: 4
- vocab_size: 8448 (TTS codec tokens)

**Talker Submodules:**
- `codec_head`: 1 weights
- `model`: 290 weights
- `thinker_to_talker_proj`: 2 weights


### 3. TOKEN2WAV (Vocoder - 809 weights)

Converts tokens to audio waveforms.

**Submodules:**
- `code2wav_bigvgan_model`: 449 weights
- `code2wav_dit_model`: 360 weights


---

## FLUX Integration Options

### Option A: Full Embedding (4-bit quantized)

```
Flux-Apex + Qwen-Omni (4-bit) = ~8.6 GB total

'voice': {
    'type': 'qwen_omni',
    'quantization': '4bit',
    'thinker': {...},   # Text + Audio + Vision reasoning
    'talker': {...},    # Speech synthesis
    'token2wav': {...}, # Vocoder
}
```

**Pros:**
- Full multimodal capabilities (text + audio + vision)
- No external downloads needed
- Complete any-to-any generation

**Cons:**
- Large file size (8.6 GB)
- Complex integration

### Option B: Thinker Only (Text + Audio Input)

```
Flux-Apex + Thinker (4-bit) = ~7.5 GB

'voice': {
    'type': 'qwen_omni_thinker',
    'quantization': '4bit',
    'thinker': {...},   # Text + Audio understanding
}
```

**Pros:**
- Smaller than full model
- Still gets audio understanding
- Text generation capabilities

**Cons:**
- No speech output
- Still large

### Option C: SVD Compressed Thinker

```
Flux-Apex + Thinker (SVD 25%) = ~7.8 GB

'voice': {
    'type': 'qwen_omni_thinker_svd',
    'compression': 'svd',
    'rank_ratio': 0.25,
    'thinker_compressed': {...},
}
```

**Pros:**
- Better quality preservation than 4-bit
- Smaller than full precision

**Cons:**
- Runtime decompression overhead

### Option D: Separate Modular Files (Recommended Start)

```
Flux-Apex-V1.flx         (5.79 GB) - Core model
Flux-Voice-Omni.flx      (2.8 GB)  - Thinker + Talker (4-bit)
Flux-Vocoder-Omni.flx    (0.5 GB)  - Token2Wav only

Total: 9.1 GB (but loadable separately)
```

**Pros:**
- Load only what you need
- Modular upgrades
- Backward compatible

**Cons:**
- Multiple files
- Slightly more complex loading

---

## Special Token IDs

| Token Type | ID |
|------------|-----|
| audio_start | 151647 |
| audio_end | 151648 |
| audio_token | 151646 |
| image_token | 151655 |
| video_token | 151656 |
| vision_start | 151652 |
| vision_end | 151653 |
| bos | 151644 |
| eos | 151645 |
| pad | 151643 |
| user | 872 |

---

## Component Mapping to FLUX

| Qwen Omni | FLUX Component | Notes |
|-----------|----------------|-------|
| thinker.audio_encoder | adapters.audio_to_wave | Audio → 3584-dim → wave |
| thinker.text_model | voice.llm | Main text generation |
| thinker.vision_tower | adapters.vision_to_wave | Vision → wave |
| talker | voice.talker | Wave → speech tokens |
| token2wav | voice.vocoder | Tokens → audio |

---

## Recommended Implementation Path

1. **Phase 1**: Download and test Qwen-Omni standalone
2. **Phase 2**: Create 4-bit quantized version locally
3. **Phase 3**: Create FLUXVoiceOmni wrapper class
4. **Phase 4**: Add to .flx format as 'voice' component
5. **Phase 5**: Bridge thinker hidden states ↔ FLUX 432-dim waves
6. **Phase 6**: Mark byte decoder as legacy
7. **Phase 7**: Test generation quality

---

*This document is auto-generated for FLUX integration planning.*
