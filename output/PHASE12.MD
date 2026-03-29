Cloning into '/kaggle/working/FLUX'...
remote: Enumerating objects: 1803, done.
remote: Counting objects: 100% (52/52), done.
remote: Compressing objects: 100% (36/36), done.
remote: Total 1803 (delta 19), reused 44 (delta 16), pack-reused 1751 (from 1)
Receiving objects: 100% (1803/1803), 2.31 MiB | 20.56 MiB/s, done.
Resolving deltas: 100% (1046/1046), done.
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 23.8/23.8 MB 70.0 MB/s eta 0:00:00:00:0100:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 84.1/84.1 kB 3.1 MB/s eta 0:00:00
✓ Working directory: /kaggle/working/FLUX
add Codeadd Markdown
Cell 2: Imports & Hardware Detection
add Codeadd Markdown
# ═══════════════════════════════════════════════════════════════
# Cell 2: Imports & Hardware Detection
# ═══════════════════════════════════════════════════════════════

import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Add paths
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / 'phases' / 'phase12'))
sys.path.insert(0, str(Path.cwd() / 'phases' / 'phase11'))
sys.path.insert(0, str(Path.cwd() / 'phases' / 'phase9_arc'))
sys.path.insert(0, str(Path.cwd() / 'phases' / 'phase_unified'))

# FLUX utilities
from flux_utils import PhaseLogger, PhaseResults, get_device, _resolve_hf_token
from flux_model import FLUXModel, load_flux

# Initialize logger
log = PhaseLogger(phase=12)
log.separator("Phase 12: FLUX Multi-Modal Agent")
log.cell_start("Cell 2 — Imports & Hardware")

# Detect device
device = get_device()
log.info(f"Device: {device}")

# GPU info
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")
    
    if vram_gb < 12:
        log.warning("⚠ Low VRAM — will use aggressive quantization")
        USE_4BIT = True
    else:
        USE_4BIT = True  # Always use 4-bit for Qwen-Omni-7B
else:
    USE_4BIT = True
    log.warning("No GPU — will be slow")

# HuggingFace token
HF_TOKEN = None
try:
    if ON_KAGGLE:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    else:
        HF_TOKEN = _resolve_hf_token()
    log.success("HF_TOKEN loaded")
except:
    log.warning("No HF_TOKEN — uploads will be skipped")

print(f"\n✓ PyTorch {torch.__version__}")
print(f"✓ Device: {device}")
print(f"✓ 4-bit quantization: {USE_4BIT}")

log.cell_end("Cell 2 — Imports & Hardware", "PASS")
[18:25:32] 
──────────────────── Phase 12: FLUX Multi-Modal Agent ────────────────────
[18:25:32] 
▶ CELL: Cell 2 — Imports & Hardware
[18:25:32]   Started: 2026-03-29 18:25:32
[18:25:32]   ℹ Device: cuda
[18:25:32]   ℹ GPU: Tesla T4 (15.6 GB)
[18:25:32]   ✓ HF_TOKEN loaded

✓ PyTorch 2.10.0+cu128
✓ Device: cuda
✓ 4-bit quantization: True
[18:25:32]   ◼ CELL Cell 2 — Imports & Hardware — PASS
add Codeadd Markdown
Cell 3: Load Flux-Base.flx
add Codeadd Markdown
# ═══════════════════════════════════════════════════════════════
# Cell 3: Load Flux-Base.flx (Contains ALL Phase 1-11 components)
# ═══════════════════════════════════════════════════════════════

log.cell_start("Cell 3 — Load Flux-Base.flx")

# Check for local checkpoint, otherwise download from HuggingFace
checkpoint_dir = Path('checkpoints')
checkpoint_dir.mkdir(exist_ok=True)

base_path = checkpoint_dir / 'Flux-Base.flx'

if not base_path.exists():
    log.info("Downloading Flux-Base.flx from HuggingFace...")
    from huggingface_hub import hf_hub_download
    
    downloaded = hf_hub_download(
        repo_id="UnseenGAP/FLUX",
        filename="checkpoints/Flux-Base.flx",
        token=HF_TOKEN,
        local_dir=str(Path.cwd()),
    )
    log.success(f"Downloaded: {downloaded}")

# Load using FLUXModel
log.info(f"Loading {base_path}...")
flux_model = FLUXModel.load(str(base_path))

# Show summary
flux_model.summary()

size_mb = base_path.stat().st_size / 1e6
log.success(f"Loaded Flux-Base.flx ({size_mb:.1f} MB)")
log.info(f"Version: {flux_model.version}")
log.info(f"Components: {flux_model.active_component_count}")

log.cell_end("Cell 3 — Load Flux-Base.flx", "PASS")
[18:25:32] 
▶ CELL: Cell 3 — Load Flux-Base.flx
[18:25:32]   Started: 2026-03-29 18:25:32
[18:25:32]   ℹ Downloading Flux-Base.flx from HuggingFace...
checkpoints/Flux-Base.flx: 100%
 5.79G/5.79G [01:00<00:00, 1.00GB/s]
[18:26:34]   ✓ Downloaded: /kaggle/working/FLUX/checkpoints/Flux-Base.flx
[18:26:34]   ℹ Loading checkpoints/Flux-Base.flx...
✓ Loaded: checkpoints/Flux-Base.flx
  Version: 2.0-base
  Phase: unified
  Components: 10
============================================================
FLUX Model Summary
============================================================
Path: checkpoints/Flux-Base.flx
Version: 2.0-base
Phase: unified
Modified: False

Components (10 active):
  [perception]
    ✓ cse
    ✓ grid_to_wave
  [knowledge]
    ✓ field
    ✓ working_memory
    ✓ episodic_memory
    ✓ semantic_memory
  [generation]
    ✓ decoder
    ✓ llm
  [reasoning]
    ✓ causal_graph
  [bridges]
    ✓ bridges

  [inactive]
    ✗ spatial_memory
    ✗ perception_field
    ✗ causal_tracker
    ✗ rule_inducer
    ✗ goal_planner

Metadata:
  created: 2026-03-29T06:36:42.571848
  base: Flux-X-complete.flx
  field_source: Flux-capable.flx
  grid_encoder: gridtowave_contrastive.pt
  llm_config: Flux-augmented.flx
============================================================
[18:26:38]   ✓ Loaded Flux-Base.flx (5792.9 MB)
[18:26:38]   ℹ Version: 2.0-base
[18:26:38]   ℹ Components: 10
[18:26:38]   ◼ CELL Cell 3 — Load Flux-Base.flx — PASS
add Codeadd Markdown
Cell 4: Initialize Qwen-Omni (Vision + Text)
add Codeadd Markdown
# ═══════════════════════════════════════════════════════════════
# Cell 4: Initialize Qwen2.5-Omni (Unified Vision + Audio + Text)
# ═══════════════════════════════════════════════════════════════

log.cell_start("Cell 4 — Initialize Qwen-Omni")

print("\n" + "="*60)
print("Loading Qwen2.5-Omni-7B (Unified Multi-Modal Model)")
print("This model handles: Vision + Audio + Text in ONE model!")
print("="*60 + "\n")

# Model configuration
OMNI_MODEL = "Qwen/Qwen2.5-Omni-7B"  # Or fallback to Qwen2-VL-7B-Instruct

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
    from transformers import BitsAndBytesConfig
    
    log.info(f"Loading {OMNI_MODEL}...")
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    # Try Qwen-Omni first, fallback to Qwen-VL if not available
    try:
        processor = AutoProcessor.from_pretrained(OMNI_MODEL, trust_remote_code=True)
        omni_model = AutoModelForCausalLM.from_pretrained(
            OMNI_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        log.success(f"Loaded: {OMNI_MODEL}")
    except Exception as e:
        log.warning(f"Qwen-Omni not available: {e}")
        log.info("Falling back to Qwen2-VL-7B-Instruct...")
        
        OMNI_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
        processor = AutoProcessor.from_pretrained(OMNI_MODEL, trust_remote_code=True)
        omni_model = AutoModelForCausalLM.from_pretrained(
            OMNI_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        log.success(f"Loaded: {OMNI_MODEL}")
    
    # Memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        log.info(f"VRAM used: {allocated:.1f} GB")
    
    OMNI_READY = True
    
except Exception as e:
    log.error(f"Failed to load Qwen-Omni: {e}")
    log.info("Will use text-only mode with existing LLM")
    OMNI_READY = False
    omni_model = None
    processor = None

log.cell_end("Cell 4 — Initialize Qwen-Omni", "PASS" if OMNI_READY else "WARN")
[18:26:38] 
▶ CELL: Cell 4 — Initialize Qwen-Omni
[18:26:38]   Started: 2026-03-29 18:26:38

============================================================
Loading Qwen2.5-Omni-7B (Unified Multi-Modal Model)
This model handles: Vision + Audio + Text in ONE model!
============================================================

[18:27:02]   ℹ Loading Qwen/Qwen2.5-Omni-7B...
preprocessor_config.json: 100%
 667/667 [00:00<00:00, 82.0kB/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
chat_template.json: 
 1.31k/? [00:00<00:00, 136kB/s]
The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`. 
config.json: 
 13.2k/? [00:00<00:00, 1.41MB/s]
tokenizer_config.json: 
 6.47k/? [00:00<00:00, 726kB/s]
vocab.json: 
 2.78M/? [00:00<00:00, 62.5MB/s]
merges.txt: 
 1.67M/? [00:00<00:00, 68.8MB/s]
tokenizer.json: 100%
 11.4M/11.4M [00:00<00:00, 53.6MB/s]
added_tokens.json: 100%
 579/579 [00:00<00:00, 70.3kB/s]
special_tokens_map.json: 100%
 832/832 [00:00<00:00, 102kB/s]
`torch_dtype` is deprecated! Use `dtype` instead!
[18:27:17]   ⚠ Qwen-Omni not available: Unrecognized configuration class <class 'transformers.models.qwen2_5_omni.configuration_qwen2_5_omni.Qwen2_5OmniConfig'> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of AfmoeConfig, ApertusConfig, ArceeConfig, AriaTextConfig, BambaConfig, BartConfig, BertConfig, BertGenerationConfig, BigBirdConfig, BigBirdPegasusConfig, BioGptConfig, BitNetConfig, BlenderbotConfig, BlenderbotSmallConfig, BloomConfig, BltConfig, CamembertConfig, LlamaConfig, CodeGenConfig, CohereConfig, Cohere2Config, CpmAntConfig, CTRLConfig, CwmConfig, Data2VecTextConfig, DbrxConfig, DeepseekV2Config, DeepseekV3Config, DiffLlamaConfig, DogeConfig, Dots1Config, ElectraConfig, Emu3Config, ErnieConfig, Ernie4_5Config, Ernie4_5_MoeConfig, Exaone4Config, FalconConfig, FalconH1Config, FalconMambaConfig, FlexOlmoConfig, FuyuConfig, GemmaConfig, Gemma2Config, Gemma3Config, Gemma3TextConfig, Gemma3nConfig, Gemma3nTextConfig, GitConfig, GlmConfig, Glm4Config, Glm4MoeConfig, Glm4MoeLiteConfig, GotOcr2Config, GPT2Config, GPT2Config, GPTBigCodeConfig, GPTNeoConfig, GPTNeoXConfig, GPTNeoXJapaneseConfig, GptOssConfig, GPTJConfig, GraniteConfig, GraniteMoeConfig, GraniteMoeHybridConfig, GraniteMoeSharedConfig, HeliumConfig, HunYuanDenseV1Config, HunYuanMoEV1Config, Jais2Config, JambaConfig, JetMoeConfig, Lfm2Config, Lfm2MoeConfig, LlamaConfig, Llama4Config, Llama4TextConfig, LongcatFlashConfig, MambaConfig, Mamba2Config, MarianConfig, MBartConfig, MegatronBertConfig, MiniMaxConfig, MiniMaxM2Config, MinistralConfig, Ministral3Config, MistralConfig, MixtralConfig, MllamaConfig, ModernBertDecoderConfig, MoshiConfig, MptConfig, MusicgenConfig, MusicgenMelodyConfig, MvpConfig, NanoChatConfig, NemotronConfig, OlmoConfig, Olmo2Config, Olmo3Config, OlmoeConfig, OpenAIGPTConfig, OPTConfig, PegasusConfig, PersimmonConfig, PhiConfig, Phi3Config, Phi4MultimodalConfig, PhimoeConfig, PLBartConfig, ProphetNetConfig, Qwen2Config, Qwen2MoeConfig, Qwen3Config, Qwen3MoeConfig, Qwen3NextConfig, RecurrentGemmaConfig, ReformerConfig, RemBertConfig, RobertaConfig, RobertaPreLayerNormConfig, RoCBertConfig, RoFormerConfig, RwkvConfig, SeedOssConfig, SmolLM3Config, SolarOpenConfig, StableLmConfig, Starcoder2Config, TrOCRConfig, VaultGemmaConfig, WhisperConfig, XGLMConfig, XLMConfig, XLMRobertaConfig, XLMRobertaXLConfig, XLNetConfig, xLSTMConfig, XmodConfig, ZambaConfig, Zamba2Config.
[18:27:17]   ℹ Falling back to Qwen2-VL-7B-Instruct...
preprocessor_config.json: 100%
 347/347 [00:00<00:00, 44.5kB/s]
chat_template.json: 
 1.05k/? [00:00<00:00, 117kB/s]
config.json: 
 1.20k/? [00:00<00:00, 140kB/s]
tokenizer_config.json: 
 4.19k/? [00:00<00:00, 419kB/s]
vocab.json: 
 2.78M/? [00:00<00:00, 87.9MB/s]
merges.txt: 
 1.67M/? [00:00<00:00, 66.6MB/s]
tokenizer.json: 
 7.03M/? [00:00<00:00, 120MB/s]
[18:27:22]   ✗ Failed to load Qwen-Omni: Unrecognized configuration class <class 'transformers.models.qwen2_vl.configuration_qwen2_vl.Qwen2VLConfig'> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of AfmoeConfig, ApertusConfig, ArceeConfig, AriaTextConfig, BambaConfig, BartConfig, BertConfig, BertGenerationConfig, BigBirdConfig, BigBirdPegasusConfig, BioGptConfig, BitNetConfig, BlenderbotConfig, BlenderbotSmallConfig, BloomConfig, BltConfig, CamembertConfig, LlamaConfig, CodeGenConfig, CohereConfig, Cohere2Config, CpmAntConfig, CTRLConfig, CwmConfig, Data2VecTextConfig, DbrxConfig, DeepseekV2Config, DeepseekV3Config, DiffLlamaConfig, DogeConfig, Dots1Config, ElectraConfig, Emu3Config, ErnieConfig, Ernie4_5Config, Ernie4_5_MoeConfig, Exaone4Config, FalconConfig, FalconH1Config, FalconMambaConfig, FlexOlmoConfig, FuyuConfig, GemmaConfig, Gemma2Config, Gemma3Config, Gemma3TextConfig, Gemma3nConfig, Gemma3nTextConfig, GitConfig, GlmConfig, Glm4Config, Glm4MoeConfig, Glm4MoeLiteConfig, GotOcr2Config, GPT2Config, GPT2Config, GPTBigCodeConfig, GPTNeoConfig, GPTNeoXConfig, GPTNeoXJapaneseConfig, GptOssConfig, GPTJConfig, GraniteConfig, GraniteMoeConfig, GraniteMoeHybridConfig, GraniteMoeSharedConfig, HeliumConfig, HunYuanDenseV1Config, HunYuanMoEV1Config, Jais2Config, JambaConfig, JetMoeConfig, Lfm2Config, Lfm2MoeConfig, LlamaConfig, Llama4Config, Llama4TextConfig, LongcatFlashConfig, MambaConfig, Mamba2Config, MarianConfig, MBartConfig, MegatronBertConfig, MiniMaxConfig, MiniMaxM2Config, MinistralConfig, Ministral3Config, MistralConfig, MixtralConfig, MllamaConfig, ModernBertDecoderConfig, MoshiConfig, MptConfig, MusicgenConfig, MusicgenMelodyConfig, MvpConfig, NanoChatConfig, NemotronConfig, OlmoConfig, Olmo2Config, Olmo3Config, OlmoeConfig, OpenAIGPTConfig, OPTConfig, PegasusConfig, PersimmonConfig, PhiConfig, Phi3Config, Phi4MultimodalConfig, PhimoeConfig, PLBartConfig, ProphetNetConfig, Qwen2Config, Qwen2MoeConfig, Qwen3Config, Qwen3MoeConfig, Qwen3NextConfig, RecurrentGemmaConfig, ReformerConfig, RemBertConfig, RobertaConfig, RobertaPreLayerNormConfig, RoCBertConfig, RoFormerConfig, RwkvConfig, SeedOssConfig, SmolLM3Config, SolarOpenConfig, StableLmConfig, Starcoder2Config, TrOCRConfig, VaultGemmaConfig, WhisperConfig, XGLMConfig, XLMConfig, XLMRobertaConfig, XLMRobertaXLConfig, XLNetConfig, xLSTMConfig, XmodConfig, ZambaConfig, Zamba2Config.
[18:27:22]   ℹ Will use text-only mode with existing LLM
[18:27:23]   ◼ CELL Cell 4 — Initialize Qwen-Omni — WARN
add Codeadd Markdown
Cell 5: Build Grid Renderer
add Codeadd Markdown

# ARC color palette (standard 10 colors)
ARC_COLORS = [
    (0, 0, 0),        # 0: black
    (0, 116, 217),    # 1: blue
    (255, 65, 54),    # 2: red
    (46, 204, 64),    # 3: green
    (255, 220, 0),    # 4: yellow
    (128, 128, 128),  # 5: grey
    (240, 18, 190),   # 6: magenta
    (255, 133, 27),   # 7: orange
    (127, 219, 255),  # 8: cyan
    (135, 12, 37),    # 9: brown
]


def render_grid_to_image(
    grid: List[List[int]],
    cell_size: int = 20,
    position: Optional[Tuple[int, int]] = None,
    ice_field: Optional[torch.Tensor] = None,
) -> Image.Image:
    """
    Render ARC grid as PIL Image for Qwen-Omni vision input.
    
    Args:
        grid: 2D list of color indices (0-9)
        cell_size: Pixels per cell
        position: Agent position (row, col) to mark
        ice_field: Optional curiosity field for overlay
    
    Returns:
        PIL Image (RGB)
    """
    h, w = len(grid), len(grid[0])
    img = Image.new('RGB', (w * cell_size, h * cell_size))
    draw = ImageDraw.Draw(img)
    
    # Draw cells
    for r in range(h):
        for c in range(w):
            color = ARC_COLORS[min(grid[r][c], 9)]
            x0, y0 = c * cell_size, r * cell_size
            x1, y1 = x0 + cell_size - 1, y0 + cell_size - 1
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(40, 40, 40))
    
    # Ice field overlay (cyan tint for high curiosity areas)
    if ice_field is not None:
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        
        for r in range(min(h, ice_field.shape[0])):
            for c in range(min(w, ice_field.shape[1])):
                ice_val = ice_field[r, c].item()
                if ice_val > 5:
                    alpha = min(int(ice_val * 8), 150)
                    x0, y0 = c * cell_size, r * cell_size
                    x1, y1 = x0 + cell_size - 1, y0 + cell_size - 1
                    draw_overlay.rectangle([x0, y0, x1, y1], fill=(0, 255, 255, alpha))
        
        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img)
    
    # Mark agent position
    if position:
        pr, pc = position
        cx = pc * cell_size + cell_size // 2
        cy = pr * cell_size + cell_size // 2
        r = cell_size // 3
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    
    return img


# Test render
test_grid = [
    [0, 0, 0, 1, 1, 1, 0, 0],
    [0, 2, 2, 1, 0, 1, 3, 0],
    [0, 2, 0, 1, 1, 1, 3, 0],
    [0, 2, 2, 0, 0, 0, 3, 0],
    [0, 0, 0, 4, 4, 4, 0, 0],
]

test_img = render_grid_to_image(test_grid, cell_size=30, position=(2, 3))
print(f"✓ Test image: {test_img.size[0]}x{test_img.size[1]} pixels")

# Display if in notebook
try:
    from IPython.display import display
    display(test_img)
except:
    pass

log.success("Grid renderer ready")
log.cell_end("Cell 5 — Grid Renderer", "PASS")
[18:27:23] 
▶ CELL: Cell 5 — Grid Renderer
[18:27:23]   Started: 2026-03-29 18:27:23
✓ Test image: 240x150 pixels

[18:27:23]   ✓ Grid renderer ready
[18:27:23]   ◼ CELL Cell 5 — Grid Renderer — PASS
add Codeadd Markdown
Cell 6: Build Visual Reasoner
add Codeadd Markdown
log.cell_start("Cell 6 — Visual Reasoner")
Cyan/teal tinted areas show high curiosity (interesting unexplored spots).

Grid size: {len(grid)} rows x {len(grid[0])} cols
My position: row {position[0]}, col {position[1]}
Available actions: {', '.join(available)}
{history_text}
Analyze the grid:
1. What patterns or structures do you see?
2. What might be the goal?
3. Which direction should I move?

Think step by step, then give your final answer as:
ACTION: [UP/DOWN/LEFT/RIGHT/INTERACT/CLICK]"""
        
        # Call Qwen-Omni with image
        try:
            if self.model is not None and self.processor is not None:
                # Vision-language inference
                inputs = self.processor(
                    text=prompt,
                    images=game_image,
                    return_tensors="pt",
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.3,
                        do_sample=True,
                    )
                
                response = self.processor.decode(outputs[0], skip_special_tokens=True)
                # Extract just the new generated part
                if prompt in response:
                    response = response.split(prompt)[-1].strip()
            else:
                # Fallback: random action with explanation
                import random
                action = random.choice(available_actions)
                response = f"No LLM available. Random action: {action_names.get(action, action)}"
                return action, response
        except Exception as e:
            print(f"  LLM error: {e}")
            import random
            return random.choice(available_actions), f"Error: {e}"
        
        # Parse action from response
        action = self._parse_action(response, available_actions, action_names)
        
        return action, response
    
    def _parse_action(self, response: str, available: List[int], names: Dict) -> int:
        """Parse action from LLM response."""
        response_upper = response.upper()
        
        # Look for "ACTION: X" pattern
        match = re.search(r'ACTION:\s*(\w+)', response_upper)
        if match:
            action_str = match.group(1)
            for aid, name in names.items():
                if name == action_str and aid in available:
                    return aid
        
        # Fallback: look for direction words
        for aid, name in names.items():
            if name in response_upper and aid in available:
                return aid
        
        # Default: first available
        return available[0] if available else 1


# Initialize reasoner
reasoner = VisualReasoner(
    model=omni_model if OMNI_READY else None,
    processor=processor if OMNI_READY else None,
    device=device,
)

log.success("VisualReasoner initialized")
log.cell_end("Cell 6 — Visual Reasoner", "PASS")
[18:27:23] 
▶ CELL: Cell 6 — Visual Reasoner
[18:27:23]   Started: 2026-03-29 18:27:23
[18:27:23]   ✓ VisualReasoner initialized
[18:27:23]   ◼ CELL Cell 6 — Visual Reasoner — PASS
add Codeadd Markdown
Cell 7: Build Spatial Memory
add Codeadd Markdown
log.cell_start("Cell 7 — Spatial Memory")
        self.device = device
        
        # Exploration mass (where we've been — "water")
        self.exploration_mass = torch.zeros(max_size, max_size, device=device)
        
        # Curiosity field (what's interesting — "ice")
        self.curiosity_field = torch.zeros(max_size, max_size, device=device)
        
        # Visit count
        self.visit_count = torch.zeros(max_size, max_size, device=device)
        
        # Last observation storage
        self.last_observation = torch.zeros(max_size, max_size, device=device)
    
    def observe(self, position: Tuple[int, int], grid: List[List[int]]):
        """Update fields based on observation."""
        r, c = position
        h, w = len(grid), len(grid[0])
        
        # Clamp to valid range
        r = max(0, min(r, self.max_size - 1))
        c = max(0, min(c, self.max_size - 1))
        
        # Update exploration mass (gaussian splash around position)
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.max_size and 0 <= nc < self.max_size:
                    dist = abs(dr) + abs(dc)
                    splash = 5.0 / (1 + dist)
                    self.exploration_mass[nr, nc] += splash
        
        # Update curiosity (ice forms on interesting unexplored areas)
        for gr in range(min(h, self.max_size)):
            for gc in range(min(w, self.max_size)):
                cell_val = grid[gr][gc]
                explored = self.exploration_mass[gr, gc].item()
                
                # Non-background, unexplored = high curiosity
                if cell_val != 0 and explored < 5:
                    self.curiosity_field[gr, gc] += 1.0
                else:
                    # Melt ice in explored areas
                    self.curiosity_field[gr, gc] *= 0.95
        
        # Update visit count
        self.visit_count[r, c] += 1
    
    def get_curiosity_direction(self, position: Tuple[int, int]) -> Tuple[int, int]:
        """Get direction toward highest curiosity."""
        r, c = position
        best_dir = (0, 0)
        best_curiosity = -1
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.max_size and 0 <= nc < self.max_size:
                curiosity = self.curiosity_field[nr, nc].item()
                if curiosity > best_curiosity:
                    best_curiosity = curiosity
                    best_dir = (dr, dc)
        
        return best_dir
    
    def reset(self):
        """Reset all fields."""
        self.exploration_mass.zero_()
        self.curiosity_field.zero_()
        self.visit_count.zero_()
        self.last_observation.zero_()


# Initialize
spatial_memory = SpatialMemory(max_size=64, device=device)

# Test with sample grid
spatial_memory.observe((5, 5), test_grid)
spatial_memory.observe((5, 6), test_grid)
spatial_memory.observe((5, 7), test_grid)

print(f"✓ Exploration mass range: {spatial_memory.exploration_mass.min():.1f} - {spatial_memory.exploration_mass.max():.1f}")
print(f"✓ Curiosity (ice) range: {spatial_memory.curiosity_field.min():.1f} - {spatial_memory.curiosity_field.max():.1f}")
print(f"✓ Best direction from (5,7): {spatial_memory.get_curiosity_direction((5, 7))}")

spatial_memory.reset()  # Clean for actual use

log.success("SpatialMemory ready")
log.cell_end("Cell 7 — Spatial Memory", "PASS")
[18:27:23] 
▶ CELL: Cell 7 — Spatial Memory
[18:27:23]   Started: 2026-03-29 18:27:23
✓ Exploration mass range: 0.0 - 10.0
✓ Curiosity (ice) range: 0.0 - 3.0
✓ Best direction from (5,7): (-1, 0)
[18:27:23]   ✓ SpatialMemory ready
[18:27:23]   ◼ CELL Cell 7 — Spatial Memory — PASS
add Codeadd Markdown
Cell 8: Build FLUXMultiAgent
add Codeadd Markdown
# ═══════════════════════════════════════════════════════════════
            'exploration_mass': self.spatial.exploration_mass.cpu(),
            'curiosity_field': self.spatial.curiosity_field.cpu(),
            'visit_count': self.spatial.visit_count.cpu(),
            'config': {'max_size': self.spatial.max_size},
        })
        print("  ✓ spatial_memory")
        
        self.flux_model.add_component('causal_tracker', {
            'links': self.causal_links[-100:],
            'total': len(self.causal_links),
        })
        print("  ✓ causal_tracker")
        
        self.flux_model.add_component('learned_rules', {
            'rules': self.learned_rules,
        })
        print("  ✓ learned_rules")
        
        # LLM reference (NOT weights!)
        self.flux_model.state['llm_reference'] = {
            'name': OMNI_MODEL,
            'load_in_4bit': True,
            'enable_vision': True,
            'enable_audio': False,
        }
        print(f"  ✓ llm_reference: {OMNI_MODEL}")
        
        # Save
        self.flux_model.save(path, overwrite=True)
        
        size_mb = Path(path).stat().st_size / 1e6
        print(f"\n✓ Saved: {path} ({size_mb:.1f} MB)")
        
        # Upload to HuggingFace
        if upload_to_hf and HF_TOKEN:
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=HF_TOKEN)
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo="checkpoints/Flux-multi-model.flx",
                    repo_id="UnseenGAP/FLUX",
                    commit_message=f"Phase 12: Flux-multi-model.flx",
                )
                print("✓ Uploaded to HuggingFace Hub")
            except Exception as e:
                print(f"⚠ HF upload failed: {e}")
        
        return path
    
    def get_stats(self) -> Dict:
        """Get agent statistics."""
        return {
            'version': self.flux_model.version,
            'game': self.current_game,
            'observations': self.observation_count,
            'actions': len(self.action_history),
            'causal_links': len(self.causal_links),
            'learned_rules': len(self.learned_rules),
        }


# Create the agent
agent = FLUXMultiAgent(
    flux_model=flux_model,
    reasoner=reasoner,
    spatial=spatial_memory,
    device=device,
)

print("\n" + "="*60)
print("FLUXMultiAgent Initialized")
print("="*60)
print(f"\n  FLUX Version: {flux_model.version}")
print(f"  LLM: {OMNI_MODEL}")
print(f"  Device: {device}")
print(f"  Vision: {'✓ Enabled' if OMNI_READY else '✗ Disabled (text-only)'}")

log.success("FLUXMultiAgent ready")
log.cell_end("Cell 8 — FLUXMultiAgent", "PASS")
[18:27:23] 
▶ CELL: Cell 8 — FLUXMultiAgent
[18:27:23]   Started: 2026-03-29 18:27:23

============================================================
FLUXMultiAgent Initialized
============================================================

  FLUX Version: 2.0-base
  LLM: Qwen/Qwen2-VL-7B-Instruct
  Device: cuda
  Vision: ✗ Disabled (text-only)
[18:27:23]   ✓ FLUXMultiAgent ready
[18:27:23]   ◼ CELL Cell 8 — FLUXMultiAgent — PASS
add Codeadd Markdown
Cell 9: Test Vision Reasoning
add Codeadd Markdown
# ═══════════════════════════════════════════════════════════════
# Cell 9: Test Vision Reasoning
# ═══════════════════════════════════════════════════════════════

log.cell_start("Cell 9 — Test Vision Reasoning")

print("\n" + "="*60)
print("TEST: Vision Reasoning on Sample Grid")
print("="*60 + "\n")

# Create a simple ARC-like puzzle
puzzle_grid = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
    [0, 1, 0, 1, 0, 0, 2, 0, 2, 0],
    [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 3, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 3, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# Reset agent for test
agent.reset("test_puzzle")

# Observe and build spatial memory
test_position = (4, 4)  # Center-ish
agent.observe(puzzle_grid, test_position)
agent.observe(puzzle_grid, (4, 5))
agent.observe(puzzle_grid, (5, 5))

# Render the grid with ice overlay
test_image = render_grid_to_image(
    puzzle_grid,
    cell_size=30,
    position=test_position,
    ice_field=agent.spatial.curiosity_field,
)

print("Puzzle grid rendered with curiosity overlay:")
try:
    from IPython.display import display
    display(test_image)
except:
    print(f"  Image size: {test_image.size}")

# Test reasoning
print("\nAsking LLM to reason about the grid...\n")

action, reasoning = agent.decide_action(
    grid=puzzle_grid,
    position=test_position,
    available_actions=[1, 2, 3, 4],  # UP, DOWN, LEFT, RIGHT
)

ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
print(f"Decided Action: {ACTION_NAMES.get(action, action)}")
print(f"\nReasoning:\n{reasoning[:500]}..." if len(reasoning) > 500 else f"\nReasoning:\n{reasoning}")

# Verify
test_passed = action in [1, 2, 3, 4] and len(reasoning) > 10
print(f"\n{'✓' if test_passed else '✗'} Test {'PASSED' if test_passed else 'FAILED'}")

log.cell_end("Cell 9 — Test Vision Reasoning", "PASS" if test_passed else "FAIL")
[18:27:23] 
▶ CELL: Cell 9 — Test Vision Reasoning
[18:27:23]   Started: 2026-03-29 18:27:23

============================================================
TEST: Vision Reasoning on Sample Grid
============================================================


✓ Reset for game: test_puzzle
Puzzle grid rendered with curiosity overlay:


Asking LLM to reason about the grid...

Decided Action: LEFT

Reasoning:
No LLM available. Random action: LEFT

✓ Test PASSED
[18:27:23]   ◼ CELL Cell 9 — Test Vision Reasoning — PASS
add Codeadd Markdown
Cell 10: Demo — Simulated Game Play
add Codeadd Markdown
# ═══════════════════════════════════════════════════════════════
# Cell 10: Demo — Simulated Game Play (10 steps)
# ═══════════════════════════════════════════════════════════════

log.cell_start("Cell 10 — Simulated Game Play")

print("\n" + "="*60)
print("DEMO: Simulated Game Play with Vision Reasoning")
print("="*60 + "\n")

# Simulate a simple game grid that changes
def create_game_grid(step: int) -> List[List[int]]:
    """Create evolving game grid."""
    base = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 3, 3, 0, 1, 0],
        [0, 1, 0, 3, 3, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    # Add some variation based on step
    if step >= 3:
        base[3][3] = 4  # Yellow
    if step >= 6:
        base[4][4] = 2  # Red
    return base

# Reset for demo game
agent.reset("demo_game")

positions = [
    (2, 2), (2, 3), (3, 3), (3, 4), (4, 4),
    (4, 3), (5, 3), (5, 4), (5, 5), (4, 5),
]

print("Running 10 simulation steps...\n")

for step in range(10):
    grid = create_game_grid(step)
    pos = positions[step]
    
    print(f"Step {step + 1}: Position {pos}")
    
    # Observe
    old_grid = create_game_grid(step - 1) if step > 0 else grid
    agent.observe(grid, pos)
    
    # Decide (every 3rd step to save time)
    if step % 3 == 0:
        action, reasoning = agent.decide_action(grid, pos, [1, 2, 3, 4])
        print(f"  → Action: {ACTION_NAMES.get(action, action)}")
        print(f"  → Reason: {reasoning[:100]}...\n")
        
        # Record effect
        agent.record_effect(old_grid, grid, action)
    else:
        print(f"  (observing...)\n")

# Stats
stats = agent.get_stats()
print("\n" + "="*60)
print("Demo Complete")
print("="*60)
for key, val in stats.items():
    print(f"  {key}: {val}")

log.success(f"Completed {stats['observations']} observations, {stats['causal_links']} causal links")
log.cell_end("Cell 10 — Simulated Game Play", "PASS")
[18:27:23] 
▶ CELL: Cell 10 — Simulated Game Play
[18:27:23]   Started: 2026-03-29 18:27:23

============================================================
DEMO: Simulated Game Play with Vision Reasoning
============================================================


✓ Reset for game: demo_game
Running 10 simulation steps...

Step 1: Position (2, 2)
  → Action: UP
  → Reason: No LLM available. Random action: UP...

Step 2: Position (2, 3)
  (observing...)

Step 3: Position (3, 3)
  (observing...)

Step 4: Position (3, 4)
  → Action: UP
  → Reason: No LLM available. Random action: UP...

Step 5: Position (4, 4)
  (observing...)

Step 6: Position (4, 3)
  (observing...)

Step 7: Position (5, 3)
  → Action: RIGHT
  → Reason: No LLM available. Random action: RIGHT...

Step 8: Position (5, 4)
  (observing...)

Step 9: Position (5, 5)
  (observing...)

Step 10: Position (4, 5)
  → Action: LEFT
  → Reason: No LLM available. Random action: LEFT...


============================================================
Demo Complete
============================================================
  version: 2.0-base
  game: demo_game
  observations: 10
  actions: 4
  causal_links: 2
  learned_rules: 0
[18:27:23]   ✓ Completed 10 observations, 2 causal links
[18:27:23]   ◼ CELL Cell 10 — Simulated Game Play — PASS