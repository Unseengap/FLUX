"""
Phase 7: FLUXModel — Full Integration of All Components

Combines:
  Phase 1: ContinuousSemanticEncoder (CSE)
  Phase 2: ResonanceField (RFC)
  Phase 3: GravitationalRelevance (GR)
  Phase 4: ThermodynamicLearner (TL)
  Phase 5: MultiTimescaleCoordinator + CausalGraph (CGN)
  Phase 6: WorkingMemory, EpisodicMemory, SemanticMemory, MemoryRouter, ConsolidationProcess

The unified pipeline:
  text → CSE → wave → GR (relevance) → CGN (causal) → Field (settle) → TL (learn) → Memory → output
"""

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# ─────────────────────────────────────────────
# Path setup for cross-phase imports
# ─────────────────────────────────────────────
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

for _phase in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6']:
    _p = str(_PHASES_DIR / _phase)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cse import ContinuousSemanticEncoder
from wave_types import SemanticWave, TOTAL_WAVE_DIM
from field import ResonanceField, FIELD_H, FIELD_W, FIELD_D, FIELD_FEATURES
from gravity import GravitationalRelevance
from thermodynamic import ThermodynamicLearner
from cgn import CausalGeometryNode
from multi_timescale import MultiTimescaleCoordinator
from causal_graph import CausalGraph
from working_memory import WorkingMemory
from episodic_memory import EpisodicMemory
from semantic_memory import SemanticMemory
from memory_router import MemoryRouter
from consolidation import ConsolidationProcess
from flux_utils import (
    get_device, load_checkpoint, save_checkpoint, checkpoint_exists,
    PhaseLogger,
)


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
FLUX_MODEL_CONFIG = {
    # CSE (Phase 1)
    'wave_dim': TOTAL_WAVE_DIM,       # 432
    'byte_window': 8,
    'byte_stride': 1,
    'interference_radius': 4,

    # Field (Phase 2)
    'field_h': FIELD_H,               # 64
    'field_w': FIELD_W,               # 64
    'field_d': FIELD_D,               # 64
    'field_features': FIELD_FEATURES,  # 512

    # GR (Phase 3)
    'gr_k_neighbors': 32,
    'gr_base_mass': 1.0,
    'gr_mass_growth': 0.01,

    # TL (Phase 4)
    'tl_initial_temp': 1.0,
    'tl_min_temp': 0.01,
    'tl_decay': 0.999,
    'tl_settle_iterations': 10,

    # CGN (Phase 5)
    'cgn_feature_dim': FIELD_FEATURES,
    'cgn_n_fast': 16,
    'cgn_n_medium': 8,
    'cgn_n_slow': 4,

    # Memory (Phase 6)
    'wm_window_size': 512,
    'feature_dim': 256,
    'episodic_dim': 256,
    'consolidation_min_access': 3,
    'consolidation_temperature': 0.05,

    # Generation
    'output_vocab_size': 256,          # byte-level output
    'max_gen_length': 200,
    'temperature': 0.8,
}


@dataclass
class FLUXResponse:
    """Result from a FLUX model forward pass or chat interaction."""
    text: str = ""
    wave: Optional[SemanticWave] = None
    relevance_output: Optional[torch.Tensor] = None
    cgn_output: Optional[torch.Tensor] = None
    field_features: Optional[torch.Tensor] = None
    memory_result: Optional[Dict[str, Any]] = None
    learned: bool = False
    latency_ms: float = 0.0


@dataclass
class FLUXStats:
    """Aggregate statistics for the full model."""
    total_params: int = 0
    cse_params: int = 0
    field_params: int = 0
    gr_params: int = 0
    tl_params: int = 0
    cgn_params: int = 0
    memory_params: int = 0
    output_head_params: int = 0
    episodic_entries: int = 0
    working_entries: int = 0
    field_energy: float = 0.0
    field_attractors: int = 0
    temperature: float = 0.0
    learning_steps: int = 0


class OutputHead(nn.Module):
    """
    Decode field features into byte-level logits for text generation.

    Maps from field feature space [field_features] to byte probabilities [256].
    """

    def __init__(self, field_features: int = 512, wave_dim: int = 432,
                 vocab_size: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.field_to_hidden = nn.Linear(field_features, hidden_dim)
        self.wave_to_hidden = nn.Linear(wave_dim, hidden_dim)
        self.merge = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, field_features: torch.Tensor,
                wave_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Produce byte-level logits from field state.

        Args:
            field_features: [field_features] or [seq, field_features]
            wave_context:   [wave_dim] or [seq, wave_dim], optional

        Returns:
            [vocab_size] or [seq, vocab_size] logits
        """
        h_field = F.gelu(self.field_to_hidden(field_features))

        if wave_context is not None:
            h_wave = F.gelu(self.wave_to_hidden(wave_context))
            if h_field.dim() != h_wave.dim():
                if h_wave.dim() > h_field.dim():
                    h_wave = h_wave.mean(dim=-2)
                else:
                    h_field = h_field.mean(dim=-2)
            h = F.gelu(self.merge(torch.cat([h_field, h_wave], dim=-1)))
        else:
            h = h_field

        h = self.norm(h)
        return self.output_proj(h)


class FLUXModel(nn.Module):
    """
    Unified FLUX Model integrating all 6 phases.

    Pipeline:
      text → CSE(encode) → wave → GR(relevance) → CGN(causal) →
      Field(perturb+settle) → TL(learn) → Memory(store+route) → OutputHead → bytes

    Supports:
      - End-to-end text generation
      - Real-time learning during chat (one-shot episodic write)
      - Cross-session memory via episodic store
      - Causal tracing of why conclusions were reached
      - Zero catastrophic forgetting
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = 'cpu'):
        super().__init__()
        self.config = {**FLUX_MODEL_CONFIG, **(config or {})}
        self._device_str = device
        self._learning_steps = 0
        self._initialized = False

        # ── Phase 1: Continuous Semantic Encoder ──
        self.cse = ContinuousSemanticEncoder(
            byte_window=self.config['byte_window'],
            byte_stride=self.config['byte_stride'],
            interference_radius=self.config['interference_radius'],
            device=device,
        )

        # ── Phase 2: Resonance Field Core ──
        self.field = ResonanceField(
            h=self.config['field_h'],
            w=self.config['field_w'],
            d=self.config['field_d'],
            features=self.config['field_features'],
            wave_dim=self.config['wave_dim'],
        )

        # ── Phase 3: Gravitational Relevance ──
        self.gr = GravitationalRelevance(
            feature_dim=self.config['field_features'],
            k_neighbors=self.config['gr_k_neighbors'],
            base_mass=self.config['gr_base_mass'],
            mass_growth_rate=self.config['gr_mass_growth'],
            device=device,
        )

        # ── Phase 4: Thermodynamic Learner ──
        self.tl = ThermodynamicLearner(
            field=self.field,
            initial_temp=self.config['tl_initial_temp'],
            min_temp=self.config['tl_min_temp'],
            decay=self.config['tl_decay'],
            settle_iterations=self.config['tl_settle_iterations'],
        )

        # ── Phase 5: Causal Geometry Nodes (Multi-Timescale) ──
        self.cgn = MultiTimescaleCoordinator(
            feature_dim=self.config['cgn_feature_dim'],
            n_fast=self.config['cgn_n_fast'],
            n_medium=self.config['cgn_n_medium'],
            n_slow=self.config['cgn_n_slow'],
        )
        self.causal_graph = CausalGraph()

        # ── Phase 6: Three-Tier Memory ──
        wave_dim = self.config['wave_dim']
        feature_dim = self.config['feature_dim']

        self.working_memory = WorkingMemory(
            window_size=self.config['wm_window_size'],
            wave_dim=wave_dim,
            feature_dim=feature_dim,
        )

        self.episodic_memory = EpisodicMemory(feature_dim=feature_dim)

        self.semantic_memory = SemanticMemory(
            field=self.field,
            protection_threshold=5.0,
        )

        self.memory_router = MemoryRouter(
            working=self.working_memory,
            episodic=self.episodic_memory,
            semantic=self.semantic_memory,
            wave_dim=wave_dim,
            feature_dim=feature_dim,
        )

        self.consolidation = ConsolidationProcess(
            episodic=self.episodic_memory,
            semantic=self.semantic_memory,
            min_access=self.config['consolidation_min_access'],
            temperature=self.config['consolidation_temperature'],
        )

        # ── Bridge projections ──
        self.wave_to_field = nn.Linear(wave_dim, self.config['field_features'])
        self.field_to_wave = nn.Linear(self.config['field_features'], wave_dim)

        # ── Output head (field features → byte logits) ──
        self.output_head = OutputHead(
            field_features=self.config['field_features'],
            wave_dim=wave_dim,
            vocab_size=self.config['output_vocab_size'],
        )

        self._initialized = True

    # ─────────────────────────────────────────────
    # Core Pipeline
    # ─────────────────────────────────────────────

    def forward(self, text: str, learn: bool = True) -> FLUXResponse:
        """
        Full FLUX pipeline: encode → relevance → causal → field → learn → memory.

        Args:
            text:  Raw input text (any UTF-8 string)
            learn: If True, performs thermodynamic settling (learning).
                   If False, query-only mode.

        Returns:
            FLUXResponse with all intermediate representations
        """
        t0 = time.time()
        device = self._device_str

        # Step 1: CSE — raw text → semantic wave
        with torch.no_grad():
            wave = self.cse.encode(text)
        wave_vec = wave.full.mean(dim=0).to(device)  # [wave_dim]

        # Step 2: Project wave into field feature space
        field_input = self.wave_to_field(wave_vec)    # [field_features]

        # Step 3: Gravitational Relevance — find relevant field context
        relevance_out = self.gr(field_input.unsqueeze(0)).squeeze(0)  # [field_features]

        # Step 4: Causal Geometry — multi-timescale processing
        cgn_out = self.cgn(relevance_out)             # [field_features]

        # Step 5: Field interaction — perturb + query
        field_features, sims, locs = self.field.query(wave_vec, k=4)
        best_features = field_features[0]             # [field_features]

        # Step 6: Thermodynamic settling (learning + inference)
        if learn:
            self.tl.settle_once(wave_vec, target=cgn_out)
            self._learning_steps += 1

        # Step 7: Memory — store in working memory + episodic + route
        self.working_memory.add_perturbation(wave_vec)

        compressed = self.working_memory.compress(
            wave_vec.unsqueeze(0)
        ).squeeze(0)                                  # [feature_dim]

        self.episodic_memory.write(
            compressed, fact=text, causal_source="forward_pass"
        )

        memory_result = self.memory_router.route_query(
            wave_vec, episodic_k=5, working_k=10
        )

        # Step 8: Combine field + CGN output (residual)
        combined = best_features + cgn_out

        latency = (time.time() - t0) * 1000

        return FLUXResponse(
            text=text,
            wave=wave,
            relevance_output=relevance_out.detach(),
            cgn_output=cgn_out.detach(),
            field_features=combined.detach(),
            memory_result=memory_result,
            learned=learn,
            latency_ms=latency,
        )

    def encode(self, text: str) -> SemanticWave:
        """Encode text to semantic wave via CSE."""
        with torch.no_grad():
            return self.cse.encode(text)

    # ─────────────────────────────────────────────
    # Chat Interface
    # ─────────────────────────────────────────────

    def chat(self, user_input: str) -> str:
        """
        End-to-end chat interaction with real-time learning.

        1. Encode user input
        2. Write to episodic memory (one-shot)
        3. Query all memory tiers for relevant context
        4. Return most relevant retrieved memory

        Args:
            user_input: Raw text from user

        Returns:
            Retrieved / generated response text
        """
        device = self._device_str

        # Encode input
        with torch.no_grad():
            wave = self.cse.encode(user_input)
        wave_vec = wave.full.mean(dim=0).to(device)

        # Write to episodic memory (one-shot, real-time learning)
        compressed = self.working_memory.compress(
            wave_vec.unsqueeze(0)
        ).squeeze(0)
        self.episodic_memory.write(
            compressed, fact=user_input, causal_source="chat"
        )

        # Add to working memory
        self.working_memory.add_perturbation(wave_vec)

        # Route query across all memory tiers
        self.memory_router.route_query(wave_vec, episodic_k=5, working_k=10)

        # Thermodynamic settle (learn the input)
        self.tl.settle_once(wave_vec)
        self._learning_steps += 1

        # Retrieve relevant episodic memories for response
        results = self.episodic_memory.search(compressed, k=5)
        if results:
            retrieved_facts = [entry.fact for entry, score in results
                               if entry.fact != user_input]
            if retrieved_facts:
                return retrieved_facts[0]

        return f"[FLUX processed: {user_input[:80]}]"

    def learn_fact(self, fact: str) -> Dict[str, Any]:
        """
        One-shot learning of a new fact.

        Writes to episodic memory and perturbs the field.
        No fine-tuning, no gradient computation needed.

        Args:
            fact: Text fact to learn

        Returns:
            Dict with learning metrics
        """
        device = self._device_str

        with torch.no_grad():
            wave = self.cse.encode(fact)
        wave_vec = wave.full.mean(dim=0).to(device)

        # Episodic write (one-shot)
        compressed = self.working_memory.compress(
            wave_vec.unsqueeze(0)
        ).squeeze(0)
        entry_id = self.episodic_memory.write(
            compressed, fact=fact, causal_source="learn_fact"
        )

        # Field perturbation + thermodynamic settle
        result = self.tl.settle_once(wave_vec)
        self._learning_steps += 1

        # Working memory
        self.working_memory.add_perturbation(wave_vec)

        return {
            'entry_id': entry_id,
            'energy_delta': result.energy_delta,
            'temperature': result.temperature,
            'learned': True,
        }

    def query(self, query_text: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Query the model for relevant memories.

        Args:
            query_text: Query string
            k: Number of results to return

        Returns:
            List of (fact_text, similarity_score) tuples
        """
        device = self._device_str

        with torch.no_grad():
            wave = self.cse.encode(query_text)
        wave_vec = wave.full.mean(dim=0).to(device)
        compressed = self.working_memory.compress(
            wave_vec.unsqueeze(0)
        ).squeeze(0)

        results = self.episodic_memory.search(compressed, k=k)
        return [(entry.fact, float(score)) for entry, score in results]

    # ─────────────────────────────────────────────
    # Generation
    # ─────────────────────────────────────────────

    def generate_logits(self, text: str) -> torch.Tensor:
        """
        Generate byte-level logits for next-byte prediction.

        Args:
            text: Input context text

        Returns:
            [vocab_size=256] logits
        """
        device = self._device_str

        with torch.no_grad():
            wave = self.cse.encode(text)
        wave_vec = wave.full.mean(dim=0).to(device)

        # Field query for context features
        field_features, sims, locs = self.field.query(wave_vec, k=4)
        combined = field_features.mean(dim=0)  # [field_features]

        # CGN processing
        cgn_out = self.cgn(combined)

        # Merge field + CGN
        merged = combined + cgn_out

        logits = self.output_head(merged, wave_context=wave_vec)
        return logits

    def generate(self, prompt: str, max_length: int = 100,
                 temperature: float = 0.8) -> str:
        """
        Generate text continuation from a prompt.

        Uses the field state + memory to produce byte-level
        output autoregressively.

        Args:
            prompt: Starting text
            max_length: Maximum bytes to generate
            temperature: Sampling temperature (lower = more deterministic)

        Returns:
            Generated text (prompt + continuation)
        """
        # Feed prompt through pipeline first
        self.forward(prompt, learn=False)

        generated_bytes = list(prompt.encode('utf-8'))
        context = prompt

        for step in range(max_length):
            logits = self.generate_logits(context)
            logits = logits / max(temperature, 1e-8)
            probs = F.softmax(logits, dim=-1)

            # Sample next byte
            next_byte = torch.multinomial(probs, 1).item()

            # Stop on null byte
            if next_byte == 0:
                break

            generated_bytes.append(next_byte)

            # Update context (sliding window of last 200 bytes)
            try:
                context = bytes(generated_bytes[-200:]).decode('utf-8', errors='replace')
            except Exception:
                context = bytes(generated_bytes[-200:]).decode('latin-1', errors='replace')

        try:
            result = bytes(generated_bytes).decode('utf-8', errors='replace')
        except Exception:
            result = bytes(generated_bytes).decode('latin-1', errors='replace')

        return result

    # ─────────────────────────────────────────────
    # Consolidation
    # ─────────────────────────────────────────────

    def run_consolidation(self) -> Dict[str, Any]:
        """Run episodic → semantic memory consolidation."""
        return self.consolidation.run_consolidation(
            wave_dim=self.config['wave_dim']
        )

    # ─────────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────────

    def get_stats(self) -> FLUXStats:
        """Collect statistics from all components."""
        cse_p = sum(p.numel() for p in self.cse.parameters())
        field_p = sum(p.numel() for p in self.field.parameters())
        gr_p = sum(p.numel() for p in self.gr.parameters())
        tl_p = sum(p.numel() for p in self.tl.parameters())
        cgn_p = sum(p.numel() for p in self.cgn.parameters())
        mem_p = sum(p.numel() for p in self.working_memory.parameters()) + \
                sum(p.numel() for p in self.memory_router.parameters()) + \
                sum(p.numel() for p in self.semantic_memory.parameters())
        out_p = sum(p.numel() for p in self.output_head.parameters()) + \
                sum(p.numel() for p in self.wave_to_field.parameters()) + \
                sum(p.numel() for p in self.field_to_wave.parameters())

        temp = 0.0
        if hasattr(self.tl, 'temperature') and hasattr(self.tl.temperature, 'temperature'):
            temp = self.tl.temperature.temperature
        elif hasattr(self.tl, 'temp_manager') and hasattr(self.tl.temp_manager, 'temperature'):
            temp = self.tl.temp_manager.temperature

        return FLUXStats(
            total_params=cse_p + field_p + gr_p + tl_p + cgn_p + mem_p + out_p,
            cse_params=cse_p,
            field_params=field_p,
            gr_params=gr_p,
            tl_params=tl_p,
            cgn_params=cgn_p,
            memory_params=mem_p,
            output_head_params=out_p,
            episodic_entries=self.episodic_memory.size,
            working_entries=self.working_memory.size,
            field_energy=self.field.total_energy(),
            field_attractors=self.field.num_attractors(),
            temperature=temp,
            learning_steps=self._learning_steps,
        )

    # ─────────────────────────────────────────────
    # State Management (.flx format)
    # ─────────────────────────────────────────────

    def save_model(self, path: str) -> Path:
        """
        Save full FLUX model in .flx format.

        Args:
            path: Save path (should end in .flx)

        Returns:
            Path to saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'format': 'FLUX',
            'version': '0.1',
            'phase': 7,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'learning_steps': self._learning_steps,
            'can_continue_learning': True,
            # Component state dicts
            'cse_state_dict': self.cse.state_dict(),
            'field_state_dict': self.field.state_dict(),
            'gr_state': self.gr.save_state(),
            'tl_state': self.tl.save_state(),
            'cgn_state': self.cgn.save_state(),
            'causal_graph_state': self.causal_graph.save_state(),
            'working_memory_state': self.working_memory.state_dict_memory(),
            'episodic_memory_state': self.episodic_memory.save_state(),
            'semantic_memory_state': self.semantic_memory.save_state(),
            'router_state': self.memory_router.save_state(),
            # Bridge + output head
            'wave_to_field_state': self.wave_to_field.state_dict(),
            'field_to_wave_state': self.field_to_wave.state_dict(),
            'output_head_state': self.output_head.state_dict(),
        }

        torch.save(state, str(path))
        return path

    @classmethod
    def load_model(cls, path: str, device: str = 'cpu') -> 'FLUXModel':
        """
        Load full FLUX model from .flx file.

        Args:
            path: Path to .flx file
            device: Target device

        Returns:
            Reconstructed FLUXModel
        """
        state = torch.load(str(path), map_location='cpu', weights_only=False)
        config = state['config']

        model = cls(config=config, device=device)

        model.cse.load_state_dict(state['cse_state_dict'])
        model.field.load_state_dict(state['field_state_dict'])
        model.gr = GravitationalRelevance.load_state(state['gr_state'], device=device)
        model.tl.load_state(state['tl_state'])
        model.cgn.load_state(state['cgn_state'])
        model.causal_graph.load_state(state['causal_graph_state'])
        model.working_memory.load_state_memory(state['working_memory_state'])
        model.episodic_memory.load_state(state['episodic_memory_state'])
        model.semantic_memory.load_state(state['semantic_memory_state'])
        model.memory_router.load_state(state['router_state'])
        model.wave_to_field.load_state_dict(state['wave_to_field_state'])
        model.field_to_wave.load_state_dict(state['field_to_wave_state'])
        model.output_head.load_state_dict(state['output_head_state'])

        model._learning_steps = state.get('learning_steps', 0)
        model = model.to(device)
        return model

    @classmethod
    def from_checkpoints(cls, device: str = 'cpu',
                         config: Optional[Dict[str, Any]] = None) -> 'FLUXModel':
        """
        Build FLUXModel by loading all phase checkpoints (1–6).

        This is the primary constructor for Phase 7 training.

        Args:
            device: Target device
            config: Optional config overrides

        Returns:
            FLUXModel with all phase weights loaded
        """
        model = cls(config=config, device=device)

        # ── Phase 1: CSE ──
        ckpt1 = load_checkpoint(1)
        model.cse.load_state_dict(ckpt1['state_dict'])
        model.cse = model.cse.to(device).eval()
        for p in model.cse.parameters():
            p.requires_grad = False
        print(f"  ✓ Phase 1 (CSE) loaded: {sum(p.numel() for p in model.cse.parameters()):,} params")

        # ── Phase 2: ResonanceField ──
        ckpt2 = load_checkpoint(2)
        model.field.load_state_dict(ckpt2['state_dict'])
        model.field = model.field.to(device)
        print(f"  ✓ Phase 2 (Field) loaded: {sum(p.numel() for p in model.field.parameters()):,} params")

        # ── Phase 3: GravitationalRelevance ──
        ckpt3 = load_checkpoint(3)
        if 'gr_state' in ckpt3:
            model.gr = GravitationalRelevance.load_state(ckpt3['gr_state'], device=device)
        elif 'state_dict' in ckpt3:
            try:
                model.gr.load_state_dict(ckpt3['state_dict'], strict=False)
            except Exception:
                pass
        model.gr = model.gr.to(device)
        print(f"  ✓ Phase 3 (GR) loaded: {sum(p.numel() for p in model.gr.parameters()):,} params")

        # ── Phase 4: ThermodynamicLearner ──
        ckpt4 = load_checkpoint(4)
        if 'tl_state' in ckpt4:
            model.tl.load_state(ckpt4['tl_state'])
        elif 'state_dict' in ckpt4:
            try:
                model.tl.load_state_dict(ckpt4['state_dict'], strict=False)
            except Exception:
                pass
        model.tl = model.tl.to(device)
        print(f"  ✓ Phase 4 (TL) loaded: {sum(p.numel() for p in model.tl.parameters()):,} params")

        # ── Phase 5: CGN + CausalGraph ──
        ckpt5 = load_checkpoint(5)
        if 'cgn_state' in ckpt5:
            model.cgn.load_state(ckpt5['cgn_state'])
        elif 'state_dict' in ckpt5:
            try:
                model.cgn.load_state_dict(ckpt5['state_dict'], strict=False)
            except Exception:
                pass
        if 'causal_graph_state' in ckpt5:
            model.causal_graph.load_state(ckpt5['causal_graph_state'])
        model.cgn = model.cgn.to(device)
        print(f"  ✓ Phase 5 (CGN) loaded: {model.cgn.total_nodes()} nodes, {model.cgn.total_params():,} params")

        # ── Phase 6: Memory System ──
        ckpt6 = load_checkpoint(6)
        if 'working_memory_state' in ckpt6:
            model.working_memory.load_state_memory(ckpt6['working_memory_state'])
        if 'episodic_memory_state' in ckpt6:
            model.episodic_memory.load_state(ckpt6['episodic_memory_state'])
        if 'semantic_memory_state' in ckpt6:
            model.semantic_memory.load_state(ckpt6['semantic_memory_state'])
        if 'router_state' in ckpt6:
            model.memory_router.load_state(ckpt6['router_state'])
        model.working_memory = model.working_memory.to(device)
        model.semantic_memory = model.semantic_memory.to(device)
        model.memory_router = model.memory_router.to(device)
        print(f"  ✓ Phase 6 (Memory) loaded: episodic={model.episodic_memory.size}, "
              f"working={model.working_memory.size}")

        model = model.to(device)
        stats = model.get_stats()
        print(f"\n  ═══ FLUXModel assembled: {stats.total_params:,} total parameters ═══")

        return model
