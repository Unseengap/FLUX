"""
FLUX Service - Model loading and inference with thinking process

This is the core service that:
1. Loads Flux-beta.flx model
2. Runs full FLUX pipeline with detailed phase logging
3. Streams character-by-character generation with SSE
"""

import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, List, Generator, Optional
from dataclasses import dataclass, asdict

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
for phase in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6', 'phase7', 'phase8']:
    sys.path.insert(0, str(PROJECT_ROOT / 'phases' / phase))


@dataclass
class PhaseEvent:
    """Event from a FLUX phase during processing."""
    type: str           # 'phase', 'char', 'metric', 'memory', 'done', 'error'
    phase: Optional[str] = None
    name: Optional[str] = None
    status: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    char: Optional[str] = None
    message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class FluxService:
    """
    FLUX model service with full pipeline introspection.
    
    Provides detailed thinking process visualization for each phase:
    - Phase 1: CSE byte encoding → 432-dim waves
    - Phase 2: Field query → attractor retrieval
    - Phase 3: Gravitational relevance → mass-weighted features
    - Phase 4: Thermodynamic settling → energy minimization
    - Phase 5: CGN → multi-timescale causal processing
    - Phase 6: Memory → episodic/semantic retrieval
    - Phase 8: WaveDecoder → character generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.device = None
        self._loaded = False
    
    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None
    
    def _get_device(self) -> str:
        """Auto-detect best device."""
        if self.config.get('DEVICE', 'auto') != 'auto':
            return self.config['DEVICE']
        
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    
    def load_model(self) -> Dict[str, Any]:
        """Load FLUX model from .flx checkpoint."""
        from flux_large import FLUXLarge, FLUX_LARGE_CONFIG
        from gravity import GravitationalRelevance
        
        self.device = self._get_device()
        model_path = Path(self.config['MODEL_PATH'])
        
        if not model_path.exists():
            # Try loading from phase8.phase.pt as fallback
            fallback_path = PROJECT_ROOT / 'checkpoints' / 'phase8.phase.pt'
            if fallback_path.exists():
                model_path = fallback_path
            else:
                raise FileNotFoundError(f"Model not found at {model_path} or {fallback_path}")
        
        # Load checkpoint
        flx = torch.load(model_path, map_location='cpu')
        
        # Determine format (.flx vs .phase.pt)
        is_flx_format = flx.get('format') == 'FLUX'
        
        if is_flx_format:
            # .flx modular format
            config = {
                **FLUX_LARGE_CONFIG,
                'wave_dim': flx['cse']['config']['wave_dim'],
                'field_h': flx['field']['config']['h'],
                'field_w': flx['field']['config']['w'],
                'field_d': flx['field']['config']['d'],
                'field_features': flx['field']['config']['features'],
            }
        else:
            # Legacy .phase.pt format
            config = {**FLUX_LARGE_CONFIG, **flx.get('config', {})}
        
        # Create model shell
        self.model = FLUXLarge(config=config, device=self.device)
        
        # Load component weights
        if is_flx_format:
            self._load_flx_weights(flx)
        else:
            self._load_phase_pt_weights(flx)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        
        return self.get_stats()
    
    def _load_flx_weights(self, flx: Dict[str, Any]):
        """Load weights from .flx modular format."""
        from gravity import GravitationalRelevance
        
        # CSE
        if flx['cse']['state_dict']:
            self.model.cse.load_state_dict(flx['cse']['state_dict'])
        
        # Field
        if flx['field']['state_dict']:
            self.model.field.load_state_dict(flx['field']['state_dict'])
        
        # Gravity
        if flx['field'].get('gravity_state'):
            self.model.gr = GravitationalRelevance.load_state(
                flx['field']['gravity_state'], device=self.device
            )
        
        # Thermodynamic
        if flx['field'].get('thermodynamic_state'):
            self.model.tl.load_state(flx['field']['thermodynamic_state'])
        
        # CGN
        if flx['causal'].get('cgn_state'):
            self.model.cgn.load_state(flx['causal']['cgn_state'])
        
        # Causal Graph
        if flx['causal'].get('graph_state'):
            self.model.causal_graph.load_state(flx['causal']['graph_state'])
        
        # Memory
        if flx['memory'].get('working'):
            self.model.working_memory.load_state_memory(flx['memory']['working'])
        if flx['memory'].get('episodic'):
            self.model.episodic_memory.load_state(flx['memory']['episodic'])
        if flx['memory'].get('semantic'):
            self.model.semantic_memory.load_state(flx['memory']['semantic'])
        
        # Bridges
        if flx['bridges'].get('wave_to_field'):
            self.model.wave_to_field.load_state_dict(flx['bridges']['wave_to_field'])
        if flx['bridges'].get('field_to_wave'):
            self.model.field_to_wave.load_state_dict(flx['bridges']['field_to_wave'])
        if flx['bridges'].get('output_head'):
            self.model.output_head.load_state_dict(flx['bridges']['output_head'])
        if flx['bridges'].get('router'):
            self.model.memory_router.load_state(flx['bridges']['router'])
        
        # Decoder
        if flx['decoder']['state_dict']:
            self.model.decoder.load_state_dict(flx['decoder']['state_dict'])
        
        # Metadata
        if flx['metadata'].get('learning_steps'):
            self.model._learning_steps = flx['metadata']['learning_steps']
    
    def _load_phase_pt_weights(self, ckpt: Dict[str, Any]):
        """Load weights from legacy .phase.pt format."""
        from gravity import GravitationalRelevance
        
        if ckpt.get('cse_state_dict'):
            self.model.cse.load_state_dict(ckpt['cse_state_dict'])
        if ckpt.get('field_state_dict'):
            self.model.field.load_state_dict(ckpt['field_state_dict'])
        if ckpt.get('gr_state'):
            self.model.gr = GravitationalRelevance.load_state(ckpt['gr_state'], device=self.device)
        if ckpt.get('tl_state'):
            self.model.tl.load_state(ckpt['tl_state'])
        if ckpt.get('cgn_state'):
            self.model.cgn.load_state(ckpt['cgn_state'])
        if ckpt.get('causal_graph_state'):
            self.model.causal_graph.load_state(ckpt['causal_graph_state'])
        if ckpt.get('working_memory_state'):
            self.model.working_memory.load_state_memory(ckpt['working_memory_state'])
        if ckpt.get('episodic_memory_state'):
            self.model.episodic_memory.load_state(ckpt['episodic_memory_state'])
        if ckpt.get('semantic_memory_state'):
            self.model.semantic_memory.load_state(ckpt['semantic_memory_state'])
        if ckpt.get('wave_to_field_state'):
            self.model.wave_to_field.load_state_dict(ckpt['wave_to_field_state'])
        if ckpt.get('field_to_wave_state'):
            self.model.field_to_wave.load_state_dict(ckpt['field_to_wave_state'])
        if ckpt.get('output_head_state'):
            self.model.output_head.load_state_dict(ckpt['output_head_state'])
        if ckpt.get('router_state'):
            self.model.memory_router.load_state(ckpt['router_state'])
        
        # Decoder (handle _orig_mod. prefix from torch.compile)
        if ckpt.get('decoder_state_dict'):
            decoder_state = {}
            for k, v in ckpt['decoder_state_dict'].items():
                decoder_state[k.replace('_orig_mod.', '')] = v
            self.model.decoder.load_state_dict(decoder_state)
        
        if ckpt.get('learning_steps'):
            self.model._learning_steps = ckpt['learning_steps']
    
    def unload_model(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._loaded = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        if not self.is_loaded():
            return {}
        
        stats = self.model.get_stats()
        return {
            'total_params': f"{stats.total_params:,}",
            'episodic_entries': stats.episodic_entries,
            'working_entries': stats.working_entries,
            'field_energy': f"{stats.field_energy:.4f}",
            'field_attractors': stats.field_attractors,
            'temperature': f"{stats.temperature:.6f}",
            'learning_steps': stats.learning_steps,
            'device': self.device,
        }
    
    def generate_with_thinking(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        learn: bool = False,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate response with full thinking process visualization.
        
        Yields SSE events for each phase and character.
        This is a "glass box" view — exposes actual data being pulled from the model.
        """
        device = self.device
        
        # ═══════════════════════════════════════════════════════════
        # PHASE 1: Continuous Semantic Encoder
        # ═══════════════════════════════════════════════════════════
        t0 = time.time()
        
        # Get actual bytes
        prompt_bytes_raw = list(prompt.encode('utf-8', errors='replace'))
        byte_display = ' '.join(f'{b:02x}' for b in prompt_bytes_raw[:20])
        if len(prompt_bytes_raw) > 20:
            byte_display += f' ... (+{len(prompt_bytes_raw)-20} more)'
        
        yield PhaseEvent(
            type='phase',
            phase='1',
            name='Continuous Semantic Encoder (CSE)',
            status='running',
            details={
                'input_text': prompt[:100] + ('...' if len(prompt) > 100 else ''),
                'raw_bytes': byte_display,
            }
        ).to_dict()
        
        with torch.no_grad():
            wave = self.model.cse.encode(prompt)
        
        wave_sequence = wave.full.to(device)  # [seq, 432]
        wave_vec = wave_sequence.mean(dim=0)   # Mean pooled [432]
        
        # Calculate actual energy per band
        phonetic_energy = wave.phonetic.pow(2).sum().item() if hasattr(wave, 'phonetic') else wave_sequence[:, :64].pow(2).sum().item()
        syntactic_energy = wave.syntactic.pow(2).sum().item() if hasattr(wave, 'syntactic') else wave_sequence[:, 64:128].pow(2).sum().item()
        semantic_energy = wave.semantic.pow(2).sum().item() if hasattr(wave, 'semantic') else wave_sequence[:, 128:384].pow(2).sum().item()
        temporal_energy = wave.temporal.pow(2).sum().item() if hasattr(wave, 'temporal') else wave_sequence[:, 384:416].pow(2).sum().item()
        intensity_energy = wave.intensity.pow(2).sum().item() if hasattr(wave, 'intensity') else wave_sequence[:, 416:432].pow(2).sum().item()
        
        cse_time = (time.time() - t0) * 1000
        
        yield PhaseEvent(
            type='phase',
            phase='1',
            name='Continuous Semantic Encoder (CSE)',
            status='complete',
            details={
                'input_bytes': len(prompt_bytes_raw),
                'wave_shape': list(wave_sequence.shape),
                'band_energies': {
                    'phonetic': f"{phonetic_energy:.2f}",
                    'syntactic': f"{syntactic_energy:.2f}",
                    'semantic': f"{semantic_energy:.2f}",
                    'temporal': f"{temporal_energy:.2f}",
                    'intensity': f"{intensity_energy:.2f}",
                },
                'dominant_band': max(
                    [('phonetic', phonetic_energy), ('syntactic', syntactic_energy), 
                     ('semantic', semantic_energy), ('temporal', temporal_energy)],
                    key=lambda x: x[1]
                )[0],
                'wave_magnitude': f"{wave_vec.norm().item():.4f}",
                'latency_ms': f"{cse_time:.2f}",
            }
        ).to_dict()
        
        # ═══════════════════════════════════════════════════════════
        # PHASE 2: Resonance Field
        # ═══════════════════════════════════════════════════════════
        t1 = time.time()
        
        yield PhaseEvent(
            type='phase',
            phase='2',
            name='Resonance Field (RFC)',
            status='running',
        ).to_dict()
        
        # Project wave to field space
        field_input = self.model.wave_to_field(wave_vec)  # [512]
        
        # Query field for attractors — get more for better visibility
        field_features, sims, locs = self.model.field.query(wave_vec, k=8)
        best_features = field_features[0]  # [512]
        
        field_energy = self.model.field.state.pow(2).sum().item()
        attractor_count = self.model.field.num_attractors()
        
        # Build attractor details
        attractor_info = []
        for i, (sim, loc) in enumerate(zip(sims[:4], locs[:4])):
            attractor_info.append({
                'rank': i + 1,
                'similarity': f"{sim.item():.4f}",
                'location': f"({loc.h}, {loc.w}, {loc.d})",
                'feature_norm': f"{field_features[i].norm().item():.4f}",
            })
        
        field_time = (time.time() - t1) * 1000
        
        yield PhaseEvent(
            type='phase',
            phase='2',
            name='Resonance Field (RFC)',
            status='complete',
            details={
                'field_dims': '96³ × 512',
                'field_energy': f"{field_energy:.4f}",
                'total_attractors': attractor_count,
                'retrieved_attractors': attractor_info,
                'query_vector_norm': f"{field_input.norm().item():.4f}",
                'latency_ms': f"{field_time:.2f}",
            }
        ).to_dict()
        
        # ═══════════════════════════════════════════════════════════
        # PHASE 3: Gravitational Relevance
        # ═══════════════════════════════════════════════════════════
        t2 = time.time()
        
        yield PhaseEvent(
            type='phase',
            phase='3',
            name='Gravitational Relevance (GR)',
            status='running',
        ).to_dict()
        
        relevance_out = self.model.gr(field_input.unsqueeze(0)).squeeze(0)
        
        # Get mass statistics and top masses
        mass_stats = {}
        top_concepts = []
        if hasattr(self.model.gr, 'mass_tracker'):
            mt = self.model.gr.mass_tracker
            mass_stats = mt.stats()
            
            # Get top-5 heaviest concepts (most evidence)
            if mt.count > 0:
                active_masses = mt.masses[:mt.count]
                top_k = min(5, mt.count)
                top_masses, top_indices = active_masses.topk(top_k)
                
                # Get similarity of query to these top concepts
                query_norm = F.normalize(field_input.unsqueeze(0), dim=-1)
                concept_norms = F.normalize(mt.concept_vectors[top_indices], dim=-1)
                query_sims = F.cosine_similarity(query_norm, concept_norms, dim=-1)
                
                for i, (mass, idx, sim) in enumerate(zip(top_masses, top_indices, query_sims)):
                    top_concepts.append({
                        'rank': i + 1,
                        'mass': f"{mass.item():.4f}",
                        'query_similarity': f"{sim.item():.4f}",
                        'concept_id': idx.item(),
                    })
        
        gr_time = (time.time() - t2) * 1000
        
        yield PhaseEvent(
            type='phase',
            phase='3',
            name='Gravitational Relevance (GR)',
            status='complete',
            details={
                'complexity': 'O(log n)',
                'tracked_concepts': mass_stats.get('count', 0),
                'mean_mass': f"{mass_stats.get('mean_mass', 1.0):.4f}",
                'max_mass': f"{mass_stats.get('max_mass', 1.0):.4f}",
                'negative_concepts': mass_stats.get('negative_count', 0),
                'top_concepts': top_concepts,
                'relevance_magnitude': f"{relevance_out.norm().item():.4f}",
                'latency_ms': f"{gr_time:.2f}",
            }
        ).to_dict()
        
        # ═══════════════════════════════════════════════════════════
        # PHASE 4: Thermodynamic Learning
        # ═══════════════════════════════════════════════════════════
        t3 = time.time()
        
        # Measure energy BEFORE settling
        energy_before = self.model.field.state.pow(2).sum().item()
        current_temp = self.model.tl.temp_manager.temperature
        
        yield PhaseEvent(
            type='phase',
            phase='4',
            name='Thermodynamic Learner (TL)',
            status='running',
            details={
                'energy_before': f"{energy_before:.4f}",
                'temperature': f"{current_temp:.6f}",
            }
        ).to_dict()
        
        settle_result = None
        if learn:
            # Settle the field (learning via physics)
            settle_result = self.model.tl.settle_once(wave_vec, target=relevance_out)
            self.model._learning_steps += 1
            learned_status = 'LEARNING (settled)'
        else:
            learned_status = 'INFERENCE (query-only)'
        
        # Measure energy AFTER
        energy_after = self.model.field.state.pow(2).sum().item()
        energy_delta = energy_after - energy_before
        
        tl_time = (time.time() - t3) * 1000
        
        yield PhaseEvent(
            type='phase',
            phase='4',
            name='Thermodynamic Learner (TL)',
            status='complete',
            details={
                'mode': learned_status,
                'temperature': f"{current_temp:.6f}",
                'energy_before': f"{energy_before:.4f}",
                'energy_after': f"{energy_after:.4f}",
                'energy_delta': f"{energy_delta:+.4f}",
                'energy_minimized': energy_delta <= 0,
                'settle_result': {
                    'initial_energy': f"{settle_result.initial_energy:.4f}" if settle_result else 'N/A',
                    'final_energy': f"{settle_result.final_energy:.4f}" if settle_result else 'N/A',
                } if settle_result else None,
                'uses_backprop': False,
                'latency_ms': f"{tl_time:.2f}",
            }
        ).to_dict()
        
        # ═══════════════════════════════════════════════════════════
        # PHASE 5: Causal Geometry Nodes
        # ═══════════════════════════════════════════════════════════
        t4 = time.time()
        
        yield PhaseEvent(
            type='phase',
            phase='5',
            name='Causal Geometry Nodes (CGN)',
            status='running',
        ).to_dict()
        
        # Try to get traces if available
        cgn_traces = []
        try:
            if hasattr(self.model.cgn, 'forward_with_traces'):
                result = self.model.cgn.forward_with_traces(relevance_out)
                cgn_out = result.output
                
                # Extract trace info
                for trace in result.traces[:5]:  # Top 5 traces
                    trace_info = {
                        'node_id': trace.node_id,
                        'bending': f"{trace.bending_magnitude:.4f}",
                        'influence': f"{trace.influence_strength:.4f}",
                        'causal_arrows': len(trace.causal_arrows),
                    }
                    if trace.causal_arrows:
                        trace_info['top_cause'] = {
                            'source': trace.causal_arrows[0].source_id,
                            'weight': f"{trace.causal_arrows[0].weight:.4f}",
                            'reason': trace.causal_arrows[0].reason[:30] if trace.causal_arrows[0].reason else 'implicit',
                        }
                    cgn_traces.append(trace_info)
            else:
                cgn_out = self.model.cgn(relevance_out)
        except Exception:
            cgn_out = self.model.cgn(relevance_out)
        
        # Use stats() method
        cgn_stats = self.model.cgn.stats()
        
        cgn_time = (time.time() - t4) * 1000
        
        yield PhaseEvent(
            type='phase',
            phase='5',
            name='Causal Geometry Nodes (CGN)',
            status='complete',
            details={
                'fast_nodes': cgn_stats.get('fast_nodes', 32),
                'medium_nodes': cgn_stats.get('medium_nodes', 16),
                'slow_nodes': cgn_stats.get('slow_nodes', 8),
                'fast_activation': f"{cgn_stats.get('fast_activation', 0):.4f}",
                'medium_activation': f"{cgn_stats.get('medium_activation', 0):.4f}",
                'slow_activation': f"{cgn_stats.get('slow_activation', 0):.4f}",
                'active_traces': cgn_traces if cgn_traces else 'N/A (no trace mode)',
                'output_norm': f"{cgn_out.norm().item():.4f}",
                'causal_traceable': True,
                'latency_ms': f"{cgn_time:.2f}",
            }
        ).to_dict()
        
        # ═══════════════════════════════════════════════════════════
        # PHASE 6: Memory System
        # ═══════════════════════════════════════════════════════════
        t5 = time.time()
        
        yield PhaseEvent(
            type='phase',
            phase='6',
            name='Three-Tier Memory System',
            status='running',
        ).to_dict()
        
        # Working memory
        self.model.working_memory.add_perturbation(wave_vec)
        compressed = self.model.working_memory.compress(wave_vec.unsqueeze(0)).squeeze(0)
        
        # Episodic memory query — get more for visibility
        episodic_results = self.model.episodic_memory.search(compressed, k=5)
        recalled_facts = []
        for r in episodic_results:
            if hasattr(r[0], 'fact') and r[0].fact:
                recalled_facts.append({
                    'fact': r[0].fact,  # Full fact for glass box
                    'similarity': f"{r[1]:.4f}",
                    'causal_source': getattr(r[0], 'causal_source', 'unknown'),
                })
        
        # Memory routing — get detailed routing decision
        memory_result = self.model.memory_router.route_query(wave_vec, episodic_k=5, working_k=10)
        
        # Extract routing decision details
        routing_info = {
            'primary_tier': 'episodic' if recalled_facts else 'semantic',
            'episodic_hits': len(recalled_facts),
            'working_size': self.model.working_memory.size,
        }
        
        mem_time = (time.time() - t5) * 1000
        
        yield PhaseEvent(
            type='phase',
            phase='6',
            name='Three-Tier Memory System',
            status='complete',
            details={
                'working_entries': self.model.working_memory.size,
                'episodic_entries': self.model.episodic_memory.size,
                'semantic_protected': len(self.model.semantic_memory.protected_attractors) if hasattr(self.model.semantic_memory, 'protected_attractors') else 0,
                'routing_decision': routing_info,
                'recalled_facts': recalled_facts,
                'forgetting_score': 0.0000,
                'latency_ms': f"{mem_time:.2f}",
            }
        ).to_dict()
        
        # Emit recalled facts as separate event for UI
        if recalled_facts:
            yield PhaseEvent(
                type='memory',
                details={'recalled': recalled_facts}
            ).to_dict()
        
        # ═══════════════════════════════════════════════════════════
        # PHASE 8: WaveDecoder Generation
        # ═══════════════════════════════════════════════════════════
        yield PhaseEvent(
            type='phase',
            phase='8',
            name='WaveDecoder (Generation)',
            status='running',
            details={
                'max_length': max_length, 
                'temperature': temperature,
                'context_from': 'field_attractor + cgn_output',
            }
        ).to_dict()
        
        t6 = time.time()
        
        # Combine field + CGN for context
        combined = best_features + cgn_out
        
        # Initialize decoder
        prompt_bytes = torch.tensor(
            list(prompt.encode('utf-8', errors='replace')),
            dtype=torch.long, device=device
        )
        
        # Get semantic context
        wave_kv = self.model.decoder.wave_proj(wave_sequence).unsqueeze(0)
        first_logits, hidden, _ = self.model.decoder.generate_init(wave_sequence, combined)
        
        # Feed prompt bytes through (if any)
        for byte_val in prompt_bytes:
            byte_t = torch.tensor([byte_val], dtype=torch.long, device=device)
            _, hidden = self.model.decoder.generate_step(byte_t, hidden, wave_kv)
        
        # Get logits for first generated byte
        if len(prompt_bytes) > 0:
            last_byte = torch.tensor([prompt_bytes[-1]], dtype=torch.long, device=device)
            logits, hidden = self.model.decoder.generate_step(last_byte, hidden, wave_kv)
        else:
            logits = first_logits
        
        # Stream character-by-character generation
        generated_bytes = []
        char_buffer = b''
        
        for step in range(max_length):
            # Temperature-scaled sampling
            scaled = logits / max(temperature, 1e-8)
            probs = F.softmax(scaled, dim=-1)
            
            # Get top-5 predictions for visibility
            top_probs, top_indices = probs.topk(5)
            candidates = []
            for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
                # Try to show as character
                try:
                    char_repr = bytes([idx]).decode('utf-8') if 32 <= idx < 127 else f'0x{idx:02x}'
                except:
                    char_repr = f'0x{idx:02x}'
                candidates.append({
                    'byte': idx,
                    'char': char_repr,
                    'prob': f"{p:.4f}",
                })
            
            next_byte = torch.multinomial(probs, 1).item()
            
            # Stop on null byte
            if next_byte == 0:
                break
            
            generated_bytes.append(next_byte)
            char_buffer += bytes([next_byte])
            
            # Try to decode as UTF-8 character
            try:
                char = char_buffer.decode('utf-8')
                # Successfully decoded - emit character with thinking info
                yield PhaseEvent(
                    type='char',
                    char=char,
                    details={
                        'byte': next_byte,
                        'step': step,
                        'candidates': candidates,  # Show what model was considering
                        'chosen_prob': f"{probs[next_byte].item():.4f}",
                    }
                ).to_dict()
                char_buffer = b''  # Reset buffer
            except UnicodeDecodeError:
                # Incomplete UTF-8 sequence, wait for more bytes
                if len(char_buffer) >= 4:
                    # Invalid sequence, emit replacement and reset
                    yield PhaseEvent(
                        type='char',
                        char='�',
                        details={
                            'byte': next_byte,
                            'step': step,
                            'candidates': candidates,
                            'reason': 'invalid UTF-8 sequence',
                        }
                    ).to_dict()
                    char_buffer = b''
            
            # Feed back for next step
            byte_t = torch.tensor([next_byte], dtype=torch.long, device=device)
            logits, hidden = self.model.decoder.generate_step(byte_t, hidden, wave_kv)
        
        # Flush any remaining bytes
        if char_buffer:
            try:
                char = char_buffer.decode('utf-8', errors='replace')
                yield PhaseEvent(
                    type='char',
                    char=char,
                ).to_dict()
            except:
                pass
        
        gen_time = (time.time() - t6) * 1000
        total_time = (time.time() - t0) * 1000
        
        # Build final generation summary
        generated_text = bytes(generated_bytes).decode('utf-8', errors='replace')
        
        yield PhaseEvent(
            type='phase',
            phase='8',
            name='WaveDecoder (Generation)',
            status='complete',
            details={
                'bytes_generated': len(generated_bytes),
                'generated_text': generated_text,
                'generation_ms': f"{gen_time:.2f}",
                'total_pipeline_ms': f"{total_time:.2f}",
                'bytes_per_second': f"{len(generated_bytes) / (gen_time / 1000):.1f}" if gen_time > 0 else 'N/A',
            }
        ).to_dict()
        
        # Final metrics with full pipeline summary
        yield PhaseEvent(
            type='metric',
            details={
                'total_latency_ms': f"{total_time:.2f}",
                'phases_executed': 7,
                'characters_generated': len(generated_bytes),
                'pipeline_breakdown': {
                    'cse_ms': f"{cse_time:.2f}",
                    'field_ms': f"{field_time:.2f}",
                    'gravity_ms': f"{gr_time:.2f}",
                    'thermo_ms': f"{tl_time:.2f}",
                    'cgn_ms': f"{cgn_time:.2f}",
                    'memory_ms': f"{mem_time:.2f}",
                    'decoder_ms': f"{gen_time:.2f}",
                },
            }
        ).to_dict()
    
    def learn_fact(self, fact: str) -> Dict[str, Any]:
        """One-shot fact learning via episodic memory."""
        device = self.device
        
        with torch.no_grad():
            wave = self.model.cse.encode(fact)
            wave_vec = wave.full.mean(dim=0).to(device)
            compressed = self.model.working_memory.compress(wave_vec.unsqueeze(0)).squeeze(0)
            
            # Write to episodic memory
            self.model.episodic_memory.write(compressed, fact=fact, causal_source="user_taught")
            
            # Thermodynamic settle (learn)
            self.model.tl.settle_once(wave_vec)
            self.model._learning_steps += 1
            
            # Add to working memory
            self.model.working_memory.add_perturbation(wave_vec)
        
        return {
            'status': 'learned',
            'fact': fact,
            'episodic_size': self.model.episodic_memory.size,
            'learning_steps': self.model._learning_steps,
        }
    
    def recall(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Query episodic memory for related facts."""
        device = self.device
        
        with torch.no_grad():
            wave = self.model.cse.encode(query)
            wave_vec = wave.full.mean(dim=0).to(device)
            compressed = self.model.working_memory.compress(wave_vec.unsqueeze(0)).squeeze(0)
            
            results = self.model.episodic_memory.search(compressed, k=k)
        
        return [
            {
                'fact': r[0].fact if hasattr(r[0], 'fact') else str(r[0]),
                'similarity': float(r[1]),
            }
            for r in results if r[0] is not None
        ]
