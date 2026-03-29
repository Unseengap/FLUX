# FLUX Unified Model Specification

**Phase: UNIFIED** — The Complete FLUX Stack for ARC-AGI-3

---

## Overview

This specification defines how to combine existing checkpoints into a single, configurable FLUX model with all necessary components for ARC-AGI-3 reasoning tasks.

### Model Lineage

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FLUX-UNIFIED.flx                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FROM Flux-X-complete.flx (BASE):                                      │
│  ├── CSE (ContinuousSemanticEncoder) — text ↔ wave                     │
│  ├── ResonanceField (96³ × 512) — semantic landscape                   │
│  ├── Memory (Working + Episodic + Semantic)                            │
│  ├── WaveDecoder — wave → text (byte-level)                            │
│  ├── CGN (CausalGeometryNode) — causal structure                       │
│  └── Bridges (wave↔field, router, output_head)                         │
│                                                                         │
│  FROM gridtowave_contrastive.pt (INJECT):                              │
│  └── GridToWave — trained ARC grid encoder                             │
│                                                                         │
│  FROM Flux-capable.flx (INJECT):                                       │
│  └── Enriched Field — 155k diverse samples as attractors               │
│                                                                         │
│  FROM Flux-augmented.flx (ADD):                                        │
│  └── LLM Bridge — Qwen2.5-3B as voice driver                           │
│                                                                         │
│  NEW (IMPLEMENT):                                                       │
│  ├── CausalTracker — action → effect learning                          │
│  ├── RuleInducer — pattern → rule abstraction                          │
│  ├── GoalPlanner — objective → sub-goal decomposition                  │
│  ├── SpatialMemory — Ice & Water navigation                            │
│  └── PerceptionField — active vision system                            │
│                                                                         │
│  SKIP:                                                                  │
│  └── Flux-X-enriched.flx — GPT-2 SVD not needed                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture

### Component Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FLUX-UNIFIED                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    COGNITIVE LAYER (NEW)                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │   │
│  │  │ GoalPlanner │  │ RuleInducer │  │ CausalTracker           │  │   │
│  │  │ (sub-goals) │  │ (patterns)  │  │ (action→effect)         │  │   │
│  │  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │   │
│  │         └────────────────┼─────────────────────┘                │   │
│  └──────────────────────────┼──────────────────────────────────────┘   │
│                             │                                          │
│  ┌──────────────────────────┼──────────────────────────────────────┐   │
│  │                    GENERATION LAYER                              │   │
│  │  ┌───────────────────────┴───────────────────────────────────┐  │   │
│  │  │                    Output Router                           │  │   │
│  │  │  ┌─────────────────┐         ┌─────────────────────────┐  │  │   │
│  │  │  │   LLM Bridge    │◄───────►│   ByteDecoder           │  │  │   │
│  │  │  │ (Qwen2.5-3B)    │  learns │   (learns from LLM)     │  │  │   │
│  │  │  │ [PRIMARY VOICE] │  from   │   [FALLBACK/REALTIME]   │  │  │   │
│  │  │  └─────────────────┘         └─────────────────────────┘  │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                             │                                          │
│  ┌──────────────────────────┼──────────────────────────────────────┐   │
│  │                    KNOWLEDGE LAYER                               │   │
│  │  ┌───────────────────────┴───────────────────────────────────┐  │   │
│  │  │              ResonanceField (96³ × 512)                    │  │   │
│  │  │          [Enriched with 155k samples from capable.flx]    │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  │                                                                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │   │
│  │  │  Working    │  │  Episodic   │  │  Semantic               │  │   │
│  │  │  Memory     │  │  Memory     │  │  Memory                 │  │   │
│  │  │ (session)   │  │ (facts)     │  │ (deep field)            │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                             │                                          │
│  ┌──────────────────────────┼──────────────────────────────────────┐   │
│  │                    PERCEPTION LAYER                              │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │   │
│  │  │ CSE             │  │ GridToWave      │  │ PerceptionField │  │   │
│  │  │ (text→wave)     │  │ (grid→wave)     │  │ (active vision) │  │   │
│  │  │ [Phase 1]       │  │ [TRAINED]       │  │ [NEW]           │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘  │   │
│  │                                                                  │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │              SpatialMemory (Ice & Water)                   │  │   │
│  │  │  exploration_mass + curiosity_field + change_detection    │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Runtime Configuration

The `.flx` file includes a `runtime_config` dict that controls component behavior at inference:

```python
FLUX_RUNTIME_CONFIG = {
    # ─── PERCEPTION ───
    'perception': {
        'cse_enabled': True,           # Text encoding via CSE
        'grid_encoder_enabled': True,  # ARC grid encoding
        'spatial_memory_enabled': True, # Ice & Water fields
        'perception_field_enabled': True, # Active vision
    },
    
    # ─── MEMORY ───
    'memory': {
        'working_memory_enabled': True,   # Session context
        'episodic_memory_enabled': True,  # Fact storage (can disable)
        'semantic_memory_enabled': True,  # Deep field queries
        'memory_write_enabled': True,     # Allow new memories
    },
    
    # ─── GENERATION ───
    'generation': {
        'llm_primary': True,              # LLM leads generation
        'byte_decoder_enabled': True,     # Byte decoder available
        'byte_decoder_learns_from_llm': True, # Distillation mode
        'generation_mode': 'llm',         # 'llm' | 'byte' | 'hybrid'
    },
    
    # ─── REASONING ───
    'reasoning': {
        'causal_tracker_enabled': True,   # Track action→effect
        'rule_inducer_enabled': True,     # Learn rules from patterns
        'goal_planner_enabled': True,     # Decompose objectives
        'hypothesis_testing': True,       # Test inferred rules
    },
    
    # ─── LEARNING ───
    'learning': {
        'realtime_learning': True,        # Learn during inference
        'field_update_enabled': True,     # Modify field attractors
        'causal_graph_update': True,      # Update causal links
        'temperature': 0.3,               # Thermodynamic temperature
        'learning_rate': 0.01,            # Field update rate
    },
    
    # ─── FIELD ───
    'field': {
        'gravity_enabled': True,          # Gravitational relevance
        'interference_enabled': True,     # Wave interference
        'thermodynamic_enabled': True,    # Energy settling
        'field_source': 'capable',        # 'complete' | 'capable' | 'custom'
    },
    
    # ─── LLM ───
    'llm': {
        'model_name': 'Qwen/Qwen2.5-3B-Instruct',
        'quantization': '4bit',
        'max_tokens': 512,
        'temperature': 0.7,
        'use_flux_context': True,         # Inject FLUX memories into prompt
        'flux_context_limit': 10,         # Max memories to inject
    },
}
```

---

## Component Specifications

### 1. CausalTracker (NEW)

**Purpose:** Track action → effect relationships to learn interaction rules.

```python
class CausalTracker:
    """
    Records what happens when agent takes actions.
    Builds causal graph: (position, action) → [changes]
    """
    
    def record(self, position, action, grid_before, grid_after):
        """Record an action and its effects."""
        changes = self.detect_changes(grid_before, grid_after)
        if changes:
            self.causal_links.append({
                'trigger': {'position': position, 'action': action},
                'effects': changes,
                'timestamp': self.step_count,
            })
    
    def query_effects(self, position, action) -> List[Effect]:
        """What happened last time I did this?"""
        
    def find_cause(self, effect_position) -> Optional[Trigger]:
        """What caused this change?"""
```

**Integration:** Called after every env.step() to log (action, effect).

---

### 2. RuleInducer (NEW)

**Purpose:** Abstract patterns from causal links into reusable rules.

```python
class RuleInducer:
    """
    Finds patterns in causal links and abstracts them to rules.
    Example: "stepping on + always changes corner" → Rule
    """
    
    def analyze_causal_links(self) -> List[Rule]:
        """Find recurring patterns in causal tracker."""
        
    def induce_rule(self, pattern: Pattern) -> Rule:
        """
        Abstract a pattern into a general rule.
        Pattern: [(pos_A, action_X) → change_at_B] × N times
        Rule: "action_X at color_C triggers change at corner"
        """
    
    def test_rule(self, rule: Rule, env) -> float:
        """Test a rule by predicting and verifying."""
```

**Integration:** Periodically analyzes causal tracker to extract rules.

---

### 3. GoalPlanner (NEW)

**Purpose:** Decompose high-level objectives into actionable sub-goals.

```python
class GoalPlanner:
    """
    Takes objective and current state, outputs sub-goal sequence.
    Uses learned rules to inform planning.
    """
    
    def __init__(self):
        self.goal_stack = []
        self.achieved = set()
    
    def set_objective(self, objective: str):
        """Parse objective and create goal stack."""
        # Example: "exit door" →
        # [Goal('collect', 'yellow'), Goal('trigger', '+'), Goal('exit', 'door')]
    
    def next_subgoal(self, state) -> Goal:
        """Get the next unachieved goal."""
        
    def is_achieved(self, goal: Goal, state) -> bool:
        """Check if a goal is complete."""
        
    def replan(self, state, blocked_goal: Goal):
        """Replan when a goal is blocked."""
```

**Integration:** Drives navigation targets based on current sub-goal.

---

### 4. PerceptionField (NEW)

**Purpose:** Active vision system with foveal focus and peripheral awareness.

```python
class PerceptionField:
    """
    FLUX's eyes — actively monitors regions of interest.
    """
    
    def __init__(self):
        self.fovea_center = (0, 0)
        self.fovea_radius = 3
        self.attention_map = None  # [H, W] attention weights
        self.watch_list = []       # Objects being tracked
    
    def focus(self, position, radius=3):
        """Shift foveal attention to position."""
    
    def peripheral_scan(self, grid) -> Optional[Tuple[int, int]]:
        """Low-res scan for changes outside focus."""
    
    def track_object(self, object_id, position):
        """Add object to tracking list."""
    
    def predict_next(self, grid, action) -> Grid:
        """What do we expect to see after action?"""
    
    def check_surprise(self, predicted, actual) -> List[Surprise]:
        """Detect expectation violations."""
```

**Integration:** Provides attention-weighted observations to spatial memory.

---

## Checkpoint Combination Process

### Step 1: Load Base (Flux-X-complete.flx)

```python
base = torch.load('Flux-X-complete.flx', map_location='cpu')

# Extract components
cse_state = base['cse']['state_dict']
field_state = base['field']['state_dict']
memory_state = base['memory']
decoder_state = base['decoder']['state_dict']
cgn_state = base['causal']['state_dict']
bridges_state = base['bridges']
```

### Step 2: Inject Trained GridToWave

```python
grid_ckpt = torch.load('gridtowave_contrastive.pt', map_location='cpu')

# Replace untrained GridToWave with trained version
grid_to_wave = GridToWave(wave_dim=432, device=device)
grid_to_wave.load_state_dict(grid_ckpt['encoder_state_dict'])
```

### Step 3: Inject Enriched Field from Flux-capable

```python
capable = torch.load('Flux-capable.flx', map_location='cpu')

# Use the enriched field (155k samples) instead of base field
enriched_field_state = capable['field']['state_dict']
field.load_state_dict(enriched_field_state)
```

### Step 4: Add LLM Bridge

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load quantized LLM
bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)
llm = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-3B-Instruct',
    quantization_config=bnb_config,
    device_map='auto',
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
```

### Step 5: Initialize New Components

```python
# Cognitive layer (new implementations)
causal_tracker = CausalTracker(device=device)
rule_inducer = RuleInducer(causal_tracker)
goal_planner = GoalPlanner()

# Perception (existing + new)
spatial_memory = SpatialMemory(max_size=30, device=device)
perception_field = PerceptionField(device=device)
```

### Step 6: Save Unified Model

```python
unified = {
    'format': 'FLUX',
    'version': '2.0-unified',
    
    # Perception
    'cse': {'config': cse_config, 'state_dict': cse_state},
    'grid_to_wave': {'config': grid_config, 'state_dict': grid_to_wave.state_dict()},
    'spatial_memory': {'config': sm_config, 'state_dict': spatial_memory.state_dict()},
    'perception_field': {'config': pf_config, 'state_dict': perception_field.state_dict()},
    
    # Knowledge
    'field': {'config': field_config, 'state_dict': enriched_field_state},
    'memory': memory_state,
    
    # Generation
    'decoder': {'config': decoder_config, 'state_dict': decoder_state},
    'llm': {'model_name': 'Qwen/Qwen2.5-3B-Instruct', 'quantization': '4bit'},
    
    # Reasoning (new)
    'causal_tracker': {'state_dict': causal_tracker.state_dict()},
    'rule_inducer': {'rules': [], 'patterns': []},
    'goal_planner': {'goal_templates': []},
    
    # Causal graph
    'causal': {'config': cgn_config, 'state_dict': cgn_state},
    'bridges': bridges_state,
    
    # Runtime config
    'runtime_config': FLUX_RUNTIME_CONFIG,
    
    # Metadata
    'metadata': {
        'created': datetime.now().isoformat(),
        'base': 'Flux-X-complete.flx',
        'field_source': 'Flux-capable.flx (155k samples)',
        'grid_encoder': 'gridtowave_contrastive.pt',
        'llm': 'Qwen2.5-3B-Instruct 4bit',
    },
}

torch.save(unified, 'FLUX-UNIFIED.flx')
```

---

## Inference Flow

```python
class FLUXUnified:
    """
    The complete FLUX model for ARC-AGI-3.
    """
    
    def __init__(self, checkpoint_path: str, config_override: dict = None):
        self.load(checkpoint_path)
        if config_override:
            self.config.update(config_override)
    
    def process_grid(self, grid: List[List[int]]) -> Action:
        """
        Full perception → reasoning → action pipeline.
        """
        # 1. PERCEPTION
        if self.config['perception']['grid_encoder_enabled']:
            wave = self.grid_to_wave.encode(grid)
        
        if self.config['perception']['spatial_memory_enabled']:
            self.spatial_memory.observe(self.position, grid)
        
        if self.config['perception']['perception_field_enabled']:
            surprises = self.perception_field.check_surprise(
                self.predicted_grid, grid
            )
        
        # 2. MEMORY
        if self.config['memory']['episodic_memory_enabled']:
            relevant = self.episodic_memory.query(wave)
        
        # 3. REASONING
        if self.config['reasoning']['causal_tracker_enabled']:
            self.causal_tracker.record(
                self.last_position, self.last_action,
                self.last_grid, grid
            )
        
        if self.config['reasoning']['goal_planner_enabled']:
            subgoal = self.goal_planner.next_subgoal(grid)
        
        # 4. PLANNING
        target = self.compute_target(subgoal, grid)
        action = self.spatial_memory.get_next_action(self.position, target)
        
        # 5. PREDICTION
        if self.config['perception']['perception_field_enabled']:
            self.predicted_grid = self.perception_field.predict_next(grid, action)
        
        # 6. LEARNING
        if self.config['learning']['realtime_learning']:
            if self.config['reasoning']['rule_inducer_enabled']:
                self.rule_inducer.analyze_causal_links()
        
        return action
    
    def generate_text(self, prompt: str) -> str:
        """
        Generate text response using configured mode.
        """
        mode = self.config['generation']['generation_mode']
        
        if mode == 'llm' and self.config['generation']['llm_primary']:
            # LLM-led generation with FLUX context
            context = self._get_flux_context(prompt)
            response = self.llm.generate(context + prompt)
            
            # Distill to byte decoder if enabled
            if self.config['generation']['byte_decoder_learns_from_llm']:
                self._distill_to_decoder(prompt, response)
            
            return response
        
        elif mode == 'byte':
            # Pure FLUX byte decoder
            wave = self.cse.encode(prompt)
            return self.decoder.decode(wave)
        
        else:  # hybrid
            # LLM generates, decoder refines
            llm_output = self.llm.generate(prompt)
            wave = self.cse.encode(llm_output)
            return self.decoder.decode(wave)
```

---

## File Format Update

The `.flx` format v2.0 adds:

```python
{
    'format': 'FLUX',
    'version': '2.0-unified',
    
    # NEW: Runtime configuration
    'runtime_config': FLUX_RUNTIME_CONFIG,
    
    # NEW: Component enable flags (for loading)
    'components': {
        'cse': True,
        'grid_to_wave': True,
        'spatial_memory': True,
        'perception_field': True,
        'field': True,
        'working_memory': True,
        'episodic_memory': True,
        'semantic_memory': True,
        'decoder': True,
        'llm': True,
        'causal_tracker': True,
        'rule_inducer': True,
        'goal_planner': True,
        'causal_graph': True,
    },
    
    # All component states...
}
```

---

## Acceptance Criteria

### Perception
- [ ] GridToWave produces discriminative wave encodings (from contrastive training)
- [ ] SpatialMemory tracks exploration and curiosity
- [ ] PerceptionField detects expectation violations

### Knowledge
- [ ] Field contains 155k+ attractor samples (from Flux-capable)
- [ ] Episodic memory can be enabled/disabled at runtime
- [ ] Memory writes can be toggled

### Generation
- [ ] LLM generates fluent text with FLUX context
- [ ] ByteDecoder learns from LLM outputs
- [ ] Generation mode switchable (llm/byte/hybrid)

### Reasoning
- [ ] CausalTracker records action → effect links
- [ ] RuleInducer extracts patterns into rules
- [ ] GoalPlanner decomposes objectives into sub-goals

### Integration
- [ ] All components loadable from single FLUX-UNIFIED.flx
- [ ] Runtime config controls component activation
- [ ] Real-time learning toggleable

---

## Next Steps

1. **Implement CausalTracker** — [phases/phase_unified/causal_tracker.py]
2. **Implement RuleInducer** — [phases/phase_unified/rule_inducer.py]
3. **Implement GoalPlanner** — [phases/phase_unified/goal_planner.py]
4. **Implement PerceptionField** — [phases/phase_unified/perception_field.py]
5. **Update flux_format.py** — Add runtime config support
6. **Create combination script** — Merge checkpoints into FLUX-UNIFIED.flx
7. **Create demo notebook** — Test full pipeline on ARC-AGI-3 game

---

*Specification Version: 1.0*
*Created: 2026-03-28*
