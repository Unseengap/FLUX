# Phase 8.5: Teacher-Guided ABC School for FLUX

## Overview

Phase 8.5 takes the trained `Flux-beta.flx` model and teaches it to generate coherent text through a **teacher-guided curriculum** with **reinforcement learning via surprise correction**.

Unlike Phase 8 (raw training on OpenWebText), Phase 8.5 uses:
1. **Gemini Teacher**: An external LLM that scores FLUX outputs and provides corrections
2. **Surprise-Based Self-Correction**: FLUX's thermodynamic surprise signal drives RL-style updates
3. **Grade-Based Subjects**: Model must PASS all subjects within a grade before advancing
4. **Episodic Memory Growth**: Every correct answer → new episodic entry → better field attractors

**Source Model**: `Flux-beta.flx` (Phase 8, 69M params, ~74 episodic entries)
**Target**: Coherent English generation + 500+ episodic entries + stable field attractors

---

## The Core Insight: Learning Through Correction

```
┌──────────────────────────────────────────────────────────────────┐
│                    FLUX ABC SCHOOL LOOP                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. FLUX generates text from prompt                              │
│        ↓                                                         │
│  2. Gemini Teacher scores (0-10) + provides correction           │
│        ↓                                                         │
│  3. FLUX computes SURPRISE = |expected - actual| score           │
│        ↓                                                         │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  if surprise HIGH:                                          │ │
│  │    → Train decoder on teacher's correction (learn mistake)  │ │
│  │    → Field temperature rises (thermodynamic instability)    │ │
│  │    → NO episodic memory write (don't memorize errors)       │ │
│  │                                                             │ │
│  │  if surprise LOW:                                           │ │
│  │    → Store fact in episodic memory (reinforce knowledge)    │ │
│  │    → Field temperature falls (settling toward attractor)    │ │
│  │    → Write to semantic memory if repeated 3+ times          │ │
│  └─────────────────────────────────────────────────────────────┘ │
│        ↓                                                         │
│  4. Subject test: Did FLUX pass this subject?                    │
│        ↓                                                         │
│  5. Grade advancement: All subjects passed → next grade          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Grade Structure: 6 Grades × 4 Subjects Each

### Subjects per Grade

| Subject | Tests | What FLUX Learns |
|---------|-------|------------------|
| **Spelling** | Can spell words correctly | Byte-level accuracy |
| **Grammar** | Correct verb/noun agreement | Syntactic structure |
| **Coherence** | Sentences make logical sense | Semantic flow |
| **Knowledge** | Can recall facts correctly | Episodic retrieval |

### Grade Progression

| Grade | Material | Spelling Test | Grammar Test | Coherence Test | Knowledge Test |
|-------|----------|---------------|--------------|----------------|----------------|
| K | Single letters & bytes | 80% ASCII | N/A | N/A | N/A |
| 1 | Common bigrams/trigrams | 70% bigrams | Basic patterns | N/A | N/A |
| 2 | Top-1000 words | 60% words | Subject-verb | Simple sentences | 5 facts |
| 3 | Common phrases | 50% phrases | Full agreement | Multi-sentence | 20 facts |
| 4 | Simple sentences | 40% sentences | Punctuation | Paragraph flow | 50 facts |
| 5 | Paragraphs | 30% paragraphs | Complex grammar | Story coherence | 100 facts |
| 6 | Real OpenWebText | 25% passages | All rules | Full coherence | 200 facts |

**Advancement Rule**: Must pass ALL 4 subjects (or 3 for grades K-1) to advance.

---

## Key Components

### 1. `GeminiTeacher` — The Scorer

```python
class GeminiTeacher:
    """
    External LLM teacher that scores FLUX outputs and provides corrections.
    
    Uses Gemini API (free tier: 60 calls/min) for:
    - Scoring text quality (0-10)
    - Providing corrected versions
    - Grading subject-specific tests
    """
    
    def score_generation(self, prompt: str, flux_output: str) -> TeacherFeedback:
        """
        Score FLUX's generation and provide correction.
        
        Returns:
            TeacherFeedback with:
            - score: 0-10 quality score
            - corrected_text: What FLUX should have generated
            - feedback: Brief explanation of issues
            - subject_scores: Per-subject breakdown
        """
```

### 2. `SurpriseCorrector` — The RL Bridge

```python
class SurpriseCorrector:
    """
    Connects Gemini feedback to FLUX's thermodynamic learning.
    
    Surprise = |teacher_score - flux_confidence|
    
    High surprise → Strong gradient signal → Learn from correction
    Low surprise  → Weak gradient signal → Reinforce existing knowledge
    """
    
    def compute_surprise(
        self, 
        flux_confidence: float,   # FLUX's own confidence (from field energy)
        teacher_score: float,      # Gemini's quality rating
    ) -> float:
        """
        Compute surprise signal for RL update.
        
        FLUX thought it was confident but teacher disagrees → HIGH surprise
        FLUX was uncertain and teacher confirms → LOW surprise (good learning)
        """
    
    def apply_correction(
        self,
        model: FLUXLarge,
        flux_output: str,
        teacher_correction: str,
        surprise: float,
    ) -> float:
        """
        Apply gradient update weighted by surprise.
        
        Returns:
            loss value (higher surprise = stronger update)
        """
```

### 3. `CurriculumSchool` — The Orchestrator

```python
class CurriculumSchool:
    """
    Manages the full grade/subject curriculum with teacher integration.
    
    Replaces simple CurriculumTrainer with:
    - Subject-based testing
    - Teacher-scored advancement
    - Episodic memory growth tracking
    - Grade report cards
    """
    
    def run_school(self, start_grade: int = 0) -> SchoolResult:
        """
        Run full curriculum from kindergarten to graduation.
        
        Each grade:
        1. For each subject: train until pass or max_attempts
        2. Test all subjects
        3. If all pass → advance; else → remedial training
        """
```

---

## .flx Loading

Phase 8.5 loads EXCLUSIVELY from `Flux-beta.flx`:

```python
def load_from_flx(path: Path = Path('checkpoints/Flux-beta.flx')) -> FLUXLarge:
    """
    Load FLUXLarge from .flx archive.
    
    The .flx format preserves:
    - All 14 trained components
    - Thermodynamic state (field temperature/energy)
    - Episodic memory (existing facts)
    - Physics state (gravitational masses)
    
    This is the foundation — Phase 8.5 builds on it.
    """
    flx = torch.load(path, map_location='cpu')
    
    # Verify format
    assert flx['format'] == 'FLUX', "Not a valid .flx file"
    assert flx['version'].startswith('1.'), f"Unsupported version: {flx['version']}"
    
    # Build model shell
    model = FLUXLarge(device='cpu')
    
    # Load all components
    model.cse.load_state_dict(flx['cse']['state_dict'])
    model.field.load_state_dict(flx['field']['state_dict'])
    model.gr.load_state(flx['field']['gravity_state'])
    model.tl.load_state(flx['field']['thermodynamic_state'])
    # ... etc for all 14 components
    
    return model
```

---

## Episodic Memory Growth Strategy

The key to coherent generation is **episodic density**:

| Phase | Episodic Entries | Generation Quality |
|-------|------------------|-------------------|
| Phase 8 (start) | 74 | Word fragments |
| Grade 2 complete | ~200 | Broken words |
| Grade 4 complete | ~500 | Short sentences |
| Grade 6 complete | ~1000+ | Full coherence |

**Every correct answer → episodic write:**
```python
def on_correct_answer(model: FLUXLarge, prompt: str, correct_text: str):
    """Store correct Q→A pair in episodic memory."""
    wave = model.cse.encode(prompt + " " + correct_text)
    model.episodic_memory.store(
        key=wave.full.mean(dim=0),
        value=correct_text,
        metadata={'grade': current_grade, 'subject': current_subject},
    )
```

---

## File Structure

```
phases/phase8_5/
├── PHASE_8_5_SPEC.md         ← This spec
├── gemini_teacher.py         ← GeminiTeacher class (API integration)
├── surprise_correction.py    ← SurpriseCorrector (RL bridge)
├── curriculum_school.py      ← CurriculumSchool (grade/subject orchestrator)
├── curriculum_data.py        ← Training data generation (existing + extended)
├── curriculum_trainer.py     ← Basic trainer (still used internally)
├── flx_loader.py             ← .flx format loading utilities
├── train_curriculum.py       ← CLI entry point
├── test_phase8_5_test1.py    ← Test: Gemini teacher integration
├── test_phase8_5_test2.py    ← Test: Surprise correction works
├── test_phase8_5_test3.py    ← Test: Grade advancement
├── test_phase8_5_test4.py    ← Test: Episodic growth
├── demo_phase8_5_demo1.py    ← Demo: Teacher-corrected generation
├── demo_phase8_5_demo2.py    ← Demo: Before/after comparison
├── RESULTS_PHASE_8_5.md      ← Auto-generated results
│
└── notebooks/
    └── phase8_5_school.ipynb ← Interactive Kaggle training notebook
```

---

## Acceptance Criteria

| Criterion | Target | Method |
|-----------|--------|--------|
| Loads from Flux-beta.flx | ✓ All components | Smoke test |
| Gemini teacher works | Scores + corrections | test1 |
| Surprise correction active | RL updates observed | test2 |
| Grade 4+ reached | Passes 4+ grades | Training log |
| Episodic entries > 300 | Memory growth | Runtime check |
| Spelling accuracy > 50% | Word spelling test | test3 |
| Coherence score > 40% | Sentence test | test4 |
| Checkpoint saved | phase8_5.phase.pt + .flx | Cell output |

---

## Environment Variables

```bash
# Required for Gemini teacher
GEMINI_API_KEY=your_key_here   # Free tier: 60 req/min

# Or in Kaggle secrets
# Add "GEMINI_API_KEY" to Kaggle → Add-ons → Secrets
```

---

## Dependencies

- `Flux-beta.flx` (Phase 8 trained model) — **REQUIRED**
- `google-generativeai` (Gemini API client)
- All Phase 8 modules (flux_large.py, wave_decoder.py, etc.)
- OpenWebText (HuggingFace datasets) for Grade 6
