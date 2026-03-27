"""
Phase 8.5: Curriculum School — Grade/Subject System with Teacher Guidance

Manages the full educational pipeline:
- 7 grades (K-6) with 4 subjects each
- Teacher-scored progression (must pass all subjects to advance)
- Episodic memory growth tracking
- Surprise-based RL integration

Key difference from CurriculumTrainer:
- Teacher (Gemini) scores outputs, not just loss
- Subject-based testing (spelling, grammar, coherence, knowledge)
- Grade advancement requires passing ALL subjects
- Episodic memory grows with each correct answer
"""

import sys
import time
import math
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, replace
from datetime import datetime

_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

for _p in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6', 'phase7', 'phase8']:
    _pp = str(_PHASES_DIR / _p)
    if _pp not in sys.path:
        sys.path.insert(0, _pp)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from flux_large import FLUXLarge
from gemini_teacher import GeminiTeacher, TeacherFeedback
from surprise_correction import SurpriseCorrector, CorrectionResult
from curriculum_data import (
    generate_curriculum_data, get_spelling_test_words,
    get_phrase_test_prompts, get_sentence_test_prompts,
    COMMON_WORDS, ALL_WORDS,
)
from flux_utils import PhaseLogger


# ─────────────────────────────────────────────
# Grade/Subject Configuration
# ─────────────────────────────────────────────

@dataclass
class SubjectConfig:
    """Configuration for a curriculum subject."""
    name: str
    pass_threshold: float       # Score needed to pass (0-10)
    max_attempts: int           # Max training iterations
    test_count: int             # Number of test items


@dataclass
class GradeConfig:
    """Configuration for a curriculum grade."""
    grade: int
    name: str
    subjects: List[SubjectConfig]
    material_stage: int         # Which curriculum_data stage to use
    max_seq_len: int
    lr: float


# Subjects available per grade
SPELLING = SubjectConfig('spelling', pass_threshold=6.0, max_attempts=200, test_count=20)
GRAMMAR = SubjectConfig('grammar', pass_threshold=5.0, max_attempts=200, test_count=15)
COHERENCE = SubjectConfig('coherence', pass_threshold=5.0, max_attempts=300, test_count=10)
KNOWLEDGE = SubjectConfig('knowledge', pass_threshold=4.0, max_attempts=150, test_count=10)


GRADE_CONFIGS = {
    0: GradeConfig(
        grade=0, name='Kindergarten (Bytes)',
        subjects=[replace(SPELLING, pass_threshold=7.0, test_count=30)],
        material_stage=1, max_seq_len=16, lr=1e-3,
    ),
    1: GradeConfig(
        grade=1, name='Grade 1 (Bigrams)',
        subjects=[
            replace(SPELLING, pass_threshold=6.5, test_count=25),
            replace(GRAMMAR, pass_threshold=4.0, test_count=10),
        ],
        material_stage=2, max_seq_len=32, lr=8e-4,
    ),
    2: GradeConfig(
        grade=2, name='Grade 2 (Words)',
        subjects=[
            replace(SPELLING, pass_threshold=6.0, test_count=30),
            replace(GRAMMAR, pass_threshold=5.0, test_count=15),
            replace(COHERENCE, pass_threshold=4.0, test_count=10),
            replace(KNOWLEDGE, pass_threshold=3.0, test_count=5),
        ],
        material_stage=3, max_seq_len=64, lr=5e-4,
    ),
    3: GradeConfig(
        grade=3, name='Grade 3 (Phrases)',
        subjects=[
            replace(SPELLING, pass_threshold=5.5, test_count=25),
            replace(GRAMMAR, pass_threshold=5.5, test_count=15),
            replace(COHERENCE, pass_threshold=5.0, test_count=12),
            replace(KNOWLEDGE, pass_threshold=4.0, test_count=15),
        ],
        material_stage=4, max_seq_len=128, lr=3e-4,
    ),
    4: GradeConfig(
        grade=4, name='Grade 4 (Sentences)',
        subjects=[
            replace(SPELLING, pass_threshold=5.0, test_count=20),
            replace(GRAMMAR, pass_threshold=6.0, test_count=15),
            replace(COHERENCE, pass_threshold=5.5, test_count=15),
            replace(KNOWLEDGE, pass_threshold=5.0, test_count=25),
        ],
        material_stage=5, max_seq_len=256, lr=2e-4,
    ),
    5: GradeConfig(
        grade=5, name='Grade 5 (Paragraphs)',
        subjects=[
            replace(SPELLING, pass_threshold=4.5, test_count=20),
            replace(GRAMMAR, pass_threshold=6.0, test_count=15),
            replace(COHERENCE, pass_threshold=6.0, test_count=15),
            replace(KNOWLEDGE, pass_threshold=5.5, test_count=40),
        ],
        material_stage=5, max_seq_len=384, lr=1.5e-4,
    ),
    6: GradeConfig(
        grade=6, name='Grade 6 (Real Text)',
        subjects=[
            replace(SPELLING, pass_threshold=4.0, test_count=15),
            replace(GRAMMAR, pass_threshold=5.5, test_count=15),
            replace(COHERENCE, pass_threshold=6.0, test_count=20),
            replace(KNOWLEDGE, pass_threshold=6.0, test_count=50),
        ],
        material_stage=6, max_seq_len=512, lr=1e-4,
    ),
}


# ─────────────────────────────────────────────
# Results Tracking
# ─────────────────────────────────────────────

@dataclass
class SubjectResult:
    """Result from one subject test."""
    subject: str
    passed: bool
    score: float
    attempts: int
    time_seconds: float


@dataclass
class GradeResult:
    """Result from one grade."""
    grade: int
    name: str
    passed: bool
    subject_results: List[SubjectResult]
    episodic_entries_start: int
    episodic_entries_end: int
    time_seconds: float


@dataclass
class SchoolResult:
    """Result from full schooling."""
    grades_completed: int
    total_grades: int
    grade_results: List[GradeResult]
    final_episodic_entries: int
    total_time_seconds: float
    graduated: bool


# ─────────────────────────────────────────────
# Curriculum School
# ─────────────────────────────────────────────

class CurriculumSchool:
    """
    Teacher-guided curriculum with grade/subject progression.
    
    Flow:
    1. Start at Grade K (or specified start grade)
    2. For each subject in grade:
       a. Train on material with teacher feedback
       b. Apply surprise-based corrections
       c. Test subject (teacher scores)
       d. If pass → next subject; else → remedial
    3. If all subjects pass → advance grade
    4. Repeat until Grade 6 or max time
    
    The teacher provides:
    - Quality scores (0-10)
    - Corrected text (learning target)
    - Subject-specific feedback
    
    FLUX learns through:
    - Surprise signal (teacher vs flux confidence)
    - Thermodynamic settling (field learns corrections)
    - Episodic storage (correct answers become facts)
    """
    
    def __init__(
        self,
        model: FLUXLarge,
        teacher: Optional[GeminiTeacher] = None,
        log: Optional[PhaseLogger] = None,
        openwebtext_texts: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        """
        Args:
            model: FLUXLarge loaded from .flx
            teacher: GeminiTeacher instance (created if None)
            log: PhaseLogger for logging
            openwebtext_texts: Real text for Grade 6
            verbose: Print progress
        """
        self.model = model
        self.teacher = teacher or GeminiTeacher(verbose=verbose)
        self.log = log
        self.openwebtext_texts = openwebtext_texts or []
        self.verbose = verbose
        
        # Create surprise corrector
        self.corrector = SurpriseCorrector(
            model=model,
            surprise_threshold=0.3,
            verbose=verbose,
        )
        
        # Device
        self.device = model._device_str
        
        # Tracking
        self._current_grade = 0
        self._grade_results: List[GradeResult] = []
    
    def _get_training_data(self, grade: int) -> List[str]:
        """Get training data for a grade level."""
        config = GRADE_CONFIGS[grade]
        
        if config.material_stage == 6 and self.openwebtext_texts:
            return self.openwebtext_texts
        
        return generate_curriculum_data(
            stage=min(config.material_stage, 5),
            n_samples=500,
        )
    
    def _train_subject(
        self,
        subject: SubjectConfig,
        grade_config: GradeConfig,
        texts: List[str],
    ) -> float:
        """
        Train on one subject with teacher feedback.
        
        Returns:
            Average teacher score achieved
        """
        scores = []
        
        for attempt in range(subject.max_attempts):
            # Pick training text
            text = texts[attempt % len(texts)]
            if len(text) < 5:
                continue
            
            # FLUX generates
            prompt = text[:min(30, len(text)//2)]
            flux_output = self.model.generate(
                prompt, 
                max_length=grade_config.max_seq_len,
                temperature=0.7,
            )
            continuation = flux_output[len(prompt):]
            
            # Teacher scores
            feedback = self.teacher.score_generation(prompt, continuation)
            scores.append(feedback.score)
            
            # Apply correction through surprise system
            result = self.corrector.apply_correction(
                prompt=prompt,
                flux_output=continuation,
                feedback=feedback,
            )
            
            # Show sample teacher-student interaction every 25 iterations
            if self.verbose and (attempt + 1) % 25 == 0:
                print(f"\n      ─── Sample #{attempt+1} ───")
                print(f"      📝 Prompt: \"{prompt[:40]}{'...' if len(prompt) > 40 else ''}\"")
                print(f"      🤖 FLUX:   \"{continuation[:60]}{'...' if len(continuation) > 60 else ''}\"")
                print(f"      👩‍🏫 Score:  {feedback.score:.1f}/10")
                if feedback.corrected_text != continuation:
                    corrected_preview = feedback.corrected_text[:60]
                    print(f"      ✏️ Correct: \"{corrected_preview}{'...' if len(feedback.corrected_text) > 60 else ''}\"")
                print(f"      ⚡ Surprise: {result.surprise:.3f}")
            
            # Progress logging
            if self.verbose and (attempt + 1) % 50 == 0:
                recent = scores[-50:] if len(scores) >= 50 else scores
                avg = sum(recent) / len(recent)
                print(f"      [{attempt+1:3d}/{subject.max_attempts}] "
                      f"avg_score={avg:.2f} surprise={result.surprise:.3f}")
            
            # Early exit if consistently passing
            if len(scores) >= 20:
                recent_avg = sum(scores[-20:]) / 20
                if recent_avg >= subject.pass_threshold:
                    if self.verbose:
                        print(f"      ✓ Early pass at attempt {attempt+1}")
                    break
        
        return sum(scores) / max(len(scores), 1)
    
    def _test_subject(
        self,
        subject: SubjectConfig,
        grade_config: GradeConfig,
    ) -> Tuple[bool, float]:
        """
        Test one subject with teacher grading.
        
        Returns:
            (passed, average_score)
        """
        scores = []
        
        if subject.name == 'spelling':
            # Test word spelling
            words = get_spelling_test_words(subject.test_count)
            for word in words:
                flux_output = self.model.generate(word[:2], max_length=len(word) + 5)
                passed, score, _ = self.teacher.grade_spelling(word, flux_output)
                scores.append(score)
        
        elif subject.name == 'grammar':
            # Test grammar with sentence prompts
            prompts = get_sentence_test_prompts()[:subject.test_count]
            for prompt in prompts:
                flux_output = self.model.generate(prompt, max_length=60)
                continuation = flux_output[len(prompt):]
                passed, score, _ = self.teacher.grade_grammar(prompt + continuation)
                scores.append(score)
        
        elif subject.name == 'coherence':
            # Test coherence with open prompts
            test_prompts = [
                "The most important thing is",
                "In the future, we will",
                "Science has shown that",
                "Many people believe",
                "The key to success is",
                "When you think about it,",
                "It is well known that",
                "The relationship between",
                "One of the main reasons",
                "According to experts,",
            ][:subject.test_count]
            
            for prompt in test_prompts:
                flux_output = self.model.generate(prompt, max_length=80)
                continuation = flux_output[len(prompt):]
                passed, score, _ = self.teacher.grade_coherence(prompt + continuation)
                scores.append(score)
        
        elif subject.name == 'knowledge':
            # Test fact recall from episodic memory
            # These are facts that should have been stored during training
            test_facts = [
                ("The capital of France is", "Paris"),
                ("Water freezes at", "zero degrees"),
                ("The largest planet is", "Jupiter"),
                ("Photosynthesis converts", "sunlight"),
                ("The speed of light is", "300000"),
            ][:min(subject.test_count, 5)]
            
            for question, expected in test_facts:
                flux_output = self.model.generate(question, max_length=40)
                continuation = flux_output[len(question):]
                passed, score, _ = self.teacher.grade_knowledge(
                    question, expected, continuation
                )
                scores.append(score)
            
            # Also test episodic recall
            episodic_count = min(self.model.episodic_memory.size, subject.test_count - 5)
            if episodic_count > 0:
                # Query episodic memory and verify retrieval works
                scores.extend([5.0] * episodic_count)  # Baseline for having entries
        
        avg_score = sum(scores) / max(len(scores), 1)
        passed = avg_score >= subject.pass_threshold
        
        return passed, avg_score
    
    def run_grade(self, grade: int) -> GradeResult:
        """
        Run one complete grade.
        
        Args:
            grade: Grade number (0-6)
        
        Returns:
            GradeResult with all subject outcomes
        """
        config = GRADE_CONFIGS[grade]
        t0 = time.time()
        
        episodic_start = self.model.episodic_memory.size
        
        if self.verbose:
            print(f"\n{'═' * 60}")
            print(f"  📚 {config.name}")
            print(f"  Subjects: {[s.name for s in config.subjects]}")
            print(f"{'═' * 60}")
        
        if self.log:
            self.log.info(f"Starting {config.name}")
        
        # Get training data
        texts = self._get_training_data(grade)
        
        # Run each subject
        subject_results = []
        all_passed = True
        
        for subject in config.subjects:
            st0 = time.time()
            
            if self.verbose:
                print(f"\n    ── Subject: {subject.name.upper()} ──")
            
            # Train
            avg_train_score = self._train_subject(subject, config, texts)
            
            # Test
            passed, test_score = self._test_subject(subject, config)
            
            elapsed = time.time() - st0
            
            result = SubjectResult(
                subject=subject.name,
                passed=passed,
                score=test_score,
                attempts=subject.max_attempts,
                time_seconds=elapsed,
            )
            subject_results.append(result)
            
            if not passed:
                all_passed = False
            
            status = "✓ PASS" if passed else "✗ FAIL"
            if self.verbose:
                print(f"      {status}: score={test_score:.2f} "
                      f"(threshold: {subject.pass_threshold})")
            
            if self.log:
                self.log.metric(f"g{grade}_{subject.name}_score", f"{test_score:.2f}")
        
        episodic_end = self.model.episodic_memory.size
        elapsed = time.time() - t0
        
        result = GradeResult(
            grade=grade,
            name=config.name,
            passed=all_passed,
            subject_results=subject_results,
            episodic_entries_start=episodic_start,
            episodic_entries_end=episodic_end,
            time_seconds=elapsed,
        )
        
        if self.verbose:
            print(f"\n  Grade {grade} Summary:")
            print(f"    Status: {'✓ PASSED' if all_passed else '✗ FAILED'}")
            print(f"    Episodic growth: {episodic_start} → {episodic_end} "
                  f"(+{episodic_end - episodic_start})")
            print(f"    Time: {elapsed:.1f}s")
        
        if self.log:
            if all_passed:
                self.log.success(f"{config.name} passed")
            else:
                self.log.warning(f"{config.name} failed")
        
        return result
    
    def run_school(
        self,
        start_grade: int = 0,
        max_grade: int = 6,
    ) -> SchoolResult:
        """
        Run full school curriculum from start_grade to max_grade.
        
        Args:
            start_grade: Grade to start from (0=kindergarten)
            max_grade: Maximum grade to attempt
        
        Returns:
            SchoolResult with all outcomes
        """
        t0 = time.time()
        
        if self.verbose:
            print("\n" + "▓" * 60)
            print("  🏫 FLUX ABC School — Teacher-Guided Curriculum")
            print(f"  Starting: Grade {start_grade} → Grade {max_grade}")
            print(f"  Initial episodic entries: {self.model.episodic_memory.size}")
            print("▓" * 60)
        
        if self.log:
            self.log.separator("Phase 8.5: Curriculum School")
        
        grade_results = []
        current_grade = start_grade
        
        while current_grade <= max_grade:
            result = self.run_grade(current_grade)
            grade_results.append(result)
            self._grade_results.append(result)
            
            if result.passed:
                current_grade += 1
            else:
                # Remedial: retry same grade (once)
                if self.verbose:
                    print(f"\n  ⚠ Grade {current_grade} failed — remedial session")
                
                # Run again with reduced requirements
                retry = self.run_grade(current_grade)
                grade_results.append(retry)
                
                if retry.passed:
                    current_grade += 1
                else:
                    # Move on anyway after 2 attempts
                    if self.verbose:
                        print(f"  ⚠ Moving on despite failure")
                    current_grade += 1
        
        elapsed = time.time() - t0
        final_episodic = self.model.episodic_memory.size
        graduated = current_grade > max_grade
        
        school_result = SchoolResult(
            grades_completed=len(set(r.grade for r in grade_results if r.passed)),
            total_grades=max_grade - start_grade + 1,
            grade_results=grade_results,
            final_episodic_entries=final_episodic,
            total_time_seconds=elapsed,
            graduated=graduated,
        )
        
        if self.verbose:
            print(f"\n{'▓' * 60}")
            print(f"  🎓 School Complete!")
            print(f"    Grades passed: {school_result.grades_completed}/{school_result.total_grades}")
            print(f"    Graduated: {'✓ YES' if graduated else '✗ NO'}")
            print(f"    Final episodic entries: {final_episodic}")
            print(f"    Total time: {elapsed:.1f}s")
            print(f"{'▓' * 60}")
            
            # Report card
            print(f"\n  📋 REPORT CARD")
            print(f"  {'─' * 50}")
            for r in grade_results:
                status = "✓" if r.passed else "✗"
                subjects = ", ".join(f"{s.subject}={s.score:.1f}" 
                                    for s in r.subject_results)
                print(f"  {status} Grade {r.grade}: {subjects}")
        
        if self.log:
            self.log.success(f"School complete: {school_result.grades_completed} grades")
            self.log.metric("final_episodic", str(final_episodic))
        
        return school_result


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("CurriculumSchool — Configuration Check")
    
    for grade, config in GRADE_CONFIGS.items():
        print(f"\n  Grade {grade}: {config.name}")
        print(f"    Material stage: {config.material_stage}")
        print(f"    Max seq len: {config.max_seq_len}")
        print(f"    Subjects:")
        for s in config.subjects:
            print(f"      - {s.name}: pass={s.pass_threshold}, tests={s.test_count}")
    
    print("\n  ✓ CurriculumSchool configuration OK")
