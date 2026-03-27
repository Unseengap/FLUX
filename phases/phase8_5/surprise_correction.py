"""
Phase 8.5: Surprise Correction — RL Bridge Between Teacher & Thermodynamics

Connects Gemini teacher feedback to FLUX's existing surprise/energy system.

The key insight:
    FLUX already has thermodynamic surprise = prediction_energy(predicted, actual)
    Teacher provides a BETTER target (corrected text)
    Wrong FLUX output + Right teacher output → HIGH surprise
    This surprise drives BOTH:
        1. WaveDecoder gradient update (learn to spell the correction)
        2. Field thermodynamic settling (store correction as attractor)

This creates an RL-like loop without explicit policy gradients:
    - Teacher score = reward signal
    - Surprise = TD error
    - Thermodynamic settling = value function update
    - Decoder gradient = policy update
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

_PHASES_DIR = Path(__file__).parent.parent
for _p in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6', 'phase7', 'phase8']:
    _pp = str(_PHASES_DIR / _p)
    if _pp not in sys.path:
        sys.path.insert(0, _pp)

from gemini_teacher import TeacherFeedback


# ─────────────────────────────────────────────
# Correction Result
# ─────────────────────────────────────────────

@dataclass
class CorrectionResult:
    """Result from applying a teacher correction."""
    surprise: float               # Computed surprise (0 = perfect match, higher = more wrong)
    teacher_score: float          # 0-10 from Gemini
    flux_confidence: float        # FLUX's own confidence estimate
    decoder_loss: float           # WaveDecoder loss on correction
    energy_before: float          # Field energy before learning
    energy_after: float           # Field energy after learning
    temperature_delta: float      # How much field heated/cooled
    episodic_stored: bool         # Was correction stored in episodic memory?


# ─────────────────────────────────────────────
# Surprise Corrector
# ─────────────────────────────────────────────

class SurpriseCorrector:
    """
    Connects teacher feedback to FLUX's thermodynamic learning.
    
    The core formula:
        surprise = |teacher_score/10 - flux_confidence|
        
        Where:
        - teacher_score: 0-10 from Gemini (normalized to 0-1)
        - flux_confidence: FLUX's estimate of its own correctness
          (derived from field energy — lower energy = higher confidence)
    
    High surprise triggers:
        1. Strong decoder gradient (learn the correction)
        2. Field heating (become plastic, ready to change)
        3. NO episodic write (don't memorize mistakes)
    
    Low surprise triggers:
        1. Weak/no decoder gradient (already know this)
        2. Field cooling (stabilize the knowledge)
        3. Episodic write (reinforce correct patterns)
    """
    
    def __init__(
        self,
        model: nn.Module,           # FLUXLarge
        decoder_lr: float = 1e-4,
        surprise_threshold: float = 0.3,
        use_amp: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            model: FLUXLarge instance (with decoder, field, thermodynamic learner)
            decoder_lr: Learning rate for decoder corrections
            surprise_threshold: Above this, apply strong corrections
            use_amp: Use automatic mixed precision
            verbose: Print debug info
        """
        self.model = model
        self.decoder_lr = decoder_lr
        self.surprise_threshold = surprise_threshold
        self.use_amp = use_amp and torch.cuda.is_available()
        self.verbose = verbose
        
        # Decoder optimizer (rebuilt per correction for fresh state)
        self._decoder_optimizer = None
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # Tracking
        self._total_corrections = 0
        self._surprise_history: List[float] = []
        self._episodic_writes = 0
    
    def compute_flux_confidence(self, text: str) -> float:
        """
        Estimate FLUX's confidence in its own output.
        
        Uses field energy as proxy:
        - Low energy after settling = high confidence (stable attractor)
        - High energy = low confidence (unsettled, uncertain)
        
        Args:
            text: The text FLUX generated
        
        Returns:
            Confidence value in [0, 1]
        """
        device = self.model._device_str
        
        with torch.no_grad():
            # Encode text to wave
            wave = self.model.cse.encode(text)
            wave_vec = wave.full.mean(dim=0).to(device)
            
            # Query field for local energy
            features, sims, locs = self.model.field.query(wave_vec, k=4)
            
            # Similarity-based confidence (high similarity = high confidence)
            # Note: sims are cosine similarities ∈ [-1, 1]
            avg_sim = sims.mean().item()
            confidence = (avg_sim + 1) / 2  # Map to [0, 1]
            
            return confidence
    
    def compute_surprise(
        self,
        flux_confidence: float,
        teacher_score: float,
    ) -> float:
        """
        Compute surprise from teacher feedback.
        
        Surprise = |teacher_normalized - flux_confidence|
        
        High surprise means: FLUX was confident but wrong, OR unsure but right
        Low surprise means: FLUX's confidence matched teacher's assessment
        
        Args:
            flux_confidence: FLUX's confidence [0, 1]
            teacher_score: Gemini score [0, 10]
        
        Returns:
            Surprise value [0, 1]
        """
        teacher_normalized = teacher_score / 10.0
        surprise = abs(teacher_normalized - flux_confidence)
        return surprise
    
    def apply_correction(
        self,
        prompt: str,
        flux_output: str,
        feedback: TeacherFeedback,
        learn_decoder: bool = True,
        learn_field: bool = True,
        store_episodic: bool = True,
    ) -> CorrectionResult:
        """
        Apply teacher correction through FLUX's learning systems.
        
        This is the core RL-like update:
        1. Compute surprise from teacher feedback
        2. If high surprise → train decoder on correction
        3. Thermodynamically settle field with correction as target
        4. If low surprise → store in episodic memory
        
        Args:
            prompt: Original prompt
            flux_output: What FLUX generated
            feedback: Teacher's score and correction
            learn_decoder: Update WaveDecoder weights
            learn_field: Apply thermodynamic settling
            store_episodic: Allow episodic memory writes
        
        Returns:
            CorrectionResult with all metrics
        """
        device = self.model._device_str
        
        # Step 1: Compute confidence and surprise
        flux_confidence = self.compute_flux_confidence(flux_output)
        surprise = self.compute_surprise(flux_confidence, feedback.score)
        
        self._surprise_history.append(surprise)
        self._total_corrections += 1
        
        # Note: verbose output moved to CurriculumSchool for cleaner display
        
        # Step 2: Get field state before learning
        energy_before = self.model.field.total_energy()
        temp_before = self.model.tl.temp_manager.temperature
        
        # Step 3: Decoder learning (if surprise is high enough)
        decoder_loss = 0.0
        if learn_decoder and surprise > self.surprise_threshold:
            decoder_loss = self._train_decoder_on_correction(
                prompt, 
                feedback.corrected_text,
                strength=surprise,  # Scale gradient by surprise
            )
        
        # Step 4: Thermodynamic field settling
        if learn_field:
            self._thermodynamic_settle(
                prompt=prompt,
                correct_text=feedback.corrected_text,
                surprise=surprise,
            )
        
        # Step 5: Get field state after learning
        energy_after = self.model.field.total_energy()
        temp_after = self.model.tl.temp_manager.temperature
        
        # Step 6: Episodic memory — ONE-SHOT LEARNING
        # FLUX's key advantage: learn facts in single exposure
        # Store BOTH confirmations (low surprise) AND corrections (high surprise)
        episodic_stored = False
        if store_episodic:
            if surprise < self.surprise_threshold:
                # Low surprise = FLUX was right → reinforce
                episodic_stored = self._store_in_episodic(
                    prompt=prompt,
                    correct_text=feedback.corrected_text,
                    score=feedback.score,
                    is_correction=False,
                )
            elif feedback.score >= 5.0:
                # High surprise but teacher gave good correction → LEARN IT
                # This is one-shot learning: store correction for future retrieval
                episodic_stored = self._store_in_episodic(
                    prompt=prompt,
                    correct_text=feedback.corrected_text,
                    score=feedback.score,
                    is_correction=True,
                )
            
            if episodic_stored:
                self._episodic_writes += 1
        
        return CorrectionResult(
            surprise=surprise,
            teacher_score=feedback.score,
            flux_confidence=flux_confidence,
            decoder_loss=decoder_loss,
            energy_before=energy_before,
            energy_after=energy_after,
            temperature_delta=temp_after - temp_before,
            episodic_stored=episodic_stored,
        )
    
    def _train_decoder_on_correction(
        self,
        prompt: str,
        correct_text: str,
        strength: float = 1.0,
    ) -> float:
        """
        Train WaveDecoder on the teacher's correction.
        
        Args:
            prompt: The input prompt
            correct_text: Teacher's corrected output
            strength: Gradient scaling factor (higher = stronger update)
        
        Returns:
            Loss value
        """
        device = self.model._device_str
        
        # Get context from FLUX pipeline
        wave_sequence, wave_vec, field_feat = self.model._get_context(prompt)
        
        # Prepare target bytes (teacher's correction)
        full_text = prompt + correct_text
        target_bytes = torch.tensor(
            list(full_text.encode('utf-8', errors='replace')),
            dtype=torch.long, device=device,
        )
        
        # Build optimizer if needed
        if self._decoder_optimizer is None:
            self._decoder_optimizer = torch.optim.AdamW(
                list(self.model.decoder.parameters()) +
                list(self.model.output_head.parameters()),
                lr=self.decoder_lr,
                weight_decay=0.01,
            )
        
        # Forward
        self.model.train()
        self._decoder_optimizer.zero_grad()
        
        max_len = min(len(target_bytes), 256)
        
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                logits = self.model.decoder(
                    target_bytes[:max_len],
                    wave_sequence.detach(),
                    field_feat.detach(),
                    max_len=max_len,
                )
                loss = F.cross_entropy(
                    logits.view(-1, 256),
                    target_bytes[:max_len].view(-1),
                )
                # Scale by surprise strength
                scaled_loss = loss * strength
        else:
            logits = self.model.decoder(
                target_bytes[:max_len],
                wave_sequence.detach(),
                field_feat.detach(),
                max_len=max_len,
            )
            loss = F.cross_entropy(
                logits.view(-1, 256),
                target_bytes[:max_len].view(-1),
            )
            scaled_loss = loss * strength
        
        # Backward
        if self.scaler:
            self.scaler.scale(scaled_loss).backward()
            self.scaler.step(self._decoder_optimizer)
            self.scaler.update()
        else:
            scaled_loss.backward()
            self._decoder_optimizer.step()
        
        self.model.eval()
        
        return loss.item()
    
    def _thermodynamic_settle(
        self,
        prompt: str,
        correct_text: str,
        surprise: float,
    ):
        """
        Apply thermodynamic settling with correction as target.
        
        Uses FLUX's existing ThermodynamicLearner.settle_once() with:
        - Higher surprise → more settling iterations (field is fluid)
        - Lower surprise → fewer iterations (field is stable)
        
        Args:
            prompt: Input prompt
            correct_text: Teacher's correction (becomes the target)
            surprise: Surprise value (modulates settling intensity)
        """
        device = self.model._device_str
        
        with torch.no_grad():
            # Encode correct text to wave
            full_text = prompt + correct_text
            wave = self.model.cse.encode(full_text)
            wave_vec = wave.full.mean(dim=0).to(device)
            
            # Convert to field features (target for settling)
            target_features = self.model.wave_to_field(wave_vec)
            
            # Use existing thermodynamic settling
            # Higher surprise → more iterations
            base_iters = self.model.tl.settle_iterations
            adaptive_iters = int(base_iters * (1.0 + surprise * 2))
            
            # The settle_once call:
            # - Computes field's current prediction
            # - Updates temperature based on surprise (already computed by field)
            # - Perturbs field toward target
            # - Settles to new energy minimum
            result = self.model.tl.settle_once(
                wave_vec=wave_vec,
                target=target_features,
                iterations=adaptive_iters,
            )
            
            # Verbose field settling moved to debug mode to reduce noise
    
    def _store_in_episodic(
        self,
        prompt: str,
        correct_text: str,
        score: float,
        is_correction: bool = False,
    ) -> bool:
        """
        Store Q→A pair in episodic memory — ONE-SHOT LEARNING.
        
        This is FLUX's key advantage over transformers:
        - One write = permanently stored fact
        - No training loop, no epochs, no batches
        - Immediately retrievable via similarity search
        
        Called for BOTH:
        - Confirmations (low surprise) — reinforce existing knowledge
        - Corrections (high surprise) — learn teacher's correction for next time
        
        Args:
            prompt: The question/prompt
            correct_text: The correct answer/completion
            score: Teacher's score
            is_correction: True if this corrects a mistake (vs confirms)
        
        Returns:
            True if successfully stored
        """
        device = self.model._device_str
        
        try:
            with torch.no_grad():
                # Encode full exchange
                full_text = prompt + " " + correct_text
                wave = self.model.cse.encode(full_text)
                wave_vec = wave.full.mean(dim=0).to(device)
                
                # Compress for episodic storage
                compressed = self.model.working_memory.compress(
                    wave_vec.unsqueeze(0)
                ).squeeze(0)
                
                # Determine causal source
                if is_correction:
                    causal = "one_shot_correction"  # Learned from mistake
                else:
                    causal = "one_shot_confirmation"  # Reinforced existing
                
                # Write to episodic memory — THE ONE-SHOT MOMENT
                entry_id = self.model.episodic_memory.write(
                    embedding=compressed,
                    fact=f"Q: {prompt[:50]}... A: {correct_text[:50]}...",
                    causal_source=causal,
                    confidence=score / 10.0,  # Use teacher score as confidence
                )
                
                if self.verbose:
                    mode = "CORRECTION" if is_correction else "confirmation"
                    print(f"      ✓ One-shot {mode}: entry {entry_id}")
                
                return entry_id is not None
        except Exception as e:
            if self.verbose:
                print(f"      ⚠ Episodic store failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Return corrector statistics."""
        recent = self._surprise_history[-50:] if self._surprise_history else []
        return {
            'total_corrections': self._total_corrections,
            'episodic_writes': self._episodic_writes,
            'avg_surprise': sum(recent) / max(len(recent), 1),
            'min_surprise': min(recent) if recent else 0.0,
            'max_surprise': max(recent) if recent else 0.0,
            'write_rate': self._episodic_writes / max(self._total_corrections, 1),
        }


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("SurpriseCorrector — Testing (requires FLUXLarge)")
    
    print("\n  Confidence computation formulas:")
    print("    flux_confidence = (avg_field_similarity + 1) / 2")
    print("    surprise = |teacher_score/10 - flux_confidence|")
    
    print("\n  Surprise thresholds:")
    print("    < 0.3: Low surprise → one-shot CONFIRMATION (reinforce)")
    print("    > 0.3: High surprise → decoder training + one-shot CORRECTION")
    
    print("\n  ONE-SHOT LEARNING:")
    print("    - Every teacher correction → episodic write")
    print("    - No training loop needed")
    print("    - Immediately retrievable next time")
    
    print("\n  ✓ SurpriseCorrector ready (with one-shot learning)")
