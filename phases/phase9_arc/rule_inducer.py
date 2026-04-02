"""
Phase 9 ARC: Rule Inducer

The heart of ARC reasoning. Learn transformation rules from 2-5 examples.

Strategy:
1. Encode all input/output pairs to wave space
2. Compute delta waves (transformation representation)
3. If deltas cluster → single rule
4. If deltas diverge → compositional/contextual rule
5. Use resonance field to settle on consistent rule

The physics advantage:
- Thermodynamic settling finds global minimum (consistent rule)
- Delta waves capture transformation in continuous space
- Field attractors represent learned rules
- Zero forgetting between tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import sys
from pathlib import Path
from collections import Counter

# Add project paths
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PHASES_DIR / 'phase8_8') not in sys.path:
    sys.path.insert(0, str(_PHASES_DIR / 'phase8_8'))

try:
    from phases.phase9_arc.arc_loader import ARCTask, ARCExample
    from phases.phase9_arc.object_detector import ObjectDetector, ObjectGraph, ARCObject
    from phases.phase9_arc.pattern_library import PATTERNS, Pattern, apply_sequence
except ImportError:
    from arc_loader import ARCTask, ARCExample
    from object_detector import ObjectDetector, ObjectGraph, ARCObject
    from pattern_library import PATTERNS, Pattern, apply_sequence

# Try to import grid adapters
try:
    from grid_adapters import GridToWave, WaveToGrid
    HAS_GRID_ADAPTERS = True
except ImportError:
    HAS_GRID_ADAPTERS = False
    print("⚠ GridToWave/WaveToGrid not available, using fallback")


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class RuleHypothesis:
    """Single rule hypothesis."""
    rule_id: str
    confidence: float
    pattern_ids: List[str]         # Sequence of patterns to apply
    params: Dict[str, Any] = None  # Parameters for patterns
    explanation: str = ""
    
    def apply(self, grid: Tensor, **kwargs) -> Tensor:
        """Apply this rule to a grid."""
        result = grid.clone()
        merged_kwargs = {**(self.params or {}), **kwargs}
        for pid in self.pattern_ids:
            if pid in PATTERNS:
                result = PATTERNS[pid](result, **merged_kwargs)
        return result


@dataclass
class TransformationFeatures:
    """Features extracted from input→output transformation."""
    # Size changes
    size_delta: Tuple[int, int]    # (H_out - H_in, W_out - W_in)
    size_same: bool
    
    # Color changes
    colors_in: Set[int]
    colors_out: Set[int]
    colors_added: Set[int]
    colors_removed: Set[int]
    color_count_delta: int
    
    # Object changes
    objects_in: int
    objects_out: int
    objects_delta: int
    
    # Spatial changes
    is_rotation: bool
    is_mirror_h: bool
    is_mirror_v: bool
    is_transpose: bool
    
    # Area changes
    total_area_in: int
    total_area_out: int
    area_ratio: float


# ─────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────

class FeatureExtractor:
    """Extract transformation features from input/output pairs."""
    
    def __init__(self):
        self.detector = ObjectDetector()
    
    def extract(self, input_grid: Tensor, output_grid: Tensor) -> TransformationFeatures:
        """Extract features from single input/output pair."""
        H_in, W_in = input_grid.shape
        H_out, W_out = output_grid.shape
        
        # Colors
        colors_in = set(input_grid.unique().tolist())
        colors_out = set(output_grid.unique().tolist())
        
        # Objects
        graph_in = self.detector.detect(input_grid)
        graph_out = self.detector.detect(output_grid)
        
        # Area
        area_in = (input_grid != 0).sum().item()
        area_out = (output_grid != 0).sum().item()
        
        # Spatial checks (only if same size)
        is_rotation = False
        is_mirror_h = False
        is_mirror_v = False
        is_transpose = False
        
        if (H_in, W_in) == (H_out, W_out):
            try:
                is_mirror_h = torch.all(output_grid == torch.flip(input_grid, dims=[1])).item()
                is_mirror_v = torch.all(output_grid == torch.flip(input_grid, dims=[0])).item()
            except:
                pass
        
        # Rotation check (output dims are swapped)
        if (H_in, W_in) == (W_out, H_out):
            try:
                rotated = torch.rot90(input_grid, k=-1)
                if rotated.shape == output_grid.shape:
                    is_rotation = torch.all(output_grid == rotated).item()
                is_transpose = torch.all(output_grid == input_grid.T).item()
            except:
                pass
        
        return TransformationFeatures(
            size_delta=(H_out - H_in, W_out - W_in),
            size_same=(H_in, W_in) == (H_out, W_out),
            colors_in=colors_in,
            colors_out=colors_out,
            colors_added=colors_out - colors_in,
            colors_removed=colors_in - colors_out,
            color_count_delta=len(colors_out) - len(colors_in),
            objects_in=len(graph_in),
            objects_out=len(graph_out),
            objects_delta=len(graph_out) - len(graph_in),
            is_rotation=is_rotation,
            is_mirror_h=is_mirror_h,
            is_mirror_v=is_mirror_v,
            is_transpose=is_transpose,
            total_area_in=area_in,
            total_area_out=area_out,
            area_ratio=area_out / max(area_in, 1),
        )
    
    def extract_all(self, task: ARCTask) -> List[TransformationFeatures]:
        """Extract features from all examples in task."""
        features = []
        for ex in task.train:
            input_t = torch.tensor(ex.input, dtype=torch.long)
            output_t = torch.tensor(ex.output, dtype=torch.long)
            features.append(self.extract(input_t, output_t))
        return features


# ─────────────────────────────────────────────
# Rule Matching
# ─────────────────────────────────────────────

class PatternMatcher:
    """Match observed transformations to known patterns."""
    
    def __init__(self):
        self.extractor = FeatureExtractor()
    
    def match_single_example(
        self,
        input_grid: Tensor,
        output_grid: Tensor,
    ) -> List[Tuple[str, float]]:
        """
        Try all single patterns, return those that match.
        Returns list of (pattern_id, accuracy).
        """
        matches = []
        
        for pid, pattern in PATTERNS.items():
            try:
                result = pattern(input_grid)
                
                # Handle size mismatch
                if result.shape != output_grid.shape:
                    continue
                
                # Compute accuracy
                correct = (result == output_grid).sum().item()
                total = output_grid.numel()
                accuracy = correct / total
                
                if accuracy > 0.9:  # Near-perfect match
                    matches.append((pid, accuracy))
                    
            except Exception:
                continue
        
        return sorted(matches, key=lambda x: -x[1])
    
    def match_task(self, task: ARCTask) -> List[RuleHypothesis]:
        """
        Find patterns that work for ALL training examples.
        """
        # Get matches for each example
        all_matches = []
        for ex in task.train:
            input_t = torch.tensor(ex.input, dtype=torch.long)
            output_t = torch.tensor(ex.output, dtype=torch.long)
            matches = self.match_single_example(input_t, output_t)
            all_matches.append(set(pid for pid, _ in matches))
        
        if not all_matches:
            return []
        
        # Find patterns that match ALL examples
        common = all_matches[0]
        for matches in all_matches[1:]:
            common = common & matches
        
        # Create hypotheses
        hypotheses = []
        for pid in common:
            hypotheses.append(RuleHypothesis(
                rule_id=f"single_{pid}",
                confidence=1.0,
                pattern_ids=[pid],
                explanation=f"Apply {PATTERNS[pid].name}",
            ))
        
        return hypotheses


# ─────────────────────────────────────────────
# Heuristic Rule Induction
# ─────────────────────────────────────────────

class HeuristicInducer:
    """Induce rules using heuristics (fast, no training required)."""
    
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.matcher = PatternMatcher()
    
    def induce(self, task: ARCTask) -> List[RuleHypothesis]:
        """
        Induce rule from task using heuristics.
        
        Strategy:
        1. Try single-pattern matching
        2. Try feature-based pattern selection
        3. Try 2-pattern compositions
        """
        hypotheses = []
        
        # 1. Single pattern matching
        single_matches = self.matcher.match_task(task)
        hypotheses.extend(single_matches)
        
        if hypotheses:
            return hypotheses
        
        # 2. Feature-based selection
        features = self.extractor.extract_all(task)
        feature_based = self._induce_from_features(features, task)
        hypotheses.extend(feature_based)
        
        if hypotheses:
            return hypotheses
        
        # 3. Try 2-pattern compositions
        compositions = self._try_compositions(task, max_depth=2)
        hypotheses.extend(compositions)
        
        return hypotheses
    
    def _induce_from_features(
        self,
        features: List[TransformationFeatures],
        task: ARCTask,
    ) -> List[RuleHypothesis]:
        """Induce rules based on extracted features."""
        hypotheses = []
        
        # Check for consistent features across all examples
        if len(features) == 0:
            return hypotheses
        
        # Check symmetry operations
        if all(f.is_mirror_h for f in features):
            hypotheses.append(RuleHypothesis(
                rule_id="heuristic_mirror_h",
                confidence=0.95,
                pattern_ids=["G4"],
                explanation="Mirror horizontally",
            ))
        
        if all(f.is_mirror_v for f in features):
            hypotheses.append(RuleHypothesis(
                rule_id="heuristic_mirror_v",
                confidence=0.95,
                pattern_ids=["G5"],
                explanation="Mirror vertically",
            ))
        
        if all(f.is_rotation for f in features):
            hypotheses.append(RuleHypothesis(
                rule_id="heuristic_rotate_90",
                confidence=0.95,
                pattern_ids=["G1"],
                explanation="Rotate 90° clockwise",
            ))
        
        if all(f.is_transpose for f in features):
            hypotheses.append(RuleHypothesis(
                rule_id="heuristic_transpose",
                confidence=0.95,
                pattern_ids=["G6"],
                explanation="Transpose (mirror diagonal)",
            ))
        
        # Check color operations
        colors_removed = [f.colors_removed for f in features]
        colors_added = [f.colors_added for f in features]
        
        # If same colors consistently swapped
        if all(len(r) == 1 and len(a) == 1 for r, a in zip(colors_removed, colors_added)):
            removed = [list(r)[0] for r in colors_removed]
            added = [list(a)[0] for a in colors_added]
            
            if len(set(removed)) == 1 and len(set(added)) == 1:
                hypotheses.append(RuleHypothesis(
                    rule_id="heuristic_color_swap",
                    confidence=0.9,
                    pattern_ids=["C1"],
                    params={"color1": removed[0], "color2": added[0]},
                    explanation=f"Swap color {removed[0]} with {added[0]}",
                ))
        
        # Check for size changes (scaling)
        if all(f.size_delta == (features[0].size_delta[0], features[0].size_delta[1]) for f in features):
            dh, dw = features[0].size_delta
            if dh > 0 and dw > 0:
                # Check if it's 2x or 3x scaling
                ex = task.train[0]
                h_in, w_in = len(ex.input), len(ex.input[0])
                h_out, w_out = len(ex.output), len(ex.output[0])
                
                if h_out == 2 * h_in and w_out == 2 * w_in:
                    hypotheses.append(RuleHypothesis(
                        rule_id="heuristic_scale_2x",
                        confidence=0.85,
                        pattern_ids=["G8"],
                        explanation="Scale up 2x",
                    ))
                elif h_out == 3 * h_in and w_out == 3 * w_in:
                    hypotheses.append(RuleHypothesis(
                        rule_id="heuristic_scale_3x",
                        confidence=0.85,
                        pattern_ids=["G9"],
                        explanation="Scale up 3x",
                    ))
        
        return hypotheses
    
    def _try_compositions(
        self,
        task: ARCTask,
        max_depth: int = 2,
    ) -> List[RuleHypothesis]:
        """Try composing 2+ patterns."""
        hypotheses = []
        
        # Priority patterns for composition
        priority_patterns = ["G1", "G2", "G3", "G4", "G5", "G6", "C1", "C3"]
        
        for p1 in priority_patterns:
            for p2 in priority_patterns:
                if p1 == p2:
                    continue
                
                # Check if p1 → p2 works for all examples
                works = True
                for ex in task.train:
                    input_t = torch.tensor(ex.input, dtype=torch.long)
                    output_t = torch.tensor(ex.output, dtype=torch.long)
                    
                    try:
                        result = apply_sequence(input_t, [p1, p2])
                        if result.shape != output_t.shape:
                            works = False
                            break
                        if not torch.all(result == output_t):
                            works = False
                            break
                    except:
                        works = False
                        break
                
                if works:
                    hypotheses.append(RuleHypothesis(
                        rule_id=f"compose_{p1}_{p2}",
                        confidence=0.8,
                        pattern_ids=[p1, p2],
                        explanation=f"Apply {PATTERNS[p1].name} then {PATTERNS[p2].name}",
                    ))
        
        return hypotheses


# ─────────────────────────────────────────────
# Wave-Based Rule Induction
# ─────────────────────────────────────────────

class WaveRuleInducer(nn.Module):
    """
    Learn rules using FLUX wave space.
    
    Key insight: Delta waves (output_wave - input_wave) represent
    the transformation in continuous space. If deltas cluster,
    there's a consistent rule.
    """
    
    def __init__(
        self,
        wave_dim: int = 432,
        device: str = 'cpu',
    ):
        super().__init__()
        self.wave_dim = wave_dim
        self.device = device
        
        # Initialize grid adapters if available
        if HAS_GRID_ADAPTERS:
            self.grid_encoder = GridToWave(wave_dim=wave_dim, device=device)
            self.grid_decoder = WaveToGrid(wave_dim=wave_dim, device=device)
        else:
            self.grid_encoder = None
            self.grid_decoder = None
        
        # Rule embedding (learnable)
        self.rule_embeddings = nn.Parameter(torch.randn(100, wave_dim) * 0.1)
        
        # Delta → rule classifier
        self.delta_classifier = nn.Sequential(
            nn.Linear(wave_dim, 256),
            nn.ReLU(),
            nn.Linear(256, len(PATTERNS)),
        )
        
        self.to(device)
    
    def encode_example(
        self,
        input_grid: Tensor,
        output_grid: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Encode example to wave space.
        
        Returns:
            (input_wave, output_wave, delta_wave)
        """
        if self.grid_encoder is None:
            # Fallback: flatten and project
            input_flat = input_grid.flatten().float()
            output_flat = output_grid.flatten().float()
            
            # Pad/truncate to fixed size
            max_len = 900  # 30x30
            if len(input_flat) > max_len:
                input_flat = input_flat[:max_len]
            else:
                input_flat = F.pad(input_flat, (0, max_len - len(input_flat)))
            
            if len(output_flat) > max_len:
                output_flat = output_flat[:max_len]
            else:
                output_flat = F.pad(output_flat, (0, max_len - len(output_flat)))
            
            # Simple projection (not ideal, but works for testing)
            input_wave = input_flat[:self.wave_dim]
            output_wave = output_flat[:self.wave_dim]
            
            if len(input_wave) < self.wave_dim:
                input_wave = F.pad(input_wave, (0, self.wave_dim - len(input_wave)))
            if len(output_wave) < self.wave_dim:
                output_wave = F.pad(output_wave, (0, self.wave_dim - len(output_wave)))
            
            delta_wave = output_wave - input_wave
            return input_wave, output_wave, delta_wave
        
        # Use proper grid encoder
        input_wave = self.grid_encoder.encode(input_grid, mode='holistic')
        output_wave = self.grid_encoder.encode(output_grid, mode='holistic')
        delta_wave = output_wave - input_wave
        
        return input_wave, output_wave, delta_wave
    
    def compute_delta_consistency(
        self,
        task: ARCTask,
    ) -> Tuple[Tensor, float]:
        """
        Compute average delta and consistency score.
        
        High consistency (close to 1.0) = single consistent rule
        Low consistency = compositional/contextual rule
        """
        deltas = []
        
        for ex in task.train:
            input_t = torch.tensor(ex.input, dtype=torch.long, device=self.device)
            output_t = torch.tensor(ex.output, dtype=torch.long, device=self.device)
            _, _, delta = self.encode_example(input_t, output_t)
            deltas.append(delta)
        
        if not deltas:
            return torch.zeros(self.wave_dim, device=self.device), 0.0
        
        # Stack and compute statistics
        deltas_t = torch.stack(deltas)
        avg_delta = deltas_t.mean(dim=0)
        
        # Consistency = 1 - average distance from mean
        distances = torch.norm(deltas_t - avg_delta, dim=1)
        avg_distance = distances.mean().item()
        
        # Normalize to [0, 1] range
        max_expected_distance = torch.norm(avg_delta).item() + 1e-6
        consistency = 1.0 - min(avg_distance / max_expected_distance, 1.0)
        
        return avg_delta, consistency
    
    def induce(self, task: ARCTask) -> List[RuleHypothesis]:
        """
        Induce rules using wave analysis + heuristics.
        """
        hypotheses = []
        
        # First try heuristics (fast)
        heuristic = HeuristicInducer()
        hypotheses = heuristic.induce(task)
        
        if hypotheses:
            return hypotheses
        
        # Compute delta consistency
        avg_delta, consistency = self.compute_delta_consistency(task)
        
        if consistency > 0.8:
            # High consistency → likely single rule
            # Try to classify the delta
            with torch.no_grad():
                logits = self.delta_classifier(avg_delta)
                probs = F.softmax(logits, dim=-1)
                top_idx = probs.argmax().item()
                top_conf = probs[top_idx].item()
            
            if top_conf > 0.5:
                pattern_ids = list(PATTERNS.keys())
                if top_idx < len(pattern_ids):
                    pid = pattern_ids[top_idx]
                    hypotheses.append(RuleHypothesis(
                        rule_id=f"wave_{pid}",
                        confidence=top_conf * consistency,
                        pattern_ids=[pid],
                        explanation=f"Wave-classified: {PATTERNS[pid].name}",
                    ))
        
        # If still no hypotheses, return identity with low confidence
        if not hypotheses:
            hypotheses.append(RuleHypothesis(
                rule_id="fallback_identity",
                confidence=0.1,
                pattern_ids=[],
                explanation="No rule identified (identity fallback)",
            ))
        
        return hypotheses


# ─────────────────────────────────────────────
# Main Inducer Interface
# ─────────────────────────────────────────────

class RuleInducer:
    """
    Main interface for rule induction.
    Combines heuristics and wave-based methods.
    """
    
    def __init__(
        self,
        use_waves: bool = True,
        device: str = 'cpu',
    ):
        self.heuristic = HeuristicInducer()
        self.wave_inducer = WaveRuleInducer(device=device) if use_waves else None
    
    def induce(
        self,
        task: ARCTask,
        max_hypotheses: int = 5,
    ) -> List[RuleHypothesis]:
        """
        Induce transformation rules from task.
        
        Returns ranked list of hypotheses (best first).
        """
        # Try heuristics first (fast)
        hypotheses = self.heuristic.induce(task)
        
        # Add wave-based if no good heuristic match
        if self.wave_inducer and len(hypotheses) == 0:
            wave_hypotheses = self.wave_inducer.induce(task)
            hypotheses.extend(wave_hypotheses)
        
        # Verify hypotheses on training examples
        verified = []
        for h in hypotheses:
            accuracy = self._verify_hypothesis(h, task)
            h.confidence *= accuracy
            if accuracy > 0.0:
                verified.append(h)
        
        # Sort by confidence
        verified.sort(key=lambda h: -h.confidence)
        
        return verified[:max_hypotheses]
    
    def _verify_hypothesis(
        self,
        hypothesis: RuleHypothesis,
        task: ARCTask,
    ) -> float:
        """
        Verify hypothesis on all training examples.
        Returns accuracy (0.0 to 1.0).
        """
        if not task.train:
            return 0.0
        
        total_correct = 0
        total_cells = 0
        
        for ex in task.train:
            input_t = torch.tensor(ex.input, dtype=torch.long)
            output_t = torch.tensor(ex.output, dtype=torch.long)
            
            try:
                predicted = hypothesis.apply(input_t)
                
                if predicted.shape != output_t.shape:
                    continue
                
                correct = (predicted == output_t).sum().item()
                total_correct += correct
                total_cells += output_t.numel()
                
            except Exception:
                continue
        
        return total_correct / max(total_cells, 1)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Rule Inducer Test")
    print("=" * 50)
    
    from arc_loader import ARCExample, ARCTask, generate_synthetic_tasks
    
    # Generate test tasks
    synthetic = generate_synthetic_tasks(10)
    
    inducer = RuleInducer(use_waves=False)  # No waves for speed
    
    for i, task in enumerate(synthetic[:5]):
        print(f"\n{'='*50}")
        print(f"Task: {task.task_id}")
        
        # Show first example
        ex = task.train[0]
        print(f"Input shape: {len(ex.input)}x{len(ex.input[0]) if ex.input else 0}")
        print(f"Output shape: {len(ex.output)}x{len(ex.output[0]) if ex.output else 0}")
        
        # Induce rules
        hypotheses = inducer.induce(task)
        
        print(f"Hypotheses ({len(hypotheses)}):")
        for h in hypotheses[:3]:
            print(f"  {h.rule_id}: {h.explanation} (conf={h.confidence:.2f})")
        
        # Test on test example
        if hypotheses and task.test:
            best = hypotheses[0]
            test_ex = task.test[0]
            test_input = torch.tensor(test_ex.input, dtype=torch.long)
            test_output = torch.tensor(test_ex.output, dtype=torch.long)
            
            try:
                predicted = best.apply(test_input)
                if predicted.shape == test_output.shape:
                    accuracy = (predicted == test_output).float().mean().item()
                    print(f"  Test accuracy: {accuracy*100:.1f}%")
            except:
                print("  Test failed")
