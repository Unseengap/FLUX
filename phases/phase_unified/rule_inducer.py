"""
RuleInducer — Pattern → Rule Abstraction for FLUX

Finds recurring patterns in causal links and abstracts them into
reusable rules. These rules can then be used for prediction and
planning without needing to re-observe every situation.

Physics Analogy:
    Like discovering physical laws from observations. Many observations
    of "step on blue triggers yellow" become the rule "blue activates yellow".
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict
import numpy as np

try:
    from .causal_tracker import CausalTracker, CausalLink, GridChange, EffectPattern
except ImportError:
    from causal_tracker import CausalTracker, CausalLink, GridChange, EffectPattern


# ─────────────────────────────────────────────
# Rule Data Structures
# ─────────────────────────────────────────────

@dataclass
class Rule:
    """
    An abstracted rule learned from observations.
    
    Example rule:
        "When action=toggle on color=2 (red), then color=2→3 at same position"
        
    Abstraction levels:
        - SPECIFIC: exact position (2,3) + action + color
        - POSITIONAL: relative positions + action + color
        - COLOR: action + trigger_color → effect_color
        - UNIVERSAL: action → effect type (regardless of colors)
    """
    rule_id: int
    trigger_action: int
    trigger_color: int
    effect_type: str  # 'local', 'adjacent', 'remote', 'global'
    effect_delta: Tuple[int, int]  # Relative position of effect
    effect_old_color: int
    effect_new_color: int
    
    # Statistics
    observation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    confidence: float = 0.0
    
    # Abstraction level
    abstraction: str = 'positional'  # 'specific', 'positional', 'color', 'universal'
    
    # Conditions (optional constraints)
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    def matches(
        self,
        action: int,
        trigger_color: int,
        effect_position: Optional[Tuple[int, int]] = None,
        trigger_position: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """Check if this rule matches the given trigger."""
        if action != self.trigger_action:
            return False
        if trigger_color != self.trigger_color:
            return False
        
        if effect_position is not None and trigger_position is not None:
            expected_delta = (
                effect_position[0] - trigger_position[0],
                effect_position[1] - trigger_position[1],
            )
            if expected_delta != self.effect_delta:
                return False
        
        return True
    
    def predict_effect(
        self,
        trigger_position: Tuple[int, int],
        grid: np.ndarray,
    ) -> Optional[GridChange]:
        """Predict the effect of applying this rule."""
        effect_pos = (
            trigger_position[0] + self.effect_delta[0],
            trigger_position[1] + self.effect_delta[1],
        )
        
        # Check bounds
        if not (0 <= effect_pos[0] < grid.shape[0] and 
                0 <= effect_pos[1] < grid.shape[1]):
            return None
        
        # Check precondition
        if grid[effect_pos[0], effect_pos[1]] != self.effect_old_color:
            return None
        
        return GridChange(
            position=effect_pos,
            old_value=self.effect_old_color,
            new_value=self.effect_new_color,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rule_id': self.rule_id,
            'trigger_action': self.trigger_action,
            'trigger_color': self.trigger_color,
            'effect_type': self.effect_type,
            'effect_delta': self.effect_delta,
            'effect_old_color': self.effect_old_color,
            'effect_new_color': self.effect_new_color,
            'observation_count': self.observation_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'confidence': self.confidence,
            'abstraction': self.abstraction,
            'conditions': self.conditions,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Rule':
        return cls(**d)
    
    def __str__(self) -> str:
        # Action mapping
        ACTIONS = {
            0: 'up', 1: 'down', 2: 'left', 3: 'right',
            4: 'pickup', 5: 'drop', 6: 'toggle', 7: 'done'
        }
        action_name = ACTIONS.get(self.trigger_action, f'action_{self.trigger_action}')
        return (
            f"Rule#{self.rule_id}: "
            f"color={self.trigger_color} + {action_name} → "
            f"{self.effect_old_color}→{self.effect_new_color} at Δ{self.effect_delta} "
            f"(conf={self.confidence:.2f})"
        )


@dataclass
class HypothesisTest:
    """Record of a rule hypothesis test."""
    rule_id: int
    predicted: GridChange
    actual: Optional[GridChange]
    success: bool
    timestamp: int


# ─────────────────────────────────────────────
# RuleInducer
# ─────────────────────────────────────────────

class RuleInducer(nn.Module):
    """
    Finds patterns in causal links and abstracts them to rules.
    
    Key capabilities:
    - Analyze causal tracker to find recurring patterns
    - Abstract patterns into general rules
    - Test rules by predicting and verifying
    - Track rule reliability over time
    - Prune unreliable rules
    """
    
    def __init__(
        self,
        causal_tracker: Optional[CausalTracker] = None,
        min_observations: int = 3,
        min_confidence: float = 0.5,
        device: str = 'cpu',
    ):
        super().__init__()
        self.causal_tracker = causal_tracker
        self.min_observations = min_observations
        self.min_confidence = min_confidence
        self._device = device
        
        # Rule storage
        self.rules: List[Rule] = []
        self.next_rule_id = 0
        
        # Index for fast lookup
        self.rules_by_trigger: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        
        # Hypothesis testing history
        self.hypothesis_tests: List[HypothesisTest] = []
        
        # Analysis state
        self.last_analysis_step = 0
        self.analysis_interval = 10  # Analyze every N steps
    
    @property
    def device(self) -> str:
        return self._device
    
    def set_causal_tracker(self, tracker: CausalTracker):
        """Set or update the causal tracker reference."""
        self.causal_tracker = tracker
    
    # ─────────────────────────────────────────────
    # Pattern Analysis
    # ─────────────────────────────────────────────
    
    def analyze_causal_links(self, force: bool = False) -> List[Rule]:
        """
        Analyze causal tracker to find and extract rules.
        
        Args:
            force: Analyze even if interval hasn't elapsed
            
        Returns:
            List of newly induced rules
        """
        if self.causal_tracker is None:
            return []
        
        # Check if it's time to analyze
        current_step = self.causal_tracker.step_count
        if not force and current_step - self.last_analysis_step < self.analysis_interval:
            return []
        
        self.last_analysis_step = current_step
        
        # Get confident patterns from causal tracker
        patterns = self.causal_tracker.get_confident_patterns(
            min_confidence=self.min_confidence,
            min_count=self.min_observations,
        )
        
        new_rules = []
        for pattern in patterns:
            # Check if we already have this rule
            existing = self._find_matching_rule(pattern)
            if existing is not None:
                # Update existing rule
                existing.observation_count = pattern.count
                existing.confidence = pattern.confidence
            else:
                # Create new rule
                rule = self.induce_rule(pattern)
                new_rules.append(rule)
        
        return new_rules
    
    def induce_rule(self, pattern: EffectPattern) -> Rule:
        """
        Abstract a pattern into a general rule.
        
        Args:
            pattern: EffectPattern from causal tracker
            
        Returns:
            New Rule object
        """
        rule = Rule(
            rule_id=self.next_rule_id,
            trigger_action=pattern.trigger_action,
            trigger_color=pattern.trigger_color,
            effect_type=pattern.effect_type,
            effect_delta=pattern.effect_delta,
            effect_old_color=pattern.old_color,
            effect_new_color=pattern.new_color,
            observation_count=pattern.count,
            confidence=pattern.confidence,
            abstraction='positional',
        )
        
        self.next_rule_id += 1
        self.rules.append(rule)
        
        # Index by trigger
        trigger_key = (pattern.trigger_action, pattern.trigger_color)
        self.rules_by_trigger[trigger_key].append(len(self.rules) - 1)
        
        return rule
    
    def _find_matching_rule(self, pattern: EffectPattern) -> Optional[Rule]:
        """Find existing rule that matches this pattern."""
        for rule in self.rules:
            if (rule.trigger_action == pattern.trigger_action and
                rule.trigger_color == pattern.trigger_color and
                rule.effect_delta == pattern.effect_delta and
                rule.effect_old_color == pattern.old_color and
                rule.effect_new_color == pattern.new_color):
                return rule
        return None
    
    # ─────────────────────────────────────────────
    # Rule Application
    # ─────────────────────────────────────────────
    
    def get_applicable_rules(
        self,
        action: int,
        trigger_color: int,
        min_confidence: float = 0.0,
    ) -> List[Rule]:
        """
        Get rules that apply to this action-color combination.
        
        Args:
            action: Action to query
            trigger_color: Color at trigger position
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of applicable rules, sorted by confidence
        """
        trigger_key = (action, trigger_color)
        rule_indices = self.rules_by_trigger.get(trigger_key, [])
        
        applicable = [
            self.rules[idx] for idx in rule_indices
            if self.rules[idx].confidence >= min_confidence
        ]
        
        return sorted(applicable, key=lambda r: r.confidence, reverse=True)
    
    def predict_effects(
        self,
        position: Tuple[int, int],
        action: int,
        grid: np.ndarray,
        min_confidence: float = 0.5,
    ) -> List[Tuple[Rule, GridChange]]:
        """
        Predict effects using learned rules.
        
        Args:
            position: Agent position
            action: Action to take
            grid: Current grid state
            min_confidence: Minimum rule confidence
            
        Returns:
            List of (rule, predicted_change) pairs
        """
        if isinstance(grid, Tensor):
            grid = grid.cpu().numpy()
        
        trigger_color = int(grid[position[0], position[1]])
        
        rules = self.get_applicable_rules(action, trigger_color, min_confidence)
        
        predictions = []
        for rule in rules:
            pred = rule.predict_effect(position, grid)
            if pred is not None:
                predictions.append((rule, pred))
        
        return predictions
    
    # ─────────────────────────────────────────────
    # Hypothesis Testing
    # ─────────────────────────────────────────────
    
    def test_rule(
        self,
        rule: Rule,
        trigger_position: Tuple[int, int],
        grid_before: np.ndarray,
        grid_after: np.ndarray,
    ) -> bool:
        """
        Test a rule by comparing prediction to actual outcome.
        
        Args:
            rule: Rule to test
            trigger_position: Where action was taken
            grid_before: Grid before action
            grid_after: Grid after action
            
        Returns:
            True if prediction was correct
        """
        # Get prediction
        prediction = rule.predict_effect(trigger_position, grid_before)
        
        if prediction is None:
            # Rule doesn't apply here
            return False
        
        # Check actual outcome
        actual_value = grid_after[prediction.position[0], prediction.position[1]]
        success = (actual_value == prediction.new_value)
        
        # Update rule stats
        if success:
            rule.success_count += 1
        else:
            rule.failure_count += 1
        
        # Recalculate confidence
        total = rule.success_count + rule.failure_count
        if total > 0:
            rule.confidence = rule.success_count / total
        
        # Record test
        actual_change = None
        if grid_before[prediction.position[0], prediction.position[1]] != actual_value:
            actual_change = GridChange(
                position=prediction.position,
                old_value=int(grid_before[prediction.position[0], prediction.position[1]]),
                new_value=int(actual_value),
            )
        
        self.hypothesis_tests.append(HypothesisTest(
            rule_id=rule.rule_id,
            predicted=prediction,
            actual=actual_change,
            success=success,
            timestamp=self.causal_tracker.step_count if self.causal_tracker else 0,
        ))
        
        return success
    
    def test_all_rules(
        self,
        trigger_position: Tuple[int, int],
        action: int,
        grid_before: np.ndarray,
        grid_after: np.ndarray,
    ) -> Dict[int, bool]:
        """
        Test all applicable rules against an observed outcome.
        
        Args:
            trigger_position: Where action was taken
            action: Action that was taken
            grid_before: Grid before action
            grid_after: Grid after action
            
        Returns:
            Dict mapping rule_id → success
        """
        if isinstance(grid_before, Tensor):
            grid_before = grid_before.cpu().numpy()
        if isinstance(grid_after, Tensor):
            grid_after = grid_after.cpu().numpy()
        
        trigger_color = int(grid_before[trigger_position[0], trigger_position[1]])
        rules = self.get_applicable_rules(action, trigger_color)
        
        results = {}
        for rule in rules:
            results[rule.rule_id] = self.test_rule(
                rule, trigger_position, grid_before, grid_after
            )
        
        return results
    
    # ─────────────────────────────────────────────
    # Rule Management
    # ─────────────────────────────────────────────
    
    def prune_rules(
        self,
        min_confidence: float = 0.3,
        min_tests: int = 5,
    ) -> List[Rule]:
        """
        Remove unreliable rules.
        
        Args:
            min_confidence: Minimum confidence to keep
            min_tests: Minimum test count before pruning
            
        Returns:
            List of pruned rules
        """
        pruned = []
        
        for rule in self.rules[:]:  # Copy to allow modification
            total = rule.success_count + rule.failure_count
            if total >= min_tests and rule.confidence < min_confidence:
                self.rules.remove(rule)
                pruned.append(rule)
        
        # Rebuild index
        self.rules_by_trigger.clear()
        for idx, rule in enumerate(self.rules):
            trigger_key = (rule.trigger_action, rule.trigger_color)
            self.rules_by_trigger[trigger_key].append(idx)
        
        return pruned
    
    def merge_similar_rules(self) -> int:
        """
        Merge rules that are essentially the same.
        
        Returns:
            Number of rules merged
        """
        merged_count = 0
        
        # Group by trigger
        groups = defaultdict(list)
        for rule in self.rules:
            key = (rule.trigger_action, rule.trigger_color, 
                   rule.effect_delta, rule.effect_old_color, rule.effect_new_color)
            groups[key].append(rule)
        
        # Merge groups with multiple rules
        new_rules = []
        for key, group in groups.items():
            if len(group) > 1:
                # Merge into one rule
                merged = group[0]
                for other in group[1:]:
                    merged.observation_count += other.observation_count
                    merged.success_count += other.success_count
                    merged.failure_count += other.failure_count
                    merged_count += 1
                
                # Recalculate confidence
                total = merged.success_count + merged.failure_count
                if total > 0:
                    merged.confidence = merged.success_count / total
                
                new_rules.append(merged)
            else:
                new_rules.append(group[0])
        
        self.rules = new_rules
        
        # Rebuild index
        self.rules_by_trigger.clear()
        for idx, rule in enumerate(self.rules):
            trigger_key = (rule.trigger_action, rule.trigger_color)
            self.rules_by_trigger[trigger_key].append(idx)
        
        return merged_count
    
    # ─────────────────────────────────────────────
    # State Management
    # ─────────────────────────────────────────────
    
    def reset(self):
        """Clear all rules and state."""
        self.rules.clear()
        self.rules_by_trigger.clear()
        self.hypothesis_tests.clear()
        self.next_rule_id = 0
        self.last_analysis_step = 0
    
    def state_dict(self) -> Dict[str, Any]:
        """Export state for checkpointing."""
        return {
            'rules': [rule.to_dict() for rule in self.rules],
            'next_rule_id': self.next_rule_id,
            'last_analysis_step': self.last_analysis_step,
            'min_observations': self.min_observations,
            'min_confidence': self.min_confidence,
        }
    
    def load_state_dict(self, state: Dict[str, Any]):
        """Load state from checkpoint."""
        self.reset()
        
        self.min_observations = state.get('min_observations', 3)
        self.min_confidence = state.get('min_confidence', 0.5)
        self.next_rule_id = state.get('next_rule_id', 0)
        self.last_analysis_step = state.get('last_analysis_step', 0)
        
        for rule_dict in state.get('rules', []):
            rule = Rule.from_dict(rule_dict)
            self.rules.append(rule)
            
            trigger_key = (rule.trigger_action, rule.trigger_color)
            self.rules_by_trigger[trigger_key].append(len(self.rules) - 1)
    
    def summary(self) -> str:
        """Get a summary of induced rules."""
        lines = [
            "RuleInducer Summary",
            "=" * 40,
            f"Total rules: {len(self.rules)}",
            f"Hypothesis tests: {len(self.hypothesis_tests)}",
            "",
        ]
        
        # Group by confidence
        high_conf = [r for r in self.rules if r.confidence >= 0.8]
        med_conf = [r for r in self.rules if 0.5 <= r.confidence < 0.8]
        low_conf = [r for r in self.rules if r.confidence < 0.5]
        
        lines.append(f"High confidence (≥0.8): {len(high_conf)}")
        lines.append(f"Medium confidence (0.5-0.8): {len(med_conf)}")
        lines.append(f"Low confidence (<0.5): {len(low_conf)}")
        
        if self.rules:
            lines.append("")
            lines.append("Top rules:")
            for rule in sorted(self.rules, key=lambda r: r.confidence, reverse=True)[:5]:
                lines.append(f"  {rule}")
        
        return "\n".join(lines)
    
    def forward(self, x: Tensor) -> Tensor:
        """Dummy forward for nn.Module compatibility."""
        return x


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import numpy as np
    
    print("Testing RuleInducer")
    print("=" * 40)
    
    # Create tracker and inducer
    tracker = CausalTracker()
    inducer = RuleInducer(tracker, min_observations=2, min_confidence=0.5)
    
    # Simulate multiple observations of the same pattern
    grid_before = np.array([
        [0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0],  # Color 2 at (1,1)
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    
    grid_after = np.array([
        [0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0],  # Changed to 3
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    
    # Record same pattern multiple times
    for _ in range(3):
        tracker.record((1, 1), action=6, grid_before=grid_before, grid_after=grid_after)
    
    print(f"✓ Recorded {tracker.step_count} causal links")
    
    # Analyze and induce rules
    new_rules = inducer.analyze_causal_links(force=True)
    print(f"✓ Induced {len(new_rules)} new rules")
    
    for rule in new_rules:
        print(f"  {rule}")
    
    # Test prediction
    predictions = inducer.predict_effects(
        position=(1, 1),
        action=6,
        grid=grid_before,
    )
    print(f"\n✓ Predictions for toggle at (1,1):")
    for rule, pred in predictions:
        print(f"  Rule#{rule.rule_id}: {pred.position} → {pred.new_value}")
    
    # Test hypothesis testing
    result = inducer.test_rule(
        new_rules[0],
        trigger_position=(1, 1),
        grid_before=grid_before,
        grid_after=grid_after,
    )
    print(f"\n✓ Hypothesis test: {'passed' if result else 'failed'}")
    
    # Print summary
    print()
    print(inducer.summary())
    
    # Test state dict
    state = inducer.state_dict()
    inducer2 = RuleInducer()
    inducer2.load_state_dict(state)
    print(f"\n✓ State dict round-trip: {len(inducer2.rules)} rules restored")
    
    print("\n✓ All tests passed!")
