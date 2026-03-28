"""
Phase 10: TaskRouter — Intelligent Mode Selection

Routes generation requests to appropriate mode:
- 'wave': Fast, semantic, good for chat/summaries
- 'byte': Precise, character-perfect, good for code/formulas

Decisions based on:
- Task type (chat vs code vs exact)
- Output modality (text vs image vs audio)
- Prompt characteristics (length, technical markers)
- User preference override
"""

import re
import torch
import torch.nn as nn
from typing import Tuple, Optional, List


class TaskRouter(nn.Module):
    """
    Route generation requests to appropriate mode.
    
    The router uses pattern matching + optional learned weights
    to decide between wave-mode and byte-mode generation.
    
    Modes:
    - 'wave': Fast (~6x), semantic, good for chat/summaries
    - 'byte': Precise, character-perfect, good for code/formulas
    """
    
    # Patterns that suggest byte mode (precision needed)
    # NOTE: These are checked AFTER wave patterns, so they should be
    # more concrete indicators of code/technical content
    BYTE_PATTERNS = [
        r'```',                    # Code blocks (strongest signal)
        r'^\s*def |^\s*class |^\s*import ', # Python code at line start
        r'^\s*function |^\s*const |^\s*let |^\s*var ', # JavaScript at line start
        r'\$\$.*\$\$|\$[^$]+\$',   # LaTeX math
        r'http[s]?://\S+',         # URLs (full pattern)
        r'\d+\.\d+\.\d+',          # Version numbers (semver)
        r'\d{4}-\d{2}-\d{2}',      # Dates (ISO format)
        r'["\'][^"\']{50,}["\']',  # Long exact quotes
        r'implement.*function|write.*function|debug.*code', # Code requests (specific)
        r'\.json|\.xml|\.html|\.css|\.py|\.js', # File extensions
        r'\bAPI\b|\bSQL\b|\bJSON\b', # Technical acronyms (uppercase)
        r'def \w+\(|class \w+:|import \w+', # Actual code patterns
    ]
    
    # Patterns that suggest wave mode (semantic understanding)
    # NOTE: These are checked FIRST with higher priority
    WAVE_PATTERNS = [
        r'\bexplain\b|\bdescribe\b|\bsummarize\b',  # Semantic verbs
        r'\bwhat is\b|\bhow does\b|\bwhy\b',       # Questions
        r'\btell me about\b|\bhelp me understand\b',
        r'^chat\b|^talk\b|^discuss\b',
        r'in your.*words|paraphrase',
        r'\bopinion\b|\bthink\b|\bfeel\b',
        r'\bcreative\b|\bstory\b|\bpoem\b',
        r'\boverview\b|\bintroduction\b',
        r'\btldr\b|tl;dr|\bsummary\b',
        r'\bmeaning\b|\bsignificance\b',
        r'\bhistory of\b|\bfuture of\b',
    ]
    
    # Patterns for multimodal routing
    IMAGE_PATTERNS = [
        r'draw|paint|sketch|illustrate',
        r'image of|picture of|photo of',
        r'visualize|render|generate.*image',
        r'look.*like|show.*me',
    ]
    
    def __init__(
        self,
        default_mode: str = 'wave',
        wave_dim: int = 432,
        learnable: bool = False,
    ):
        """
        Initialize TaskRouter.
        
        Args:
            default_mode: Default when no patterns match
            wave_dim: Wave dimension for learned routing
            learnable: Whether to use learned routing in addition to patterns
        """
        super().__init__()
        
        self.default_mode = default_mode
        self.wave_dim = wave_dim
        self.learnable = learnable
        
        # Compile regex patterns
        self._byte_re = re.compile('|'.join(self.BYTE_PATTERNS), re.IGNORECASE)
        self._wave_re = re.compile('|'.join(self.WAVE_PATTERNS), re.IGNORECASE)
        self._image_re = re.compile('|'.join(self.IMAGE_PATTERNS), re.IGNORECASE)
        
        # Optional learned router
        if learnable:
            self.mode_classifier = nn.Sequential(
                nn.Linear(wave_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 2),  # [wave_score, byte_score]
            )
        
        # Stats tracking
        self._route_counts = {'wave': 0, 'byte': 0}
        self._reason_counts = {}
    
    def route(
        self,
        prompt: str,
        output_modality: str = 'text',
        wave_vec: Optional[torch.Tensor] = None,
    ) -> str:
        """
        Decide generation mode for this request.
        
        Args:
            prompt: Input prompt text
            output_modality: Target modality ('text', 'image', 'audio', 'mol')
            wave_vec: Optional wave vector for learned routing
        
        Returns:
            'wave' or 'byte'
        """
        mode, _ = self.route_with_reason(prompt, output_modality, wave_vec)
        return mode
    
    def route_with_reason(
        self,
        prompt: str,
        output_modality: str = 'text',
        wave_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[str, str]:
        """
        Route and explain why.
        
        Args:
            prompt: Input prompt text
            output_modality: Target modality
            wave_vec: Optional wave vector for learned routing
        
        Returns:
            (mode, reason) tuple
        """
        # Non-text output → always wave mode (byte mode is text-only)
        if output_modality != 'text':
            reason = f"Non-text output ({output_modality}) requires wave mode"
            self._track('wave', reason)
            return 'wave', reason
        
        # Check for semantic-understanding patterns FIRST (higher priority)
        wave_match = self._wave_re.search(prompt)
        if wave_match:
            reason = f"Semantic pattern: '{wave_match.group()[:30]}'"
            self._track('wave', reason)
            return 'wave', reason
        
        # Check for precision-requiring patterns (byte mode)
        byte_match = self._byte_re.search(prompt)
        if byte_match:
            reason = f"Precision pattern: '{byte_match.group()[:30]}'"
            self._track('byte', reason)
            return 'byte', reason
        
        # Learned routing (if enabled and wave_vec provided)
        if self.learnable and wave_vec is not None:
            with torch.no_grad():
                scores = self.mode_classifier(wave_vec)
                if scores[0] > scores[1]:
                    reason = f"Learned router (wave: {scores[0]:.2f} > byte: {scores[1]:.2f})"
                    self._track('wave', reason)
                    return 'wave', reason
                else:
                    reason = f"Learned router (byte: {scores[1]:.2f} > wave: {scores[0]:.2f})"
                    self._track('byte', reason)
                    return 'byte', reason
        
        # Length-based default
        # Short prompts → byte (often exact queries)
        # Long prompts → wave (likely discussion)
        if len(prompt) > 150:
            reason = f"Long prompt ({len(prompt)} chars) → wave mode"
            self._track('wave', reason)
            return 'wave', reason
        elif len(prompt) < 30:
            reason = f"Short prompt ({len(prompt)} chars) → byte mode"
            self._track('byte', reason)
            return 'byte', reason
        
        # Default fallback
        reason = f"Default mode: {self.default_mode}"
        self._track(self.default_mode, reason)
        return self.default_mode, reason
    
    def detect_modality(self, prompt: str) -> str:
        """
        Detect intended output modality from prompt.
        
        Args:
            prompt: Input prompt text
        
        Returns:
            Detected modality: 'text', 'image', 'audio', 'mol'
        """
        if self._image_re.search(prompt):
            return 'image'
        
        # Audio patterns
        if re.search(r'speak|say|voice|audio|sound|music', prompt, re.I):
            return 'audio'
        
        # Molecule patterns
        if re.search(r'molecule|compound|chemical|smiles|inchi', prompt, re.I):
            return 'mol'
        
        return 'text'
    
    def _track(self, mode: str, reason: str) -> None:
        """Track routing decisions for analysis."""
        self._route_counts[mode] += 1
        key = reason.split(':')[0].strip() if ':' in reason else reason[:30]
        self._reason_counts[key] = self._reason_counts.get(key, 0) + 1
    
    def get_stats(self) -> dict:
        """Get routing statistics."""
        total = sum(self._route_counts.values())
        return {
            'total_routes': total,
            'wave_ratio': self._route_counts['wave'] / max(1, total),
            'byte_ratio': self._route_counts['byte'] / max(1, total),
            'route_counts': self._route_counts.copy(),
            'reason_counts': self._reason_counts.copy(),
        }
    
    def reset_stats(self) -> None:
        """Reset routing statistics."""
        self._route_counts = {'wave': 0, 'byte': 0}
        self._reason_counts = {}
    
    def forward(
        self,
        wave_vec: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for learned routing.
        
        Args:
            wave_vec: [batch, wave_dim] wave vectors
        
        Returns:
            [batch, 2] mode scores (wave, byte)
        """
        if not self.learnable:
            # Return uniform scores if not learnable
            batch = wave_vec.shape[0] if wave_vec.dim() > 1 else 1
            return torch.ones(batch, 2, device=wave_vec.device) * 0.5
        
        return self.mode_classifier(wave_vec)
    
    def state_dict_router(self) -> dict:
        """Get state dict for saving."""
        state = {
            'default_mode': self.default_mode,
            'wave_dim': self.wave_dim,
            'learnable': self.learnable,
            'route_counts': self._route_counts,
        }
        if self.learnable:
            state['classifier'] = self.mode_classifier.state_dict()
        return state
    
    def load_state_router(self, state: dict) -> None:
        """Load state from dict."""
        self.default_mode = state.get('default_mode', 'wave')
        if 'route_counts' in state:
            self._route_counts = state['route_counts']
        if self.learnable and 'classifier' in state:
            self.mode_classifier.load_state_dict(state['classifier'])


# ─────────────────────────────────────────────
# Convenience functions
# ─────────────────────────────────────────────

def analyze_prompt(prompt: str) -> dict:
    """
    Analyze a prompt without routing.
    
    Returns detailed analysis of patterns detected.
    """
    router = TaskRouter()
    
    byte_matches = router._byte_re.findall(prompt)
    wave_matches = router._wave_re.findall(prompt)
    
    mode, reason = router.route_with_reason(prompt)
    modality = router.detect_modality(prompt)
    
    return {
        'prompt': prompt[:100] + '...' if len(prompt) > 100 else prompt,
        'recommended_mode': mode,
        'reason': reason,
        'modality': modality,
        'byte_patterns_found': byte_matches[:5],
        'wave_patterns_found': wave_matches[:5],
        'prompt_length': len(prompt),
    }


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("TaskRouter (Phase 10) — Testing")
    print("=" * 60)
    
    router = TaskRouter(default_mode='wave')
    
    test_prompts = [
        # Should route to BYTE (precision)
        ("```python\ndef hello():\n    print('Hello')\n```", "byte"),
        ("https://github.com/Unseengap/FLUX", "byte"),
        ("Write a function that sorts a list", "byte"),
        ("GET /api/v2/users HTTP/1.1", "byte"),  # Actual API call
        
        # Should route to WAVE (semantic)
        ("Explain quantum computing to me like I'm five", "wave"),
        ("What is the meaning of life?", "wave"),
        ("Tell me about the history of AI", "wave"),
        ("Summarize this article in your own words", "wave"),
        
        # Edge cases
        ("Hi", "byte"),  # Short → byte default
        ("Can you help me understand why deep learning has become so popular in recent years?", "wave"),
    ]
    
    print("\nRouting Tests:")
    print("-" * 60)
    
    correct = 0
    for prompt, expected in test_prompts:
        mode, reason = router.route_with_reason(prompt)
        status = "✓" if mode == expected else "✗"
        if mode == expected:
            correct += 1
        print(f"\n  {status} '{prompt[:40]}...'")
        print(f"      Expected: {expected}, Got: {mode}")
        print(f"      Reason: {reason}")
    
    print(f"\n{'=' * 60}")
    print(f"Accuracy: {correct}/{len(test_prompts)} ({100*correct/len(test_prompts):.0f}%)")
    
    stats = router.get_stats()
    print(f"\nRouting stats:")
    print(f"  Wave: {stats['route_counts']['wave']} ({100*stats['wave_ratio']:.0f}%)")
    print(f"  Byte: {stats['route_counts']['byte']} ({100*stats['byte_ratio']:.0f}%)")
