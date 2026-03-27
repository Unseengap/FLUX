"""
Phase 8.5: Gemini Teacher — External LLM for Scoring & Correction

Uses Google's Gemini API to:
1. Score FLUX text generation quality (0-10)
2. Provide corrected versions of bad outputs
3. Grade subject-specific tests (spelling, grammar, coherence)

The teacher's score creates the SURPRISE signal that drives FLUX learning:
    FLUX confidence vs teacher score mismatch → thermodynamic surprise
    High surprise → field heats up → stronger learning
    Low surprise → field cools → knowledge consolidates

Environment:
    GEMINI_API_KEY: API key (free tier: 60 req/min)
    Or in Kaggle: Add-ons → Secrets → "GEMINI_API_KEY"
"""

import os
import time
import json
import re
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────────────────────────────────
# API Configuration
# ─────────────────────────────────────────────

def _get_api_key() -> Optional[str]:
    """Resolve Gemini API key from environment or Kaggle secrets."""
    # 1. Direct env var
    key = os.environ.get('GEMINI_API_KEY')
    if key:
        return key
    
    # 2. Kaggle secrets
    try:
        from kaggle_secrets import UserSecretsClient
        key = UserSecretsClient().get_secret("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    
    # 3. .env file
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith('GEMINI_API_KEY='):
                return line.split('=', 1)[1].strip().strip('"').strip("'")
    
    return None


# ─────────────────────────────────────────────
# Data Types
# ─────────────────────────────────────────────

@dataclass
class TeacherFeedback:
    """Feedback from the Gemini teacher."""
    score: float                   # 0-10 quality score
    corrected_text: str            # What FLUX should have generated
    feedback: str                  # Brief explanation
    subject_scores: Dict[str, float] = field(default_factory=dict)
    raw_response: str = ""         # Full API response for debugging


@dataclass
class SubjectTest:
    """Test for a specific curriculum subject."""
    subject: str                   # 'spelling', 'grammar', 'coherence', 'knowledge'
    prompt: str                    # What to generate
    expected: str                  # Gold answer (for knowledge tests)
    flux_output: str = ""          # What FLUX generated
    passed: bool = False
    score: float = 0.0
    feedback: str = ""


# ─────────────────────────────────────────────
# Gemini Teacher
# ─────────────────────────────────────────────

class GeminiTeacher:
    """
    External LLM teacher that scores FLUX outputs and provides corrections.
    
    Uses Gemini API (free tier: 60 calls/min) for:
    - Scoring text quality (0-10)
    - Providing corrected versions
    - Grading subject-specific tests
    
    The teacher acts as the "target signal" for thermodynamic learning:
    FLUX generates → teacher scores → surprise computed → field learns
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        rate_limit_delay: float = 1.0,
        verbose: bool = False,
    ):
        """
        Args:
            api_key: Gemini API key (auto-resolved if None)
            model_name: Which Gemini model to use
            rate_limit_delay: Seconds between API calls
            verbose: Print debug info
        """
        self.api_key = api_key or _get_api_key()
        self.model_name = model_name
        self.rate_limit_delay = rate_limit_delay
        self.verbose = verbose
        
        self._model = None
        self._last_call_time = 0.0
        self._call_count = 0
        self._error_count = 0
        
        # Initialize API if key available
        if self.api_key:
            self._init_api()
    
    def _init_api(self):
        """Initialize Gemini API client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            # Try models in order of preference
            models_to_try = [self.model_name, 'gemini-2.0-flash', 'gemini-1.5-flash-latest', 'gemini-pro']
            
            for model in models_to_try:
                try:
                    self._model = genai.GenerativeModel(model)
                    # Quick test to verify model works
                    _ = self._model.generate_content("test", generation_config={'max_output_tokens': 5})
                    self.model_name = model
                    if self.verbose:
                        print(f"  ✓ Gemini teacher initialized ({model})")
                    return
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠ Model {model} failed: {e}")
                    continue
            
            print("  ⚠ No Gemini models available")
            self._model = None
            
        except ImportError:
            print("  ⚠ google-generativeai not installed")
            print("    Run: pip install google-generativeai")
            self._model = None
        except Exception as e:
            print(f"  ⚠ Gemini init failed: {e}")
            self._model = None
    
    @property
    def is_available(self) -> bool:
        """True if API is ready to use."""
        return self._model is not None
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_call_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_call_time = time.time()
    
    def _call_api(self, prompt: str) -> str:
        """Make a rate-limited API call."""
        if not self.is_available:
            return ""
        
        self._rate_limit()
        self._call_count += 1
        
        try:
            response = self._model.generate_content(prompt)
            return response.text
        except Exception as e:
            self._error_count += 1
            if self.verbose:
                print(f"    ⚠ API error: {e}")
            return ""
    
    def score_generation(
        self,
        prompt: str,
        flux_output: str,
        context: str = "",
    ) -> TeacherFeedback:
        """
        Score FLUX's text generation and provide correction.
        
        This is the main interface for teacher-guided learning.
        
        Args:
            prompt: The input prompt given to FLUX
            flux_output: What FLUX generated
            context: Optional additional context
        
        Returns:
            TeacherFeedback with score, correction, and per-subject scores
        """
        if not self.is_available:
            # Fallback: heuristic scoring when API unavailable
            return self._heuristic_score(prompt, flux_output)
        
        api_prompt = f"""You are grading an AI model's text generation.

PROMPT given to the AI: "{prompt}"

AI OUTPUT: "{flux_output}"

Score this output on a scale of 0-10 where:
0 = Complete garbage/random bytes
3 = Some recognizable words but incoherent
5 = Broken sentences, partial grammar
7 = Mostly correct English, minor issues
10 = Perfect fluent English, makes sense

Also provide:
1. A CORRECTED version (what should have been generated)
2. Per-subject scores (0-10): spelling, grammar, coherence, relevance

Respond in JSON:
{{
    "score": <float>,
    "corrected_text": "<string>",
    "feedback": "<brief explanation>",
    "spelling": <float>,
    "grammar": <float>,
    "coherence": <float>,
    "relevance": <float>
}}"""
        
        response = self._call_api(api_prompt)
        
        if not response:
            return self._heuristic_score(prompt, flux_output)
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return TeacherFeedback(
                    score=float(data.get('score', 5.0)),
                    corrected_text=data.get('corrected_text', flux_output),
                    feedback=data.get('feedback', ''),
                    subject_scores={
                        'spelling': float(data.get('spelling', 5.0)),
                        'grammar': float(data.get('grammar', 5.0)),
                        'coherence': float(data.get('coherence', 5.0)),
                        'relevance': float(data.get('relevance', 5.0)),
                    },
                    raw_response=response,
                )
        except Exception as e:
            if self.verbose:
                print(f"    ⚠ Parse error: {e}")
        
        return self._heuristic_score(prompt, flux_output)
    
    def _heuristic_score(self, prompt: str, flux_output: str) -> TeacherFeedback:
        """
        Fallback scoring when API unavailable.
        Uses simple heuristics to estimate quality.
        """
        text = flux_output.strip()
        
        # Basic heuristics
        has_spaces = ' ' in text
        word_like = sum(1 for c in text if c.isalpha() or c == ' ') / max(len(text), 1)
        has_common = any(w in text.lower() for w in ['the', 'is', 'and', 'to', 'of', 'a'])
        avg_word_len = sum(len(w) for w in text.split()) / max(len(text.split()), 1)
        
        # Estimate scores
        spelling_score = word_like * 10
        grammar_score = 5.0 if has_spaces else 2.0
        coherence_score = 6.0 if has_common else 3.0
        relevance_score = 5.0  # Can't judge without API
        
        overall = (spelling_score + grammar_score + coherence_score + relevance_score) / 4
        
        return TeacherFeedback(
            score=overall,
            corrected_text=text,  # No correction without API
            feedback="(Heuristic scoring - API unavailable)",
            subject_scores={
                'spelling': spelling_score,
                'grammar': grammar_score,
                'coherence': coherence_score,
                'relevance': relevance_score,
            },
        )
    
    def grade_spelling(self, word: str, flux_spelling: str) -> Tuple[bool, float, str]:
        """
        Grade FLUX's spelling of a specific word.
        
        Args:
            word: The target word
            flux_spelling: What FLUX generated
        
        Returns:
            (passed, score, correction)
        """
        # Exact match
        if flux_spelling.strip().lower() == word.lower():
            return True, 10.0, word
        
        # Partial match (edit distance)
        def edit_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return edit_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            prev_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                curr_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = prev_row[j + 1] + 1
                    deletions = curr_row[j] + 1
                    substitutions = prev_row[j] + (c1 != c2)
                    curr_row.append(min(insertions, deletions, substitutions))
                prev_row = curr_row
            return prev_row[-1]
        
        dist = edit_distance(flux_spelling.strip().lower(), word.lower())
        max_dist = max(len(word), len(flux_spelling))
        score = max(0, 10 * (1 - dist / max_dist))
        passed = score >= 7.0  # 70% match = pass
        
        return passed, score, word
    
    def grade_grammar(self, sentence: str) -> Tuple[bool, float, str]:
        """
        Grade sentence grammar using Gemini.
        
        Returns:
            (passed, score, corrected_sentence)
        """
        if not self.is_available:
            # Simple heuristic
            has_subject = any(w in sentence.lower().split()[:3] 
                            for w in ['i', 'he', 'she', 'it', 'they', 'we', 'the', 'a'])
            has_period = sentence.strip().endswith(('.', '!', '?'))
            score = 5.0 + (2.5 if has_subject else 0) + (2.5 if has_period else 0)
            return score >= 7.0, score, sentence
        
        api_prompt = f"""Grade this sentence's grammar (0-10):
Sentence: "{sentence}"

Respond only with JSON: {{"score": <float>, "corrected": "<string>"}}"""
        
        response = self._call_api(api_prompt)
        try:
            data = json.loads(re.search(r'\{[^{}]*\}', response).group())
            score = float(data.get('score', 5.0))
            corrected = data.get('corrected', sentence)
            return score >= 7.0, score, corrected
        except Exception:
            return False, 5.0, sentence
    
    def grade_coherence(self, text: str) -> Tuple[bool, float, str]:
        """
        Grade text coherence (does it make sense?).
        
        Returns:
            (passed, score, feedback)
        """
        if not self.is_available:
            # Heuristic: longer coherent-looking text = better
            words = text.split()
            score = min(10.0, len(words) * 0.5 + 2.0)
            return score >= 6.0, score, ""
        
        api_prompt = f"""Rate this text's coherence (does it make logical sense?) from 0-10:
Text: "{text}"

Respond only with JSON: {{"score": <float>, "feedback": "<brief explanation>"}}"""
        
        response = self._call_api(api_prompt)
        try:
            data = json.loads(re.search(r'\{[^{}]*\}', response).group())
            score = float(data.get('score', 5.0))
            feedback = data.get('feedback', '')
            return score >= 6.0, score, feedback
        except Exception:
            return False, 5.0, ""
    
    def grade_knowledge(
        self,
        question: str,
        expected_answer: str,
        flux_answer: str,
    ) -> Tuple[bool, float, str]:
        """
        Grade FLUX's answer to a knowledge question.
        
        Args:
            question: The question asked
            expected_answer: Gold answer
            flux_answer: FLUX's response
        
        Returns:
            (passed, score, feedback)
        """
        # Check for key terms from expected answer
        expected_words = set(expected_answer.lower().split())
        flux_words = set(flux_answer.lower().split())
        
        overlap = len(expected_words & flux_words)
        coverage = overlap / max(len(expected_words), 1)
        
        score = coverage * 10
        passed = coverage >= 0.5  # At least half the expected words
        feedback = f"Matched {overlap}/{len(expected_words)} key terms"
        
        return passed, score, feedback
    
    def get_stats(self) -> Dict[str, Any]:
        """Return teacher usage statistics."""
        return {
            'api_available': self.is_available,
            'model': self.model_name,
            'calls': self._call_count,
            'errors': self._error_count,
            'success_rate': (self._call_count - self._error_count) / max(self._call_count, 1),
        }


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("GeminiTeacher — Testing")
    
    teacher = GeminiTeacher(verbose=True)
    
    if teacher.is_available:
        print("\n  Testing score_generation...")
        feedback = teacher.score_generation(
            prompt="The meaning of life is",
            flux_output="to find happiness and help others",
        )
        print(f"    Score: {feedback.score:.1f}")
        print(f"    Feedback: {feedback.feedback}")
    else:
        print("  ⚠ API not available — testing heuristics")
        feedback = teacher._heuristic_score(
            prompt="Hello",
            flux_output="the quick brown fox jumps over",
        )
        print(f"    Heuristic score: {feedback.score:.1f}")
        print(f"    Subjects: {feedback.subject_scores}")
    
    # Test spelling
    passed, score, _ = teacher.grade_spelling("hello", "helo")
    print(f"\n  Spelling 'hello' as 'helo': {score:.1f} ({'✓' if passed else '✗'})")
    
    passed, score, _ = teacher.grade_spelling("hello", "hello")
    print(f"  Spelling 'hello' as 'hello': {score:.1f} ({'✓' if passed else '✗'})")
    
    print("\n  ✓ GeminiTeacher tests complete")
