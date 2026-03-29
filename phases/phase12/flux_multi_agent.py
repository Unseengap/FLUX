"""
flux_multi_agent.py — Phase 12 Multi-Modal Agent

The complete FLUX agent that combines:
- Spatial Memory (visual mapping via Ice/Water fields)
- LLM Reasoning (Qwen-Omni — unified vision+audio+text)
- Causal Tracking (action → effect learning)
- Game Memory (per-game episodic storage)

This is the final assembly of all FLUX components.

Uses flux_model.py and flux_utils.py for proper .flx handling.
Loads from Flux-Base.flx, saves to Flux-multi-model.flx with FULL state.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
import sys
import numpy as np

# Add project root
FLUX_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(FLUX_ROOT))

# Use the OFFICIAL flux utilities
from flux_utils import (
    PhaseLogger,
    PhaseResults,
    get_device,
    CHECKPOINT_DIR,
    HF_REPO_ID,
)

# Local imports
from visual_reasoner import VisualReasoner
from grid_renderer import GridRenderer, ASCIIRenderer, normalize_grid
from action_parser import ActionParser, ACTION_MAP
from game_memory import GameMemory, GameMemoryManager
from qwen_omni_bridge import QwenOmniBridge, QwenOmniConfig, create_qwen_bridge


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class MultiAgentConfig:
    """Configuration for FLUXMultiAgent."""
    
    # Checkpoint settings
    flx_path: str = "checkpoints/Flux-Base.flx"
    output_path: str = "checkpoints/Flux-multi-model.flx"
    
    # LLM settings
    llm_name: str = "Qwen/Qwen2.5-Omni-7B"
    load_in_4bit: bool = True
    enable_vision: bool = True
    enable_audio: bool = False
    
    # Spatial memory settings
    spatial_max_size: int = 64
    spatial_feature_dim: int = 64
    
    # Agent settings
    max_actions_per_game: int = 200
    verbose: bool = True
    
    # Learning settings
    enable_causal_learning: bool = True
    enable_rule_induction: bool = True


# ─────────────────────────────────────────────
# FLUX Multi-Modal Agent
# ─────────────────────────────────────────────

class FLUXMultiAgent(nn.Module):
    """
    The complete FLUX agent that combines:
    - Spatial Memory (visual mapping)
    - LLM Reasoning (Qwen-Omni — unified vision+audio+text)
    - Causal Tracking (action → effect learning)
    - Game Memory (per-game episodic storage)
    
    This is the final assembly of all FLUX components.
    
    Uses Qwen2.5-Omni instead of separate models:
    - ONE model for vision + audio + text
    - Can directly "see" grid images (no ASCII conversion needed)
    - Saves ~4GB VRAM vs loading separate CLIP/Whisper
    """
    
    def __init__(
        self,
        config: Optional[MultiAgentConfig] = None,
        device: Optional[str] = None,
        log: Optional[PhaseLogger] = None,
    ):
        """
        Initialize the FLUX Multi-Modal Agent.
        
        Args:
            config: Agent configuration
            device: Device to use (auto-detect if None)
            log: PhaseLogger instance
        """
        super().__init__()
        
        self.config = config or MultiAgentConfig()
        self._device = device or get_device()
        self.log = log or PhaseLogger(phase=12)
        
        self.log.separator("Initializing FLUXMultiAgent")
        
        # ═══════════════════════════════════════════════════════════
        # Step 1: Initialize FLUX Model (from flux_model.py)
        # ═══════════════════════════════════════════════════════════
        self.flux_model = None
        self._try_load_flux_base()
        
        # ═══════════════════════════════════════════════════════════
        # Step 2: Initialize LLM with Qwen-Omni
        # ═══════════════════════════════════════════════════════════
        self.log.info("Initializing Qwen-Omni LLM bridge...")
        
        qwen_config = QwenOmniConfig(
            model_name=self.config.llm_name,
            load_in_4bit=self.config.load_in_4bit,
            enable_vision=self.config.enable_vision,
            enable_audio=self.config.enable_audio,
        )
        
        self.llm_bridge = QwenOmniBridge(config=qwen_config, device=self._device)
        llm_info = self.llm_bridge.get_info()
        self.log.success(f"LLM ready: {llm_info['model_name']} (vision={llm_info['vision']})")
        
        # ═══════════════════════════════════════════════════════════
        # Step 3: Initialize Spatial Memory
        # ═══════════════════════════════════════════════════════════
        self.spatial_memory = None
        self._init_spatial_memory()
        
        # ═══════════════════════════════════════════════════════════
        # Step 4: Initialize Visual Reasoner
        # ═══════════════════════════════════════════════════════════
        self.reasoner = VisualReasoner(
            llm_bridge=self.llm_bridge,
            spatial_memory=self.spatial_memory,
            prefer_vision=self.config.enable_vision,
            verbose=self.config.verbose,
        )
        self.log.success("VisualReasoner ready")
        
        # ═══════════════════════════════════════════════════════════
        # Step 5: Initialize Cognitive Components
        # ═══════════════════════════════════════════════════════════
        self.causal_tracker = None
        self.rule_inducer = None
        self._init_cognitive_components()
        
        # ═══════════════════════════════════════════════════════════
        # Step 6: Game State
        # ═══════════════════════════════════════════════════════════
        self.current_game = None
        self.action_history: List[Dict] = []
        self.observation_count = 0
        self.learned_rules: List[Dict] = []
        
        # Game memory manager
        self.game_memory_manager = GameMemoryManager(
            save_dir=CHECKPOINT_DIR / 'game_memories'
        )
        self.current_game_memory: Optional[GameMemory] = None
        
        # Component flags
        self._components = {
            'spatial_memory': self.spatial_memory is not None,
            'visual_reasoner': True,
            'causal_tracker': self.causal_tracker is not None,
            'rule_inducer': self.rule_inducer is not None,
            'llm_bridge': True,
        }
        
        self.log.separator("FLUXMultiAgent Initialized")
        active = sum(1 for v in self._components.values() if v)
        self.log.info(f"Active components: {active}/{len(self._components)}")
    
    def _try_load_flux_base(self):
        """Try to load Flux-Base.flx if available."""
        try:
            from flux_model import FLUXModel
            
            flx_path = Path(self.config.flx_path)
            if not flx_path.is_absolute():
                flx_path = FLUX_ROOT / flx_path
            
            if flx_path.exists():
                self.log.info(f"Loading base from {flx_path}...")
                self.flux_model = FLUXModel.load(str(flx_path))
                self.log.success(f"Loaded: version={self.flux_model.version}")
            else:
                self.log.warning(f"Flux-Base.flx not found at {flx_path}")
                
        except ImportError:
            self.log.warning("flux_model.py not available")
        except Exception as e:
            self.log.warning(f"Could not load Flux-Base.flx: {e}")
    
    def _init_spatial_memory(self):
        """Initialize spatial memory system."""
        try:
            # Try to import from phase9_arc
            sys.path.insert(0, str(FLUX_ROOT / 'phases' / 'phase9_arc'))
            from spatial_memory import SpatialMemory
            
            self.spatial_memory = SpatialMemory(
                max_size=self.config.spatial_max_size,
                feature_dim=self.config.spatial_feature_dim,
                device=self._device,
            )
            self.log.success("SpatialMemory (Ice/Water) ready")
            
        except ImportError:
            self.log.warning("SpatialMemory not available (phase9_arc missing)")
            self._create_minimal_spatial_memory()
    
    def _create_minimal_spatial_memory(self):
        """Create minimal spatial memory if full version not available."""
        
        class MinimalSpatialMemory(nn.Module):
            """Minimal spatial memory for basic tracking."""
            
            def __init__(self, max_size: int = 64, device: str = 'cpu'):
                super().__init__()
                self.max_size = max_size
                self.feature_dim = 64
                self._device = device
                
                self.register_buffer('exploration_mass', torch.zeros(max_size, max_size))
                self.register_buffer('curiosity_field', torch.zeros(max_size, max_size))
                self.register_buffer('visit_count', torch.zeros(max_size, max_size))
                self.register_buffer('last_observation', torch.zeros(max_size, max_size, 64))
            
            def reset(self):
                self.exploration_mass.zero_()
                self.curiosity_field.zero_()
                self.visit_count.zero_()
                self.last_observation.zero_()
            
            def observe(self, position, local_view, global_grid=None):
                r, c = position
                if 0 <= r < self.max_size and 0 <= c < self.max_size:
                    self.visit_count[r, c] += 1
                    self.exploration_mass[r, c] += 1.0
                return {'position': position, 'change_detected': False}
            
            def get_navigation_target(self, position, grid):
                return None
        
        self.spatial_memory = MinimalSpatialMemory(
            max_size=self.config.spatial_max_size,
            device=self._device,
        )
        self.log.info("Using minimal spatial memory")
    
    def _init_cognitive_components(self):
        """Initialize causal tracker and rule inducer."""
        try:
            sys.path.insert(0, str(FLUX_ROOT / 'phases' / 'phase_unified'))
            
            from causal_tracker import CausalTracker
            self.causal_tracker = CausalTracker(max_history=1000, device=self._device)
            self.log.success("CausalTracker ready")
            
            from rule_inducer import RuleInducer
            self.rule_inducer = RuleInducer(
                causal_tracker=self.causal_tracker,
                device=self._device,
            )
            self.log.success("RuleInducer ready")
            
        except ImportError as e:
            self.log.warning(f"Cognitive components not available: {e}")
    
    # ─────────────────────────────────────────────
    # Game Session API
    # ─────────────────────────────────────────────
    
    def reset(self, game_id: str):
        """Reset for new game."""
        self.current_game = game_id
        self.action_history = []
        self.observation_count = 0
        
        # Reset spatial memory
        if self.spatial_memory:
            self.spatial_memory.reset()
        
        # Get or create game memory
        self.current_game_memory = self.game_memory_manager.get_memory(game_id)
        self.current_game_memory.new_session()
        
        # Update reasoner
        if hasattr(self.reasoner, 'set_spatial_memory'):
            self.reasoner.set_spatial_memory(self.spatial_memory)
        
        # Teach LLM about game type
        self._teach_game_context(game_id)
        
        self.log.info(f"Reset for game: {game_id}")
    
    def _teach_game_context(self, game_id: str):
        """Teach LLM context about the game type."""
        game_contexts = {
            'ls20': (
                "ls20 is a maze puzzle where you navigate through a grid. "
                "Stepping on certain cells can rotate arrows or trigger effects. "
                "The goal is often to make arrows point in a specific direction."
            ),
            'ft09': (
                "ft09 is a logic puzzle where you click cells to transform patterns. "
                "Look for repeating patterns and click to apply transformations."
            ),
            'vc33': (
                "vc33 is an orchestration puzzle. Click cells in sequence "
                "to achieve a target pattern."
            ),
        }
        
        # Store context in game memory
        if self.current_game_memory and game_id in game_contexts:
            self.current_game_memory.goals_pending.append(game_contexts[game_id])
    
    # ─────────────────────────────────────────────
    # Core Game Loop Methods
    # ─────────────────────────────────────────────
    
    def observe(
        self,
        frame: Union[List[List[int]], np.ndarray, Tensor],
        position: Tuple[int, int],
    ) -> Dict[str, Any]:
        """
        Process observation from game.
        
        Updates spatial memory and records in episodic memory.
        
        Args:
            frame: Current game grid
            position: Agent position (row, col)
            
        Returns:
            Dict with observation results
        """
        self.observation_count += 1
        
        # Normalize frame
        grid = normalize_grid(frame)
        
        # Update spatial memory
        result = {'position': position, 'step': self.observation_count}
        
        if self.spatial_memory:
            local_view = grid.tolist() if isinstance(grid, np.ndarray) else grid
            obs_result = self.spatial_memory.observe(
                position=position,
                local_view=local_view,
                global_grid=local_view,
            )
            result['change_detected'] = obs_result.get('change_detected', False)
        
        return result
    
    def decide_action(
        self,
        frame: Union[List[List[int]], np.ndarray, Tensor],
        position: Tuple[int, int],
        available_actions: List[int],
    ) -> Tuple[int, str]:
        """
        Use LLM reasoning to decide action.
        
        Args:
            frame: Current game grid
            position: Agent position (row, col)
            available_actions: List of available action IDs
            
        Returns:
            (action_id, reasoning_text)
        """
        # Get recent history for context
        recent_history = self.action_history[-5:] if self.action_history else None
        
        # Use visual reasoner
        action, reasoning = self.reasoner.reason_about_game(
            game_id=self.current_game or "unknown",
            grid=frame,
            position=position,
            available_actions=available_actions,
            history=recent_history,
        )
        
        # Record action
        action_name = ACTION_MAP.get(action, f'ACTION{action}')
        self.action_history.append({
            'step': self.observation_count,
            'action': action,
            'action_name': action_name,
            'position': position,
            'reasoning': reasoning[:200],
        })
        
        # Record in game memory
        if self.current_game_memory:
            self.current_game_memory.record_action(
                position=position,
                action_id=action,
                action_name=action_name,
                reasoning=reasoning,
                confidence=0.8,  # Could extract from parser
            )
        
        return action, reasoning
    
    def record_effect(
        self,
        old_grid: Union[List[List[int]], np.ndarray],
        new_grid: Union[List[List[int]], np.ndarray],
        action: int,
        position: Tuple[int, int],
    ) -> List[Dict]:
        """
        Record action → effect for causal learning.
        
        Args:
            old_grid: Grid before action
            new_grid: Grid after action
            action: Action taken
            position: Position when action was taken
            
        Returns:
            List of detected changes
        """
        changes = []
        
        # Normalize grids
        old_np = normalize_grid(old_grid)
        new_np = normalize_grid(new_grid)
        
        # Detect changes
        diff = old_np != new_np
        changed_positions = np.argwhere(diff)
        
        for pos in changed_positions:
            r, c = int(pos[0]), int(pos[1])
            changes.append({
                'position': (r, c),
                'old': int(old_np[r, c]),
                'new': int(new_np[r, c]),
            })
        
        # Record in causal tracker
        if self.causal_tracker and self.config.enable_causal_learning:
            self.causal_tracker.record(
                position=position,
                action=action,
                grid_before=old_np,
                grid_after=new_np,
            )
        
        # Try rule induction
        if self.rule_inducer and self.config.enable_rule_induction:
            try:
                new_rules = self.rule_inducer.analyze_causal_links()
            except Exception:
                new_rules = []
            
            if new_rules:
                for rule in new_rules:
                    rule_dict = {
                        'trigger_action': rule.trigger_action,
                        'trigger_color': rule.trigger_color,
                        'effect_description': str(rule),
                        'confidence': rule.confidence,
                    }
                    self.learned_rules.append(rule_dict)
                    
                    # Record in game memory
                    if self.current_game_memory:
                        self.current_game_memory.record_rule(
                            trigger_action=rule.trigger_action,
                            trigger_color=rule.trigger_color,
                            effect_description=str(rule),
                            confidence=rule.confidence,
                            observations=rule.observation_count,
                        )
        
        # Record in game memory
        if self.current_game_memory and changes:
            self.current_game_memory.record_effect(
                action_id=action,
                position=position,
                changes=changes[:10],
                summary=f"{len(changes)} cells changed",
            )
        
        return changes
    
    def end_game(
        self,
        final_score: Optional[float] = None,
        final_state: str = 'FINISHED',
    ):
        """End current game and save memories."""
        if self.current_game_memory:
            key_discoveries = []
            if self.learned_rules:
                key_discoveries = [r['effect_description'][:50] for r in self.learned_rules[-3:]]
            
            self.current_game_memory.end_session(
                final_score=final_score,
                final_state=final_state,
                key_discoveries=key_discoveries,
            )
        
        # Save all memories
        self.game_memory_manager.save_all()
        
        self.log.info(f"Game ended: {final_state}, score={final_score}")
    
    # ─────────────────────────────────────────────
    # Agent Position Detection
    # ─────────────────────────────────────────────
    
    def find_position(self, grid: Union[List[List[int]], np.ndarray]) -> Tuple[int, int]:
        """
        Try to detect agent position in grid.
        
        Looks for distinctive markers that might indicate the player.
        
        Args:
            grid: Game grid
            
        Returns:
            Detected position or grid center
        """
        grid_np = normalize_grid(grid)
        h, w = grid_np.shape
        
        # Strategy 1: Look for unique colors that appear only once
        unique, counts = np.unique(grid_np, return_counts=True)
        single_count = unique[counts == 1]
        
        if len(single_count) > 0:
            # Use first single-count color as potential player
            target = single_count[0]
            pos = np.argwhere(grid_np == target)
            if len(pos) > 0:
                return (int(pos[0][0]), int(pos[0][1]))
        
        # Strategy 2: Return center
        return (h // 2, w // 2)
    
    # ─────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────
    
    def save_flx(
        self,
        path: Optional[str] = None,
        upload_to_hf: bool = False,
    ) -> str:
        """
        Save the COMPLETE multi-modal agent to .flx format.
        
        Args:
            path: Output path (uses config default if None)
            upload_to_hf: Whether to upload to HuggingFace Hub
            
        Returns:
            Path to saved file
        """
        path = path or self.config.output_path
        if not Path(path).is_absolute():
            path = str(FLUX_ROOT / path)
        
        self.log.separator("Saving Flux-multi-model.flx")
        
        # Build state dict
        state = {
            'format': 'FLUX',
            'version': '4.0-multi-modal',
            'phase': 'phase12',
            'timestamp': datetime.now().isoformat(),
            
            # Metadata
            'metadata': {
                'phase': 12,
                'description': 'FLUX Multi-Modal Agent with Visual Reasoning',
                'capabilities': [
                    'spatial_vision',
                    'llm_reasoning',
                    'causal_learning',
                    'game_memory',
                    'qwen_omni_vision' if self.llm_bridge.has_vision else 'text_only',
                ],
            },
            
            # Components
            'components': self._components,
            
            # LLM reference (not weights)
            'llm_reference': {
                'name': self.config.llm_name,
                'load_in_4bit': self.config.load_in_4bit,
                'enable_vision': self.config.enable_vision,
            },
            
            # Learned rules
            'learned_rules': self.learned_rules,
            
            # Game memories (summary)
            'game_memories': {
                'games': self.game_memory_manager.get_all_games(),
                'total_rules': self.game_memory_manager.get_total_rules(),
            },
        }
        
        # Add spatial memory state
        if self.spatial_memory:
            state['spatial_memory'] = {
                'exploration_mass': self.spatial_memory.exploration_mass.cpu(),
                'curiosity_field': self.spatial_memory.curiosity_field.cpu(),
                'visit_count': self.spatial_memory.visit_count.cpu(),
                'config': {
                    'max_size': self.spatial_memory.max_size,
                    'feature_dim': getattr(self.spatial_memory, 'feature_dim', 64),
                },
            }
        
        # Add causal tracker state
        if self.causal_tracker:
            links = []
            for link in self.causal_tracker.causal_links[-100:]:
                links.append(link.to_dict() if hasattr(link, 'to_dict') else {
                    'action': link.trigger_action,
                    'position': link.trigger_position,
                })
            
            state['causal_tracker'] = {
                'links': links,
                'total_observations': len(self.causal_tracker.causal_links),
            }
        
        # Save
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        
        size_mb = Path(path).stat().st_size / 1e6
        self.log.success(f"Saved: {path} ({size_mb:.1f} MB)")
        
        # Upload to HuggingFace
        if upload_to_hf:
            self._upload_to_hf(path)
        
        return path
    
    def _upload_to_hf(self, path: str):
        """Upload checkpoint to HuggingFace Hub."""
        try:
            # Try to get token
            from flux_utils import _resolve_hf_token
            token = _resolve_hf_token()
            
            if token:
                from huggingface_hub import HfApi
                api = HfApi(token=token)
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo="checkpoints/Flux-multi-model.flx",
                    repo_id=HF_REPO_ID,
                    commit_message=f"Phase 12: Flux-multi-model.flx — {datetime.now().isoformat()}",
                )
                self.log.success("Uploaded to HuggingFace Hub")
            else:
                self.log.warning("No HF_TOKEN — skipping upload")
                
        except Exception as e:
            self.log.warning(f"HF upload failed: {e}")
    
    @classmethod
    def load(
        cls,
        path: str = "checkpoints/Flux-multi-model.flx",
        device: Optional[str] = None,
    ) -> 'FLUXMultiAgent':
        """
        Load a FLUXMultiAgent from saved .flx file.
        
        Args:
            path: Path to .flx file
            device: Device to use
            
        Returns:
            Loaded FLUXMultiAgent
        """
        device = device or get_device()
        log = PhaseLogger(phase=12)
        
        # Resolve path
        if not Path(path).is_absolute():
            path = str(FLUX_ROOT / path)
        
        log.info(f"Loading FLUXMultiAgent from {path}...")
        
        if not Path(path).exists():
            log.warning(f"File not found: {path}, creating new agent")
            return cls(device=device, log=log)
        
        # Load state
        state = torch.load(path, map_location='cpu', weights_only=False)
        
        # Create agent with matching config
        config = MultiAgentConfig()
        if 'llm_reference' in state:
            ref = state['llm_reference']
            config.llm_name = ref.get('name', config.llm_name)
            config.load_in_4bit = ref.get('load_in_4bit', True)
            config.enable_vision = ref.get('enable_vision', True)
        
        agent = cls(config=config, device=device, log=log)
        
        # Restore spatial memory state
        if 'spatial_memory' in state and agent.spatial_memory:
            sm_state = state['spatial_memory']
            if 'exploration_mass' in sm_state:
                agent.spatial_memory.exploration_mass.copy_(
                    sm_state['exploration_mass'].to(device)
                )
            if 'curiosity_field' in sm_state:
                agent.spatial_memory.curiosity_field.copy_(
                    sm_state['curiosity_field'].to(device)
                )
        
        # Restore learned rules
        if 'learned_rules' in state:
            agent.learned_rules = state['learned_rules']
        
        log.success("FLUXMultiAgent loaded")
        return agent
    
    # ─────────────────────────────────────────────
    # Statistics & Info
    # ─────────────────────────────────────────────
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        stats = {
            'version': '4.0-multi-modal',
            'device': self._device,
            'components': self._components,
            'current_game': self.current_game,
            'observations': self.observation_count,
            'actions_taken': len(self.action_history),
            'learned_rules': len(self.learned_rules),
            'games_played': len(self.game_memory_manager.get_all_games()),
            'llm_info': self.llm_bridge.get_info(),
        }
        
        vram = self.llm_bridge.get_vram_usage()
        if vram is not None:
            stats['vram_gb'] = round(vram, 2)
        
        return stats
    
    def summary(self):
        """Print agent summary."""
        stats = self.get_stats()
        
        print("\n" + "═" * 50)
        print("  FLUX Multi-Modal Agent (Phase 12)")
        print("═" * 50)
        print(f"  Device: {stats['device']}")
        print(f"  LLM: {stats['llm_info']['model_name']}")
        print(f"  Vision: {'✓' if stats['llm_info']['vision'] else '✗'}")
        print(f"  Components: {sum(1 for v in stats['components'].values() if v)}")
        print(f"  Rules learned: {stats['learned_rules']}")
        print(f"  Games played: {stats['games_played']}")
        if 'vram_gb' in stats:
            print(f"  VRAM: {stats['vram_gb']} GB")
        print("═" * 50 + "\n")
    
    @property
    def device(self) -> str:
        return self._device


# ─────────────────────────────────────────────
# Factory Function
# ─────────────────────────────────────────────

def create_agent(
    enable_vision: bool = True,
    load_in_4bit: bool = True,
    device: Optional[str] = None,
    verbose: bool = True,
) -> FLUXMultiAgent:
    """
    Create a FLUX Multi-Modal Agent with sensible defaults.
    
    Args:
        enable_vision: Enable vision capabilities
        load_in_4bit: Use 4-bit quantization
        device: Device to use
        verbose: Print status messages
        
    Returns:
        Configured FLUXMultiAgent
    """
    config = MultiAgentConfig(
        enable_vision=enable_vision,
        load_in_4bit=load_in_4bit,
        verbose=verbose,
    )
    
    return FLUXMultiAgent(config=config, device=device)


# ─────────────────────────────────────────────
# Module Exports
# ─────────────────────────────────────────────

__all__ = [
    'FLUXMultiAgent',
    'MultiAgentConfig',
    'create_agent',
]
