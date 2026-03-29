# FLUX Unified Cognitive Layer
# Phase: UNIFIED — The Complete FLUX Stack for ARC-AGI-3

# ARC-AGI-3 Interface (actions, states, frames, sessions)
try:
    from .arc_interface import (
        GameAction, GameState, GameFrame, ActionCommand,
        GameScore, LevelScore, ACTION_ID_TO_NAME, ACTION_DELTAS,
        get_action_delta, apply_action_to_position, grid_diff,
        find_agent_position, find_goal_position
    )
    from .arc_session import (
        ARCSession, SessionConfig, OperationMode, Scorecard,
        GameRecording, RecordingEntry, create_session
    )
except ImportError:
    pass  # Allow imports without arc_interface for backward compat

# Cognitive layer components
try:
    from .causal_tracker import CausalTracker, CausalLink, GridChange, EffectPattern
    from .rule_inducer import RuleInducer, Rule, HypothesisTest
    from .goal_planner import GoalPlanner, Goal, GoalType, GoalStatus, GoalTemplate
    from .perception_field import PerceptionField, TrackedObject, Surprise
except ImportError:
    # Fallback for standalone execution
    try:
        from causal_tracker import CausalTracker, CausalLink, GridChange, EffectPattern
        from rule_inducer import RuleInducer, Rule, HypothesisTest
        from goal_planner import GoalPlanner, Goal, GoalType, GoalStatus, GoalTemplate
        from perception_field import PerceptionField, TrackedObject, Surprise
    except ImportError:
        pass  # Allow partial imports

__all__ = [
    # ARC-AGI-3 Interface
    'GameAction',
    'GameState',
    'GameFrame',
    'ActionCommand',
    'GameScore',
    'LevelScore',
    'ACTION_ID_TO_NAME',
    'ACTION_DELTAS',
    'get_action_delta',
    'apply_action_to_position',
    'grid_diff',
    'find_agent_position',
    'find_goal_position',
    
    # Session Management
    'ARCSession',
    'SessionConfig',
    'OperationMode',
    'Scorecard',
    'GameRecording',
    'RecordingEntry',
    'create_session',
    
    # Cognitive Components
    'CausalTracker',
    'CausalLink',
    'GridChange',
    'EffectPattern',
    'RuleInducer',
    'Rule',
    'HypothesisTest',
    'GoalPlanner',
    'Goal',
    'GoalType',
    'GoalStatus',
    'GoalTemplate',
    'PerceptionField',
    'TrackedObject',
    'Surprise',
]
