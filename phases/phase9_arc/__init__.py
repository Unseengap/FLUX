"""
Phase 9 ARC: ARC-AGI Reasoning Components

Specialized components for solving ARC-AGI puzzles using FLUX.

Key components:
- SpatialMemory: Curiosity-driven exploration with ice/water navigation
- PatternLibrary: Library of common ARC patterns
- RuleInducer: Learn transformation rules from examples
- ARCAgent: Main agent for ARC puzzle solving
- ObjectDetector: Detect objects in ARC grids
- ARCLoader: Load ARC tasks from dataset

Usage:
    from phases.phase9_arc import SpatialMemory, ARCAgent, PatternLibrary
    
    spatial = SpatialMemory(grid_size=(64, 64))
    spatial.observe(position, cell_value)
    curiosity = spatial.get_curiosity_map()
    
    agent = ARCAgent(model)
    solution = agent.solve(task)
"""

try:
    from .spatial_memory import (
        SpatialMemory,
        ExplorationTracker,
        CuriosityField,
        update_exploration,
    )
    from .pattern_library import (
        PatternLibrary,
        Pattern,
        match_pattern,
        register_pattern,
    )
    from .rule_inducer import (
        RuleInducer,
        TransformRule,
        induce_rule,
        apply_rule,
    )
    from .arc_agent import (
        ARCAgent,
        solve_arc_task,
        evaluate_solution,
    )
    from .object_detector import (
        ObjectDetector,
        ARCObject,
        detect_objects,
        segment_grid,
    )
    from .arc_loader import (
        ARCLoader,
        ARCTask,
        load_training_tasks,
        load_evaluation_tasks,
    )
except ImportError:
    # Allow partial imports for standalone execution
    pass

__all__ = [
    # Spatial memory
    'SpatialMemory',
    'ExplorationTracker',
    'CuriosityField',
    'update_exploration',
    
    # Pattern library
    'PatternLibrary',
    'Pattern',
    'match_pattern',
    'register_pattern',
    
    # Rule inducer
    'RuleInducer',
    'TransformRule',
    'induce_rule',
    'apply_rule',
    
    # ARC agent
    'ARCAgent',
    'solve_arc_task',
    'evaluate_solution',
    
    # Object detection
    'ObjectDetector',
    'ARCObject',
    'detect_objects',
    'segment_grid',
    
    # Data loading
    'ARCLoader',
    'ARCTask',
    'load_training_tasks',
    'load_evaluation_tasks',
]
