"""
GoalPlanner — Proactive goal and planning system.

Goals are NOT always running. They're created when:
1. User explicitly asks for something complex
2. A task requires multiple steps
3. FLUX notices a recurring pattern

Usage:
    planner = GoalPlanner(flx_state, executor)
    
    # Create a goal
    goal = planner.create_goal(
        description="Prepare morning weather briefing",
        steps=[
            Step("Get user's location from memory", "query_memory", {"query": "user location"}),
            Step("Summarize weather", "generate_text", {"prompt": "Summarize weather"}),
        ]
    )
    
    # Check for triggered goals
    triggered = planner.check_triggers({"time": "08:00", "context": "morning"})
    
    # Execute a goal
    result = planner.execute_goal(goal.id)
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class GoalStatus(Enum):
    """Status of a goal."""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class StepStatus(Enum):
    """Status of a step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Step:
    """A single step in a goal."""
    description: str
    tool_name: str
    tool_args: Dict[str, Any]
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'description': self.description,
            'tool_name': self.tool_name,
            'tool_args': self.tool_args,
            'status': self.status.value,
            'result': str(self.result)[:200] if self.result else None,
            'error': self.error,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
        }


@dataclass
class Goal:
    """A goal with multiple steps."""
    id: str
    description: str
    steps: List[Step]
    status: GoalStatus = GoalStatus.PENDING
    priority: float = 0.5  # 0.0 - 1.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    triggers: List[str] = field(default_factory=list)  # Conditions that activate this goal
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'description': self.description,
            'steps': [s.to_dict() for s in self.steps],
            'status': self.status.value,
            'priority': self.priority,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'triggers': self.triggers,
            'context': self.context,
        }
    
    @property
    def progress(self) -> float:
        """Completion progress (0.0 - 1.0)."""
        if not self.steps:
            return 1.0
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        return completed / len(self.steps)


@dataclass
class PlanResult:
    """Result of goal planning or execution."""
    success: bool
    goal_id: str
    status: GoalStatus
    completed_steps: int
    total_steps: int
    results: List[Any]
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'goal_id': self.goal_id,
            'status': self.status.value,
            'completed_steps': self.completed_steps,
            'total_steps': self.total_steps,
            'error': self.error,
        }


class GoalPlanner:
    """
    Proactive goal and planning system.
    
    Manages goals that activate based on triggers or explicit user requests.
    Goals are multi-step plans executed via the tool executor.
    """
    
    def __init__(
        self,
        flx_state: Dict[str, Any],
        executor: 'FluxToolExecutor',
    ):
        """
        Initialize planner.
        
        Args:
            flx_state: Loaded .flx state dict
            executor: Tool executor for running steps
        """
        self.flx = flx_state
        self.executor = executor
        
        # Load persisted goals
        self.goals: Dict[str, Goal] = {}
        self._load_goals()
        
        # Pattern detection for automatic goal creation
        self.patterns: Dict[str, Dict[str, Any]] = flx_state.get('goal_patterns', {})
        
        # Execution history
        self.execution_history: List[Dict[str, Any]] = []
    
    def _load_goals(self):
        """Load goals from flx state."""
        goals_data = self.flx.get('goals', {})
        for goal_id, goal_dict in goals_data.items():
            steps = [
                Step(
                    description=s['description'],
                    tool_name=s['tool_name'],
                    tool_args=s['tool_args'],
                    status=StepStatus(s.get('status', 'pending')),
                    result=s.get('result'),
                    error=s.get('error'),
                )
                for s in goal_dict.get('steps', [])
            ]
            self.goals[goal_id] = Goal(
                id=goal_id,
                description=goal_dict['description'],
                steps=steps,
                status=GoalStatus(goal_dict.get('status', 'pending')),
                priority=goal_dict.get('priority', 0.5),
                created_at=goal_dict.get('created_at', ''),
                triggers=goal_dict.get('triggers', []),
                context=goal_dict.get('context', {}),
            )
    
    def _save_goals(self):
        """Save goals to flx state."""
        self.flx['goals'] = {
            goal_id: goal.to_dict()
            for goal_id, goal in self.goals.items()
        }
    
    def create_goal(
        self,
        description: str,
        steps: List[Step],
        priority: float = 0.5,
        triggers: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Goal:
        """
        Create a new goal.
        
        Args:
            description: What this goal accomplishes
            steps: List of steps to execute
            priority: Priority (0.0 - 1.0)
            triggers: Conditions that activate this goal
            context: Additional context
            
        Returns:
            Created Goal
        """
        goal_id = str(uuid.uuid4())[:8]
        
        goal = Goal(
            id=goal_id,
            description=description,
            steps=steps,
            priority=priority,
            triggers=triggers or [],
            context=context or {},
        )
        
        self.goals[goal_id] = goal
        self._save_goals()
        
        return goal
    
    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID."""
        return self.goals.get(goal_id)
    
    def list_goals(
        self,
        status: Optional[GoalStatus] = None,
    ) -> List[Goal]:
        """
        List goals, optionally filtered by status.
        
        Args:
            status: Filter by status
            
        Returns:
            List of matching goals
        """
        goals = list(self.goals.values())
        if status:
            goals = [g for g in goals if g.status == status]
        return sorted(goals, key=lambda g: g.priority, reverse=True)
    
    def execute_goal(self, goal_id: str) -> PlanResult:
        """
        Execute a goal's steps.
        
        Args:
            goal_id: ID of goal to execute
            
        Returns:
            PlanResult with execution status
        """
        goal = self.goals.get(goal_id)
        if goal is None:
            return PlanResult(
                success=False,
                goal_id=goal_id,
                status=GoalStatus.FAILED,
                completed_steps=0,
                total_steps=0,
                results=[],
                error=f"Goal '{goal_id}' not found"
            )
        
        # Mark as active
        goal.status = GoalStatus.ACTIVE
        goal.started_at = datetime.now().isoformat()
        
        results = []
        completed = 0
        
        for step in goal.steps:
            if step.status == StepStatus.COMPLETED:
                completed += 1
                results.append(step.result)
                continue
            
            if step.status == StepStatus.SKIPPED:
                continue
            
            # Execute step
            step.status = StepStatus.IN_PROGRESS
            step.started_at = datetime.now().isoformat()
            
            try:
                result = self.executor.execute(step.tool_name, step.tool_args)
                
                if result.success:
                    step.status = StepStatus.COMPLETED
                    step.result = result.result
                    completed += 1
                    results.append(result.result)
                else:
                    step.status = StepStatus.FAILED
                    step.error = result.error
                    goal.status = GoalStatus.FAILED
                    break
                    
            except Exception as e:
                step.status = StepStatus.FAILED
                step.error = str(e)
                goal.status = GoalStatus.FAILED
                break
            
            step.completed_at = datetime.now().isoformat()
        
        # Update goal status
        if completed == len(goal.steps):
            goal.status = GoalStatus.COMPLETED
            goal.completed_at = datetime.now().isoformat()
        
        self._save_goals()
        
        # Record execution
        self.execution_history.append({
            'goal_id': goal_id,
            'status': goal.status.value,
            'completed_steps': completed,
            'timestamp': datetime.now().isoformat(),
        })
        
        return PlanResult(
            success=goal.status == GoalStatus.COMPLETED,
            goal_id=goal_id,
            status=goal.status,
            completed_steps=completed,
            total_steps=len(goal.steps),
            results=results,
            error=goal.steps[-1].error if goal.steps and goal.steps[-1].error else None,
        )
    
    def check_triggers(self, context: Dict[str, Any]) -> List[Goal]:
        """
        Check if any goals should be triggered.
        
        Args:
            context: Current context (time, user state, etc.)
            
        Returns:
            List of triggered goals
        """
        triggered = []
        
        for goal in self.goals.values():
            if goal.status != GoalStatus.PENDING:
                continue
            
            for trigger in goal.triggers:
                if self._matches_trigger(trigger, context):
                    triggered.append(goal)
                    break
        
        return sorted(triggered, key=lambda g: g.priority, reverse=True)
    
    def _matches_trigger(self, trigger: str, context: Dict[str, Any]) -> bool:
        """Check if trigger matches context."""
        # Simple trigger matching - format: "key:value" or "key"
        if ':' in trigger:
            key, value = trigger.split(':', 1)
            return str(context.get(key, '')).lower() == value.lower()
        else:
            return trigger in context
    
    def pause_goal(self, goal_id: str) -> bool:
        """Pause a goal."""
        goal = self.goals.get(goal_id)
        if goal and goal.status == GoalStatus.ACTIVE:
            goal.status = GoalStatus.PAUSED
            self._save_goals()
            return True
        return False
    
    def resume_goal(self, goal_id: str) -> bool:
        """Resume a paused goal."""
        goal = self.goals.get(goal_id)
        if goal and goal.status == GoalStatus.PAUSED:
            goal.status = GoalStatus.ACTIVE
            self._save_goals()
            return True
        return False
    
    def cancel_goal(self, goal_id: str) -> bool:
        """Cancel a goal."""
        goal = self.goals.get(goal_id)
        if goal:
            goal.status = GoalStatus.FAILED
            goal.context['cancelled'] = True
            self._save_goals()
            return True
        return False
    
    def delete_goal(self, goal_id: str) -> bool:
        """Delete a goal."""
        if goal_id in self.goals:
            del self.goals[goal_id]
            self._save_goals()
            return True
        return False
    
    def detect_patterns(self, action_history: List[Dict[str, Any]]) -> Optional[Goal]:
        """
        Detect recurring patterns and create automatic goals.
        
        Args:
            action_history: Recent user actions
            
        Returns:
            Auto-generated goal if pattern detected
        """
        # Simple pattern detection: same action N times
        if len(action_history) < 5:
            return None
        
        # Count action types
        action_counts: Dict[str, int] = {}
        for action in action_history[-10:]:
            key = f"{action.get('type', '')}:{action.get('target', '')}"
            action_counts[key] = action_counts.get(key, 0) + 1
        
        # If any action repeated 5+ times, suggest automation
        for action_key, count in action_counts.items():
            if count >= 5:
                action_type, target = action_key.split(':', 1)
                
                # Check if pattern already exists
                pattern_id = f"auto_{action_key}"
                if pattern_id in self.patterns:
                    continue
                
                # Create automatic goal (not activated yet)
                goal = self.create_goal(
                    description=f"Auto-detected: {action_type} on {target}",
                    steps=[
                        Step(
                            description=f"Perform {action_type}",
                            tool_name=action_type,
                            tool_args={'target': target},
                        )
                    ],
                    priority=0.3,
                    triggers=[],
                    context={'auto_detected': True, 'pattern_count': count}
                )
                
                self.patterns[pattern_id] = {
                    'goal_id': goal.id,
                    'action_key': action_key,
                    'count': count,
                }
                self.flx['goal_patterns'] = self.patterns
                
                return goal
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get planner statistics."""
        status_counts = {}
        for goal in self.goals.values():
            status_counts[goal.status.value] = status_counts.get(goal.status.value, 0) + 1
        
        return {
            'total_goals': len(self.goals),
            'by_status': status_counts,
            'patterns_detected': len(self.patterns),
            'executions': len(self.execution_history),
        }
