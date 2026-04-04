"""
AutonomousAgent — Main agent that ties all autonomous components together.

This is the top-level agent that:
1. Loads from .flx file
2. Uses native JSON function calling
3. Executes tools via FluxToolExecutor
4. Manages goals via GoalPlanner
5. Ingests documents via DocumentIngester
6. Runs code via CodeSandbox

Usage:
    from phases.phase_autonomous import AutonomousAgent, autonomous_inference
    
    # Load and create agent
    agent = AutonomousAgent.from_flx('Flux-Apex-V1.flx')
    
    # Chat
    response = agent.chat("What do you know about ARC puzzles?")
    
    # Or use convenience function
    response = autonomous_inference(flx_path, "Hello!")
"""

import json
import time
import torch
from torch import Tensor
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .tool_executor import FluxToolExecutor, ToolExecution
from .code_sandbox import CodeSandbox
from .document_ingester import DocumentIngester
from .goal_planner import GoalPlanner, Goal, Step


@dataclass
class AgentResponse:
    """Response from the agent."""
    content: str
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    thinking: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'tool_calls': self.tool_calls,
            'tool_results': self.tool_results,
            'thinking': self.thinking,
            'metadata': self.metadata or {},
        }


class AutonomousAgent:
    """
    Main autonomous agent that coordinates all components.
    
    This agent:
    - Loads entirely from a .flx file
    - Uses the embedded instruct model for reasoning
    - Calls FLUX tools via native JSON function calling
    - Manages state persistently in the .flx
    """
    
    def __init__(
        self,
        flx_state: Dict[str, Any],
        flx_path: Optional[str] = None,
        device: str = 'cpu',
    ):
        """
        Initialize the agent.
        
        Args:
            flx_state: Loaded .flx state dict
            flx_path: Path to .flx file (for saving)
            device: Target device
        """
        self.flx = flx_state
        self.flx_path = flx_path
        self.device = device
        
        # Initialize components
        self.executor = FluxToolExecutor(flx_state, device)
        self.sandbox = CodeSandbox()
        self.ingester = DocumentIngester(flx_state)
        self.planner = GoalPlanner(flx_state, self.executor)
        
        # Conversation history
        self.history: List[Dict[str, Any]] = []
        
        # Session metadata
        self.session_start = datetime.now().isoformat()
        self.turn_count = 0
        
        # Model info
        self.model_version = flx_state.get('version', 'unknown')
    
    @classmethod
    def from_flx(cls, flx_path: str, device: str = 'cpu') -> 'AutonomousAgent':
        """
        Create agent from .flx file.
        
        Args:
            flx_path: Path to .flx file
            device: Target device
            
        Returns:
            AutonomousAgent instance
        """
        flx_state = torch.load(flx_path, map_location='cpu', weights_only=False)
        return cls(flx_state, flx_path, device)
    
    def get_system_prompt(self) -> str:
        """Get system prompt for the agent."""
        return """You are FLUX, an autonomous cognitive system based on physics-inspired architecture.

Your architecture:
- You (Instruct Model) are the BRAIN - you reason and orchestrate
- The Coder Model handles all code generation and execution
- Multiple sandboxes can run in parallel

Your capabilities:
1. **Wave Encoding**: Convert text/images/grids into 432-dimensional semantic waves
2. **Knowledge Retrieval**: Search the resonance field and episodic memory
3. **Causal Reasoning**: Predict effects and trace causal chains
4. **Code Execution**: Delegate to coder model (parallel sandboxes available)
5. **Document Processing**: Ingest and search documents
6. **Goal Planning**: Create and execute multi-step plans

**IMPORTANT**: For ANY coding task, use `delegate_to_coder` instead of `run_code`.
The coder model will generate and execute code while you stay free for reasoning.

Available functions:
- encode_text, encode_grid: Convert input to waves
- query_field, recall_memory, store_memory: Access knowledge
- predict_effect, query_cgn: Causal reasoning
- delegate_to_coder: Delegate a coding task (PREFERRED)
- delegate_parallel: Run multiple coding tasks in parallel
- run_code: Direct code execution (use delegate_to_coder instead)
- ingest_document: Process documents
- create_tool: Create reusable tools

Think step-by-step. Delegate coding tasks. Synthesize results into responses."""
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get available tools in native JSON format."""
        # Load from tools_v2.json
        try:
            from phases.phase_orchestrator.native_json_orchestrator import load_native_tools
            return load_native_tools()
        except ImportError:
            # Fallback to built-in definitions
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "query_memory",
                        "description": "Search episodic memory for facts",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "limit": {"type": "integer", "default": 5}
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "store_memory",
                        "description": "Store a new fact",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "importance": {"type": "number", "default": 0.5}
                            },
                            "required": ["content"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "run_code",
                        "description": "Execute Python code",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string"},
                                "timeout": {"type": "integer", "default": 30}
                            },
                            "required": ["code"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "delegate_to_coder",
                        "description": "Delegate a coding task to the coder model. Use this instead of run_code - the coder will generate and execute code while you stay free for reasoning.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string", "description": "Description of what the code should do"},
                                "context": {"type": "object", "description": "Input data/parameters"}
                            },
                            "required": ["task"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "delegate_parallel",
                        "description": "Run multiple coding tasks in parallel. Use when you have several independent calculations.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "tasks": {"type": "array", "items": {"type": "string"}, "description": "List of task descriptions"},
                                "context": {"type": "object", "description": "Shared context for all tasks"}
                            },
                            "required": ["tasks"]
                        }
                    }
                },
            ]
    
    def process_turn(
        self,
        user_input: str,
        max_tool_calls: int = 5,
    ) -> AgentResponse:
        """
        Process a single conversation turn.
        
        Args:
            user_input: User's message
            max_tool_calls: Maximum tool calls per turn
            
        Returns:
            AgentResponse with content and tool usage
        """
        self.turn_count += 1
        
        # Add to history
        self.history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Check for triggered goals
        context = {"user_input": user_input, "turn": self.turn_count}
        triggered_goals = self.planner.check_triggers(context)
        
        tool_calls = []
        tool_results = []
        
        # Execute triggered goals first
        for goal in triggered_goals[:1]:  # Limit to one auto-goal per turn
            result = self.planner.execute_goal(goal.id)
            tool_calls.append({"type": "goal", "goal_id": goal.id})
            tool_results.append(result.to_dict())
        
        # Process explicit commands
        response_content = self._generate_response(user_input, tool_calls, tool_results, max_tool_calls)
        
        # Add to history
        self.history.append({
            "role": "assistant",
            "content": response_content,
            "tool_calls": [tc for tc in tool_calls if tc.get("type") != "goal"],
            "timestamp": datetime.now().isoformat(),
        })
        
        return AgentResponse(
            content=response_content,
            tool_calls=tool_calls,
            tool_results=tool_results,
            metadata={
                "turn": self.turn_count,
                "triggered_goals": [g.id for g in triggered_goals],
            }
        )
    
    def _generate_response(
        self,
        user_input: str,
        tool_calls: List[Dict],
        tool_results: List[Dict],
        max_tool_calls: int,
    ) -> str:
        """
        Generate response using VLM with tool calling.
        
        This is where the VLM would be called. For now, we simulate
        basic tool detection and response generation.
        """
        input_lower = user_input.lower()
        
        # Detect tool needs from input
        if "remember" in input_lower or "store" in input_lower:
            # Extract content to remember
            content = user_input.replace("remember", "").replace("store", "").strip()
            result = self.executor.execute("store_memory", {"content": content, "importance": 0.8})
            tool_calls.append({"name": "store_memory", "args": {"content": content}})
            tool_results.append(result.to_dict())
            return f"I've stored that in my memory: '{content}'"
        
        elif "recall" in input_lower or "what do you know" in input_lower:
            query = input_lower.replace("recall", "").replace("what do you know about", "").strip()
            result = self.executor.execute("query_memory", {"query": query, "limit": 5})
            tool_calls.append({"name": "query_memory", "args": {"query": query}})
            tool_results.append(result.to_dict())
            
            if result.success and result.result:
                memories = result.result
                if memories:
                    return f"Here's what I remember about '{query}':\n" + "\n".join(
                        f"- {m['content']}" for m in memories[:5]
                    )
            return f"I don't have any memories about '{query}' yet."
        
        elif "calculate" in input_lower or "compute" in input_lower or "=" in input_lower:
            # Extract code
            code = user_input.replace("calculate", "").replace("compute", "").strip()
            if not any(kw in code for kw in ['print', 'result', '=']):
                code = f"result = {code}\nprint(result)"
            
            result = self.sandbox.execute(code)
            tool_calls.append({"name": "run_code", "args": {"code": code}})
            tool_results.append(result.to_dict())
            
            if result.success:
                return f"Result: {result.output.strip() or result.result}"
            else:
                return f"Error: {result.error}"
        
        elif "create goal" in input_lower or "plan" in input_lower:
            # Simple goal creation
            description = user_input.replace("create goal", "").replace("plan", "").strip()
            goal = self.planner.create_goal(
                description=description,
                steps=[
                    Step("Analyze request", "query_memory", {"query": description}),
                    Step("Generate response", "generate_text", {"prompt": description}),
                ],
            )
            return f"Created goal '{goal.id}': {description}"
        
        elif "execute goal" in input_lower:
            # Extract goal ID
            parts = user_input.split()
            goal_id = parts[-1] if parts else ""
            result = self.planner.execute_goal(goal_id)
            return f"Goal {goal_id}: {result.status.value} ({result.completed_steps}/{result.total_steps} steps)"
        
        elif "list goals" in input_lower:
            goals = self.planner.list_goals()
            if goals:
                return "Goals:\n" + "\n".join(
                    f"- [{g.id}] {g.description} ({g.status.value})"
                    for g in goals
                )
            return "No goals created yet."
        
        elif "ingest" in input_lower or "read file" in input_lower:
            return "Document ingestion requires file content. Use ingest_document tool with content and filename."
        
        elif "list documents" in input_lower:
            docs = self.ingester.list_documents()
            if docs:
                return "Ingested documents:\n" + "\n".join(
                    f"- {d['filename']} ({d['total_chunks']} chunks)"
                    for d in docs
                )
            return "No documents ingested yet."
        
        elif "help" in input_lower:
            return """I can help you with:
- **Memory**: "remember X", "recall Y", "what do you know about Z"
- **Calculations**: "calculate 2+2", "compute sqrt(16)"
- **Goals**: "create goal X", "list goals", "execute goal ID"
- **Documents**: "list documents", ingest files via tool
- **Code**: Run code safely in my sandbox

What would you like to do?"""
        
        else:
            # Generic response
            return f"I heard: '{user_input}'. I'm an autonomous FLUX agent with wave encoding, memory, code execution, and goal planning capabilities. Ask me to remember something, calculate, or create a goal!"
    
    def chat(self, user_input: str) -> str:
        """Simple chat interface."""
        response = self.process_turn(user_input)
        return response.content
    
    def save(self, path: Optional[str] = None):
        """Save state back to .flx file."""
        save_path = path or self.flx_path
        if save_path:
            torch.save(self.flx, save_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "model_version": self.model_version,
            "session_start": self.session_start,
            "turn_count": self.turn_count,
            "history_length": len(self.history),
            "executor_stats": self.executor.get_stats(),
            "planner_stats": self.planner.get_stats(),
            "documents": len(self.ingester.documents),
        }
    
    def reset_session(self):
        """Reset session state (keeps persisted data)."""
        self.history = []
        self.turn_count = 0
        self.session_start = datetime.now().isoformat()
        self.executor.clear_cache()


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

def autonomous_inference(
    flx_path: str,
    user_input: str,
    device: str = 'cpu',
) -> str:
    """
    Run autonomous inference.
    
    Args:
        flx_path: Path to .flx file
        user_input: User's message
        device: Target device
        
    Returns:
        Agent response
    """
    agent = AutonomousAgent.from_flx(flx_path, device)
    return agent.chat(user_input)


def create_agent(flx_path: str = None, device: str = 'cpu') -> AutonomousAgent:
    """
    Create an autonomous agent.
    
    Args:
        flx_path: Path to .flx file (optional, creates empty state if None)
        device: Target device
        
    Returns:
        AutonomousAgent instance
    """
    if flx_path and Path(flx_path).exists():
        return AutonomousAgent.from_flx(flx_path, device)
    else:
        # Create empty state
        empty_state = {
            'version': '8.2-autonomous',
            'format': 'FLUX',
            'memory': {'state_dict': {'episodic': {'metadata': []}}},
            'documents': {},
            'goals': {},
            'user_tools': {},
        }
        return AutonomousAgent(empty_state, flx_path, device)
