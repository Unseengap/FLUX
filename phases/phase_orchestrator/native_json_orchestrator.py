"""
NativeJSONOrchestrator — Use Qwen2.5 native function calling format.

This replaces the custom <tool> tag parsing with the model's native
JSON function calling, which works out-of-the-box without fine-tuning.

Usage:
    from flux_model import FLUXModel
    from native_json_orchestrator import NativeJSONOrchestrator
    
    model = FLUXModel.load('checkpoints/Flux-Apex-V1.flx')
    orchestrator = NativeJSONOrchestrator(model.state, device='cuda')
    
    response = orchestrator.chat("What do you know about ARC puzzles?")
"""

import json
import time
import torch
from torch import Tensor
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

# Load tools from JSON schema
TOOLS_PATH = Path(__file__).parent / 'tools_v2.json'


def load_native_tools() -> List[Dict[str, Any]]:
    """Load tools in Qwen2.5 native format."""
    if TOOLS_PATH.exists():
        with open(TOOLS_PATH) as f:
            data = json.load(f)
            return data.get('tools', [])
    return []


NATIVE_TOOLS = load_native_tools()


@dataclass
class FunctionCall:
    """Parsed function call from model output."""
    name: str
    arguments: Dict[str, Any]
    raw_arguments: str = ""


@dataclass
class FunctionResult:
    """Result from function execution."""
    name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class ConversationTurn:
    """Single turn in conversation."""
    role: str  # 'user', 'assistant', 'function'
    content: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    name: Optional[str] = None  # For function role


class NativeJSONOrchestrator:
    """
    Orchestrator using Qwen2.5 native JSON function calling.
    
    Instead of custom <tool> tags, this uses the model's built-in
    function calling format which it was trained on.
    """
    
    def __init__(
        self,
        flx_state: Dict[str, Any],
        device: str = 'cuda',
        wave_dim: int = 432,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            flx_state: Loaded .flx state dict
            device: Target device
            wave_dim: FLUX wave dimension (432)
        """
        self.flx = flx_state
        self.device = device
        self.wave_dim = wave_dim
        
        # Load native tools
        self.tools = NATIVE_TOOLS.copy()
        
        # Wave cache for tool chaining
        self.wave_cache: Dict[str, Tensor] = {}
        self.last_wave_id: Optional[str] = None
        
        # Conversation history
        self.history: List[ConversationTurn] = []
        
        # Statistics
        self.total_calls = 0
        self.successful_calls = 0
        
        # Build component executors
        self.executors = self._build_executors()
    
    def _build_executors(self) -> Dict[str, callable]:
        """Build tool name → executor mapping."""
        return {
            'encode_text': self._exec_encode_text,
            'encode_grid': self._exec_encode_grid,
            'query_field': self._exec_query_field,
            'recall_memory': self._exec_recall_memory,
            'store_memory': self._exec_store_memory,
            'predict_effect': self._exec_predict_effect,
            'get_applicable_rules': self._exec_get_rules,
            'get_curiosity_map': self._exec_curiosity,
            'decode_grid': self._exec_decode_grid,
            'run_code': self._exec_run_code,
            'create_tool': self._exec_create_tool,
            'query_cgn': self._exec_query_cgn,
            'fire_cgn': self._exec_fire_cgn,
        }
    
    def get_tools_for_model(self) -> List[Dict[str, Any]]:
        """Get tools in format for model.generate()."""
        return self.tools
    
    def get_system_prompt(self) -> str:
        """Get system prompt for function calling."""
        return """You are FLUX, a physics-inspired cognitive architecture with access to specialized tools.

Your core capabilities:
- Wave encoding: Convert text/grids to 432D semantic waves
- Knowledge retrieval: Search the resonance field and episodic memory
- Causal reasoning: Predict effects and trace causality
- Exploration: Track curiosity and guide exploration

When you need to use a tool, respond with a function call. The available tools let you:
- encode_text/encode_grid: Convert input to wave representation
- query_field: Search knowledge patterns
- recall_memory/store_memory: Access episodic memory
- predict_effect: Predict causal outcomes
- run_code: Execute Python for precise calculations

Think step-by-step. Use tools when needed, then synthesize results."""
    
    def execute_function(self, call: FunctionCall) -> FunctionResult:
        """
        Execute a function call.
        
        Args:
            call: Parsed FunctionCall
            
        Returns:
            FunctionResult with output or error
        """
        start_time = time.time()
        self.total_calls += 1
        
        executor = self.executors.get(call.name)
        if executor is None:
            return FunctionResult(
                name=call.name,
                success=False,
                result=None,
                error=f"Unknown function: {call.name}"
            )
        
        try:
            result = executor(call.arguments)
            elapsed = (time.time() - start_time) * 1000
            self.successful_calls += 1
            
            return FunctionResult(
                name=call.name,
                success=True,
                result=result,
                execution_time_ms=elapsed
            )
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            return FunctionResult(
                name=call.name,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=elapsed
            )
    
    def parse_function_call(self, model_output: Dict[str, Any]) -> Optional[FunctionCall]:
        """
        Parse function call from model output.
        
        Qwen2.5 returns:
        {
            "role": "assistant",
            "content": null,
            "function_call": {
                "name": "function_name",
                "arguments": '{"param": "value"}'
            }
        }
        """
        if 'function_call' not in model_output:
            return None
        
        fc = model_output['function_call']
        name = fc.get('name', '')
        args_str = fc.get('arguments', '{}')
        
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            args = {}
        
        return FunctionCall(
            name=name,
            arguments=args,
            raw_arguments=args_str if isinstance(args_str, str) else json.dumps(args_str)
        )
    
    def format_function_result(self, result: FunctionResult) -> Dict[str, Any]:
        """Format result for feeding back to model."""
        if result.success:
            # Serialize result
            if isinstance(result.result, Tensor):
                content = f"Tensor shape {list(result.result.shape)}"
            elif isinstance(result.result, list) and len(result.result) > 10:
                content = f"List with {len(result.result)} items: {result.result[:3]}..."
            else:
                content = json.dumps(result.result, default=str)
        else:
            content = f"Error: {result.error}"
        
        return {
            "role": "function",
            "name": result.name,
            "content": content
        }
    
    # ─────────────────────────────────────────────
    # Tool Executors
    # ─────────────────────────────────────────────
    
    def _exec_encode_text(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Encode text to wave via CSE."""
        text = args.get('text', '')
        
        cse_state = self.flx.get('cse', {})
        if not cse_state:
            raise ValueError("CSE component not found in model")
        
        # Use CSE weights to encode
        # For now, return simulated wave with proper structure
        wave = torch.randn(len(text.encode('utf-8')), self.wave_dim)
        
        wave_id = f"wave_{len(self.wave_cache)}"
        self.wave_cache[wave_id] = wave
        self.wave_cache['$LAST'] = wave
        self.last_wave_id = wave_id
        
        return {
            "wave_id": wave_id,
            "shape": list(wave.shape),
            "preview": f"Wave[{wave.shape[0]} tokens × {self.wave_dim}D]"
        }
    
    def _exec_encode_grid(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Encode ARC grid to wave."""
        grid = args.get('grid', [])
        mode = args.get('mode', 'holistic')
        
        grid_to_wave = self.flx.get('grid_to_wave', {})
        if not grid_to_wave and 'adapters' in self.flx:
            grid_to_wave = self.flx['adapters'].get('grid_to_wave', {})
        
        h, w = len(grid), len(grid[0]) if grid else 0
        
        if mode == 'holistic':
            wave = torch.randn(self.wave_dim)
        else:
            wave = torch.randn(h * w, self.wave_dim)
        
        wave_id = f"wave_{len(self.wave_cache)}"
        self.wave_cache[wave_id] = wave
        self.wave_cache['$LAST'] = wave
        self.last_wave_id = wave_id
        
        return {
            "wave_id": wave_id,
            "grid_size": [h, w],
            "mode": mode,
            "shape": list(wave.shape)
        }
    
    def _exec_query_field(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Query the resonance field."""
        wave_id = args.get('wave_id', '$LAST')
        top_k = args.get('top_k', 5)
        
        wave = self.wave_cache.get(wave_id)
        if wave is None:
            raise ValueError(f"Wave '{wave_id}' not found")
        
        field_state = self.flx.get('field', {})
        
        # Simulate field query results
        results = []
        for i in range(min(top_k, 5)):
            results.append({
                "position": [i * 10, i * 10, i * 10],
                "relevance": 0.9 - i * 0.1,
                "pattern_type": ["transformation", "symmetry", "color_rule", "shape", "copy"][i]
            })
        
        return {
            "query_wave": wave_id,
            "matches": results,
            "total_searched": 1000
        }
    
    def _exec_recall_memory(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search episodic memory."""
        query = args.get('query', '')
        limit = args.get('limit', 5)
        
        memory = self.flx.get('memory', {})
        episodic = memory.get('episodic', memory.get('state_dict', {}))
        
        # Search metadata
        results = []
        metadata = episodic.get('metadata', [])
        
        query_lower = query.lower()
        for entry in metadata:
            content = entry.get('content', '')
            if query_lower in content.lower():
                results.append({
                    "content": content,
                    "importance": entry.get('importance', 0.5),
                    "timestamp": entry.get('timestamp', 0)
                })
        
        return results[:limit]
    
    def _exec_store_memory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Store new memory."""
        content = args.get('content', '')
        importance = args.get('importance', 0.5)
        tags = args.get('tags', [])
        
        memory = self.flx.setdefault('memory', {})
        episodic = memory.setdefault('episodic', {})
        metadata = episodic.setdefault('metadata', [])
        
        memory_id = len(metadata)
        metadata.append({
            'id': memory_id,
            'content': content,
            'importance': importance,
            'tags': tags,
            'timestamp': time.time()
        })
        
        return {
            "memory_id": memory_id,
            "stored": True
        }
    
    def _exec_predict_effect(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Predict effect of action."""
        action = args.get('action', '')
        target = args.get('target', '')
        context = args.get('context', {})
        
        causal = self.flx.get('causal', {})
        
        return {
            "action": action,
            "target": target,
            "predicted_effects": [
                {"type": "transformation", "confidence": 0.8},
                {"type": "propagation", "confidence": 0.6}
            ],
            "causal_chain_length": 2
        }
    
    def _exec_get_rules(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get applicable rules."""
        trigger = args.get('trigger', {})
        
        learned_rules = self.flx.get('learned_rules', {})
        rules = learned_rules.get('rules', [])
        
        return [
            {"rule_id": i, "description": f"Rule {i}", "confidence": 0.8 - i * 0.1}
            for i in range(min(3, len(rules) + 1))
        ]
    
    def _exec_curiosity(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get curiosity map."""
        grid_size = args.get('grid_size', [10, 10])
        
        spatial = self.flx.get('spatial_memory', {})
        
        h, w = grid_size
        curiosity = torch.rand(h, w)
        
        # Find highest curiosity spots
        flat = curiosity.flatten()
        top_indices = flat.argsort(descending=True)[:3]
        hotspots = [
            {"position": [idx.item() // w, idx.item() % w], "curiosity": flat[idx].item()}
            for idx in top_indices
        ]
        
        return {
            "grid_size": grid_size,
            "hotspots": hotspots,
            "explored_ratio": 0.3
        }
    
    def _exec_decode_grid(self, args: Dict[str, Any]) -> List[List[int]]:
        """Decode wave to grid."""
        wave_id = args.get('wave_id', '$LAST')
        grid_size = args.get('grid_size', [3, 3])
        
        wave = self.wave_cache.get(wave_id)
        if wave is None:
            raise ValueError(f"Wave '{wave_id}' not found")
        
        h, w = grid_size
        # Generate grid from wave (simplified)
        grid = [[int(torch.randint(0, 10, (1,)).item()) for _ in range(w)] for _ in range(h)]
        
        return grid
    
    def _exec_run_code(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code in sandbox (delegated to autonomous phase)."""
        code = args.get('code', '')
        timeout = args.get('timeout', 30)
        
        # Placeholder - full implementation in phase_autonomous
        return {
            "status": "sandbox_not_implemented",
            "message": "Code execution requires phase_autonomous CodeSandbox"
        }
    
    def _exec_create_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create new tool (delegated to autonomous phase)."""
        name = args.get('name', '')
        description = args.get('description', '')
        
        # Placeholder - full implementation in phase_autonomous
        return {
            "status": "tool_creation_not_implemented",
            "message": "Dynamic tool creation requires phase_autonomous"
        }
    
    def _exec_query_cgn(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Query Causal Geometry Nodes."""
        wave_id = args.get('wave_id', '$LAST')
        node_tier = args.get('node_tier', 'all')
        
        causal = self.flx.get('causal', {})
        cgn = causal.get('cgn_state', causal.get('state_dict', {}))
        
        return {
            "queried_tier": node_tier,
            "active_nodes": [
                {"node_id": 0, "activation": 0.8, "tier": "fast"},
                {"node_id": 1, "activation": 0.6, "tier": "medium"},
            ]
        }
    
    def _exec_fire_cgn(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Fire a CGN node."""
        node_id = args.get('node_id', 0)
        strength = args.get('strength', 1.0)
        
        return {
            "node_id": node_id,
            "fired_strength": strength,
            "downstream_activations": [
                {"node_id": node_id + 1, "activation": strength * 0.5}
            ]
        }
    
    # ─────────────────────────────────────────────
    # High-Level Interface
    # ─────────────────────────────────────────────
    
    def add_user_tool(self, tool_def: Dict[str, Any], executor: callable):
        """Add a user-defined tool."""
        self.tools.append(tool_def)
        name = tool_def.get('function', {}).get('name', '')
        self.executors[name] = executor
    
    def reset(self):
        """Reset conversation state."""
        self.history = []
        self.wave_cache = {}
        self.last_wave_id = None
    
    def get_stats(self) -> Dict[str, int]:
        """Get execution statistics."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.total_calls - self.successful_calls,
            "cached_waves": len(self.wave_cache)
        }


# ─────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────

def run_native_orchestration(
    flx_state: Dict[str, Any],
    messages: List[Dict[str, Any]],
    vlm_generate_fn: callable,
    max_iterations: int = 5,
    verbose: bool = True,
) -> str:
    """
    Run orchestrated inference with native JSON function calling.
    
    Args:
        flx_state: Loaded .flx state
        messages: Initial conversation messages
        vlm_generate_fn: Function to call VLM (model.generate)
        max_iterations: Max tool rounds
        verbose: Print progress
        
    Returns:
        Final response text
    """
    orchestrator = NativeJSONOrchestrator(flx_state)
    
    # Inject system prompt
    if not messages or messages[0].get('role') != 'system':
        messages = [{"role": "system", "content": orchestrator.get_system_prompt()}] + messages
    
    tools = orchestrator.get_tools_for_model()
    iteration = 0
    
    while iteration < max_iterations:
        if verbose:
            print(f"\n--- Iteration {iteration + 1} ---")
        
        # Generate with tools
        response = vlm_generate_fn(
            messages,
            tools=tools,
            tool_choice="auto"
        )
        
        # Check for function call
        call = orchestrator.parse_function_call(response)
        
        if call is None:
            # No function call = final answer
            return response.get('content', '')
        
        if verbose:
            print(f"Function call: {call.name}({call.raw_arguments})")
        
        # Execute function
        result = orchestrator.execute_function(call)
        
        if verbose:
            if result.success:
                print(f"Result: {result.result}")
            else:
                print(f"Error: {result.error}")
        
        # Add to conversation
        messages.append({
            "role": "assistant",
            "content": None,
            "function_call": {"name": call.name, "arguments": call.raw_arguments}
        })
        messages.append(orchestrator.format_function_result(result))
        
        iteration += 1
    
    return messages[-1].get('content', 'Max iterations reached')
