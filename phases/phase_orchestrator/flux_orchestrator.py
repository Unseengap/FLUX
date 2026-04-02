"""
FLUXOrchestrator — VLM-driven component orchestration.

The VLM becomes the brain that decides which FLUX components to invoke.
Tool calls are parsed from VLM output and dispatched to the appropriate
component, with results fed back into the VLM context.

Usage:
    from flux_model import FLUXModel
    from flux_orchestrator import FLUXOrchestrator, orchestrated_inference
    
    model = FLUXModel.load('checkpoints/Flux-Apex-V1.flx')
    orchestrator = FLUXOrchestrator(model, device='cuda')
    
    response = orchestrated_inference(
        orchestrator=orchestrator,
        vlm=model.vlm,
        user_input="Solve this ARC puzzle",
        grid=[[0,1,0],[1,2,1],[0,1,0]],
    )
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import re
from dataclasses import dataclass

try:
    from phases.phase_orchestrator.tool_registry import FLUX_TOOLS, ToolDefinition, get_tool
    from phases.phase_orchestrator.system_prompt import FLUX_SYSTEM_PROMPT
except ImportError:
    try:
        from .tool_registry import FLUX_TOOLS, ToolDefinition, get_tool
        from .system_prompt import FLUX_SYSTEM_PROMPT
    except ImportError:
        from tool_registry import FLUX_TOOLS, ToolDefinition, get_tool
        from system_prompt import FLUX_SYSTEM_PROMPT


@dataclass
class ToolCall:
    """Parsed tool call from VLM output."""
    name: str
    params: Dict[str, Any]
    raw: str
    

@dataclass 
class ToolResult:
    """Result from tool execution."""
    success: bool
    output: Any
    tool_name: str
    wave: Optional[Tensor] = None  # For tools that produce waves
    error: Optional[str] = None
    execution_time_ms: float = 0.0


class FLUXOrchestrator(nn.Module):
    """
    Dispatches VLM tool calls to FLUX components.
    
    The VLM generates text with <tool>...</tool> tags.
    This class parses those calls and executes them on the
    appropriate FLUX component, returning results for the
    VLM to incorporate.
    
    Architecture:
        VLM Output → Parse Tool Calls → Dispatch → Execute → Format Results → VLM
                                          ↓
                              ┌───────────────────────┐
                              │   FLUX Components     │
                              │ - CSE (encode)        │
                              │ - Field (query)       │
                              │ - Memory (store/recall)│
                              │ - CausalTracker       │
                              │ - Adapters            │
                              │ - SpatialMemory       │
                              └───────────────────────┘
    """
    
    def __init__(
        self,
        flx_model: 'FLUXModel',
        device: str = 'cuda',
        wave_dim: int = 432,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            flx_model: Loaded FLUXModel with all components
            device: Target device
            wave_dim: FLUX wave dimension (432)
        """
        super().__init__()
        self.device = device
        self.wave_dim = wave_dim
        self.flx = flx_model
        
        # Build component lookup
        self.components = self._build_component_map(flx_model)
        
        # Context accumulator for this turn
        self.context_waves: List[Tensor] = []
        self.last_wave: Optional[Tensor] = None
        self.last_grid: Optional[List[List[int]]] = None
        self.input_grid: Optional[List[List[int]]] = None
        self.input_image: Optional[Tensor] = None
        
        # Tool call history for this turn
        self.history: List[Tuple[ToolCall, ToolResult]] = []
        
        # Statistics
        self.total_tool_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
    
    def _build_component_map(self, flx) -> Dict[str, Any]:
        """Build a flat map of component paths to instances."""
        components = {}
        
        # Get state dict if model is loaded from .flx
        state = getattr(flx, 'state', {})
        
        # Direct components from state
        for name in ['cse', 'field', 'causal_tracker', 'spatial_memory']:
            if name in state:
                components[name] = ComponentWrapper(name, state[name], self.device)
            elif hasattr(flx, name):
                components[name] = getattr(flx, name)
        
        # Memory tiers
        if 'memory' in state:
            memory = state['memory']
            for tier in ['episodic', 'working', 'semantic']:
                if tier in memory:
                    components[f'memory.{tier}'] = ComponentWrapper(
                        f'memory.{tier}', memory[tier], self.device
                    )
        
        # Adapters
        if 'adapters' in state:
            adapters = state['adapters']
            for adapter_name, adapter_state in adapters.items():
                components[f'adapters.{adapter_name}'] = ComponentWrapper(
                    f'adapters.{adapter_name}', adapter_state, self.device
                )
        
        # Grid to wave (top-level in some versions)
        if 'grid_to_wave' in state:
            components['adapters.grid_to_wave'] = ComponentWrapper(
                'adapters.grid_to_wave', state['grid_to_wave'], self.device
            )
        
        # Causal system
        if 'causal' in state:
            causal = state['causal']
            if 'cgn_state' in causal:
                components['causal.cgn'] = ComponentWrapper(
                    'causal.cgn', causal['cgn_state'], self.device
                )
            if 'graph_state' in causal:
                components['causal.graph'] = ComponentWrapper(
                    'causal.graph', causal['graph_state'], self.device
                )
        
        # Rule inducer / learned rules
        if 'learned_rules' in state:
            components['rule_inducer'] = ComponentWrapper(
                'rule_inducer', state['learned_rules'], self.device
            )
        
        # VLM
        if 'vlm' in state:
            components['vlm'] = ComponentWrapper(
                'vlm', state['vlm'], self.device
            )
        
        return components
    
    def parse_tool_calls(self, vlm_output: str) -> List[ToolCall]:
        """
        Extract tool calls from VLM output.
        
        Format:
            <tool>tool_name</tool>
            <params>{"param": "value"}</params>
        
        Args:
            vlm_output: Raw text from VLM
            
        Returns:
            List of parsed ToolCall objects
        """
        calls = []
        
        # Find all <tool>...</tool> blocks with optional params
        # Pattern handles both single-line and multi-line
        tool_pattern = r'<tool>\s*([\w_]+)\s*</tool>\s*(?:<params>\s*(.*?)\s*</params>)?'
        matches = re.findall(tool_pattern, vlm_output, re.DOTALL | re.IGNORECASE)
        
        for tool_name, params_str in matches:
            tool_name = tool_name.strip()
            
            # Check if tool exists
            if tool_name not in FLUX_TOOLS:
                continue
            
            # Parse params
            if params_str:
                # Replace special variables before parsing
                params_str = self._substitute_variables(params_str)
                try:
                    params = json.loads(params_str)
                except json.JSONDecodeError:
                    # Try to fix common JSON issues
                    params_str = params_str.replace("'", '"')
                    try:
                        params = json.loads(params_str)
                    except:
                        params = {}
            else:
                params = {}
            
            calls.append(ToolCall(
                name=tool_name,
                params=params,
                raw=f"<tool>{tool_name}</tool><params>{json.dumps(params)}</params>"
            ))
        
        return calls
    
    def _substitute_variables(self, params_str: str) -> str:
        """Replace $LAST_WAVE, $LAST_GRID, etc. with markers."""
        replacements = {
            '"$LAST_WAVE"': '"__LAST_WAVE__"',
            '$LAST_WAVE': '"__LAST_WAVE__"',
            '"$LAST_GRID"': '"__LAST_GRID__"',
            '$LAST_GRID': '"__LAST_GRID__"',
            '"$INPUT_GRID"': '"__INPUT_GRID__"',
            '$INPUT_GRID': '"__INPUT_GRID__"',
            '"$INPUT_IMAGE"': '"__INPUT_IMAGE__"',
            '$INPUT_IMAGE': '"__INPUT_IMAGE__"',
            '"$CONTEXT"': '"__CONTEXT__"',
            '$CONTEXT': '"__CONTEXT__"',
        }
        for old, new in replacements.items():
            params_str = params_str.replace(old, new)
        return params_str
    
    def _resolve_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve __LAST_WAVE__ etc. to actual values."""
        resolved = {}
        for key, value in params.items():
            if value == "__LAST_WAVE__":
                if self.last_wave is None:
                    raise ValueError("$LAST_WAVE referenced but no previous wave exists")
                resolved[key] = self.last_wave
            elif value == "__LAST_GRID__":
                if self.last_grid is None:
                    raise ValueError("$LAST_GRID referenced but no previous grid exists")
                resolved[key] = self.last_grid
            elif value == "__INPUT_GRID__":
                if self.input_grid is None:
                    raise ValueError("$INPUT_GRID referenced but no input grid provided")
                resolved[key] = self.input_grid
            elif value == "__INPUT_IMAGE__":
                if self.input_image is None:
                    raise ValueError("$INPUT_IMAGE referenced but no input image provided")
                resolved[key] = self.input_image
            elif value == "__CONTEXT__":
                resolved[key] = self.get_context_for_vlm()
            else:
                resolved[key] = value
        return resolved
    
    def execute_tool(self, call: ToolCall) -> ToolResult:
        """
        Execute a single tool call.
        
        Args:
            call: Parsed ToolCall
            
        Returns:
            ToolResult with output or error
        """
        import time
        start_time = time.time()
        
        self.total_tool_calls += 1
        
        # Get tool definition
        tool_def = get_tool(call.name)
        if tool_def is None:
            self.failed_calls += 1
            return ToolResult(
                success=False,
                output=None,
                tool_name=call.name,
                error=f"Unknown tool: {call.name}"
            )
        
        # Get component
        component = self.components.get(tool_def.component_path)
        if component is None:
            self.failed_calls += 1
            return ToolResult(
                success=False,
                output=None,
                tool_name=call.name,
                error=f"Component not found: {tool_def.component_path}"
            )
        
        # Resolve variable references
        try:
            params = self._resolve_params(call.params)
        except ValueError as e:
            self.failed_calls += 1
            return ToolResult(
                success=False,
                output=None,
                tool_name=call.name,
                error=str(e)
            )
        
        # Execute
        try:
            result = component.call(tool_def.method_name, params)
            
            # Track waves for chaining
            if isinstance(result, Tensor):
                if result.dim() >= 1 and result.shape[-1] == self.wave_dim:
                    self.last_wave = result
                    self.context_waves.append(result)
            
            # Track grids
            if isinstance(result, list):
                if len(result) > 0 and isinstance(result[0], list):
                    self.last_grid = result
            
            elapsed_ms = (time.time() - start_time) * 1000
            self.successful_calls += 1
            
            return ToolResult(
                success=True,
                output=result,
                tool_name=call.name,
                wave=self.last_wave if isinstance(result, Tensor) else None,
                execution_time_ms=elapsed_ms,
            )
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.failed_calls += 1
            return ToolResult(
                success=False,
                output=None,
                tool_name=call.name,
                error=str(e),
                execution_time_ms=elapsed_ms,
            )
    
    def execute_all(self, vlm_output: str) -> List[ToolResult]:
        """
        Parse and execute all tool calls in VLM output.
        
        Args:
            vlm_output: Raw VLM text containing tool calls
            
        Returns:
            List of ToolResult objects
        """
        calls = self.parse_tool_calls(vlm_output)
        results = []
        
        for call in calls:
            result = self.execute_tool(call)
            self.history.append((call, result))
            results.append(result)
        
        return results
    
    def format_results_for_vlm(self, results: List[ToolResult]) -> str:
        """
        Format tool results as context for VLM continuation.
        
        Args:
            results: List of ToolResult objects
            
        Returns:
            Formatted string for VLM context
        """
        lines = []
        
        for result in results:
            if result.success:
                # Format output based on type
                output = result.output
                if isinstance(output, Tensor):
                    output_str = f"Tensor shape {list(output.shape)}, dtype={output.dtype}"
                elif isinstance(output, list):
                    if len(output) > 10:
                        output_str = f"List with {len(output)} items"
                    elif len(output) > 0 and isinstance(output[0], list):
                        # Grid
                        output_str = f"Grid {len(output)}×{len(output[0])}: {output}"
                    else:
                        output_str = str(output)[:200]
                elif isinstance(output, dict):
                    output_str = f"Dict with keys: {list(output.keys())}"
                else:
                    output_str = str(output)[:200]
                
                lines.append(f"[{result.tool_name}] ✓ {output_str} ({result.execution_time_ms:.1f}ms)")
            else:
                lines.append(f"[{result.tool_name}] ✗ Error: {result.error}")
        
        return "\n".join(lines)
    
    def get_context_for_vlm(self) -> Optional[Tensor]:
        """
        Get accumulated context waves to inject into VLM.
        
        Returns:
            Averaged context wave [432] or None
        """
        if not self.context_waves:
            return None
        
        # Stack and average context waves
        # Handle different shapes
        waves = []
        for w in self.context_waves:
            if w.dim() == 1:
                waves.append(w.unsqueeze(0))  # [432] → [1, 432]
            elif w.dim() == 2:
                waves.append(w.mean(dim=0, keepdim=True))  # [seq, 432] → [1, 432]
            else:
                waves.append(w.view(-1, self.wave_dim).mean(dim=0, keepdim=True))
        
        stacked = torch.cat(waves, dim=0)  # [n, 432]
        return stacked.mean(dim=0)  # [432]
    
    def reset_turn(self):
        """Reset context for a new conversation turn."""
        self.context_waves = []
        self.last_wave = None
        self.last_grid = None
        self.input_grid = None
        self.input_image = None
        self.history = []
    
    def set_input(
        self,
        grid: Optional[List[List[int]]] = None,
        image: Optional[Tensor] = None,
    ):
        """Set input context for tool variable resolution."""
        self.input_grid = grid
        self.input_image = image
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            'total_tool_calls': self.total_tool_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'success_rate': self.successful_calls / max(1, self.total_tool_calls),
            'current_turn_calls': len(self.history),
            'context_waves': len(self.context_waves),
        }


class ComponentWrapper:
    """
    Wraps a FLUX component loaded from .flx state dict.
    
    Provides a unified interface for calling component methods,
    handling both nn.Module instances and raw state dicts.
    """
    
    def __init__(self, name: str, state: Dict[str, Any], device: str):
        self.name = name
        self.state = state
        self.device = device
        self._module = None
    
    def call(self, method_name: str, params: Dict[str, Any]) -> Any:
        """
        Call a method on the wrapped component.
        
        For nn.Module components, this calls the actual method.
        For state dicts, this performs the appropriate operation
        based on the method name.
        """
        # If we have a loaded module, use it directly
        if self._module is not None:
            method = getattr(self._module, method_name, None)
            if method is not None:
                return method(**params)
        
        # Otherwise, handle based on component type
        if 'state_dict' in self.state:
            # This is a serialized nn.Module
            return self._handle_serialized_module(method_name, params)
        else:
            # This is a raw state dict (e.g., memory data)
            return self._handle_raw_state(method_name, params)
    
    def _handle_serialized_module(self, method: str, params: Dict) -> Any:
        """Handle calls to serialized nn.Module components."""
        # For CSE, Field, etc. we need to reconstruct and call
        # This is a placeholder — actual implementation would
        # reconstruct the module from state_dict
        raise NotImplementedError(
            f"Module {self.name} needs reconstruction. "
            f"Load the full FLUXModel for runtime inference."
        )
    
    def _handle_raw_state(self, method: str, params: Dict) -> Any:
        """Handle calls to raw state components (memory, rules, etc.)."""
        if 'memory' in self.name:
            return self._handle_memory_call(method, params)
        elif 'rule' in self.name:
            return self._handle_rule_call(method, params)
        elif 'spatial' in self.name:
            return self._handle_spatial_call(method, params)
        else:
            raise NotImplementedError(f"No handler for {self.name}.{method}")
    
    def _handle_memory_call(self, method: str, params: Dict) -> Any:
        """Handle episodic/semantic memory calls."""
        if method == 'search':
            # Search episodic memory
            query = params.get('query', '')
            limit = params.get('limit', 5)
            
            # Simple keyword search in stored metadata
            metadata = self.state.get('metadata', [])
            results = []
            for entry in metadata:
                content = entry.get('content', '')
                if query.lower() in content.lower():
                    results.append((
                        content,
                        entry.get('timestamp', 0),
                        entry.get('importance', 0.5)
                    ))
            return results[:limit]
        
        elif method == 'store':
            # Store new memory
            content = params.get('content', '')
            importance = params.get('importance', 0.5)
            
            metadata = self.state.get('metadata', [])
            new_id = len(metadata)
            metadata.append({
                'id': new_id,
                'content': content,
                'importance': importance,
                'timestamp': len(metadata),
            })
            self.state['metadata'] = metadata
            return new_id
        
        else:
            raise NotImplementedError(f"Memory method: {method}")
    
    def _handle_rule_call(self, method: str, params: Dict) -> Any:
        """Handle rule inducer calls."""
        if method == 'match_rules':
            trigger_color = params.get('trigger_color')
            trigger_action = params.get('trigger_action')
            
            rules = self.state.get('rules', [])
            matches = []
            for rule in rules:
                if (rule.get('trigger_color') == trigger_color and
                    rule.get('trigger_action') == trigger_action):
                    matches.append(rule)
            return matches
        else:
            raise NotImplementedError(f"Rule method: {method}")
    
    def _handle_spatial_call(self, method: str, params: Dict) -> Any:
        """Handle spatial memory calls."""
        if method == 'get_curiosity':
            grid_size = params.get('grid_size', (10, 10))
            
            # Return curiosity field cropped to grid size
            curiosity = self.state.get('curiosity_field')
            if curiosity is not None:
                if isinstance(curiosity, Tensor):
                    return curiosity[:grid_size[0], :grid_size[1]]
            return torch.ones(grid_size)  # Default: all curious
        
        elif method == 'update':
            position = params.get('position', (0, 0))
            novelty = params.get('novelty', 0.5)
            
            exploration = self.state.get('exploration_mass')
            if exploration is not None and isinstance(exploration, Tensor):
                exploration[position[0], position[1]] += novelty
                return exploration[position[0], position[1]].item()
            return novelty
        
        else:
            raise NotImplementedError(f"Spatial method: {method}")


def orchestrated_inference(
    orchestrator: FLUXOrchestrator,
    vlm: Any,  # VLM module with generate() method
    user_input: str,
    image: Optional[Tensor] = None,
    grid: Optional[List[List[int]]] = None,
    max_iterations: int = 5,
    verbose: bool = True,
) -> str:
    """
    Run inference with VLM orchestrating FLUX tools.
    
    The loop:
    1. VLM generates response (may include tool calls)
    2. Parse and execute tool calls
    3. Inject results back into context
    4. Continue until VLM produces final answer (no more tools)
    
    Args:
        orchestrator: FLUXOrchestrator instance
        vlm: VLM module with generate() method
        user_input: User's question/request
        image: Optional input image
        grid: Optional input grid (for ARC puzzles)
        max_iterations: Max tool calling rounds
        verbose: Print progress
        
    Returns:
        Final VLM response
    """
    orchestrator.reset_turn()
    orchestrator.set_input(grid=grid, image=image)
    
    # Build initial messages
    messages = [
        {"role": "system", "content": FLUX_SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    
    if image is not None:
        messages[-1]["image"] = image
    
    iteration = 0
    final_response = ""
    
    while iteration < max_iterations:
        if verbose:
            print(f"\n--- Iteration {iteration + 1} ---")
        
        # Generate VLM response
        context_wave = orchestrator.get_context_for_vlm()
        response = vlm.generate(messages, context_wave=context_wave)
        
        if verbose:
            print(f"VLM: {response[:200]}...")
        
        # Check for tool calls
        tool_calls = orchestrator.parse_tool_calls(response)
        
        if not tool_calls:
            # No tools = final answer
            final_response = response
            if verbose:
                print("No more tool calls — returning final response")
            break
        
        if verbose:
            print(f"Found {len(tool_calls)} tool calls")
        
        # Execute tools
        results = orchestrator.execute_all(response)
        
        # Format results for next iteration
        tool_context = orchestrator.format_results_for_vlm(results)
        
        if verbose:
            print(f"Tool results:\n{tool_context}")
        
        # Add to conversation
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "system", "content": f"Tool results:\n{tool_context}\n\nContinue your response:"})
        
        iteration += 1
    
    if iteration >= max_iterations and not final_response:
        # Reached max iterations without final answer
        final_response = response  # Use last response
        if verbose:
            print(f"Warning: Reached max iterations ({max_iterations})")
    
    return final_response
