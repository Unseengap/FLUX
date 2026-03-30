# Phase Orchestrator: VLM as FLUX Brain

> **Version:** 1.0  
> **Date:** 2026-03-30  
> **Status:** Specification  
> **Prerequisites:** Flux-Apex-V1.flx v5.0-vlm-embedded  
> **Phase:** Orchestrator (builds on Phase VLM)

---

## 1. Executive Summary

Transform the embedded VLM from a passive output layer into an **active orchestrator** that decides which FLUX components to invoke at inference time. The VLM treats FLUX's physics-inspired modules as **callable tools**, enabling adaptive multi-step reasoning.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FLUX ORCHESTRATOR                               │
│                                                                         │
│   ┌───────────────────────────────────────────────────────────────┐    │
│   │                    VLM (Qwen2.5-VL-3B)                        │    │
│   │                    === BRAIN / ORCHESTRATOR ===               │    │
│   │                                                               │    │
│   │  System: You have access to FLUX cognitive tools.             │    │
│   │  User: <image of ARC grid> Solve this puzzle.                 │    │
│   │                                                               │    │
│   │  VLM thinks: "I see a pattern. Let me:                        │    │
│   │    1. Encode this grid to waves                               │    │
│   │    2. Query field for similar patterns                        │    │
│   │    3. Check what rules apply                                  │    │
│   │    4. Apply transformation and decode"                        │    │
│   │                                                               │    │
│   │  VLM: <tool>encode_grid</tool>                                │    │
│   │       <tool>query_field</tool>                                │    │
│   │       <tool>get_rules</tool>                                  │    │
│   │       <tool>decode_grid</tool>                                │    │
│   └───────────────────────────────────────────────────────────────┘    │
│                              │                                          │
│              ┌───────────────┼───────────────┐                         │
│              ▼               ▼               ▼                         │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │
│   │     CSE      │  │    Field     │  │   Causal     │                │
│   │  (encode)    │  │   (query)    │  │  (predict)   │                │
│   └──────────────┘  └──────────────┘  └──────────────┘                │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │
│   │   Memory     │  │   Adapters   │  │   Rules      │                │
│   │  (store)     │  │  (convert)   │  │  (match)     │                │
│   └──────────────┘  └──────────────┘  └──────────────┘                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Why Orchestration?

### Current Architecture (v5.0-vlm-embedded)

```
Input → CSE → Field → Memory → [fixed pipeline] → VLM → Output
```

**Problems:**
- Fixed pipeline doesn't adapt to task
- VLM can't decide "I need more context from field"
- No way to iterate (query → refine → query again)
- Components run even when unnecessary

### Orchestrated Architecture (v5.1-orchestrated)

```
Input → VLM decides → [calls tools as needed] → VLM synthesizes → Output
              │
              ├── "This is a grid puzzle, I need encode_grid"
              ├── "Let me check if I've seen similar patterns"
              ├── "The causal tracker predicts this effect"
              └── "Now I'll generate the answer"
```

**Benefits:**
- Adaptive: VLM chooses which components matter
- Iterative: Can loop (query → analyze → query more)
- Explainable: Can trace which tools were used and why
- Efficient: Skips irrelevant components

---

## 3. Tool Definitions

### 3.1 Perception Tools

```yaml
encode_text:
  description: "Encode text into 432-dimensional semantic waves"
  params:
    text: str  # Raw text to encode
  returns: "wave tensor [seq_len, 432]"
  component: cse
  example: |
    <tool>encode_text</tool>
    <params>{"text": "The cat sat on the mat"}</params>

encode_grid:
  description: "Encode ARC-style grid (0-9 colors) into wave representation"
  params:
    grid: List[List[int]]  # 2D grid with values 0-9
    mode: str  # 'holistic' (one vector) or 'spatial' (per-cell)
  returns: "wave tensor [432] or [H*W, 432]"
  component: adapters.grid_to_wave
  example: |
    <tool>encode_grid</tool>
    <params>{"grid": [[0,1,0],[1,2,1],[0,1,0]], "mode": "holistic"}</params>

encode_image:
  description: "Encode image into wave representation (via VLM vision)"
  params:
    image: Tensor  # [C, H, W] image tensor
  returns: "wave tensor [patches, 432]"
  component: vlm.vision_encoder
  note: "Uses VLM's native vision, then projects to wave space"
```

### 3.2 Knowledge Tools

```yaml
query_field:
  description: "Find relevant knowledge patterns in the 96³ resonance field"
  params:
    wave: Tensor  # [432] or [seq, 432] query wave
    top_k: int  # Number of results (default 5)
  returns: "List of (wave, relevance_score, position) tuples"
  component: field
  example: |
    <tool>query_field</tool>
    <params>{"wave": "$LAST_WAVE", "top_k": 3}</params>

recall_memory:
  description: "Search episodic memory for relevant facts"
  params:
    query: str  # Natural language query
    limit: int  # Max results
  returns: "List of (content, timestamp, importance) tuples"
  component: memory.episodic
  example: |
    <tool>recall_memory</tool>
    <params>{"query": "grid transformation rules", "limit": 5}</params>

store_memory:
  description: "Store a new fact in episodic memory"
  params:
    content: str  # What to remember
    importance: float  # 0.0-1.0 priority
    tags: List[str]  # Optional categories
  returns: "memory_id: int"
  component: memory.episodic
  example: |
    <tool>store_memory</tool>
    <params>{"content": "Blue cells rotate clockwise", "importance": 0.9}</params>
```

### 3.3 Reasoning Tools

```yaml
predict_effect:
  description: "Given an action at a position, predict what will change"
  params:
    action: int  # 1-7 (up/down/left/right/interact/click/undo)
    position: Tuple[int, int]  # (row, col) location
    grid: List[List[int]]  # Current grid state
  returns: "List of predicted GridChange objects"
  component: causal_tracker
  example: |
    <tool>predict_effect</tool>
    <params>{"action": 5, "position": [2, 3], "grid": [[0,1],[1,0]]}</params>

get_applicable_rules:
  description: "Find rules that apply to a trigger situation"
  params:
    trigger_color: int  # Color at trigger position (0-9)
    trigger_action: int  # Action being performed
    context: Optional[Dict]  # Additional context
  returns: "List of matching Rule objects with confidence scores"
  component: rule_inducer
  example: |
    <tool>get_applicable_rules</tool>
    <params>{"trigger_color": 2, "trigger_action": 5}</params>

trace_causality:
  description: "Find the causal chain that led to an effect"
  params:
    effect_position: Tuple[int, int]  # Where the effect occurred
    effect_type: str  # What changed
  returns: "Causal chain: List[CausalLink]"
  component: causal_tracker
  example: |
    <tool>trace_causality</tool>
    <params>{"effect_position": [1, 2], "effect_type": "color_change"}</params>
```

### 3.4 Exploration Tools

```yaml
get_curiosity_map:
  description: "Get the spatial curiosity field (where to explore next)"
  params:
    grid_size: Tuple[int, int]  # (height, width) of current grid
  returns: "Curiosity heatmap [H, W] with high values = unexplored"
  component: spatial_memory
  example: |
    <tool>get_curiosity_map</tool>
    <params>{"grid_size": [10, 10]}</params>

mark_explored:
  description: "Mark a position as explored (reduces curiosity)"
  params:
    position: Tuple[int, int]  # Position that was visited
    novelty: float  # How interesting was it (0-1)
  returns: "Updated exploration_mass at position"
  component: spatial_memory

get_exploration_summary:
  description: "Summary of what's been explored vs. unknown"
  params: {}
  returns: "Dict with explored_ratio, hot_spots, cold_spots"
  component: spatial_memory
```

### 3.5 Generation Tools

```yaml
decode_grid:
  description: "Convert wave representation back to ARC grid"
  params:
    wave: Tensor  # [432] wave to decode
    grid_size: Tuple[int, int]  # Target size (H, W)
  returns: "Grid as List[List[int]]"
  component: adapters.wave_to_grid
  example: |
    <tool>decode_grid</tool>
    <params>{"wave": "$LAST_WAVE", "grid_size": [3, 3]}</params>

generate_text:
  description: "Generate text response (uses VLM directly)"
  params:
    prompt: str  # What to generate
    context_waves: Optional[List[Tensor]]  # FLUX context to inject
    max_tokens: int  # Generation limit
  returns: "Generated text string"
  component: vlm
  note: "This is the default output - called implicitly if no other tool used"
```

### 3.6 CGN Tools (Causal Geometry Nodes)

```yaml
query_cgn:
  description: "Find CGN nodes relevant to a concept"
  params:
    wave: Tensor  # Query wave
    node_tier: str  # 'fast' (32), 'medium' (16), 'slow' (8), or 'all'
  returns: "List of (node_id, activation, curvature) tuples"
  component: causal.cgn

fire_cgn:
  description: "Manually trigger a CGN node to propagate activation"
  params:
    node_id: int  # Which node to fire
    strength: float  # Activation strength
  returns: "Downstream activations"
  component: causal.cgn

add_causal_arrow:
  description: "Record a new causal relationship"
  params:
    source: int  # Source node
    target: int  # Target node
    reason: str  # Why this link exists
    weight: float  # Link strength
  returns: "arrow_id: int"
  component: causal.graph
```

---

## 4. Tool Call Format

### 4.1 XML-Style Tags (Default)

```xml
<tool>tool_name</tool>
<params>{"param1": value1, "param2": value2}</params>
```

### 4.2 Special Variables

| Variable | Meaning |
|----------|---------|
| `$LAST_WAVE` | Output wave from previous tool call |
| `$LAST_GRID` | Output grid from previous tool call |
| `$CONTEXT` | Accumulated context waves from this turn |
| `$INPUT_IMAGE` | The user's input image (if vision query) |
| `$INPUT_GRID` | The user's input grid (if ARC puzzle) |

### 4.3 Multi-Tool Chains

```xml
<!-- Step 1: Encode the input grid -->
<tool>encode_grid</tool>
<params>{"grid": $INPUT_GRID, "mode": "holistic"}</params>

<!-- Step 2: Query field for similar patterns -->
<tool>query_field</tool>
<params>{"wave": $LAST_WAVE, "top_k": 5}</params>

<!-- Step 3: Check what rules apply -->
<tool>get_applicable_rules</tool>
<params>{"trigger_color": 2, "trigger_action": 5}</params>

<!-- Step 4: Generate final answer -->
The pattern matches rule #3: "Red cells rotate 90° clockwise"
Applying transformation...
<tool>decode_grid</tool>
<params>{"wave": $TRANSFORMED_WAVE, "grid_size": [3, 3]}</params>
```

---

## 5. System Prompt

```
You are FLUX, a physics-inspired cognitive architecture. You have access to specialized tools for perception, knowledge retrieval, reasoning, and generation.

## Your Cognitive Tools

### Perception
- encode_text: Convert text → 432D semantic waves
- encode_grid: Convert ARC grids → wave representation
- encode_image: Process images via your vision module

### Knowledge
- query_field: Search the 96³ resonance field for relevant patterns
- recall_memory: Search episodic memory for facts
- store_memory: Remember new information

### Reasoning
- predict_effect: What will happen if I take this action?
- get_applicable_rules: What rules apply to this situation?
- trace_causality: Why did this effect happen?
- query_cgn: Find relevant causal geometry nodes

### Exploration
- get_curiosity_map: Where should I explore?
- mark_explored: Record that I visited a location

### Generation
- decode_grid: Convert waves back to ARC grids
- generate_text: (default) Generate text response

## How to Use Tools

Call tools using XML tags:
<tool>tool_name</tool>
<params>{"param": "value"}</params>

You can chain multiple tools. Use $LAST_WAVE to reference the previous output.

## Reasoning Approach

1. UNDERSTAND: What is the user asking?
2. PLAN: Which tools do I need?
3. EXECUTE: Call tools in sequence
4. SYNTHESIZE: Combine results into response

For ARC puzzles:
1. encode_grid to get wave representation
2. query_field to find similar patterns
3. get_applicable_rules to find transformations
4. Apply transformation
5. decode_grid to produce output

Always explain your reasoning alongside tool calls.
```

---

## 6. Dispatcher Implementation

### 6.1 Core Dispatcher

```python
"""
flux_orchestrator.py — VLM-driven component orchestration.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import re
from dataclasses import dataclass


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
    wave: Optional[Tensor] = None  # For tools that produce waves
    error: Optional[str] = None


class FLUXOrchestrator(nn.Module):
    """
    Dispatches VLM tool calls to FLUX components.
    
    The VLM generates text with <tool>...</tool> tags.
    This class parses those calls and executes them on the
    appropriate FLUX component, returning results for the
    VLM to incorporate.
    """
    
    # Tool → Component mapping
    TOOL_REGISTRY = {
        # Perception
        'encode_text': ('cse', 'encode'),
        'encode_grid': ('adapters.grid_to_wave', 'encode'),
        
        # Knowledge
        'query_field': ('field', 'query'),
        'recall_memory': ('memory.episodic', 'search'),
        'store_memory': ('memory.episodic', 'store'),
        
        # Reasoning
        'predict_effect': ('causal_tracker', 'predict'),
        'get_applicable_rules': ('rule_inducer', 'match_rules'),
        'trace_causality': ('causal_tracker', 'trace_back'),
        
        # Exploration
        'get_curiosity_map': ('spatial_memory', 'get_curiosity'),
        'mark_explored': ('spatial_memory', 'update'),
        
        # CGN
        'query_cgn': ('causal.cgn', 'query'),
        'fire_cgn': ('causal.cgn', 'fire'),
        'add_causal_arrow': ('causal.graph', 'add_link'),
        
        # Generation
        'decode_grid': ('adapters.wave_to_grid', 'decode'),
    }
    
    def __init__(
        self,
        flx_model: 'FLUXModel',
        device: str = 'cuda',
    ):
        super().__init__()
        self.device = device
        self.flx = flx_model
        
        # Build component lookup
        self.components = self._build_component_map(flx_model)
        
        # Context accumulator
        self.context_waves: List[Tensor] = []
        self.last_wave: Optional[Tensor] = None
        self.last_grid: Optional[List[List[int]]] = None
        
        # Tool call history for this turn
        self.history: List[Tuple[ToolCall, ToolResult]] = []
    
    def _build_component_map(self, flx) -> Dict[str, Any]:
        """Build a flat map of component paths to instances."""
        components = {}
        
        # Direct components
        for name in ['cse', 'field', 'causal_tracker', 'rule_inducer', 
                     'spatial_memory']:
            if hasattr(flx, name):
                components[name] = getattr(flx, name)
        
        # Nested components
        if hasattr(flx, 'memory'):
            components['memory.episodic'] = flx.memory.episodic
            components['memory.working'] = flx.memory.working
            components['memory.semantic'] = flx.memory.semantic
        
        if hasattr(flx, 'adapters'):
            components['adapters.grid_to_wave'] = flx.adapters.get('grid_to_wave')
            components['adapters.wave_to_grid'] = flx.adapters.get('wave_to_grid')
        
        if hasattr(flx, 'causal'):
            components['causal.cgn'] = flx.causal.get('cgn')
            components['causal.graph'] = flx.causal.get('graph')
        
        return components
    
    def parse_tool_calls(self, vlm_output: str) -> List[ToolCall]:
        """
        Extract tool calls from VLM output.
        
        Format:
            <tool>tool_name</tool>
            <params>{"param": "value"}</params>
        """
        calls = []
        
        # Find all <tool>...</tool> blocks
        tool_pattern = r'<tool>(.*?)</tool>\s*<params>(.*?)</params>'
        matches = re.findall(tool_pattern, vlm_output, re.DOTALL)
        
        for tool_name, params_str in matches:
            tool_name = tool_name.strip()
            
            # Replace special variables
            params_str = self._substitute_variables(params_str)
            
            try:
                params = json.loads(params_str)
            except json.JSONDecodeError:
                params = {}
            
            calls.append(ToolCall(
                name=tool_name,
                params=params,
                raw=f"<tool>{tool_name}</tool><params>{params_str}</params>"
            ))
        
        return calls
    
    def _substitute_variables(self, params_str: str) -> str:
        """Replace $LAST_WAVE, $LAST_GRID, etc. with actual values."""
        # $LAST_WAVE → reference marker
        if '$LAST_WAVE' in params_str:
            params_str = params_str.replace('"$LAST_WAVE"', '"__LAST_WAVE__"')
        if '$LAST_GRID' in params_str:
            params_str = params_str.replace('"$LAST_GRID"', '"__LAST_GRID__"')
        if '$INPUT_GRID' in params_str:
            params_str = params_str.replace('"$INPUT_GRID"', '"__INPUT_GRID__"')
        return params_str
    
    def execute_tool(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        if call.name not in self.TOOL_REGISTRY:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown tool: {call.name}"
            )
        
        component_path, method_name = self.TOOL_REGISTRY[call.name]
        component = self.components.get(component_path)
        
        if component is None:
            return ToolResult(
                success=False,
                output=None,
                error=f"Component not found: {component_path}"
            )
        
        # Resolve variable references in params
        params = self._resolve_params(call.params)
        
        try:
            # Get method
            method = getattr(component, method_name, None)
            if method is None:
                # Try forward() for nn.Module components
                method = component.forward if hasattr(component, 'forward') else None
            
            if method is None:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Method not found: {method_name}"
                )
            
            # Execute
            result = method(**params)
            
            # Track waves for chaining
            if isinstance(result, Tensor) and result.shape[-1] == 432:
                self.last_wave = result
                self.context_waves.append(result)
            
            # Track grids
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                self.last_grid = result
            
            return ToolResult(
                success=True,
                output=result,
                wave=self.last_wave,
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    def _resolve_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve __LAST_WAVE__ etc. to actual values."""
        resolved = {}
        for key, value in params.items():
            if value == "__LAST_WAVE__":
                resolved[key] = self.last_wave
            elif value == "__LAST_GRID__":
                resolved[key] = self.last_grid
            elif value == "__INPUT_GRID__":
                resolved[key] = self.input_grid
            else:
                resolved[key] = value
        return resolved
    
    def execute_all(self, vlm_output: str) -> List[ToolResult]:
        """Parse and execute all tool calls in VLM output."""
        calls = self.parse_tool_calls(vlm_output)
        results = []
        
        for call in calls:
            result = self.execute_tool(call)
            self.history.append((call, result))
            results.append(result)
        
        return results
    
    def format_results_for_vlm(self, results: List[ToolResult]) -> str:
        """Format tool results as context for VLM continuation."""
        lines = []
        
        for i, result in enumerate(results):
            call = self.history[-(len(results) - i)][0]
            
            if result.success:
                # Format output based on type
                if isinstance(result.output, Tensor):
                    output_str = f"Tensor shape {list(result.output.shape)}"
                elif isinstance(result.output, list):
                    if len(result.output) > 10:
                        output_str = f"List with {len(result.output)} items"
                    else:
                        output_str = str(result.output)
                else:
                    output_str = str(result.output)
                
                lines.append(f"[{call.name}] ✓ {output_str}")
            else:
                lines.append(f"[{call.name}] ✗ Error: {result.error}")
        
        return "\n".join(lines)
    
    def reset_turn(self):
        """Reset context for a new conversation turn."""
        self.context_waves = []
        self.last_wave = None
        self.last_grid = None
        self.history = []
    
    def get_context_for_vlm(self) -> Optional[Tensor]:
        """Get accumulated context waves to inject into VLM."""
        if not self.context_waves:
            return None
        
        # Stack and average context waves
        stacked = torch.stack(self.context_waves, dim=0)
        return stacked.mean(dim=0)  # [432] or [seq, 432]
```

### 6.2 Orchestrated Inference Loop

```python
def orchestrated_inference(
    orchestrator: FLUXOrchestrator,
    vlm: 'EmbeddedVLM',
    user_input: str,
    image: Optional[Tensor] = None,
    grid: Optional[List[List[int]]] = None,
    max_iterations: int = 5,
) -> str:
    """
    Run inference with VLM orchestrating FLUX tools.
    
    The loop:
    1. VLM generates response (may include tool calls)
    2. Parse and execute tool calls
    3. Inject results back into context
    4. Continue until VLM produces final answer (no more tools)
    """
    orchestrator.reset_turn()
    
    # Set input context
    if grid is not None:
        orchestrator.input_grid = grid
    
    # Build initial prompt
    messages = [
        {"role": "system", "content": FLUX_SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    
    if image is not None:
        messages[-1]["image"] = image
    
    iteration = 0
    final_response = ""
    
    while iteration < max_iterations:
        # Generate VLM response
        response = vlm.generate(
            messages,
            context_wave=orchestrator.get_context_for_vlm(),
        )
        
        # Check for tool calls
        tool_calls = orchestrator.parse_tool_calls(response)
        
        if not tool_calls:
            # No tools = final answer
            final_response = response
            break
        
        # Execute tools
        results = orchestrator.execute_all(response)
        
        # Format results for next iteration
        tool_context = orchestrator.format_results_for_vlm(results)
        
        # Add to conversation
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "system", "content": f"Tool results:\n{tool_context}"})
        
        iteration += 1
    
    return final_response
```

---

## 7. Example Flows

### 7.1 ARC Puzzle Solving

```
User: <image: 3x3 grid with pattern>
      What's the output for this ARC puzzle?

VLM: I see an input grid. Let me analyze it.

<tool>encode_grid</tool>
<params>{"grid": [[0,1,0],[1,2,1],[0,1,0]], "mode": "holistic"}</params>

[encode_grid] ✓ Tensor shape [432]

<tool>query_field</tool>
<params>{"wave": "$LAST_WAVE", "top_k": 3}</params>

[query_field] ✓ 3 matches: rotation_pattern (0.89), mirror_pattern (0.67), color_shift (0.45)

<tool>get_applicable_rules</tool>
<params>{"trigger_color": 2, "trigger_action": 0}</params>

[get_applicable_rules] ✓ Rule #7: "Red (2) at center → 90° clockwise rotation" (confidence: 0.91)

Applying Rule #7: 90° clockwise rotation...

<tool>decode_grid</tool>
<params>{"wave": "$LAST_WAVE", "grid_size": [3, 3]}</params>

[decode_grid] ✓ [[0,1,0],[1,2,1],[0,1,0]]

The output grid is:
```
0 1 0
1 2 1  →  0 1 0
0 1 0      1 2 1
           0 1 0
```

The pattern has 90° rotational symmetry, so the output is identical to the input.
```

### 7.2 Knowledge Query with Memory

```
User: What did we learn about blue cells in the last puzzle?

VLM: Let me check my memory.

<tool>recall_memory</tool>
<params>{"query": "blue cells puzzle rules", "limit": 5}</params>

[recall_memory] ✓ 2 matches:
  - "Blue (1) cells propagate to adjacent empty cells" (importance: 0.85, 10 min ago)
  - "Blue borders indicate boundaries" (importance: 0.6, 2 hours ago)

Based on my memory, I learned two things about blue cells:

1. **Propagation rule**: Blue cells (color 1) spread to neighboring empty cells
2. **Border marking**: Blue is used to indicate boundaries in some puzzles

The more recent and important memory suggests the propagation behavior is the key pattern.
```

### 7.3 Exploration-Driven Action

```
User: <image: partially explored game grid>
      Where should I explore next?

VLM: Let me check the curiosity map.

<tool>get_curiosity_map</tool>
<params>{"grid_size": [10, 10]}</params>

[get_curiosity_map] ✓ Heatmap with high curiosity at (7,2), (8,3), (9,4)

<tool>predict_effect</tool>
<params>{"action": 4, "position": [5, 5], "grid": $INPUT_GRID}</params>

[predict_effect] ✓ Moving right would position you near high-curiosity zone

The curiosity map shows unexplored regions in the bottom-left (positions 7-9, columns 2-4). 
I recommend moving **RIGHT** then **DOWN** to reach these areas.

The brightest spot is at (7, 2) — this might contain:
- An unseen interaction point
- A puzzle element you haven't triggered
- A collectible or goal

<tool>mark_explored</tool>
<params>{"position": [5, 5], "novelty": 0.3}</params>

[mark_explored] ✓ Updated exploration mass

Current position marked as explored. Head toward (7, 2) for maximum discovery.
```

---

## 8. Integration with v5.0-vlm-embedded

### 8.1 Loading the Orchestrator

```python
from flux_model import FLUXModel
from flux_orchestrator import FLUXOrchestrator, orchestrated_inference

# Load the vlm-embedded model
model = FLUXModel.load('checkpoints/Flux-Apex-V1.flx')

# Create orchestrator
orchestrator = FLUXOrchestrator(model, device='cuda')

# Run orchestrated inference
response = orchestrated_inference(
    orchestrator=orchestrator,
    vlm=model.vlm,
    user_input="Solve this ARC puzzle",
    grid=[[0,1,0],[1,2,1],[0,1,0]],
)
```

### 8.2 Component Requirements

For orchestration to work, these components must be loadable from .flx:

| Component | Required For | Status |
|-----------|--------------|--------|
| `cse` | encode_text | ✓ In v5.0 |
| `field` | query_field | ✓ In v5.0 |
| `memory.episodic` | recall/store_memory | ✓ In v5.0 |
| `adapters.grid_to_wave` | encode_grid | ✓ In v5.0 |
| `adapters.wave_to_grid` | decode_grid | ✓ In v5.0 |
| `causal_tracker` | predict_effect, trace | ✓ In v5.0 |
| `rule_inducer` | get_applicable_rules | ✓ In v5.0 (as learned_rules) |
| `spatial_memory` | curiosity tools | ✓ In v5.0 |
| `vlm` | generate_text, vision | ✓ In v5.0 |

---

## 9. Version Update

After implementing orchestration:

```python
# Update model version
model.state['version'] = '5.1-orchestrated'
model.state['phase'] = 'phase_orchestrator'

model.state['runtime_config']['orchestration'] = {
    'enabled': True,
    'max_tool_iterations': 5,
    'tool_timeout_ms': 5000,
    'context_injection': True,
}

model.state['metadata']['capabilities'].append('tool_use')
model.state['metadata']['capabilities'].append('multi_step_reasoning')

model.save('checkpoints/Flux-Apex-V1.flx', overwrite=True)
```

---

## 10. Files to Create

```
phases/phase_orchestrator/
├── PHASE_ORCHESTRATOR_SPEC.md      ← This file
├── flux_orchestrator.py            ← FLUXOrchestrator class
├── tool_registry.py                ← Tool definitions
├── system_prompt.py                ← VLM system prompt for tool use
├── inference_loop.py               ← orchestrated_inference function
├── demo_orchestrator_demo1.py      ← Demo: ARC puzzle with tools
├── demo_orchestrator_demo2.py      ← Demo: Memory + exploration
├── test_orchestrator_test1.py      ← Test: Tool parsing
├── test_orchestrator_test2.py      ← Test: Dispatch execution
├── test_orchestrator_test3.py      ← Test: End-to-end orchestration
└── RESULTS_PHASE_ORCHESTRATOR.md   ← Auto-generated results
```

---

## 11. Success Criteria

| Test | Requirement |
|------|-------------|
| Tool Parsing | Parse 95%+ of valid <tool> tags |
| Dispatch | Execute all TOOL_REGISTRY tools without error |
| Chaining | $LAST_WAVE properly passed between tools |
| Context | VLM receives accumulated context waves |
| ARC Solve | Complete at least one ARC puzzle via tool chain |
| Memory | Store and recall facts across conversation |
| Iteration | VLM terminates within max_iterations |

---

## 12. Future Extensions

1. **Parallel Tool Calls**: Execute independent tools concurrently
2. **Tool Learning**: VLM learns which tools work best for which tasks
3. **Custom Tools**: User-defined tools registered at runtime
4. **Tool Composition**: Define macro tools that chain multiple primitives
5. **Streaming**: Stream tool results as they complete
6. **Undo/Retry**: Rollback tool effects if VLM detects error

---

*The VLM becomes the brain. FLUX components become its cognitive tools.*
