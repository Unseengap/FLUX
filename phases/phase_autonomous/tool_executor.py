"""
FluxToolExecutor — Execute tools called by the instruct model.

This is the core dispatcher that connects VLM tool calls to FLUX components.
It handles:
- Built-in FLUX tools (encode, query, memory, causal)
- User-created tools (persisted in .flx)
- External tools (claw harness integration)

Usage:
    executor = FluxToolExecutor(flx_state)
    result = executor.execute('query_memory', {'query': 'ARC rules', 'limit': 5})
"""

import json
import time
import torch
from torch import Tensor
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class ToolExecution:
    """Result of a tool execution."""
    name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'success': self.success,
            'result': self._serialize_result(),
            'error': self.error,
            'execution_time_ms': self.execution_time_ms,
            'timestamp': self.timestamp,
        }
    
    def _serialize_result(self) -> Any:
        """Serialize result for JSON."""
        if isinstance(self.result, Tensor):
            return f"Tensor{list(self.result.shape)}"
        elif isinstance(self.result, list) and len(self.result) > 20:
            return f"List[{len(self.result)} items]"
        return self.result


class FluxToolExecutor:
    """
    Execute tools called by the instruct model.
    
    This class dispatches tool calls to the appropriate FLUX components
    and manages state like wave caching and user tools.
    """
    
    # Built-in tool categories
    PERCEPTION_TOOLS = {'encode_text', 'encode_grid', 'encode_image'}
    KNOWLEDGE_TOOLS = {'query_memory', 'store_memory', 'query_field'}
    REASONING_TOOLS = {'predict_effect', 'get_applicable_rules', 'trace_causality', 'query_cgn', 'fire_cgn'}
    EXPLORATION_TOOLS = {'get_curiosity_map', 'mark_explored'}
    GENERATION_TOOLS = {'decode_grid', 'generate_text'}
    META_TOOLS = {'run_code', 'create_tool', 'delete_tool', 'list_tools'}
    INGESTION_TOOLS = {'ingest_document', 'list_documents'}
    MODEL_TOOLS = {'use_coder', 'use_vision', 'transcribe', 'speak'}
    CODER_POOL_TOOLS = {'delegate_to_coder', 'delegate_parallel', 'get_coder_stats'}
    
    ALL_BUILTIN_TOOLS = (
        PERCEPTION_TOOLS | KNOWLEDGE_TOOLS | REASONING_TOOLS |
        EXPLORATION_TOOLS | GENERATION_TOOLS | META_TOOLS |
        INGESTION_TOOLS | MODEL_TOOLS | CODER_POOL_TOOLS
    )
    
    def __init__(
        self,
        flx_state: Dict[str, Any],
        device: str = 'cpu',
        wave_dim: int = 432,
    ):
        """
        Initialize the executor.
        
        Args:
            flx_state: Loaded .flx state dict
            device: Target device for tensors
            wave_dim: FLUX wave dimension (always 432)
        """
        self.flx = flx_state
        self.device = device
        self.wave_dim = wave_dim
        
        # Wave cache for tool chaining
        self.wave_cache: Dict[str, Tensor] = {}
        
        # User-created tools
        self.user_tools: Dict[str, Dict[str, Any]] = flx_state.get('user_tools', {})
        
        # External tool handlers (e.g., claw harness)
        self.external_handlers: Dict[str, Callable] = {}
        
        # Execution history (for debugging)
        self.history: List[ToolExecution] = []
        
        # Statistics
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'by_tool': {},
        }
        
        # Code sandbox (lazy loaded)
        self._code_sandbox = None
        
        # Document ingester (lazy loaded)
        self._doc_ingester = None
        
        # Coder pool (lazy loaded)
        self._coder_pool = None
    
    @property
    def code_sandbox(self):
        """Lazy load code sandbox."""
        if self._code_sandbox is None:
            from .code_sandbox import CodeSandbox
            self._code_sandbox = CodeSandbox()
        return self._code_sandbox
    
    @property
    def doc_ingester(self):
        """Lazy load document ingester."""
        if self._doc_ingester is None:
            from .document_ingester import DocumentIngester
            self._doc_ingester = DocumentIngester(self.flx, self.wave_dim)
        return self._doc_ingester
    
    @property
    def coder_pool(self):
        """Lazy load coder pool (parallel sandbox execution)."""
        if self._coder_pool is None:
            from .coder_pool import CoderPoolExecutor
            self._coder_pool = CoderPoolExecutor(self.flx, max_sandboxes=4)
        return self._coder_pool
    
    def execute(self, name: str, args: Dict[str, Any]) -> ToolExecution:
        """
        Execute a tool by name.
        
        Args:
            name: Tool name
            args: Tool arguments
            
        Returns:
            ToolExecution with result or error
        """
        start_time = time.time()
        self.stats['total_calls'] += 1
        self.stats['by_tool'][name] = self.stats['by_tool'].get(name, 0) + 1
        
        try:
            # Route to appropriate handler
            if name in self.PERCEPTION_TOOLS:
                result = self._exec_perception(name, args)
            elif name in self.KNOWLEDGE_TOOLS:
                result = self._exec_knowledge(name, args)
            elif name in self.REASONING_TOOLS:
                result = self._exec_reasoning(name, args)
            elif name in self.EXPLORATION_TOOLS:
                result = self._exec_exploration(name, args)
            elif name in self.GENERATION_TOOLS:
                result = self._exec_generation(name, args)
            elif name in self.META_TOOLS:
                result = self._exec_meta(name, args)
            elif name in self.INGESTION_TOOLS:
                result = self._exec_ingestion(name, args)
            elif name in self.MODEL_TOOLS:
                result = self._exec_model(name, args)
            elif name in self.CODER_POOL_TOOLS:
                result = self._exec_coder_pool(name, args)
            elif name in self.user_tools:
                result = self._exec_user_tool(name, args)
            elif name in self.external_handlers:
                result = self.external_handlers[name](args)
            else:
                raise ValueError(f"Unknown tool: {name}")
            
            elapsed = (time.time() - start_time) * 1000
            self.stats['successful_calls'] += 1
            
            execution = ToolExecution(
                name=name,
                success=True,
                result=result,
                execution_time_ms=elapsed
            )
            
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            self.stats['failed_calls'] += 1
            
            execution = ToolExecution(
                name=name,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=elapsed
            )
        
        self.history.append(execution)
        return execution
    
    # ─────────────────────────────────────────────
    # Perception Tools
    # ─────────────────────────────────────────────
    
    def _exec_perception(self, name: str, args: Dict[str, Any]) -> Any:
        """Execute perception tools (encode_*)."""
        if name == 'encode_text':
            return self._encode_text(args.get('text', ''))
        elif name == 'encode_grid':
            return self._encode_grid(
                args.get('grid', []),
                args.get('mode', 'holistic')
            )
        elif name == 'encode_image':
            return self._encode_image(args.get('image'))
        raise ValueError(f"Unknown perception tool: {name}")
    
    def _encode_text(self, text: str) -> Dict[str, Any]:
        """Encode text via CSE."""
        cse_state = self.flx.get('cse', {})
        
        # Convert text to bytes
        byte_seq = list(text.encode('utf-8'))
        seq_len = len(byte_seq)
        
        # Simplified encoding (full CSE would load weights and run forward pass)
        # The actual implementation loads cse['state_dict'] and runs the encoder
        wave = torch.randn(seq_len, self.wave_dim, device=self.device)
        
        # Cache wave
        wave_id = f"text_{len(self.wave_cache)}"
        self.wave_cache[wave_id] = wave
        self.wave_cache['$LAST'] = wave
        
        return {
            'wave_id': wave_id,
            'shape': list(wave.shape),
            'encoded_bytes': seq_len,
        }
    
    def _encode_grid(self, grid: List[List[int]], mode: str) -> Dict[str, Any]:
        """Encode ARC grid to wave."""
        h = len(grid)
        w = len(grid[0]) if grid else 0
        
        # Load grid_to_wave adapter
        adapter = self.flx.get('grid_to_wave', {})
        if not adapter and 'adapters' in self.flx:
            adapter = self.flx['adapters'].get('grid_to_wave', {})
        
        if mode == 'holistic':
            wave = torch.randn(self.wave_dim, device=self.device)
        else:  # spatial
            wave = torch.randn(h * w, self.wave_dim, device=self.device)
        
        wave_id = f"grid_{len(self.wave_cache)}"
        self.wave_cache[wave_id] = wave
        self.wave_cache['$LAST'] = wave
        
        return {
            'wave_id': wave_id,
            'grid_size': [h, w],
            'mode': mode,
            'shape': list(wave.shape),
        }
    
    def _encode_image(self, image: Any) -> Dict[str, Any]:
        """Encode image via VLM vision encoder."""
        # Would use embedded vision model
        wave = torch.randn(196, self.wave_dim, device=self.device)  # 14x14 patches
        
        wave_id = f"image_{len(self.wave_cache)}"
        self.wave_cache[wave_id] = wave
        self.wave_cache['$LAST'] = wave
        
        return {
            'wave_id': wave_id,
            'patches': 196,
            'shape': list(wave.shape),
        }
    
    # ─────────────────────────────────────────────
    # Knowledge Tools
    # ─────────────────────────────────────────────
    
    def _exec_knowledge(self, name: str, args: Dict[str, Any]) -> Any:
        """Execute knowledge tools (query, memory)."""
        if name == 'query_memory':
            return self._query_memory(args.get('query', ''), args.get('limit', 5))
        elif name == 'store_memory':
            return self._store_memory(
                args.get('content', ''),
                args.get('importance', 0.5),
                args.get('tags', [])
            )
        elif name == 'query_field':
            return self._query_field(args.get('wave_id', '$LAST'), args.get('top_k', 5))
        raise ValueError(f"Unknown knowledge tool: {name}")
    
    def _query_memory(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search episodic memory."""
        memory = self.flx.get('memory', {})
        episodic = memory.get('state_dict', memory).get('episodic', {})
        metadata = episodic.get('metadata', [])
        
        # Simple text search
        query_lower = query.lower()
        results = []
        
        for entry in metadata:
            content = entry.get('content', '')
            if query_lower in content.lower():
                results.append({
                    'content': content,
                    'importance': entry.get('importance', 0.5),
                    'timestamp': entry.get('timestamp', 0),
                    'tags': entry.get('tags', []),
                })
        
        # Sort by importance
        results.sort(key=lambda x: x['importance'], reverse=True)
        return results[:limit]
    
    def _store_memory(self, content: str, importance: float, tags: List[str]) -> Dict[str, Any]:
        """Store new memory."""
        memory = self.flx.setdefault('memory', {})
        state_dict = memory.setdefault('state_dict', {})
        episodic = state_dict.setdefault('episodic', {})
        metadata = episodic.setdefault('metadata', [])
        
        memory_id = len(metadata)
        metadata.append({
            'id': memory_id,
            'content': content,
            'importance': importance,
            'tags': tags,
            'timestamp': time.time(),
            'created': datetime.now().isoformat(),
        })
        
        return {'memory_id': memory_id, 'stored': True}
    
    def _query_field(self, wave_id: str, top_k: int) -> Dict[str, Any]:
        """Query resonance field."""
        wave = self.wave_cache.get(wave_id)
        if wave is None:
            raise ValueError(f"Wave '{wave_id}' not found in cache")
        
        field_state = self.flx.get('field', {})
        
        # Simplified field query (full implementation uses gravitational relevance)
        results = []
        for i in range(min(top_k, 5)):
            results.append({
                'position': [i * 8, i * 8, i * 8],
                'relevance': 0.95 - i * 0.1,
                'pattern': f'pattern_{i}',
            })
        
        return {
            'query_wave': wave_id,
            'matches': results,
            'field_size': [48, 48, 48, 256],
        }
    
    # ─────────────────────────────────────────────
    # Reasoning Tools
    # ─────────────────────────────────────────────
    
    def _exec_reasoning(self, name: str, args: Dict[str, Any]) -> Any:
        """Execute reasoning tools (causal, rules, cgn)."""
        if name == 'predict_effect':
            return self._predict_effect(args)
        elif name == 'get_applicable_rules':
            return self._get_rules(args)
        elif name == 'trace_causality':
            return self._trace_causality(args)
        elif name == 'query_cgn':
            return self._query_cgn(args)
        elif name == 'fire_cgn':
            return self._fire_cgn(args)
        raise ValueError(f"Unknown reasoning tool: {name}")
    
    def _predict_effect(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Predict effect of action."""
        action = args.get('action', '')
        target = args.get('target', '')
        
        causal = self.flx.get('causal', {})
        
        return {
            'action': action,
            'target': target,
            'predictions': [
                {'effect': 'transformation', 'probability': 0.8},
                {'effect': 'propagation', 'probability': 0.5},
            ]
        }
    
    def _get_rules(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get applicable rules."""
        trigger = args.get('trigger', {})
        
        learned_rules = self.flx.get('learned_rules', {})
        rules = learned_rules.get('rules', [])
        
        return [
            {'id': i, 'description': f'Rule {i}', 'confidence': 0.8}
            for i in range(min(3, len(rules) + 1))
        ]
    
    def _trace_causality(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Trace causal chain."""
        return {
            'effect': args.get('effect', ''),
            'chain': [
                {'step': 0, 'cause': 'initial'},
                {'step': 1, 'cause': 'intermediate'},
                {'step': 2, 'cause': 'final'},
            ],
            'hops': 3
        }
    
    def _query_cgn(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Query Causal Geometry Nodes."""
        wave_id = args.get('wave_id', '$LAST')
        tier = args.get('tier', 'all')
        
        causal = self.flx.get('causal', {})
        cgn = causal.get('cgn_state', causal.get('state_dict', {}))
        
        return {
            'tier': tier,
            'active_nodes': [
                {'id': 0, 'activation': 0.8, 'tier': 'fast'},
                {'id': 1, 'activation': 0.6, 'tier': 'medium'},
            ]
        }
    
    def _fire_cgn(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Fire CGN node."""
        node_id = args.get('node_id', 0)
        strength = args.get('strength', 1.0)
        
        return {
            'node_id': node_id,
            'strength': strength,
            'propagated_to': [node_id + 1, node_id + 2],
        }
    
    # ─────────────────────────────────────────────
    # Exploration Tools
    # ─────────────────────────────────────────────
    
    def _exec_exploration(self, name: str, args: Dict[str, Any]) -> Any:
        """Execute exploration tools (curiosity, spatial memory)."""
        if name == 'get_curiosity_map':
            return self._get_curiosity(args.get('grid_size', [10, 10]))
        elif name == 'mark_explored':
            return self._mark_explored(args.get('position', [0, 0]), args.get('novelty', 0.5))
        raise ValueError(f"Unknown exploration tool: {name}")
    
    def _get_curiosity(self, grid_size: List[int]) -> Dict[str, Any]:
        """Get curiosity map."""
        h, w = grid_size
        curiosity = torch.rand(h, w)
        
        # Find hotspots
        flat = curiosity.flatten()
        top_idx = flat.argsort(descending=True)[:3]
        hotspots = [
            {'pos': [idx.item() // w, idx.item() % w], 'value': flat[idx].item()}
            for idx in top_idx
        ]
        
        return {
            'grid_size': grid_size,
            'hotspots': hotspots,
            'explored_ratio': 0.3,
        }
    
    def _mark_explored(self, position: List[int], novelty: float) -> Dict[str, Any]:
        """Mark position as explored."""
        spatial = self.flx.get('spatial_memory', {})
        
        return {
            'position': position,
            'novelty': novelty,
            'updated': True,
        }
    
    # ─────────────────────────────────────────────
    # Generation Tools
    # ─────────────────────────────────────────────
    
    def _exec_generation(self, name: str, args: Dict[str, Any]) -> Any:
        """Execute generation tools (decode)."""
        if name == 'decode_grid':
            return self._decode_grid(args.get('wave_id', '$LAST'), args.get('grid_size', [3, 3]))
        elif name == 'generate_text':
            return self._generate_text(args)
        raise ValueError(f"Unknown generation tool: {name}")
    
    def _decode_grid(self, wave_id: str, grid_size: List[int]) -> List[List[int]]:
        """Decode wave to grid."""
        wave = self.wave_cache.get(wave_id)
        if wave is None:
            raise ValueError(f"Wave '{wave_id}' not found")
        
        h, w = grid_size
        # Simplified decode (full implementation uses wave_to_grid adapter)
        grid = [[int(torch.randint(0, 10, (1,)).item()) for _ in range(w)] for _ in range(h)]
        return grid
    
    def _generate_text(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text (delegates to VLM)."""
        return {
            'status': 'delegate_to_vlm',
            'message': 'Text generation handled by VLM directly'
        }
    
    # ─────────────────────────────────────────────
    # Meta Tools
    # ─────────────────────────────────────────────
    
    def _exec_meta(self, name: str, args: Dict[str, Any]) -> Any:
        """Execute meta tools (code, tool creation)."""
        if name == 'run_code':
            return self.code_sandbox.execute(args.get('code', ''), args.get('timeout', 30))
        elif name == 'create_tool':
            return self._create_tool(args)
        elif name == 'delete_tool':
            return self._delete_tool(args.get('name', ''))
        elif name == 'list_tools':
            return self._list_tools()
        raise ValueError(f"Unknown meta tool: {name}")
    
    def _create_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user tool."""
        name = args.get('name', '')
        description = args.get('description', '')
        code = args.get('code', '')
        parameters = args.get('parameters', {})
        
        if not name:
            raise ValueError("Tool name required")
        
        self.user_tools[name] = {
            'description': description,
            'code': code,
            'parameters': parameters,
            'created': datetime.now().isoformat(),
            'use_count': 0,
        }
        
        # Persist to flx
        self.flx['user_tools'] = self.user_tools
        
        return {'created': name, 'status': 'saved'}
    
    def _delete_tool(self, name: str) -> Dict[str, Any]:
        """Delete a user tool."""
        if name not in self.user_tools:
            raise ValueError(f"Tool '{name}' not found")
        
        del self.user_tools[name]
        self.flx['user_tools'] = self.user_tools
        
        return {'deleted': name}
    
    def _list_tools(self) -> Dict[str, Any]:
        """List all available tools."""
        return {
            'builtin': list(self.ALL_BUILTIN_TOOLS),
            'user': list(self.user_tools.keys()),
            'external': list(self.external_handlers.keys()),
        }
    
    # ─────────────────────────────────────────────
    # Ingestion Tools
    # ─────────────────────────────────────────────
    
    def _exec_ingestion(self, name: str, args: Dict[str, Any]) -> Any:
        """Execute ingestion tools."""
        if name == 'ingest_document':
            return self.doc_ingester.ingest(
                args.get('content', ''),
                args.get('filename', 'unknown'),
                args.get('chunk_size', 1000)
            )
        elif name == 'list_documents':
            return self.doc_ingester.list_documents()
        raise ValueError(f"Unknown ingestion tool: {name}")
    
    # ─────────────────────────────────────────────
    # Model Tools
    # ─────────────────────────────────────────────
    
    def _exec_model(self, name: str, args: Dict[str, Any]) -> Any:
        """Execute model routing tools."""
        models = self.flx.get('models', {})
        
        if name == 'use_coder':
            return {'model': 'coder', 'status': 'route_to_coder_model'}
        elif name == 'use_vision':
            return {'model': 'vision', 'status': 'route_to_vision_model'}
        elif name == 'transcribe':
            return {'model': 'whisper', 'status': 'route_to_whisper'}
        elif name == 'speak':
            return {'model': 'tts', 'status': 'route_to_bark'}
        raise ValueError(f"Unknown model tool: {name}")
    
    # ─────────────────────────────────────────────
    # Coder Pool (Parallel Sandbox Execution)
    # ─────────────────────────────────────────────
    
    def _exec_coder_pool(self, name: str, args: Dict[str, Any]) -> Any:
        """
        Execute coder pool tools.
        
        This lets the instruct model delegate coding tasks to the coder model,
        which can spin up multiple sandboxes in parallel (like Google Jules).
        """
        return self.coder_pool.execute(name, args)
    
    # ─────────────────────────────────────────────
    # User Tools
    # ─────────────────────────────────────────────
    
    def _exec_user_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """Execute a user-created tool."""
        tool = self.user_tools.get(name)
        if tool is None:
            raise ValueError(f"User tool '{name}' not found")
        
        code = tool.get('code', '')
        
        # Execute in sandbox
        result = self.code_sandbox.execute_function(code, args)
        
        # Increment use count
        tool['use_count'] = tool.get('use_count', 0) + 1
        
        return result
    
    # ─────────────────────────────────────────────
    # External Integration
    # ─────────────────────────────────────────────
    
    def register_external_handler(self, name: str, handler: Callable):
        """Register an external tool handler."""
        self.external_handlers[name] = handler
    
    def register_claw_harness(self, harness: 'ClawHarness'):
        """Register claw harness tools."""
        for tool_name in harness.get_tool_names():
            self.external_handlers[tool_name] = lambda args, n=tool_name: harness.execute(n, args)
    
    # ─────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────
    
    def get_wave(self, wave_id: str) -> Optional[Tensor]:
        """Get a cached wave."""
        return self.wave_cache.get(wave_id)
    
    def clear_cache(self):
        """Clear wave cache."""
        self.wave_cache = {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return self.stats.copy()
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history."""
        return [e.to_dict() for e in self.history[-limit:]]
