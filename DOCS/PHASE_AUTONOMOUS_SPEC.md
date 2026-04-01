# FLUX Autonomous Architecture — Phase 4 Specification

**Version:** 1.0  
**Status:** IN PROGRESS  
**Goal:** Self-contained, self-modifying AGI that runs from a single .flx file

> **Implementation Notebook:** `notebooks/flux_autonomous_complete.ipynb`

---

## Executive Summary

This document captures the architectural changes needed to transform FLUX from a "model with external code" into a **true autonomous cognitive system** that:

1. Runs entirely from a single `.flx` file (no external codebase)
2. Uses native JSON tool calling (no custom `<tool>` tags)
3. Creates, saves, and reuses tools dynamically
4. Sets goals and plans only when needed
5. Ingests documents/files of any format
6. Executes code when precision is required
7. Learns and evolves with use

---

## Table of Contents

1. [Current State vs Target State](#current-state-vs-target-state)
2. [Tool Calling: Native JSON Format](#tool-calling-native-json-format)
3. [Self-Contained Runtime](#self-contained-runtime)
4. [Dynamic Tool Creation](#dynamic-tool-creation)
5. [Goal & Planning System](#goal--planning-system)
6. [Document Ingestion](#document-ingestion)
7. [Code Execution](#code-execution)
8. [Implementation Phases](#implementation-phases)
9. [Codebase Cleanup](#codebase-cleanup)

---

## Current State vs Target State

### What We Have (v7.1)

```
Flux-Apex-V1.flx (14-15 GB)
├── models/           ✅ 11 embedded models
├── config/           ✅ Runtime settings
├── memories/         ✅ Field, episodic, semantic
├── orchestration/    ⚠️ Custom <tool> format (broken)
└── code/             ❌ External (needs codebase repo)
```

**Problems:**
- Needs external Python codebase to run
- Custom `<tool>` tags don't work with instruct models
- No dynamic tool creation
- No document ingestion
- Goals/plans are hardcoded, not learned

### What We Want (v8.0)

```
Flux-Apex-V1.flx (15-16 GB)
├── models/           ✅ All embedded models
├── config/           ✅ Runtime settings  
├── memories/         ✅ Persistent knowledge
├── tools/            ✅ JSON schema definitions (native format)
├── runtime/          ✅ All Python code embedded
│   ├── core/         ✅ flux_model, flux_utils, flux_format
│   ├── components/   ✅ CSE, field, memory, causal
│   ├── orchestrator/ ✅ Tool dispatch, model routing
│   └── bootstrap.py  ✅ 50-line self-extractor
├── user_tools/       ✅ Dynamically created tools
├── documents/        ✅ Ingested files (waves + metadata)
└── goals/            ✅ Learned goal patterns
```

**Result:** Drop the .flx file anywhere with PyTorch, run 3 lines, FLUX wakes up fully autonomous.

---

## Tool Calling: Native JSON Format

### Why Not Custom `<tool>` Tags

| Custom Tags | Native JSON |
|-------------|-------------|
| Needs fine-tuning | Already works |
| Fragile regex parsing | Structured output |
| Model doesn't know format | Model trained on format |
| Breaks with model updates | Standard across versions |

### Qwen2.5-Instruct Native Format

```python
# Define tools in JSON Schema format
FLUX_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_memory",
            "description": "Search FLUX's episodic memory for relevant facts",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "encode_to_wave",
            "description": "Convert text to 432D semantic wave using CSE",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to encode"
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_field",
            "description": "Search the 96³ resonance field for related knowledge",
            "parameters": {
                "type": "object",
                "properties": {
                    "wave": {
                        "type": "string",
                        "description": "Wave ID from previous encode, or '$LAST'"
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5
                    }
                },
                "required": ["wave"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Execute Python code for precise calculations or data processing",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Max execution time in seconds",
                        "default": 30
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_tool",
            "description": "Create a new reusable tool and save it for later",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Tool name (snake_case)"
                    },
                    "description": {
                        "type": "string",
                        "description": "What the tool does"
                    },
                    "code": {
                        "type": "string",
                        "description": "Python function body"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "JSON Schema for parameters"
                    }
                },
                "required": ["name", "description", "code", "parameters"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ingest_document",
            "description": "Process and store a document in FLUX memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Document content (text, or base64 for binary)"
                    },
                    "filename": {
                        "type": "string",
                        "description": "Original filename with extension"
                    },
                    "chunk_size": {
                        "type": "integer",
                        "default": 1000
                    }
                },
                "required": ["content", "filename"]
            }
        }
    }
]
```

### Model Response Format

```python
# Qwen responds with structured function_call
response = {
    "role": "assistant",
    "content": None,
    "function_call": {
        "name": "query_memory",
        "arguments": '{"query": "what did user say about Paris", "limit": 3}'
    }
}

# Parse and execute
import json
call = response["function_call"]
args = json.loads(call["arguments"])
result = execute_tool(call["name"], args)

# Feed result back
messages.append({
    "role": "function",
    "name": call["name"],
    "content": json.dumps(result)
})
```

### Tool Executor Implementation

```python
class FluxToolExecutor:
    """Execute tools called by the instruct model."""
    
    def __init__(self, flx_state: dict):
        self.flx = flx_state
        self.wave_cache = {}  # Store recent waves for $LAST reference
        self.user_tools = flx_state.get('user_tools', {})
    
    def execute(self, name: str, args: dict) -> dict:
        """Dispatch tool call to appropriate handler."""
        
        # Built-in tools
        if name == "query_memory":
            return self._query_memory(args)
        elif name == "encode_to_wave":
            return self._encode_wave(args)
        elif name == "query_field":
            return self._query_field(args)
        elif name == "run_code":
            return self._run_code(args)
        elif name == "create_tool":
            return self._create_tool(args)
        elif name == "ingest_document":
            return self._ingest_document(args)
        
        # User-created tools
        elif name in self.user_tools:
            return self._run_user_tool(name, args)
        
        else:
            return {"error": f"Unknown tool: {name}"}
    
    def _query_memory(self, args: dict) -> dict:
        """Search episodic memory."""
        # Use embedded all-MiniLM for query encoding
        # Search FAISS index in memory component
        query = args["query"]
        limit = args.get("limit", 5)
        
        # Implementation uses self.flx['memory']['episodic']
        results = search_episodic(self.flx, query, limit)
        return {"results": results}
    
    def _encode_wave(self, args: dict) -> dict:
        """Encode text to wave via CSE."""
        text = args["text"]
        wave = run_cse(self.flx['cse'], text)
        wave_id = f"wave_{len(self.wave_cache)}"
        self.wave_cache[wave_id] = wave
        self.wave_cache["$LAST"] = wave
        return {"wave_id": wave_id, "dim": 432}
    
    def _run_code(self, args: dict) -> dict:
        """Execute Python code in sandbox."""
        code = args["code"]
        timeout = args.get("timeout", 30)
        
        # Restricted execution environment
        allowed_modules = ['math', 'json', 're', 'datetime', 'collections']
        result = sandbox_exec(code, timeout, allowed_modules)
        return {"output": result}
    
    def _create_tool(self, args: dict) -> dict:
        """Create and save a new tool."""
        name = args["name"]
        description = args["description"]
        code = args["code"]
        parameters = args["parameters"]
        
        # Store in user_tools
        self.user_tools[name] = {
            "description": description,
            "code": code,
            "parameters": parameters,
            "created": datetime.now().isoformat()
        }
        
        # Persist to .flx
        self.flx['user_tools'] = self.user_tools
        
        return {"created": name, "status": "saved"}
```

---

## Self-Contained Runtime

### What Gets Embedded

| Category | Files | Lines | Purpose |
|----------|-------|-------|---------|
| **Core** | flux_model.py, flux_utils.py, flux_format.py | ~2,500 | Load/save, utilities |
| **Components** | cse.py, field.py, gravity.py, thermodynamic.py, cgn.py | ~3,000 | Physics engine |
| **Memory** | working_memory.py, episodic_memory.py, semantic_memory.py | ~2,000 | Three-tier memory |
| **Orchestrator** | tool_executor.py, model_router.py | ~1,000 | Tool dispatch |
| **Agent** | unified_agent.py, goal_planner.py | ~1,500 | Autonomous behavior |
| **Bootstrap** | bootstrap.py | ~100 | Self-extraction |
| **Total** | | ~10,000 | Full runtime |

### Embedding Process

```python
import ast
import torch
from pathlib import Path

def bundle_codebase(root: Path) -> dict:
    """Bundle all Python code into a dict."""
    
    code_bundle = {}
    
    # Core files
    for f in ['flux_model.py', 'flux_utils.py']:
        code_bundle[f] = (root / f).read_text()
    
    # Phase components
    phases = ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6', 'phase8']
    for phase in phases:
        phase_dir = root / 'phases' / phase
        for py_file in phase_dir.glob('*.py'):
            if not py_file.name.startswith(('test_', 'demo_', 'train_')):
                key = f"phases/{phase}/{py_file.name}"
                code_bundle[key] = py_file.read_text()
    
    # Orchestrator
    orch_dir = root / 'phases' / 'phase_orchestrator'
    for py_file in orch_dir.glob('*.py'):
        key = f"orchestrator/{py_file.name}"
        code_bundle[key] = py_file.read_text()
    
    # Validate syntax
    for name, source in code_bundle.items():
        try:
            ast.parse(source)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in {name}: {e}")
    
    return code_bundle

def embed_runtime(flx_path: Path):
    """Add runtime code to .flx file."""
    
    flx = torch.load(flx_path, map_location='cpu', weights_only=False)
    
    # Bundle code
    root = Path(__file__).parent
    code_bundle = bundle_codebase(root)
    
    # Bootstrap script
    bootstrap = '''
import sys
import types
import torch

def wake_up(flx_path: str):
    """Bootstrap FLUX from .flx file."""
    
    # Load the model
    flx = torch.load(flx_path, map_location='cpu', weights_only=False)
    
    # Extract and register all modules
    code = flx['runtime']['code']
    modules = {}
    
    for path, source in code.items():
        # Convert path to module name
        mod_name = path.replace('/', '.').replace('.py', '')
        
        # Create module
        module = types.ModuleType(mod_name)
        module.__file__ = path
        
        # Execute in module namespace
        exec(compile(source, path, 'exec'), module.__dict__)
        
        # Register
        sys.modules[mod_name] = module
        modules[mod_name] = module
    
    # Initialize orchestrator
    from orchestrator.tool_executor import FluxToolExecutor
    from orchestrator.model_router import ModelRouter
    
    executor = FluxToolExecutor(flx)
    router = ModelRouter(flx)
    
    return {
        'flx': flx,
        'executor': executor,
        'router': router,
        'modules': modules,
    }

if __name__ == '__main__':
    import sys
    flux = wake_up(sys.argv[1] if len(sys.argv) > 1 else 'Flux-Apex-V1.flx')
    print("FLUX is awake.")
'''
    
    flx['runtime'] = {
        'code': code_bundle,
        'bootstrap': bootstrap,
        'version': '8.0-autonomous',
        'embedded_at': datetime.now().isoformat(),
    }
    
    torch.save(flx, flx_path)
    print(f"✓ Runtime embedded: {len(code_bundle)} files")
```

### Running From .flx Only

```python
# This is ALL you need — no codebase, no pip install (except torch)
import torch
flx = torch.load('Flux-Apex-V1.flx', map_location='cpu', weights_only=False)
exec(flx['runtime']['bootstrap'])
flux = wake_up('Flux-Apex-V1.flx')

# Now FLUX is fully operational
response = flux['router'].chat("What do you remember about me?")
```

---

## Dynamic Tool Creation

### How FLUX Creates New Tools

```
User: "I need a tool that converts temperatures between Celsius and Fahrenheit"

FLUX (thinks): This is a reusable utility. I'll create a tool.

FLUX (calls create_tool):
{
    "name": "convert_temperature",
    "description": "Convert between Celsius and Fahrenheit",
    "code": "def convert_temperature(value, from_unit, to_unit):\n    if from_unit == 'C' and to_unit == 'F':\n        return value * 9/5 + 32\n    elif from_unit == 'F' and to_unit == 'C':\n        return (value - 32) * 5/9\n    else:\n        return value",
    "parameters": {
        "type": "object",
        "properties": {
            "value": {"type": "number"},
            "from_unit": {"type": "string", "enum": ["C", "F"]},
            "to_unit": {"type": "string", "enum": ["C", "F"]}
        },
        "required": ["value", "from_unit", "to_unit"]
    }
}

FLUX: "Done. I created 'convert_temperature'. Want me to convert something?"
```

### Tool Persistence

```python
# Tools are saved in flx['user_tools']
flx['user_tools'] = {
    'convert_temperature': {
        'description': 'Convert between Celsius and Fahrenheit',
        'code': '...',
        'parameters': {...},
        'created': '2026-04-01T10:30:00',
        'use_count': 0,
    },
    'analyze_csv': {
        'description': 'Parse and analyze CSV data',
        'code': '...',
        'parameters': {...},
        'created': '2026-04-01T11:00:00',
        'use_count': 5,
    }
}

# Auto-saved to .flx after creation
torch.save(flx, 'Flux-Apex-V1.flx')
```

---

## Goal & Planning System

### When Goals Are Created

Goals are NOT always running. They're created when:
1. User explicitly asks for something complex
2. A task requires multiple steps
3. FLUX notices a recurring pattern

### Goal Structure

```python
@dataclass
class Goal:
    id: str
    description: str
    status: str  # 'active', 'paused', 'completed', 'failed'
    priority: float  # 0.0 - 1.0
    created: str
    steps: List[Step]
    context: dict  # Why this goal exists
    triggers: List[str]  # What activates this goal

@dataclass  
class Step:
    description: str
    tool_calls: List[dict]  # Tools to execute
    status: str
    result: Any
```

### Example: Proactive Goal

```python
# FLUX notices user asks about weather every morning
goal = Goal(
    id="morning_weather",
    description="Prepare morning weather briefing",
    status="active",
    priority=0.7,
    created="2026-04-01T08:00:00",
    steps=[
        Step("Get user's location from memory", [...]),
        Step("Fetch weather data", [...]),
        Step("Summarize for user", [...]),
    ],
    context={"pattern": "user asks 'weather' 5/5 mornings"},
    triggers=["time:morning", "user:wakes_up"],
)
```

### Goal Execution

```python
class GoalPlanner:
    def __init__(self, flx: dict, executor: FluxToolExecutor):
        self.flx = flx
        self.executor = executor
        self.active_goals = flx.get('goals', {}).get('active', [])
    
    def check_triggers(self, context: dict) -> List[Goal]:
        """Check if any goals should activate."""
        triggered = []
        for goal in self.active_goals:
            if self._matches_triggers(goal, context):
                triggered.append(goal)
        return triggered
    
    def execute_goal(self, goal: Goal) -> dict:
        """Execute a goal's steps."""
        results = []
        for step in goal.steps:
            if step.status == 'completed':
                continue
            
            for tool_call in step.tool_calls:
                result = self.executor.execute(
                    tool_call['name'],
                    tool_call['args']
                )
                results.append(result)
            
            step.status = 'completed'
            step.result = results[-1]
        
        goal.status = 'completed'
        self._save_goal(goal)
        return {"goal": goal.id, "results": results}
```

---

## Document Ingestion

### Supported Formats

| Format | Handler | Output |
|--------|---------|--------|
| `.txt`, `.md` | Direct text | Waves + chunks |
| `.pdf` | PyMuPDF extract | Text → waves |
| `.docx` | python-docx | Text → waves |
| `.json` | Parse & stringify | Structured waves |
| `.csv` | pandas read | Row waves |
| `.html` | BeautifulSoup | Text → waves |
| `.py`, `.js` | Syntax-aware chunks | Code waves |
| Images | VL model encode | Visual waves |
| Audio | Whisper transcribe | Text → waves |

### Ingestion Pipeline

```python
class DocumentIngester:
    def __init__(self, flx: dict):
        self.flx = flx
        self.cse = load_cse(flx)
        self.chunker = SemanticChunker(chunk_size=1000, overlap=100)
    
    def ingest(self, content: str, filename: str) -> dict:
        """Ingest document into FLUX memory."""
        
        # Detect format
        ext = Path(filename).suffix.lower()
        
        # Extract text
        if ext == '.pdf':
            text = self._extract_pdf(content)
        elif ext in ['.txt', '.md']:
            text = content
        elif ext == '.json':
            text = self._stringify_json(content)
        else:
            text = content  # Fallback
        
        # Chunk semantically
        chunks = self.chunker.chunk(text)
        
        # Encode each chunk to wave
        waves = []
        for i, chunk in enumerate(chunks):
            wave = self.cse.encode(chunk)
            wave_id = f"doc_{filename}_{i}"
            
            # Store in episodic memory
            self._store_chunk(wave_id, chunk, wave, {
                'source': filename,
                'chunk_index': i,
                'total_chunks': len(chunks),
            })
            
            waves.append(wave_id)
        
        # Update field with document embedding
        doc_wave = self._aggregate_waves(waves)
        self._perturb_field(doc_wave, filename)
        
        return {
            'filename': filename,
            'chunks': len(chunks),
            'wave_ids': waves,
            'status': 'ingested'
        }
```

### User Adds Documents

```
User: [uploads research_paper.pdf]

FLUX: "I'll read this. Give me a moment..."

FLUX (internally):
1. ingest_document(content=base64, filename="research_paper.pdf")
2. Extract 15 pages → 23 chunks
3. Encode to waves → store in episodic memory
4. Perturb field with aggregate embedding

FLUX: "Done. I've read 'research_paper.pdf' (15 pages, 23 sections).
       Key topics: machine learning, attention mechanisms, transformers.
       Ask me anything about it."
```

---

## Code Execution

### When FLUX Runs Code

1. **Math/calculations** — Precision required
2. **Data processing** — CSV, JSON manipulation
3. **Validation** — Check if something is syntactically correct
4. **Tool creation** — Building new capabilities
5. **Complex logic** — When natural language is ambiguous

### Sandboxed Execution

```python
import ast
import sys
from io import StringIO
from typing import Any
import signal

class CodeSandbox:
    """Safe Python execution environment."""
    
    ALLOWED_MODULES = {
        'math', 'json', 're', 'datetime', 'collections',
        'itertools', 'functools', 'operator', 'string',
        'statistics', 'random', 'base64', 'hashlib',
    }
    
    FORBIDDEN_NODES = {
        ast.Import, ast.ImportFrom,  # No arbitrary imports
        ast.Delete,  # No deletions
        ast.Global, ast.Nonlocal,  # No scope manipulation
    }
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    def execute(self, code: str) -> dict:
        """Execute code safely."""
        
        # Validate AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"error": f"Syntax error: {e}"}
        
        # Check for forbidden operations
        for node in ast.walk(tree):
            if type(node) in self.FORBIDDEN_NODES:
                return {"error": f"Forbidden operation: {type(node).__name__}"}
        
        # Prepare restricted globals
        safe_globals = {
            '__builtins__': {
                'print': print, 'len': len, 'range': range,
                'int': int, 'float': float, 'str': str,
                'list': list, 'dict': dict, 'set': set,
                'True': True, 'False': False, 'None': None,
                'min': min, 'max': max, 'sum': sum, 'abs': abs,
                'round': round, 'sorted': sorted, 'enumerate': enumerate,
                'zip': zip, 'map': map, 'filter': filter,
                'isinstance': isinstance, 'type': type,
            }
        }
        
        # Add allowed modules
        import importlib
        for mod_name in self.ALLOWED_MODULES:
            safe_globals[mod_name] = importlib.import_module(mod_name)
        
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        result = None
        try:
            # Timeout handling
            def timeout_handler(signum, frame):
                raise TimeoutError("Code execution timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)
            
            # Execute
            exec(compile(tree, '<sandbox>', 'exec'), safe_globals)
            
            signal.alarm(0)  # Cancel timeout
            
            output = sys.stdout.getvalue()
            
            # Get last expression value if any
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                result = eval(compile(
                    ast.Expression(tree.body[-1].value),
                    '<sandbox>', 'eval'
                ), safe_globals)
            
            return {
                "output": output,
                "result": result,
                "status": "success"
            }
            
        except TimeoutError:
            return {"error": "Execution timed out"}
        except Exception as e:
            return {"error": str(e)}
        finally:
            sys.stdout = old_stdout
            signal.alarm(0)
```

### Example: FLUX Uses Code

```
User: "What's the compound interest on $10,000 at 5% for 10 years?"

FLUX (thinks): This needs precise calculation. I'll use code.

FLUX (calls run_code):
{
    "code": "principal = 10000\nrate = 0.05\nyears = 10\nresult = principal * (1 + rate) ** years\nprint(f'Final amount: ${result:.2f}')\nprint(f'Interest earned: ${result - principal:.2f}')"
}

Output:
Final amount: $16288.95
Interest earned: $6288.95

FLUX: "After 10 years at 5% compound interest, your $10,000 would grow to 
       **$16,288.95**. That's $6,288.95 in interest."
```

---

## Implementation Phases

### Phase 4.1: Tool Format Migration (1-2 days)

| Task | Status |
|------|--------|
| Convert `tool_registry.py` to JSON schema format | ✅ |
| Remove all `<tool>` tag parsing code | ⬜ |
| Implement `FluxToolExecutor` class | ✅ |
| Test with Qwen2.5-Instruct native tool calling | ✅ |
| Update orchestrator to use JSON format | ✅ |

> **Files:** `phases/phase_orchestrator/tools_v2.json`, `notebooks/flux_autonomous_complete.ipynb`

### Phase 4.2: Code Embedding (1-2 days)

| Task | Status |
|------|--------|
| Create `bundle_codebase()` function | ✅ |
| Write bootstrap.py self-extractor | ✅ |
| Test wake_up() from .flx only | ⬜ |
| Validate all imports resolve correctly | ⬜ |
| Add to save workflow in flux_model.py | ⬜ |

> **Implementation:** See Cell 13-14 in `notebooks/flux_autonomous_complete.ipynb`

### Phase 4.3: Dynamic Tools (1 day)

| Task | Status |
|------|--------|
| Implement `create_tool` tool | ✅ |
| Add `user_tools` storage to .flx | ⬜ |
| Test tool creation and persistence | ⬜ |
| Add tool deletion/update capabilities | ⬜ |

### Phase 4.4: Document Ingestion (2-3 days)

| Task | Status |
|------|--------|
| Create `DocumentIngester` class | ⬜ |
| Add format handlers (PDF, DOCX, etc.) | ⬜ |
| Implement semantic chunking | ⬜ |
| Wire to episodic memory storage | ⬜ |
| Test with various document types | ⬜ |

### Phase 4.5: Code Execution (1 day)

| Task | Status |
|------|--------|
| Implement `CodeSandbox` class | ⬜ |
| Add `run_code` tool | ⬜ |
| Test security restrictions | ⬜ |
| Add timeout handling | ⬜ |

### Phase 4.6: Goal System (2-3 days)

| Task | Status |
|------|--------|
| Implement `Goal` and `Step` dataclasses | ⬜ |
| Create `GoalPlanner` class | ⬜ |
| Add trigger detection | ⬜ |
| Wire to episodic memory for pattern learning | ⬜ |
| Test proactive goal activation | ⬜ |

### Phase 4.7: Codebase Cleanup (1-2 days)

| Task | Status |
|------|--------|
| Remove deprecated `<tool>` tag code | ⬜ |
| Remove unused phase code (1.5, 2.5, 3.5, 9) | ⬜ |
| Consolidate orchestrator code | ⬜ |
| Update all documentation | ⬜ |
| Final validation test | ⬜ |

**Total Estimated Time:** 10-14 days

---

## Codebase Cleanup

### Files to REMOVE (Not in model, just concepts)

```
phases/phase1_5/     # Causal encoding — merged into phase5
phases/phase2_5/     # Dynamic field — never trained
phases/phase3_5/     # Personal fabric — future phase
phases/phase9/       # Wave generation — superseded by VLM
```

### Files to KEEP (Actually used)

```
phases/phase1/       # CSE — in model
phases/phase2/       # Field — in model  
phases/phase3/       # Gravity — in model
phases/phase4/       # Thermodynamic — in model
phases/phase5/       # CGN — in model
phases/phase6/       # Memory — in model
phases/phase7/       # FLUXModel assembly
phases/phase8/       # WaveDecoder (legacy but present)
phases/phase8_8/     # Grid adapters — in model
phases/phase8_9/     # Multi-modal adapters — in model
phases/phase_orchestrator/  # Tool dispatch — needed
phases/phase_unified/       # Agent loop — needed
```

### Code to REFACTOR

| File | Change |
|------|--------|
| `tool_registry.py` | Convert to JSON schema format |
| `flux_orchestrator.py` | Remove `<tool>` parsing, use native JSON |
| `system_prompt.py` | Update for JSON tool format |
| `flux_model.py` | Add runtime embedding methods |

---

## Success Criteria

Phase 4 is complete when:

1. ✅ FLUX runs from .flx file alone (no external codebase)
2. ✅ Tool calling works with Qwen's native JSON format
3. ✅ Users can create tools that persist across sessions
4. ✅ Documents of any format can be ingested
5. ✅ Code execution works in sandbox
6. ✅ Goals activate automatically based on patterns
7. ✅ Codebase is cleaned of deprecated code

---

## Appendix: Full Tool List (JSON Schema)

See `phases/phase_orchestrator/tools_v2.json` for complete tool definitions.

### Built-in Tools

| Tool | Category | Purpose |
|------|----------|---------|
| `query_memory` | Knowledge | Search episodic memory |
| `store_memory` | Knowledge | Save new fact |
| `encode_to_wave` | Perception | Text → 432D wave |
| `query_field` | Knowledge | Search resonance field |
| `run_code` | Execution | Python sandbox |
| `create_tool` | Meta | Make new tool |
| `delete_tool` | Meta | Remove user tool |
| `ingest_document` | Ingestion | Add document to memory |
| `list_documents` | Ingestion | Show ingested docs |
| `set_goal` | Planning | Create explicit goal |
| `check_goals` | Planning | View active goals |
| `use_coder` | Models | Route to Coder model |
| `use_vision` | Models | Route to VL model |
| `transcribe` | Models | Route to Whisper |
| `speak` | Models | Route to Bark TTS |

---

*Document created: April 1, 2026*  
*To be executed after Phase 3 validation testing is complete*
