"""
CoderPool — Parallel sandbox execution via the Coder model.

The instruct model delegates coding tasks to the coder model, which can
spin up multiple sandboxes in parallel. This keeps the instruct model
free for reasoning while coder handles all code generation and execution.

Architecture (like Google Jules):

    ┌─────────────────────────────────────────────────────────────────┐
    │                    INSTRUCT MODEL (Brain)                       │
    │              Qwen2.5-1.5B-Instruct                              │
    │                                                                 │
    │  "I need to calculate X, analyze Y, and process Z"              │
    │                         │                                        │
    │               delegate_to_coder()                               │
    │                         ↓                                        │
    │  ┌─────────────────────────────────────────────────────────┐    │
    │  │                    CODER POOL                            │    │
    │  │              Qwen2.5-Coder-1.5B-Instruct                 │    │
    │  │                                                         │    │
    │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │    │
    │  │  │ Sandbox  │  │ Sandbox  │  │ Sandbox  │  ...         │    │
    │  │  │    1     │  │    2     │  │    3     │              │    │
    │  │  │ (calc X) │  │(analyze Y│  │(process Z│              │    │
    │  │  └────┬─────┘  └────┬─────┘  └────┬─────┘              │    │
    │  │       │             │             │                     │    │
    │  │       └─────────────┴─────────────┘                     │    │
    │  │                     │                                    │    │
    │  │              [Results Pool]                              │    │
    │  └─────────────────────┬───────────────────────────────────┘    │
    │                        ↓                                        │
    │              Results back to Instruct                           │
    └─────────────────────────────────────────────────────────────────┘

Usage:
    from phases.phase_autonomous.coder_pool import CoderPool, CodingTask
    
    pool = CoderPool(flx_state, max_sandboxes=4)
    
    # Single task
    result = pool.execute_task(CodingTask(
        description="Calculate compound interest",
        context={"principal": 10000, "rate": 0.05, "years": 10}
    ))
    
    # Parallel tasks
    results = pool.execute_parallel([
        CodingTask("Calculate X"),
        CodingTask("Analyze Y"),
        CodingTask("Process Z"),
    ])
"""

import json
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .code_sandbox import CodeSandbox, SandboxResult


class TaskStatus(Enum):
    """Status of a coding task."""
    PENDING = "pending"
    GENERATING = "generating"  # Coder is writing code
    EXECUTING = "executing"    # Sandbox is running
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CodingTask:
    """A task to be handled by the coder model."""
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    language: str = "python"
    timeout: int = 30
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    priority: int = 0  # Higher = more urgent
    
    def to_prompt(self) -> str:
        """Convert task to prompt for coder model."""
        context_str = ""
        if self.context:
            context_str = "\n\nContext:\n" + json.dumps(self.context, indent=2)
        
        return f"""Write Python code to: {self.description}

Requirements:
- Write clean, efficient code
- Print the result at the end
- Handle edge cases
- No external dependencies beyond math, json, datetime, collections
{context_str}

Code:"""


@dataclass
class TaskResult:
    """Result from a coding task."""
    task_id: str
    status: TaskStatus
    generated_code: Optional[str] = None
    sandbox_result: Optional[SandboxResult] = None
    error: Optional[str] = None
    generation_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    @property
    def success(self) -> bool:
        return self.status == TaskStatus.COMPLETED and self.sandbox_result and self.sandbox_result.success
    
    @property
    def output(self) -> str:
        if self.sandbox_result:
            return self.sandbox_result.output
        return ""
    
    @property
    def result_value(self) -> Any:
        if self.sandbox_result:
            return self.sandbox_result.result
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'status': self.status.value,
            'success': self.success,
            'generated_code': self.generated_code,
            'output': self.output,
            'result': str(self.result_value) if self.result_value else None,
            'error': self.error,
            'total_time_ms': self.total_time_ms,
        }


class CoderPool:
    """
    Pool of sandboxes managed by the coder model.
    
    The instruct model delegates coding tasks here. The coder model
    generates code, and sandboxes execute it in parallel.
    """
    
    def __init__(
        self,
        flx_state: Dict[str, Any],
        max_sandboxes: int = 4,
        sandbox_timeout: int = 30,
        coder_generate_fn: Optional[Callable] = None,
    ):
        """
        Initialize the coder pool.
        
        Args:
            flx_state: Loaded .flx state dict
            max_sandboxes: Maximum concurrent sandboxes
            sandbox_timeout: Default timeout per sandbox
            coder_generate_fn: Function to call coder model (optional)
        """
        self.flx = flx_state
        self.max_sandboxes = max_sandboxes
        self.sandbox_timeout = sandbox_timeout
        
        # Coder model generate function
        # If not provided, we use a simple code generation heuristic
        self._coder_generate = coder_generate_fn or self._default_code_generator
        
        # Sandbox pool
        self._sandboxes: List[CodeSandbox] = [
            CodeSandbox(timeout=sandbox_timeout)
            for _ in range(max_sandboxes)
        ]
        self._sandbox_locks = [threading.Lock() for _ in range(max_sandboxes)]
        
        # Task queue and results
        self._pending_tasks: List[CodingTask] = []
        self._results: Dict[str, TaskResult] = {}
        
        # Thread pool for parallel execution
        self._executor = ThreadPoolExecutor(max_workers=max_sandboxes)
        
        # Statistics
        self.stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'total_code_generated': 0,
            'total_execution_time_ms': 0,
        }
    
    def _default_code_generator(self, prompt: str) -> str:
        """
        Default code generator when no coder model is connected.
        
        This is a simple heuristic for common tasks. In production,
        this would be replaced with actual Qwen2.5-Coder calls.
        """
        prompt_lower = prompt.lower()
        
        # Extract numbers from context
        import re
        numbers = re.findall(r'\d+\.?\d*', prompt)
        
        if 'compound interest' in prompt_lower or 'interest' in prompt_lower:
            return """
# Compound interest calculation
principal = float(input_context.get('principal', 10000))
rate = float(input_context.get('rate', 0.05))
years = int(input_context.get('years', 10))

final_amount = principal * (1 + rate) ** years
interest_earned = final_amount - principal

print(f"Principal: ${principal:,.2f}")
print(f"Rate: {rate*100:.1f}%")
print(f"Years: {years}")
print(f"Final Amount: ${final_amount:,.2f}")
print(f"Interest Earned: ${interest_earned:,.2f}")

result = {'final_amount': final_amount, 'interest': interest_earned}
"""
        
        elif 'fibonacci' in prompt_lower:
            return """
# Fibonacci sequence
n = int(input_context.get('n', 10))

def fib(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

sequence = [fib(i) for i in range(n)]
print(f"First {n} Fibonacci numbers: {sequence}")
result = sequence
"""
        
        elif 'factorial' in prompt_lower:
            return """
# Factorial calculation
import math
n = int(input_context.get('n', 10))
result = math.factorial(n)
print(f"{n}! = {result}")
"""
        
        elif 'prime' in prompt_lower:
            return """
# Prime number check/generation
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

n = int(input_context.get('n', 100))
if input_context.get('check', False):
    result = is_prime(n)
    print(f"{n} is {'prime' if result else 'not prime'}")
else:
    primes = [x for x in range(2, n+1) if is_prime(x)]
    print(f"Primes up to {n}: {primes}")
    result = primes
"""
        
        elif 'sort' in prompt_lower:
            return """
# Sorting
data = input_context.get('data', [3, 1, 4, 1, 5, 9, 2, 6])
result = sorted(data)
print(f"Sorted: {result}")
"""
        
        elif 'sum' in prompt_lower or 'add' in prompt_lower:
            nums = [float(n) for n in numbers] if numbers else [1, 2, 3, 4, 5]
            return f"""
# Sum calculation
numbers = {nums}
result = sum(numbers)
print(f"Sum of {{numbers}} = {{result}}")
"""
        
        elif 'average' in prompt_lower or 'mean' in prompt_lower:
            return """
# Average calculation
import statistics
data = input_context.get('data', [1, 2, 3, 4, 5])
result = statistics.mean(data)
print(f"Average of {data} = {result}")
"""
        
        elif 'sqrt' in prompt_lower or 'square root' in prompt_lower:
            return """
# Square root
import math
n = float(input_context.get('n', 16))
result = math.sqrt(n)
print(f"√{n} = {result}")
"""
        
        else:
            # Generic fallback
            return """
# Generic task
print("Task: " + input_context.get('description', 'Unknown'))
data = input_context.get('data', None)
if data:
    print(f"Input data: {data}")
    result = data
else:
    result = "No specific code generated for this task"
    print(result)
"""
    
    def _acquire_sandbox(self) -> tuple[int, CodeSandbox]:
        """Acquire an available sandbox."""
        for i, lock in enumerate(self._sandbox_locks):
            if lock.acquire(blocking=False):
                return i, self._sandboxes[i]
        
        # All busy, wait for first available
        for i, lock in enumerate(self._sandbox_locks):
            lock.acquire(blocking=True)
            return i, self._sandboxes[i]
        
        raise RuntimeError("No sandbox available")
    
    def _release_sandbox(self, index: int):
        """Release a sandbox back to the pool."""
        self._sandbox_locks[index].release()
    
    def execute_task(self, task: CodingTask) -> TaskResult:
        """
        Execute a single coding task.
        
        Args:
            task: The coding task
            
        Returns:
            TaskResult with generated code and execution result
        """
        start_time = time.time()
        self.stats['total_tasks'] += 1
        
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.PENDING,
        )
        
        try:
            # Phase 1: Generate code via coder model
            result.status = TaskStatus.GENERATING
            gen_start = time.time()
            
            prompt = task.to_prompt()
            generated_code = self._coder_generate(prompt)
            
            result.generated_code = generated_code
            result.generation_time_ms = (time.time() - gen_start) * 1000
            self.stats['total_code_generated'] += 1
            
            # Phase 2: Execute in sandbox
            result.status = TaskStatus.EXECUTING
            exec_start = time.time()
            
            # Acquire sandbox
            sandbox_idx, sandbox = self._acquire_sandbox()
            
            try:
                # Wrap code with context injection
                wrapped_code = f"""
input_context = {json.dumps(task.context)}
{generated_code}
"""
                sandbox_result = sandbox.execute(wrapped_code, task.timeout)
                result.sandbox_result = sandbox_result
                
            finally:
                self._release_sandbox(sandbox_idx)
            
            result.execution_time_ms = (time.time() - exec_start) * 1000
            
            # Set final status
            if sandbox_result.success:
                result.status = TaskStatus.COMPLETED
                self.stats['successful_tasks'] += 1
            else:
                result.status = TaskStatus.FAILED
                result.error = sandbox_result.error
                self.stats['failed_tasks'] += 1
                
        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = str(e)
            self.stats['failed_tasks'] += 1
        
        result.total_time_ms = (time.time() - start_time) * 1000
        self.stats['total_execution_time_ms'] += result.total_time_ms
        
        # Store result
        self._results[task.task_id] = result
        
        return result
    
    def execute_parallel(
        self,
        tasks: List[CodingTask],
        wait: bool = True,
    ) -> List[TaskResult]:
        """
        Execute multiple tasks in parallel.
        
        Args:
            tasks: List of coding tasks
            wait: Whether to wait for all to complete
            
        Returns:
            List of TaskResults (in same order as input)
        """
        futures = {}
        
        for task in tasks:
            future = self._executor.submit(self.execute_task, task)
            futures[future] = task.task_id
        
        results = {}
        
        if wait:
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result()
                    results[task_id] = result
                except Exception as e:
                    results[task_id] = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.FAILED,
                        error=str(e)
                    )
        
        # Return in original order
        return [results.get(t.task_id, self._results.get(t.task_id)) for t in tasks]
    
    def delegate(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        parallel_subtasks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        High-level delegation API for the instruct model.
        
        Args:
            description: What needs to be done
            context: Context data
            parallel_subtasks: If provided, execute these in parallel
            
        Returns:
            Aggregated results
        """
        if parallel_subtasks:
            # Execute subtasks in parallel
            tasks = [
                CodingTask(
                    description=subtask,
                    context=context or {},
                )
                for subtask in parallel_subtasks
            ]
            
            results = self.execute_parallel(tasks)
            
            return {
                'type': 'parallel',
                'total_tasks': len(tasks),
                'successful': sum(1 for r in results if r.success),
                'results': [r.to_dict() for r in results],
            }
        else:
            # Single task
            task = CodingTask(
                description=description,
                context=context or {},
            )
            
            result = self.execute_task(task)
            
            return {
                'type': 'single',
                'success': result.success,
                'output': result.output,
                'result': result.result_value,
                'code': result.generated_code,
                'error': result.error,
            }
    
    def get_result(self, task_id: str) -> Optional[TaskResult]:
        """Get result for a specific task."""
        return self._results.get(task_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            **self.stats,
            'active_sandboxes': sum(1 for lock in self._sandbox_locks if lock.locked()),
            'total_sandboxes': self.max_sandboxes,
        }
    
    def shutdown(self):
        """Shutdown the pool."""
        self._executor.shutdown(wait=True)


# ─────────────────────────────────────────────
# Integration with FluxToolExecutor
# ─────────────────────────────────────────────

def create_coder_pool_tools() -> Dict[str, Dict[str, Any]]:
    """
    Create tool definitions for coder pool operations.
    
    These tools let the instruct model delegate to coder.
    """
    return {
        "delegate_to_coder": {
            "type": "function",
            "function": {
                "name": "delegate_to_coder",
                "description": "Delegate a coding task to the coder model. Use this instead of run_code when you need code to be GENERATED and then executed. The coder model will write the code and execute it in a sandbox.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Description of what code needs to do"
                        },
                        "context": {
                            "type": "object",
                            "description": "Input data/parameters for the code"
                        }
                    },
                    "required": ["task"]
                }
            }
        },
        "delegate_parallel": {
            "type": "function",
            "function": {
                "name": "delegate_parallel",
                "description": "Delegate multiple coding tasks to run in parallel. Use when you have several independent calculations or data processing steps.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tasks": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of task descriptions to execute in parallel"
                        },
                        "context": {
                            "type": "object",
                            "description": "Shared context for all tasks"
                        }
                    },
                    "required": ["tasks"]
                }
            }
        },
        "get_coder_stats": {
            "type": "function",
            "function": {
                "name": "get_coder_stats",
                "description": "Get statistics about the coder pool (active sandboxes, success rate, etc.)",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    }


class CoderPoolExecutor:
    """
    Executor for coder pool tools.
    
    Integrates with FluxToolExecutor to handle delegation.
    """
    
    def __init__(self, flx_state: Dict[str, Any], max_sandboxes: int = 4):
        self.pool = CoderPool(flx_state, max_sandboxes=max_sandboxes)
    
    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a coder pool tool."""
        if name == "delegate_to_coder":
            return self.pool.delegate(
                description=args.get('task', ''),
                context=args.get('context', {}),
            )
        
        elif name == "delegate_parallel":
            return self.pool.delegate(
                description="parallel execution",
                context=args.get('context', {}),
                parallel_subtasks=args.get('tasks', []),
            )
        
        elif name == "get_coder_stats":
            return self.pool.get_stats()
        
        else:
            return {'error': f"Unknown coder tool: {name}"}
    
    def shutdown(self):
        """Shutdown the pool."""
        self.pool.shutdown()
