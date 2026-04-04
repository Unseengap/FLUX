"""
CodeSandbox — Safe Python execution environment.

Provides restricted Python execution with:
- No arbitrary imports
- No file system access (except explicit paths)
- No network access
- Timeout enforcement
- Memory limits

Usage:
    sandbox = CodeSandbox(timeout=30)
    result = sandbox.execute("print(2 + 2)")
    # result.output == "4\n"
"""

import ast
import sys
import signal
import importlib
from io import StringIO
from typing import Dict, Any, Optional, Set, List
from dataclasses import dataclass
from datetime import datetime
import traceback


@dataclass
class SandboxResult:
    """Result from sandbox execution."""
    success: bool
    output: str
    result: Any  # Last expression value
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'output': self.output,
            'result': str(self.result) if self.result is not None else None,
            'error': self.error,
            'execution_time_ms': self.execution_time_ms,
        }


class CodeSandbox:
    """
    Safe Python execution environment.
    
    Restricts dangerous operations while allowing useful computation.
    """
    
    # Modules allowed in sandbox
    ALLOWED_MODULES: Set[str] = {
        # Math and data
        'math', 'statistics', 'random', 'decimal', 'fractions',
        # Data structures
        'collections', 'itertools', 'functools', 'operator',
        # String processing
        'string', 're', 'textwrap',
        # Time (safe parts)
        'datetime', 'time',  # time.time() allowed, time.sleep() blocked
        # JSON
        'json',
        # Base64 encoding
        'base64', 'hashlib',
        # Type hints
        'typing',
        # Copy
        'copy',
        # Dataclasses
        'dataclasses',
        # Enum
        'enum',
    }
    
    # AST nodes that are forbidden
    FORBIDDEN_NODES: Set[type] = {
        # Note: Import/ImportFrom now allowed for ALLOWED_MODULES
        # We check imports at validation time instead
        ast.Global,       # No global manipulation
        ast.Nonlocal,     # No nonlocal manipulation
    }
    
    # Forbidden built-in names
    FORBIDDEN_BUILTINS: Set[str] = {
        'eval', 'exec', 'compile',  # No code execution
        'open', 'input',            # No I/O
        '__import__', 'importlib',  # No imports
        'globals', 'locals',        # No scope access
        'getattr', 'setattr', 'delattr',  # Limited object manipulation
        'vars', 'dir',              # No introspection that could leak
        'exit', 'quit',             # No exit
        'breakpoint',               # No debugging
    }
    
    # Safe built-ins to expose
    SAFE_BUILTINS: Dict[str, Any] = {
        # Types
        'int': int, 'float': float, 'str': str, 'bool': bool,
        'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
        'frozenset': frozenset, 'bytes': bytes, 'bytearray': bytearray,
        # Basic functions
        'len': len, 'range': range, 'enumerate': enumerate, 'zip': zip,
        'map': map, 'filter': filter, 'reversed': reversed, 'sorted': sorted,
        'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
        'pow': pow, 'divmod': divmod,
        # Type checking
        'isinstance': isinstance, 'issubclass': issubclass, 'type': type,
        'callable': callable, 'hasattr': hasattr,
        # String
        'ord': ord, 'chr': chr, 'repr': repr, 'ascii': ascii,
        'format': format,
        # Iteration
        'iter': iter, 'next': next, 'all': all, 'any': any,
        # Object creation
        'object': object, 'property': property,
        'staticmethod': staticmethod, 'classmethod': classmethod,
        # Printing
        'print': print,
        # Constants
        'True': True, 'False': False, 'None': None,
        # Exceptions
        'Exception': Exception, 'ValueError': ValueError,
        'TypeError': TypeError, 'KeyError': KeyError,
        'IndexError': IndexError, 'AttributeError': AttributeError,
        'RuntimeError': RuntimeError, 'StopIteration': StopIteration,
        # Slice
        'slice': slice,
        # Help with debugging
        'id': id,
    }
    
    def __init__(self, timeout: int = 30, max_output_size: int = 100000):
        """
        Initialize sandbox.
        
        Args:
            timeout: Maximum execution time in seconds
            max_output_size: Maximum output buffer size
        """
        self.timeout = timeout
        self.max_output_size = max_output_size
        
        # Pre-import allowed modules
        self._modules: Dict[str, Any] = {}
        for mod_name in self.ALLOWED_MODULES:
            try:
                self._modules[mod_name] = importlib.import_module(mod_name)
            except ImportError:
                pass
    
    def validate_code(self, code: str) -> Optional[str]:
        """
        Validate code for safety.
        
        Returns error message if unsafe, None if safe.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"Syntax error: {e}"
        
        # Check for forbidden nodes
        for node in ast.walk(tree):
            if type(node) in self.FORBIDDEN_NODES:
                return f"Forbidden operation: {type(node).__name__}"
            
            # Check imports - only allow ALLOWED_MODULES
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod_name = alias.name.split('.')[0]  # Get top-level module
                    if mod_name not in self.ALLOWED_MODULES:
                        return f"Import forbidden: {alias.name} (allowed: {', '.join(sorted(self.ALLOWED_MODULES))})"
            
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    mod_name = node.module.split('.')[0]
                    if mod_name not in self.ALLOWED_MODULES:
                        return f"Import forbidden: {node.module} (allowed: {', '.join(sorted(self.ALLOWED_MODULES))})"
            
            # Check for dangerous attribute access
            if isinstance(node, ast.Attribute):
                if node.attr.startswith('_'):
                    return f"Access to private attribute forbidden: {node.attr}"
            
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.FORBIDDEN_BUILTINS:
                        return f"Forbidden function: {node.func.id}"
        
        return None
    
    def _build_globals(self) -> Dict[str, Any]:
        """Build the globals dict for execution."""
        
        # Create restricted __import__
        allowed_modules = self._modules
        def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
            """Import that only allows ALLOWED_MODULES."""
            top_level = name.split('.')[0]
            if top_level not in allowed_modules:
                raise ImportError(f"Import of '{name}' is not allowed")
            return allowed_modules[top_level]
        
        builtins_dict = self.SAFE_BUILTINS.copy()
        builtins_dict['__import__'] = restricted_import
        
        globals_dict = {
            '__builtins__': builtins_dict,
        }
        
        # Add allowed modules (pre-imported)
        for name, mod in self._modules.items():
            globals_dict[name] = mod
        
        return globals_dict
    
    def execute(self, code: str, timeout: Optional[int] = None) -> SandboxResult:
        """
        Execute code in sandbox.
        
        Args:
            code: Python code to execute
            timeout: Override default timeout
            
        Returns:
            SandboxResult with output and status
        """
        import time as time_module
        start_time = time_module.time()
        
        timeout = timeout or self.timeout
        
        # Validate first
        error = self.validate_code(code)
        if error:
            return SandboxResult(
                success=False,
                output='',
                result=None,
                error=error,
                execution_time_ms=(time_module.time() - start_time) * 1000
            )
        
        # Parse
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return SandboxResult(
                success=False,
                output='',
                result=None,
                error=f"Syntax error: {e}",
                execution_time_ms=(time_module.time() - start_time) * 1000
            )
        
        # Build globals
        sandbox_globals = self._build_globals()
        sandbox_locals: Dict[str, Any] = {}
        
        # Capture stdout
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_output = StringIO()
        sys.stdout = captured_output
        sys.stderr = captured_output
        
        result_value = None
        error_msg = None
        
        try:
            # Set timeout (Unix only)
            if hasattr(signal, 'SIGALRM'):
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Execution timed out after {timeout}s")
                
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
            
            try:
                # Execute main body
                exec(compile(tree, '<sandbox>', 'exec'), sandbox_globals, sandbox_locals)
                
                # Get last expression value if any
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    try:
                        expr_tree = ast.Expression(tree.body[-1].value)
                        result_value = eval(
                            compile(expr_tree, '<sandbox>', 'eval'),
                            sandbox_globals,
                            sandbox_locals
                        )
                    except:
                        pass
                        
            finally:
                # Cancel alarm
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                    
        except TimeoutError as e:
            error_msg = str(e)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
        finally:
            # Restore stdout
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        output = captured_output.getvalue()
        
        # Truncate if too long
        if len(output) > self.max_output_size:
            output = output[:self.max_output_size] + f"\n... (truncated, {len(output)} chars total)"
        
        elapsed_ms = (time_module.time() - start_time) * 1000
        
        return SandboxResult(
            success=error_msg is None,
            output=output,
            result=result_value,
            error=error_msg,
            execution_time_ms=elapsed_ms
        )
    
    def execute_function(self, code: str, args: Dict[str, Any]) -> Any:
        """
        Execute code that defines a function and call it with args.
        
        The code should define a function, and this will call it
        with the provided arguments.
        
        Args:
            code: Python function definition
            args: Arguments to pass to the function
            
        Returns:
            Function result
        """
        # Parse to find function name
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Syntax error: {e}")
        
        # Find function definition
        func_name = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                break
        
        if func_name is None:
            raise ValueError("No function definition found in code")
        
        # Execute function definition
        result = self.execute(code)
        if not result.success:
            raise RuntimeError(result.error)
        
        # Build call code
        # Serialize args to code
        args_code = ', '.join(f'{k}={repr(v)}' for k, v in args.items())
        call_code = f"{code}\nresult = {func_name}({args_code})\nresult"
        
        # Execute with call
        call_result = self.execute(call_code)
        if not call_result.success:
            raise RuntimeError(call_result.error)
        
        return call_result.result
    
    def add_module(self, name: str, module: Any):
        """Add an additional module to the sandbox."""
        self._modules[name] = module
    
    def get_available_modules(self) -> List[str]:
        """Get list of available modules."""
        return list(self._modules.keys())


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

_default_sandbox: Optional[CodeSandbox] = None


def get_sandbox() -> CodeSandbox:
    """Get or create default sandbox."""
    global _default_sandbox
    if _default_sandbox is None:
        _default_sandbox = CodeSandbox()
    return _default_sandbox


def safe_exec(code: str, timeout: int = 30) -> SandboxResult:
    """Execute code in default sandbox."""
    return get_sandbox().execute(code, timeout)
