# HANDOFF: Bootstrap Module Registration Fix

**Date:** April 3, 2026  
**Issue:** Embedded modules not loading correctly from `.flx` file  
**Priority:** HIGH — Blocks autonomous agent functionality  
**Status:** Diagnosed but NOT fixed in embedded runtime

---

## Problem Summary

When `bootstrap.py` extracts embedded Python modules from `Flux-Apex-V1.flx`, the `__init__.py` files are being registered with WRONG module names, causing all submodule imports to fail.

### Symptoms

1. **Cell 9 output** shows `__init__.py` files failing:
   ```
   ⚠ phases/phase_autonomous/__init__.py (error: No module named 'phases.phase_autonomous.__init__')
   ```

2. **Cell 9 module check** shows all new modules missing:
   ```
   New modules check:
     ✗ phases.phase_autonomous
     ✗ phases.phase_autonomous.tool_executor
     ✗ phases.phase_autonomous.code_sandbox
   ```

3. **Cell 10** import fails:
   ```
   ModuleNotFoundError: No module named 'phases.phase_autonomous'
   ```

---

## Root Cause Analysis

### The Fix That Was Applied (but not correctly embedded)

In `/Users/admin/Desktop/flux/bootstrap.py`, the `EmbeddedModuleFinder` class was updated to handle `__init__.py` files:

```python
# Around line 85-100 in bootstrap.py
def find_module(self, fullname, path=None):
    if fullname in self.modules:
        return self
    return None

def load_module(self, fullname):
    if fullname in sys.modules:
        return sys.modules[fullname]
    
    source, file_path = self.modules[fullname]
    
    # CRITICAL FIX: Handle __init__.py → package name mapping
    module_name = fullname
    is_package = file_path.endswith('__init__.py')
    
    if module_name.endswith('.__init__'):
        # Strip .__init__ suffix for proper package registration
        pkg_name = module_name[:-9]  # Remove '.__init__'
        module_name = pkg_name
    
    # Create module with correct name
    module = types.ModuleType(module_name)
    module.__file__ = file_path
    module.__loader__ = self
    
    if is_package:
        module.__package__ = module_name
        module.__path__ = [str(Path(file_path).parent)]
    else:
        module.__package__ = module_name.rsplit('.', 1)[0] if '.' in module_name else ''
    
    sys.modules[module_name] = module
    # ... exec code ...
```

### Why It's Not Working

The fix exists in the **disk version** of `bootstrap.py`, but:

1. The **embedded bootstrap** inside `Flux-Apex-V1.flx` was NOT updated with the fix
2. Cell 7 claims to load fresh bootstrap: `✓ Loading fresh bootstrap.py from repo`
3. But the bootstrap that runs in Cell 9 is the OLD version that was embedded before the fix

### Evidence

From Cell 9 output:
```
⚠ phases/phase2/__init__.py (error: No module named 'phases.phase2.__init__')
```

This error message format `No module named 'phases.phase2.__init__'` indicates the bootstrap is trying to register modules WITH the `.__init__` suffix, which is the OLD buggy behavior.

---

## Files Involved

### Primary Files

| File | Location | Purpose |
|------|----------|---------|
| `bootstrap.py` | `/Users/admin/Desktop/flux/bootstrap.py` | Self-extractor with import hook — **HAS THE FIX** |
| `flux_autonomous_embed.ipynb` | `/Users/admin/Desktop/flux/notebooks/flux_autonomous_embed.ipynb` | Embedding notebook |
| `Flux-Apex-V1.flx` | `checkpoints/Flux-Apex-V1.flx` (or HuggingFace) | Model file with embedded runtime |

### Embedded Runtime Structure in .flx

```python
flx['runtime'] = {
    'version': '8.3-autonomous',
    'bootstrap': '...',  # <- THIS IS THE PROBLEM: Contains OLD bootstrap code
    'code': {
        'phases/phase_autonomous/__init__.py': '<compressed>',
        'phases/phase_autonomous/tool_executor.py': '<compressed>',
        # ... 99 total files
    },
    'metadata': {...}
}
```

---

## What Needs to Happen

### Option A: Re-run notebook with correct bootstrap embedding

The notebook's Cell 7 already has code to load fresh bootstrap:
```python
bootstrap_path = ROOT / 'bootstrap.py'
if bootstrap_path.exists():
    bootstrap_code = bootstrap_path.read_text()
    print("✓ Loading fresh bootstrap.py from repo")
```

But something went wrong. The agent needs to:

1. **Verify** `bootstrap.py` on disk has the `__init__.py` fix
2. **Ensure** Cell 7 actually embeds it (not some cached version)
3. **Re-run** the full notebook
4. **Verify** Cell 9 shows proper module loading

### Option B: Fix bootstrap.py if the fix is incomplete

Check `/Users/admin/Desktop/flux/bootstrap.py` lines 80-150 for the `EmbeddedModuleFinder` class.

The fix should ensure:
1. `__init__.py` files map to package names WITHOUT `.__init__` suffix
2. Packages have `is_package=True` and `__path__` attribute
3. Modules are registered in `sys.modules` under the CORRECT name

### Expected Correct Behavior

When Cell 9 runs, it should show:
```
✓ phases/phase_autonomous/__init__.py
✓ phases/phase_autonomous/tool_executor.py
...

New modules check:
  ✓ phases.phase_autonomous
  ✓ phases.phase_autonomous.tool_executor
```

NOT:
```
⚠ phases/phase_autonomous/__init__.py (error: No module named 'phases.phase_autonomous.__init__')
```

---

## Relevant Code Sections

### bootstrap.py — EmbeddedModuleFinder (check this)

```python
# /Users/admin/Desktop/flux/bootstrap.py
# Look for class EmbeddedModuleFinder around line 50-150

class EmbeddedModuleFinder:
    """Import hook for embedded modules."""
    
    def __init__(self, modules: Dict[str, Tuple[str, str]]):
        # modules = {'phases.phase_autonomous': (source_code, 'phases/phase_autonomous/__init__.py'), ...}
        self.modules = modules
    
    def find_module(self, fullname, path=None):
        # Should return self if fullname is in self.modules
        ...
    
    def load_module(self, fullname):
        # CRITICAL: Must handle __init__.py properly
        # If file_path ends with __init__.py:
        #   - Register as package (is_package=True)
        #   - Use package name, not package.__init__
        #   - Set __path__ attribute
        ...
```

### flux_autonomous_embed.ipynb — Cell 7 (embedding logic)

```python
# Cell 7: Update runtime and metadata
# ...
# ALWAYS load fresh bootstrap.py from disk (contains fixed import hook)
bootstrap_path = ROOT / 'bootstrap.py'
if bootstrap_path.exists():
    bootstrap_code = bootstrap_path.read_text()
    print("✓ Loading fresh bootstrap.py from repo")
# ...
new_runtime = {
    'bootstrap': bootstrap_code,  # FRESH bootstrap with fixed import hook
    ...
}
```

---

## Verification Steps

After fixing, run these checks:

### 1. Check bootstrap.py on disk
```python
# In the repo directory
with open('bootstrap.py') as f:
    content = f.read()
    
# Should contain the __init__ fix
assert '.__init__' in content or 'is_package' in content
print("✓ bootstrap.py has package handling")
```

### 2. Check embedded bootstrap in .flx
```python
import torch
flx = torch.load('checkpoints/Flux-Apex-V1.flx', map_location='cpu', weights_only=False)
bootstrap = flx['runtime']['bootstrap']

# Should contain the fix
assert 'is_package' in bootstrap or '.__init__' in bootstrap
print("✓ Embedded bootstrap has fix")
```

### 3. Test module registration
```python
from bootstrap import wake_up
result = wake_up('checkpoints/Flux-Apex-V1.flx', device='cpu')

# Check modules are registered correctly
assert 'phases.phase_autonomous' in result['modules']
assert 'phases.phase_autonomous.tool_executor' in result['modules']
print("✓ Modules registered correctly")
```

### 4. Test imports work
```python
from phases.phase_autonomous import FluxToolExecutor, CodeSandbox, CoderPool
print("✓ Imports work")
```

---

## Context Documents

| Document | Path | Purpose |
|----------|------|---------|
| Codebase Embed Spec | `DOCS/FLUX_CODEBASE_EMBED_SPEC.md` | How runtime embedding works |
| Apex Model Spec | `DOCS/FLUX_APEX_V1.md` | Model structure and components |
| Autonomous Spec | `DOCS/PHASE_AUTONOMOUS_SPEC.md` | What phase_autonomous should do |
| Copilot Instructions | `.github/copilot-instructions.md` | Project conventions |

---

## Phase Autonomous Files (should all load)

| File | Lines | Purpose |
|------|-------|---------|
| `phases/phase_autonomous/__init__.py` | 102 | Package exports |
| `phases/phase_autonomous/tool_executor.py` | 694 | 28 built-in tools |
| `phases/phase_autonomous/code_sandbox.py` | 404 | Safe Python execution |
| `phases/phase_autonomous/coder_pool.py` | 632 | Parallel sandbox pool |
| `phases/phase_autonomous/document_ingester.py` | 461 | Multi-format ingestion |
| `phases/phase_autonomous/goal_planner.py` | 488 | Proactive planning |
| `phases/phase_autonomous/autonomous_agent.py` | 481 | Main agent coordinator |

---

## Summary for Next Agent

**Problem:** `bootstrap.py` embedded in `.flx` file has bug where `__init__.py` files are registered as `package.__init__` instead of `package`, breaking all imports.

**Fix exists:** The disk version of `bootstrap.py` has the fix, but the embedded version in the saved `.flx` doesn't.

**Action needed:**
1. Verify `bootstrap.py` fix is correct
2. Re-embed fresh `bootstrap.py` into `.flx`
3. Re-save the model
4. Verify Cell 9 and Cell 10 pass

**Success criteria:**
- Cell 9 shows `✓ phases.phase_autonomous` (not `⚠`)
- Cell 10 imports succeed without workarounds
- No need for Cell 10.5 re-embedding step
