# FLUX v2 Notebooks Specification

> All v2 notebooks live in `v2/V2_NOTEBOOKS/`.  
> All results (logs, graphs, metrics) are saved to `v2/V2_results/<phase_name>/`.  
> Every notebook is compatible with **Google Colab** and **Kaggle** without modification.

---

## HuggingFace Credentials

```python
HF_TOKEN      = os.environ.get("HF_TOKEN", "")      # resolved in Cell 3 — never hardcode
GITHUB_TOKEN  = os.environ.get("GITHUB_TOKEN", "")  # resolved in Cell 3 — never hardcode
HF_USER       = "UnseenGAP"
HF_REPO_ID    = "UnseenGAP/FLUX"
```

Models are saved to and loaded from `UnseenGAP/FLUX` on HuggingFace Hub.  
**Tests and demos always load from HuggingFace** — never from local files.

Both tokens are resolved in Cell 3 using this priority order:
1. `os.environ` (set in Cell 2 from the env var)
2. Colab secrets (`google.colab.userdata`)
3. Kaggle secrets (`kaggle_secrets.UserSecretsClient`)
4. Warning printed if still missing — cell does NOT raise, only blocks later cells

> **Critical:** Always call `.strip()` on resolved token strings.  
> Colab and Kaggle secrets silently append `\r\n`, which causes  
> `Illegal header value b'Bearer '` when passed to `HfApi(token=...)`.  
> This error means the token is **empty or contains whitespace**, not that the token is wrong.

---

## GitHub Repository

```python
GITHUB_REPO = "https://github.com/Unseengap/FLUX.git"
BRANCH      = "v2"
```

---

## Standard Cell Order (All Phases)

| Cell # | Name | Run When |
|--------|------|----------|
| 1 | SETUP & CLONE | First time only — installs deps, clones v2 branch |
| 2 | **REFRESH** | Before every run — clears state, pulls latest, wipes results |
| 3 | HARDWARE & CREDENTIALS | After Refresh — resolves tokens, downloads prior phase checkpoint |
| 4 | SMOKE TEST | After Hardware — verifies imports work |
| 5 | TRAINING | Core training loop with full PhaseLogger output |
| 6 | **TRAINING DIAGNOSTICS** | Immediately after training — catch show-stoppers |
| 7 | UPLOAD TO HUGGINGFACE | After passing diagnostics |
| 8 | TEST 1 — loads from HF | Validates primary metric |
| 9 | TEST 2 — loads from HF | Secondary metric |
| 10 | TEST 3 — loads from HF | Tertiary metric |
| 11 | DEMO 1 — loads from HF | Visual/interactive demo |
| 12 | DEMO 2 — loads from HF | Visual/interactive demo |
| 13 | SAVE RESULTS | Saves all logs + graphs to V2_results/<phase>/ then pushes to GitHub |
| 14 | FINAL SUMMARY | Markdown summary block |

---

## REFRESH Cell Behaviour (Cell 2)

The Refresh cell is the most important infrastructure cell.  
Run it at the start of every session and after fixing any bug.

```
1. %reset -f                    → clear Python namespace completely
2. Re-define all constants      → REPO_PATH, RESULTS_DIR, HF_TOKEN, GITHUB_TOKEN, etc.
3. torch.cuda.empty_cache()     → free GPU VRAM
4. gc.collect()                 → free CPU RAM
5. git pull origin v2           → pull latest code from GitHub
6. shutil.rmtree(RESULTS_DIR)   → delete previous results (start fresh)
7. os.makedirs(RESULTS_DIR)     → recreate clean results directory
8. Verify key files exist        → catch missing files immediately
9. Print summary                → confirm refresh succeeded (shows HF_TOKEN / GITHUB_TOKEN status)
```

---

## Hardware & Credentials Cell (Cell 3)

Cell 3 does three things beyond hardware detection:

**Token resolution** (both HF_TOKEN and GITHUB_TOKEN):
```
env var → Colab secrets (google.colab.userdata) → Kaggle secrets → warning
```
Tokens are written back to the notebook-scope variables so all later cells can use them.

**HuggingFace login** via `huggingface_hub.login()`.

**Prior phase checkpoint download** — every phase from Phase 2 onward must download the
previous phase's checkpoint from HuggingFace before training starts, because each phase
runs in a fresh Colab/Kaggle session with no local files:

```python
# Phase N downloads phase N-1 checkpoint in Cell 3
if not os.path.exists(LOCAL_P_PREV_CKPT):
    hf_hub_download(repo_id=HF_REPO_ID, filename=HF_P_PREV_PATH,
                    token=HF_TOKEN, local_dir=CHECKPOINT_DIR)
```

This must happen in Cell 3 (not Cell 5) so the smoke test in Cell 4 can verify
the downloaded checkpoint loads correctly before training starts.

---

## Upload Cell (Cell 7) — Token Re-Resolution

**Every upload cell must re-resolve `HF_TOKEN` independently.**  
Do NOT rely on the notebook-scope `HF_TOKEN` variable alone. The `%reset -f` in Cell 2
clears all variables, and if the user skips Cell 3 or runs cells out of order, `HF_TOKEN`
will be an empty string, producing `Illegal header value b'Bearer '`.

Required pattern for every Cell 7 (copy exactly):

```python
# ── Re-resolve HF_TOKEN (guards against %reset -f clearing the variable) ──────
_hf_token = HF_TOKEN if 'HF_TOKEN' in dir() else ''
_hf_token = _hf_token or os.environ.get('HF_TOKEN', '')

if not _hf_token:
    try:
        from google.colab import userdata
        _hf_token = (userdata.get('HF_TOKEN') or '').strip()
    except Exception:
        pass

if not _hf_token:
    try:
        from kaggle_secrets import UserSecretsClient
        _hf_token = (UserSecretsClient().get_secret('HF_TOKEN') or '').strip()
    except Exception:
        pass

if not _hf_token:
    log.error("HF_TOKEN is empty — add it via Colab/Kaggle secrets and re-run Cell 3")
    raise ValueError("HF_TOKEN is required for upload. Run Cell 3 first.")

HF_TOKEN = _hf_token
os.environ['HF_TOKEN'] = _hf_token
log.success(f"HF_TOKEN resolved (len={len(_hf_token)})")

_api = HfApi(token=_hf_token)   # ← always pass _hf_token, never bare HF_TOKEN
```

The `log.success(f"HF_TOKEN resolved (len=XX)")` line is intentional — it confirms  
the token was actually read before a large (500+ MB) upload begins.

---

## Training Cell (Cell 5) — Module Reload

Always force-reload the training module before calling `train_field()` (or equivalent).  
This prevents stale bytecode from a failed prior run being used:

```python
import train_field as _tf_module
importlib.reload(_tf_module)
from train_field import train_field
```

This pattern must appear in every Cell 5, immediately before calling the training function.

---

## Final Gate Discrepancy (Known Behaviour)

At the end of `train_field()` (and equivalent training scripts), a final gate check  
runs on the **in-memory model** after the last `field.settle()` call. This sometimes  
reports `⚠ NOT YET` even when the **saved checkpoint** (which captured the best gate  
crossing during training) actually passes all thresholds.

**Rule:** The saved checkpoint is the source of truth.  
Cell 6 (Training Diagnostics) loads from the checkpoint file — its verdict supersedes  
the final in-memory gate printed at the end of Cell 5.

Do not block progression on the end-of-training gate printout alone. Trust Cell 6.

---

## Training Diagnostics Cell (Cell 6)

Run immediately after training finishes. Checks for show-stoppers:

```
✓ Checkpoint file exists on disk
✓ Checkpoint loads without error
✓ All required state_dict keys present
✓ Loss value is finite (no NaN/Inf)
✓ Loss is below reasonable threshold
✓ Model forward pass runs without error
✓ Wave shape is correct ([N, TOTAL_WAVE_DIM])
✓ Wave is not degenerate (std > 0)
✓ Mini decode gate: at least some text decoded
```

If any FAIL check triggers → **do not proceed to next cell**.  
WARN checks are noted but do not block progression.

Diagnostics are saved to `V2_results/<phase>/diagnostics.json`.

## Cell 13 — Save Results

After generating the results report and uploading logs to HuggingFace, Cell 13
pushes to GitHub using this exact sequence (required to avoid GitHub Push Protection):

```
1. Scrub tokens from log files  → regex replaces hf_*/ghp_* with [TOKEN_REDACTED]
2. git fetch origin v2          → sync remote state
3. git reset --soft origin/v2   → discard any local commits that may contain tokens
4. git add <result files>       → stage only clean scrubbed files
5. git commit                   → commit only if status shows changes
6. git push origin v2           → push to GitHub
7. Restore clean remote URL     → strip token from git config (always, via finally block)
```

Git authentication uses HTTPS token injection into the remote URL:
```python
_auth_url = clean_url.replace('https://', f'https://USER:GITHUB_TOKEN@')
```
The token is **always** stripped back out in the `finally` block, never stored permanently.```
v2/V2_results/
└── phase1/
│   ├── logs/
│   │   └── phase1.log          (full PhaseLogger output)
│   ├── graphs/
│   │   ├── training_loss.png
│   │   ├── decode_gate.png
│   │   └── wave_similarity.png
│   ├── metrics.json            (all numeric results)
│   ├── diagnostics.json        (show-stopper check results)
│   └── RESULTS_PHASE_1.md      (auto-generated by PhaseResults)
└── phase2/
    ├── logs/
    ├── graphs/
    ├── metrics.json
    ├── diagnostics.json
    └── RESULTS_PHASE_2.md
```

---

## HuggingFace Upload Convention

| Phase | Local checkpoint file | HF filename |
|-------|-----------------------|-------------|
| 1 | `checkpoints/phase1_v2.phase.pt` | `v2/phase1_v2.phase.pt` |
| 2 | `checkpoints/phase2_v2.phase.pt` | `v2/phase2_v2.phase.pt` |

Upload after training, load before tests/demos.

---

## Environment Detection

```python
import os

if os.path.exists('/kaggle/working'):
    RUNTIME  = 'kaggle'
    WORK_DIR = '/kaggle/working'
elif os.path.exists('/content'):
    RUNTIME  = 'colab'
    WORK_DIR = '/content'
else:
    RUNTIME  = 'local'
    WORK_DIR = os.path.expanduser('~')

REPO_PATH   = f'{WORK_DIR}/FLUX'
RESULTS_DIR = f'{REPO_PATH}/v2/V2_results/phase<N>'
```

---

## Colab vs Kaggle Differences (Handled Automatically)

| Feature | Colab | Kaggle |
|---------|-------|--------|
| Working dir | `/content/` | `/kaggle/working/` |
| GPU | T4 / A100 / L4 | P100 / T4 |
| Session length | 12h | 9h |
| Secrets | `google.colab.userdata` | `kaggle_secrets.UserSecretsClient` |
| Output artifacts | Download via Colab UI | `/kaggle/working/` auto-output |

---

## Notebook Files

| File | Phase | Description |
|------|-------|-------------|
| `phase1_v2.ipynb` | 1 | Wave Codec: CSE + WaveChunker + WaveToText joint training |
| `phase2_v2.ipynb` | 2 | Resonance Field with decode-preserving bridge |
