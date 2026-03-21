"""
Benchmark: GravitationalRelevance vs PyTorch MultiheadAttention.

Key fix: torch.cuda.synchronize() before/after timing on GPU.
Without sync, GPU ops appear to take ~0.4ms (just kernel launch
overhead) because perf_counter returns before the GPU finishes.
"""
import torch
import torch.nn as nn
import time
from typing import List, Tuple

SEQ_LENGTHS = [128, 256, 512, 1024, 2048, 4096]
FEATURE_DIM = 512
N_HEADS = 8
WARMUP_ITERS = 3
BENCH_ITERS = 10


def _sync(device: str):
    """Synchronize GPU if CUDA, otherwise no-op."""
    if 'cuda' in str(device) and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_gr(gr, seq_len: int, device: str) -> Tuple[float, float]:
    """Benchmark GravitationalRelevance forward pass.

    Returns:
        (mean_ms, std_ms)
    """
    x = torch.randn(1, seq_len, FEATURE_DIM, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = gr(x)
    _sync(device)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(BENCH_ITERS):
            _sync(device)
            start = time.perf_counter()
            _ = gr(x)
            _sync(device)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

    mean_ms = sum(times) / len(times)
    std_ms = (sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5
    return mean_ms, std_ms


def benchmark_attention(seq_len: int, device: str) -> Tuple[float, float]:
    """Benchmark PyTorch MultiheadAttention forward pass.

    Returns:
        (mean_ms, std_ms) or ('OOM', 0.0) if out of memory.
    """
    mha = nn.MultiheadAttention(
        embed_dim=FEATURE_DIM, num_heads=N_HEADS, batch_first=True
    ).to(device)
    mha.eval()

    try:
        x = torch.randn(1, seq_len, FEATURE_DIM, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(WARMUP_ITERS):
                _ = mha(x, x, x)
        _sync(device)

        # Timed runs
        times = []
        with torch.no_grad():
            for _ in range(BENCH_ITERS):
                _sync(device)
                start = time.perf_counter()
                _ = mha(x, x, x)
                _sync(device)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

        mean_ms = sum(times) / len(times)
        std_ms = (sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5
        return mean_ms, std_ms

    except RuntimeError:
        return 'OOM', 0.0
    finally:
        del mha
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_benchmark(gr, device='cuda') -> List[dict]:
    """Run GR vs attention benchmark across sequence lengths."""
    results = []
    print(f"\n{'Seq Len':>10} {'GR (ms)':>12} {'Attn (ms)':>12} {'Speedup':>10}")
    print("-" * 48)

    for seq_len in SEQ_LENGTHS:
        gr_ms, gr_std = benchmark_gr(gr, seq_len, device)
        attn_result = benchmark_attention(seq_len, device)

        if attn_result[0] == 'OOM':
            attn_ms = 'OOM'
            speedup_val = float('inf')
            speedup_str = 'OOM'
        else:
            attn_ms = attn_result[0]
            speedup_val = attn_ms / gr_ms if gr_ms > 0 else 0.0
            speedup_str = f"{speedup_val:.1f}x"

        attn_str = f"{attn_ms:.1f}" if isinstance(attn_ms, float) else attn_ms
        print(f"{seq_len:>10} {gr_ms:>10.1f}ms {attn_str:>10}ms {speedup_str:>10}")

        results.append({
            'seq_len': seq_len,
            'gr_ms': gr_ms,
            'attention_ms': attn_ms if isinstance(attn_ms, float) else 0.0,
            'speedup': speedup_val if isinstance(attn_ms, float) else float('inf'),
        })

    return results
