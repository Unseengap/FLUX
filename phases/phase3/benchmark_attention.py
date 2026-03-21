"""
Benchmark: GravitationalRelevance vs PyTorch MultiheadAttention.

HONEST ASSESSMENT:
  GR's current implementation is NOT faster than PyTorch attention.
  PyTorch attention is a single fused CUDA kernel. GR involves:
    - CPU↔GPU transfers (spatial index uses numpy)
    - Python-level loops in mass_tracker.observe()
    - Index rebuild on every forward pass

  The O(log n) claim is about the *spatial search* asymptotic
  complexity, not the full forward pass. At today's sequence lengths
  (< 8K tokens), attention's fused CUDA kernel dominates.

  GR's theoretical advantage would emerge at 100K+ tokens with a
  GPU-native spatial index implementation. For Phase 3, we benchmark
  honestly and note where the architectural advantage lies.
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
    """Run GR vs attention benchmark across sequence lengths.

    Reports honestly: GR is currently slower due to Python/CPU overhead.
    The comparison shows the theoretical scaling difference (O(n²) vs O(n log n)).
    """
    results = []
    print(f"\n{'Seq Len':>10} {'GR (ms)':>12} {'Attn (ms)':>12} {'Ratio':>10} {'Scaling':>12}")
    print("-" * 60)

    prev_gr_ms = None
    prev_attn_ms = None

    for seq_len in SEQ_LENGTHS:
        gr_ms, gr_std = benchmark_gr(gr, seq_len, device)
        attn_result = benchmark_attention(seq_len, device)

        if attn_result[0] == 'OOM':
            attn_ms = None
            ratio_str = 'Attn OOM'
        else:
            attn_ms = attn_result[0]
            ratio = gr_ms / attn_ms if attn_ms > 0 else float('inf')
            ratio_str = f"{ratio:.1f}x slower"

        # Compute scaling factor from previous seq length
        if prev_gr_ms is not None and prev_attn_ms is not None and attn_ms is not None:
            gr_growth = gr_ms / prev_gr_ms
            attn_growth = attn_ms / prev_attn_ms
            scaling_str = f"GR:{gr_growth:.1f}x A:{attn_growth:.1f}x"
        else:
            scaling_str = "--"

        attn_str = f"{attn_ms:.1f}" if attn_ms is not None else "OOM"
        print(f"{seq_len:>10} {gr_ms:>10.1f}ms {attn_str:>10}ms {ratio_str:>10} {scaling_str:>12}")

        results.append({
            'seq_len': seq_len,
            'gr_ms': gr_ms,
            'attention_ms': attn_ms if attn_ms is not None else 0.0,
            'gr_growth': gr_ms / prev_gr_ms if prev_gr_ms else 1.0,
            'attn_growth': attn_ms / prev_attn_ms if (prev_attn_ms and attn_ms) else 1.0,
        })

        prev_gr_ms = gr_ms
        prev_attn_ms = attn_ms

    # Print scaling analysis
    print(f"\n  Scaling analysis (when seq doubles):")
    print(f"    O(n²) attention: growth should be ~4x per doubling")
    print(f"    O(n log n) GR:   growth should be ~2x per doubling")
    gr_growths = [r['gr_growth'] for r in results[1:]]
    attn_growths = [r['attn_growth'] for r in results[1:]]
    if gr_growths:
        print(f"    GR avg growth:   {sum(gr_growths)/len(gr_growths):.2f}x")
    if attn_growths:
        print(f"    Attn avg growth: {sum(attn_growths)/len(attn_growths):.2f}x")
    print(f"\n  Note: GR is slower in absolute time due to Python/CPU spatial index.")
    print(f"  The O(log n) advantage is in scaling rate, not constant factors.")
    print(f"  A CUDA-native spatial index would close the gap at seq > 10K.")

    return results
