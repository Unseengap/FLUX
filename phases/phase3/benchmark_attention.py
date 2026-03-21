import torch
import torch.nn as nn
import time
from typing import List, Tuple

SEQ_LENGTHS = [128, 256, 512, 1024, 2048, 4096]
FEATURE_DIM = 512
N_HEADS = 8

def benchmark_gr(gr, seq_len, device):
    x = torch.randn(1, seq_len, FEATURE_DIM, device=device)
    for _ in range(3): _ = gr(x)
    times = []
    for _ in range(10):
        start = time.perf_counter()
        _ = gr(x)
        times.append((time.perf_counter() - start) * 1000)
    return sum(times) / len(times), 0.0

def benchmark_attention(seq_len, device):
    mha = nn.MultiheadAttention(embed_dim=FEATURE_DIM, num_heads=N_HEADS, batch_first=True).to(device)
    try:
        x = torch.randn(1, seq_len, FEATURE_DIM, device=device)
        for _ in range(3): _ = mha(x, x, x)
        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = mha(x, x, x)
            times.append((time.perf_counter() - start) * 1000)
        return sum(times) / len(times), 0.0
    except RuntimeError: return 'OOM', 0.0

def run_benchmark(gr, device='cuda'):
    results = []
    print(f"{'Seq Len':>10} {'GR (ms)':>12} {'Attn (ms)':>12} {'Speedup':>10}")
    for seq_len in SEQ_LENGTHS:
        gr_ms, _ = benchmark_gr(gr, seq_len, device)
        attn_ms, _ = benchmark_attention(seq_len, device)
        if attn_ms == 'OOM': speedup = 'inf'
        else: speedup = f"{attn_ms/gr_ms:.1f}x"
        print(f"{seq_len:>10} {gr_ms:>12.1f} {str(attn_ms):>12} {speedup:>10}")
        results.append({'seq_len': seq_len, 'gr_ms': gr_ms, 'attention_ms': attn_ms, 'speedup': speedup})
    return results
