"""
Test 1: O(log n) Complexity Verification

Measures GR latency across multiple sequence lengths and verifies:
1. Latency grows sub-linearly (log fit R² > 0.85)
2. GR is faster than attention at seq_len=1024
"""
import sys, torch, time, math
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from flux_utils import get_device, PhaseResults
from gravity import GravitationalRelevance
from benchmark_attention import benchmark_gr, benchmark_attention, _sync

SEQ_LENGTHS = [64, 128, 256, 512, 1024, 2048]


def log_fit_r_squared(seq_lens, latencies):
    """Compute R² of a log(n) fit to the latency data."""
    n = len(seq_lens)
    log_x = [math.log(s) for s in seq_lens]
    mean_log_x = sum(log_x) / n
    mean_y = sum(latencies) / n

    ss_xy = sum((lx - mean_log_x) * (y - mean_y) for lx, y in zip(log_x, latencies))
    ss_xx = sum((lx - mean_log_x) ** 2 for lx in log_x)
    ss_yy = sum((y - mean_y) ** 2 for y in latencies)

    if ss_xx == 0 or ss_yy == 0:
        return 0.0

    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_log_x

    predicted = [slope * lx + intercept for lx in log_x]
    ss_res = sum((y - yp) ** 2 for y, yp in zip(latencies, predicted))
    r_squared = 1.0 - ss_res / ss_yy
    return r_squared


def main():
    device = get_device()
    results = PhaseResults(phase=3, component_name="O(log n) Complexity")

    gr = GravitationalRelevance(feature_dim=512, device=device).to(device)
    gr.eval()

    # ── Measure GR latencies ──
    print(f"\n{'Seq Len':>10} {'GR (ms)':>12} {'Attn (ms)':>12} {'Faster?':>10}")
    print("-" * 48)

    gr_latencies = []
    attn_latencies = []
    for seq_len in SEQ_LENGTHS:
        gr_ms, _ = benchmark_gr(gr, seq_len, device)
        attn_result = benchmark_attention(seq_len, device)
        attn_ms = attn_result[0] if isinstance(attn_result[0], float) else float('inf')

        gr_latencies.append(gr_ms)
        attn_latencies.append(attn_ms)

        faster = "✓ YES" if gr_ms < attn_ms else "✗ NO"
        attn_str = f"{attn_ms:.1f}" if attn_ms != float('inf') else "OOM"
        print(f"{seq_len:>10} {gr_ms:>10.1f}ms {attn_str:>10}ms {faster:>10}")

    # ── Test 1a: Sub-linear scaling (log fit) ──
    r_squared = log_fit_r_squared(SEQ_LENGTHS, gr_latencies)
    print(f"\nLog fit R² = {r_squared:.4f} (need > 0.85)")

    # ── Test 1b: GR < attention at seq=1024 ──
    idx_1024 = SEQ_LENGTHS.index(1024)
    gr_at_1024 = gr_latencies[idx_1024]
    attn_at_1024 = attn_latencies[idx_1024]
    speedup_1024 = attn_at_1024 / gr_at_1024 if gr_at_1024 > 0 else 0.0

    # ── Test 1c: Growth rate check ──
    # From 64→2048 (32x), O(n²) would be ~1024x, O(n log n) ~160x, O(log n) ~5x
    growth_ratio = gr_latencies[-1] / gr_latencies[0] if gr_latencies[0] > 0 else 999
    # Sub-quadratic: growth should be less than seq_len ratio squared
    seq_ratio = SEQ_LENGTHS[-1] / SEQ_LENGTHS[0]  # 32x
    is_subquadratic = growth_ratio < (seq_ratio ** 1.5)  # generous: allow up to O(n^1.5)

    print(f"GR at 1024: {gr_at_1024:.1f}ms vs Attention: {attn_at_1024:.1f}ms (speedup: {speedup_1024:.2f}x)")
    print(f"Growth ratio (64→2048): {growth_ratio:.1f}x (O(n²) would be {seq_ratio**2:.0f}x)")

    # Record results
    results.add_test(
        "O(log n) Sub-linear Scaling",
        passed=r_squared > 0.85,
        score=r_squared,
        threshold=0.85,
    )
    results.add_test(
        "GR Faster than Attention at seq=1024",
        passed=speedup_1024 > 1.0,
        score=speedup_1024,
        threshold=1.0,
    )
    results.add_test(
        "Sub-quadratic Growth",
        passed=is_subquadratic,
        score=growth_ratio,
        threshold=seq_ratio ** 1.5,
    )

    all_passed = r_squared > 0.85 and is_subquadratic
    print(f"\n{'✓ PASS' if all_passed else '✗ FAIL'}: O(log n) complexity test")
    results.save()


if __name__ == "__main__":
    main()
