"""
PHASE 1 DEMO 2: Show constructive and destructive interference.

Run: python demo_phase1_demo2.py

Shows how wave interference affects similarity between word pairs:
    Constructive: similar words → similarity INCREASES
    Destructive: opposite words → similarity DECREASES
    Neutral: unrelated words → minimal change
"""

import sys
from pathlib import Path

PHASE_DIR = Path(__file__).parent
PROJECT_ROOT = PHASE_DIR.parent.parent
sys.path.insert(0, str(PHASE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

from cse import ContinuousSemanticEncoder
from flux_utils import load_checkpoint


def compute_similarity(cse: ContinuousSemanticEncoder, w1: str, w2: str) -> float:
    """Compute cosine similarity of semantic mean vectors."""
    with torch.no_grad():
        wave1 = cse.encode(w1)
        wave2 = cse.encode(w2)
    v1 = wave1.semantic.mean(dim=0)
    v2 = wave2.semantic.mean(dim=0)
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


def main():
    console = Console()
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]FLUX Phase 1 Demo 2: Interference Patterns")
    console.print("[bold cyan]=" * 60)

    # ── Load trained CSE ──
    console.print("\n  Loading Phase 1 checkpoint...")
    checkpoint = load_checkpoint(1)
    config = checkpoint['config']
    cse = ContinuousSemanticEncoder(**config, device='cpu')
    cse.load_state_dict(checkpoint['state_dict'])
    cse.eval()

    # ── Constructive Interference (similar words) ──
    console.print("\n[bold green]── Constructive Interference (Similar Words) ──")
    constructive_pairs = [
        ("king", "queen"),
        ("dog", "cat"),
        ("happy", "joyful"),
        ("car", "vehicle"),
        ("ocean", "sea"),
        ("doctor", "physician"),
    ]

    table = Table(title="Similar Words → High Similarity (Constructive)")
    table.add_column("Word 1", style="cyan")
    table.add_column("Word 2", style="cyan")
    table.add_column("Cosine Similarity", justify="right")
    table.add_column("Status", justify="center")

    for w1, w2 in constructive_pairs:
        sim = compute_similarity(cse, w1, w2)
        status = "[green]✓ constructive" if sim > 0.5 else "[yellow]~ weak"
        table.add_row(w1, w2, f"{sim:.4f}", status)

    console.print(table)

    # ── Destructive Interference (opposite words) ──
    console.print("\n[bold red]── Destructive Interference (Opposite Words) ──")
    destructive_pairs = [
        ("hot", "cold"),
        ("happy", "sad"),
        ("love", "hate"),
        ("big", "small"),
        ("light", "dark"),
    ]

    table = Table(title="Opposite Words → Low Similarity (Destructive)")
    table.add_column("Word 1", style="red")
    table.add_column("Word 2", style="red")
    table.add_column("Cosine Similarity", justify="right")
    table.add_column("Status", justify="center")

    for w1, w2 in destructive_pairs:
        sim = compute_similarity(cse, w1, w2)
        status = "[green]✓ destructive" if sim < 0.3 else "[yellow]~ partial"
        table.add_row(w1, w2, f"{sim:.4f}", status)

    console.print(table)

    # ── Neutral (unrelated words) ──
    console.print("\n[bold yellow]── Neutral (Unrelated Words) ──")
    neutral_pairs = [
        ("cat", "democracy"),
        ("Paris", "banana"),
        ("happy", "concrete"),
        ("ocean", "keyboard"),
        ("doctor", "algebra"),
    ]

    table = Table(title="Unrelated Words → Near-Zero Similarity (Neutral)")
    table.add_column("Word 1", style="yellow")
    table.add_column("Word 2", style="yellow")
    table.add_column("Cosine Similarity", justify="right")
    table.add_column("Status", justify="center")

    for w1, w2 in neutral_pairs:
        sim = compute_similarity(cse, w1, w2)
        status = "[green]✓ neutral" if abs(sim) < 0.3 else "[yellow]~ some signal"
        table.add_row(w1, w2, f"{sim:.4f}", status)

    console.print(table)

    # ── Summary ──
    console.print("\n[bold]── Summary ──")
    all_constructive = [compute_similarity(cse, w1, w2) for w1, w2 in constructive_pairs]
    all_destructive = [compute_similarity(cse, w1, w2) for w1, w2 in destructive_pairs]
    all_neutral = [compute_similarity(cse, w1, w2) for w1, w2 in neutral_pairs]

    avg_con = sum(all_constructive) / len(all_constructive)
    avg_des = sum(all_destructive) / len(all_destructive)
    avg_neu = sum(all_neutral) / len(all_neutral)

    console.print(f"  Average similarity (similar words):   [green]{avg_con:.4f}")
    console.print(f"  Average similarity (opposite words):  [red]{avg_des:.4f}")
    console.print(f"  Average similarity (unrelated words): [yellow]{avg_neu:.4f}")
    console.print(f"\n  Separation (similar - opposite): {avg_con - avg_des:.4f}")
    console.print("  ✓ Demo 2 complete\n")


if __name__ == '__main__':
    main()
