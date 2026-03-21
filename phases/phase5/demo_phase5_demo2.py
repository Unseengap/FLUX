import sys
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from multi_timescale import MultiTimescaleCoordination

def demo_timescale_activation():
    print("DEMO: Fast vs Slow Node Activation")
    feature_dim = 512
    mtc = MultiTimescaleCoordination(feature_dim)

    signal = torch.randn(feature_dim)
    print("\nProcessing signal through multiple timescales...")

    # Simulate a few steps
    for i in range(3):
        out = mtc(signal)
        print(f"Step {i+1}: Mean activation = {out.abs().mean().item():.4f}")

    print("\nFast nodes (syntax) react immediately to pattern fluctuations.")
    print("Slow nodes (concepts) integrate information over many steps to form deep attractors.")

if __name__ == "__main__":
    demo_timescale_activation()
