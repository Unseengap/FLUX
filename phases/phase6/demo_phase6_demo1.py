import sys
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from working_memory import WorkingMemory

def demo_cross_session_memory():
    print("DEMO: Cross-session Memory (Simplified)")
    wm = WorkingMemory(window_size=10)

    print("\nAdding current session context...")
    for i in range(5):
        wm.add_perturbation(torch.randn(512))

    ctx = wm.get_current_context()
    print(f"Working memory contains {len(ctx)} recent perturbations.")

    print("\nIn a real cross-session demo, episodic memory would load facts from disk.")
    print("This confirms the working memory tier is active and managing context.")

if __name__ == "__main__":
    demo_cross_session_memory()
