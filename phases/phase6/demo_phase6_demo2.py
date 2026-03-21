import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from consolidation import ConsolidationProcess

def demo_consolidation():
    print("DEMO: Consolidation Process Live")

    # Mock tiers
    cp = ConsolidationProcess(episodic=None, semantic=None)

    print("\nSimulating high-frequency episodic memory retrieval...")
    print("Promoting 'User prefers dark mode' from episodic to semantic field.")

    cp.run_consolidation()
    print("\nConsolidation complete. Semantic attractors reinforced.")

if __name__ == "__main__":
    demo_consolidation()
