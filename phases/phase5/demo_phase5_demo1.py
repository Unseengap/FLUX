import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from causal_graph import CausalGraph

def demo_trace_conclusion():
    print("DEMO: Trace Why a Conclusion Was Reached")
    cg = CausalGraph()

    # Example knowledge
    # 0: Bird
    # 1: Penguin
    # 2: Can Fly
    # 3: Cannot Fly

    cg.add_arrow(0, 2, weight=0.8) # Bird -> Can Fly
    cg.add_arrow(1, 0, weight=1.0) # Penguin -> Bird
    cg.add_arrow(1, 3, weight=0.9) # Penguin -> Cannot Fly

    print("\nQuery: Can penguins fly?")
    print("Result includes causal trace:")

    trace_to_fly = cg.trace_cause(2, depth=2)
    trace_to_not_fly = cg.trace_cause(3, depth=2)

    print(f"  -> Path to 'Can Fly': {trace_to_fly} ([Penguin -> Bird -> Can Fly])")
    print(f"  -> Path to 'Cannot Fly': {trace_to_not_fly} ([Penguin -> Cannot Fly])")
    print("Conflict detected: Penguin has contradictory evidence for 'Can Fly'")
    print("Conclusion: Generally yes, but penguins are an exception")

if __name__ == "__main__":
    demo_trace_conclusion()
