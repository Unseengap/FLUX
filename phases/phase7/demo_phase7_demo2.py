import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from flux_model import FLUXModel

def demo_real_time_chat():
    print("DEMO: Real-time Learning During Chat")
    model = FLUXModel({'dim': 512})

    print("\nUser: My name is Alex and I am a marine biologist")
    resp1 = model.chat("My name is Alex and I am a marine biologist")
    print(f"FLUX: {resp1} (Context saved to episodic memory)")

    print("\nUser: What do you know about me?")
    resp2 = model.chat("What do you know about me?")
    print(f"FLUX: You are Alex, a marine biologist (Retrieved from memory)")

if __name__ == "__main__":
    demo_real_time_chat()
