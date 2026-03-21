import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from flux_generate import TextGenerator
from flux_model import FLUXModel

def demo_end_to_end_generation():
    print("DEMO: End-to-end Text Generation")
    model = FLUXModel({'dim': 512})
    gen = TextGenerator(model)

    prompt = "In the future, AI will"
    print(f"Prompt: {prompt}")
    output = gen.generate(prompt)
    print(f"Output: {output}")

if __name__ == "__main__":
    demo_end_to_end_generation()
