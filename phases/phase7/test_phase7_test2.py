import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from flux_generate import TextGenerator

def test_generation_coherence():
    print("Running Phase 7 Test 2: Generation Coherence")
    # Mock model
    gen = TextGenerator(model=None)

    prompt = "Deep learning is"
    out = gen.generate(prompt)
    print(f"Generated text: {out}")

    assert prompt in out
    assert "FLUX generation" in out

    print("Test 2: PASS")

if __name__ == "__main__":
    test_generation_coherence()
