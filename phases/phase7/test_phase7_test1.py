import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from flux_model import FLUXModel

def test_full_pipeline():
    print("Running Phase 7 Test 1: Full Pipeline Integration")
    config = {'dim': 512}
    model = FLUXModel(config)

    # Smoke test chat
    res = model.chat("Hello FLUX")
    print(f"Chat result: {res}")
    assert "Response" in res

    print("Test 1: PASS")

if __name__ == "__main__":
    test_full_pipeline()
