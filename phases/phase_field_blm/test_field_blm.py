"""
Test script for Field-Based BLM components.
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

def test_byte_wave_encoder():
    """Test ByteWaveEncoder."""
    from byte_wave_encoder import ByteWaveEncoder, ByteWaveConfig
    
    print("Testing ByteWaveEncoder...")
    
    config = ByteWaveConfig(wave_dim=432)
    encoder = ByteWaveEncoder(config)
    
    params = sum(p.numel() for p in encoder.parameters())
    print(f"  Parameters: {params:,}")
    assert params < 150_000, f"Too many parameters: {params}"
    
    # Test encode_bytes (full sequence)
    x = torch.randint(0, 256, (2, 64))  # batch=2, seq=64
    waves = encoder.encode_bytes(x)
    
    assert waves.shape == (2, 64, 432), f"Wrong shape: {waves.shape}"
    print(f"  encode_bytes output shape: {waves.shape}")
    
    # Test forward (context wave)
    context = encoder(x)
    assert context.shape == (2, 432), f"Wrong context shape: {context.shape}"
    print(f"  forward (context) output shape: {context.shape}")
    
    print("  ✓ ByteWaveEncoder OK\n")
    return True


def test_resonance_field():
    """Test ResonanceField."""
    from resonance_field import ResonanceField, FieldConfig
    
    print("Testing ResonanceField...")
    
    config = FieldConfig(dims=(32, 32, 32), features=256, wave_dim=432)
    field = ResonanceField(config)
    
    # Test deposit
    wave = torch.randn(432)
    field.deposit(wave, next_byte=ord('a'), evidence=1.0)
    
    assert field.unique_attractors == 1
    print(f"  Deposited 1 attractor")
    
    # Test query
    result, conf, pos = field.query(wave)
    assert result == ord('a'), f"Wrong prediction: {result}"
    print(f"  Query result: '{chr(result)}' with confidence {conf:.3f}")
    
    # Test similar wave
    similar = wave + torch.randn_like(wave) * 0.1
    result2, conf2, _ = field.query(similar)
    assert result2 == ord('a'), "Similar wave should give same result"
    print(f"  Similar wave: '{chr(result2)}' with confidence {conf2:.3f}")
    
    print("  ✓ ResonanceField OK\n")
    return True


def test_thermodynamic_settler():
    """Test ThermodynamicSettler."""
    from resonance_field import ResonanceField
    from thermodynamic_settler import ThermodynamicSettler
    
    print("Testing ThermodynamicSettler...")
    
    field = ResonanceField()
    settler = ThermodynamicSettler(field)
    
    # Test learning
    wave = torch.randn(432)
    
    # Learn association 10 times
    for _ in range(10):
        settler.learn(wave, ord('b'))
    
    # Predict
    settler.reset_temperature()
    pred, conf = settler.settle(wave)
    
    assert pred == ord('b'), f"Wrong prediction: {pred}"
    print(f"  Prediction: '{chr(pred)}' with confidence {conf:.3f}")
    
    # Test predict_and_learn
    wave2 = torch.randn(432)
    pred, conf, correct = settler.predict_and_learn(wave2, ord('c'))
    print(f"  First prediction (no prior): {pred}, correct={correct}")
    
    # After learning
    for _ in range(10):
        pred, conf, correct = settler.predict_and_learn(wave2, ord('c'))
    
    assert pred == ord('c'), "Should learn association"
    print(f"  After 10 iterations: '{chr(pred)}', correct={correct}")
    
    print("  ✓ ThermodynamicSettler OK\n")
    return True


def test_field_blm():
    """Test complete FieldBLM."""
    from field_blm import FieldBLM, FieldBLMConfig
    
    print("Testing FieldBLM...")
    
    config = FieldBLMConfig(
        wave_dim=432,
        context_window=32,
        field_dims=(32, 32, 32),
    )
    model = FieldBLM(config)
    
    params = model.num_parameters
    print(f"  Parameters: {params:,}")
    assert params < 150_000, f"Too many parameters: {params}"
    
    # Test training
    texts = [
        "Hello, world!",
        "The quick brown fox.",
        "Test test test.",
    ]
    
    for text in texts:
        result = model.train_on_text(text)
        print(f"  Trained on '{text[:20]}...' - acc={result['accuracy']:.2%}")
    
    # Test generation
    output = model.generate("Hello", max_bytes=20)
    print(f"  Generated: '{output}'")
    
    # Test save/load
    model.save('/tmp/field_blm_test.pt')
    model2 = FieldBLM.load('/tmp/field_blm_test.pt')
    
    assert model2.field.unique_attractors == model.field.unique_attractors
    print(f"  Save/load OK - {model2.field.unique_attractors} attractors preserved")
    
    print("  ✓ FieldBLM OK\n")
    return True


def main():
    print("=" * 60)
    print("Field-Based BLM Component Tests")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("ByteWaveEncoder", test_byte_wave_encoder()))
    results.append(("ResonanceField", test_resonance_field()))
    results.append(("ThermodynamicSettler", test_thermodynamic_settler()))
    results.append(("FieldBLM", test_field_blm()))
    
    print("=" * 60)
    print("Results:")
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print()
    print(f"{'All tests passed!' if all_passed else 'Some tests failed!'}")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
