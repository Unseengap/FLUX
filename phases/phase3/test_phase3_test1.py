import torch, time
from gravity import GravitationalRelevance
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gr = GravitationalRelevance(feature_dim=512, device=device).to(device)
    seq_lens = [128, 1024]
    lats = []
    for sl in seq_lens:
        x = torch.randn(1, sl, 512, device=device)
        for _ in range(3): gr(x)
        s = time.perf_counter()
        for _ in range(5): gr(x)
        lats.append((time.perf_counter()-s)/5)
    print(f"Lats: {lats}")
    assert lats[1] < lats[0] * 20
    print("✓ Complexity passed")
if __name__ == "__main__": main()
