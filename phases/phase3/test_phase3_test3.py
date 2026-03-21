import torch
from gravity import GravitationalRelevance
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gr = GravitationalRelevance(feature_dim=512, device=device).to(device)
    v = torch.randn(1, 512, device=device)
    gr.spatial_index.add(v)
    gr.mass_tracker.observe(v)
    gr.contradict(v, strength=100.0)
    m = gr.mass_tracker.get_masses(torch.tensor([0], device=device))
    assert m[0] < 0
    print("✓ Repulsion passed")
if __name__ == "__main__": main()
