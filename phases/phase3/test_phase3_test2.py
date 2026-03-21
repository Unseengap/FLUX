import torch
from gravity import GravitationalRelevance
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gr = GravitationalRelevance(feature_dim=512, device=device).to(device)
    v = torch.randn(10, 512, device=device)
    gr.spatial_index.add(v)
    idx = gr.spatial_index.search(v[0], k=1)
    assert idx.item() == 0
    print("✓ Precision passed")
if __name__ == "__main__": main()
