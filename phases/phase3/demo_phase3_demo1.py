import torch
from benchmark_attention import run_benchmark
from gravity import GravitationalRelevance
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gr = GravitationalRelevance(device=device).to(device)
    run_benchmark(gr, device=device)
if __name__ == "__main__": main()
