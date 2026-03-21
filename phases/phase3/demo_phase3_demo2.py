import torch
from mass_tracker import MassTracker
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tracker = MassTracker(device=device)
    tracker.observe(torch.randn(10, 512, device=device))
    print(tracker.stats())
if __name__ == "__main__": main()
