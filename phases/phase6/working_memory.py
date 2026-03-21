import torch
from collections import deque

class WorkingMemory:
    """
    Tier 1: Rolling field window for recent context.
    Analogy: Your desk — what you're looking at right now.
    """
    def __init__(self, window_size: int = 512):
        self.window_size = window_size
        self.context = deque(maxlen=window_size)

    def add_perturbation(self, wave_vector: torch.Tensor):
        self.context.append(wave_vector.detach())

    def get_current_context(self) -> torch.Tensor:
        if not self.context:
            return torch.empty(0)
        return torch.stack(list(self.context))

    def clear(self):
        self.context.clear()
