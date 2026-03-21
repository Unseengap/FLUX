"""
Local energy computations for thermodynamic settling.
"""
import torch.nn.functional as F
def compute_local_energy(state, target):
    return F.mse_loss(state, target)
