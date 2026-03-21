import torch, hashlib
from typing import List
class FabricIdentity:
    def __init__(self, cse): self.cse = cse
    def derive_identity_wave(self, phrases: List[str]):
        waves = [self.cse.encode(p).semantic.mean(dim=0) for p in phrases]
        combined = torch.zeros_like(waves[0])
        for i, w in enumerate(waves): combined += w * (1.0/(i+1))
        return torch.nn.functional.normalize(combined.unsqueeze(0)).squeeze(0)
    def derive_fabric_id(self, wave):
        return hashlib.sha256(wave.detach().cpu().numpy().tobytes()).hexdigest()
