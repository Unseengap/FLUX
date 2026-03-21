import torch
from datetime import datetime
from consent_layer import ConsentLayer
class PersonalFabric:
    def __init__(self, fabric_id, field, cse, cwc, device='cpu'):
        self.fabric_id, self.field, self.cse, self.cwc, self.device = fabric_id, field, cse, cwc, device
        self.personal_zone = set(); self.active_handshakes = {}; self.consent_layer = ConsentLayer()
        self.interaction_count = 0
    def initialize_identity(self, wave): self.identity_wave = wave.to(self.device)
    def perturb_personal(self, text, strength=1.0):
        wave = self.cse.encode(text)
        loc = self.field.perturb(wave.full.mean(dim=0), strength=strength)
        self.personal_zone.add(loc.h*self.field.w*self.field.d + loc.w*self.field.d + loc.d)
        self.interaction_count += 1
    def query_personal(self, text, k=8):
        wave = self.cse.encode(text)
        return self.field.query(wave.full.mean(dim=0), k=k)
