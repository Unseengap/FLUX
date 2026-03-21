class AssistantTraitsEncoder:
    def __init__(self, cse, cwc, field, device='cpu'): self.cse, self.cwc, self.field, self.device = cse, cwc, field, device
    def seed_all_traits(self, strength=0.8):
        for ex in ["Answer directly.", "Simple language.", "Be helpful."]:
            wave = self.cse.encode(ex)
            self.field.perturb(wave.full.mean(dim=0), strength=strength)
