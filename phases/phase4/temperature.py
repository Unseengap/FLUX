class TemperatureManager:
    def __init__(self, initial=1.0, min_t=0.01, decay=0.999): self.current_t, self.min_t, self.decay = initial, min_t, decay
    def step(self): self.current_t = max(self.min_t, self.current_t * self.decay); return self.current_t
