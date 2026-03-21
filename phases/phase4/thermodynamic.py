from temperature import TemperatureManager
class ThermodynamicLearner:
    def __init__(self, field, tm=None): self.field, self.tm = field, tm or TemperatureManager()
    def settle_step(self, wave, target, iterations=5):
        t = self.tm.step()
        for _ in range(iterations):
            self.field.perturb(wave, strength=t)
            self.field.settle(steps=1)
