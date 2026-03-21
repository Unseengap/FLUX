from thermodynamic import ThermodynamicLearner
class OnlineLearner:
    def __init__(self, tl): self.tl = tl
    def learn_fact(self, wave, target): self.tl.settle_step(wave, target, iterations=10)
