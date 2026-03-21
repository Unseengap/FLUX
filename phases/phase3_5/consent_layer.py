from enum import Enum
class ConsentRule(Enum):
    ALLOW = 'allow'; BLOCK = 'block'; REVIEW = 'review'
class ConsentLayer:
    def __init__(self): self.blocked = set(); self.trusted = set()
    def evaluate_incoming(self, wave, source, fabric):
        if source in self.blocked: return ConsentRule.BLOCK
        if source in self.trusted: return ConsentRule.ALLOW
        return ConsentRule.REVIEW
    def set_rule(self, source, rule):
        if rule == ConsentRule.BLOCK: self.blocked.add(source)
        elif rule == ConsentRule.ALLOW: self.trusted.add(source)
