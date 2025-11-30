from collections import deque

class EMASmoother:
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.state = {}

    def update(self, scores: dict):
        out = {}
        for k,v in scores.items():
            if k not in self.state:
                self.state[k] = v
            else:
                self.state[k] = self.alpha * v + (1-self.alpha) * self.state[k]
            out[k] = self.state[k]
        return out
