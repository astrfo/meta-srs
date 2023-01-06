import numpy as np

class Environment:
    def __init__(self, K):
        self.K = K
        self.prob = np.array([0.4, 0.5, 0.6, 0.8])

    def play(self, arm):
        if self.prob[arm] > np.random.rand():
            return 1
        else:
            return 0