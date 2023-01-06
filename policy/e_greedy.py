import numpy as np

class e_greedy:
    def __init__(self, K):
        self.K = K
        self.eps = 0.01

    def initialize(self):
        self.n = np.zeros(self.K)
        self.V = np.zeros(self.K)
    
    def select_arm(self):
        if self.eps < np.random.random():
            return np.random.choice(np.where(self.V == self.V.max())[0])
        else:
            return np.random.randint(self.K)

    def update(self, arm, reward):
        self.n[arm] += 1
        self.V[arm] += (reward - self.V[arm]) / self.n[arm]