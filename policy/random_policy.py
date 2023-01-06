import numpy as np

class random_policy:
    def __init__(self, K):
        self.K = K

    def initialize(self):
        pass
    
    def select_arm(self):
        return np.random.randint(self.K)

    def update(self, arm, reward):
        pass