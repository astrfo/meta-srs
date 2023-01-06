import copy as cp
import numpy as np

class MetaBandit:
    def __init__(self, K, upper_agent, srs_agent, srsch_agent):
        self.K = K
        self.upper_agent = upper_agent
        self.srs_agent = srs_agent
        self.srsch_agent = srsch_agent
        self.select_agent = 0

    def initialize(self):
        self.upper_agent.initialize()
        self.srs_agent.initialize()
        self.srsch_agent.initialize()
        self.lower_count = np.zeros(2)

    def select_arm(self):
        self.select_agent = self.upper_agent.select_arm()
        if self.select_agent == 0:
            self.lower_count[0] += 1
            return self.srs_agent.select_arm()
        else:
            self.lower_count[1] += 1
            return self.srsch_agent.select_arm()

    def update(self, arm, reward):
        self.upper_agent.update(self.select_agent, reward)
        self.srs_agent.update(arm, reward)
        self.srsch_agent.update(arm, reward)