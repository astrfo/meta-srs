import os
from datetime import datetime
from tqdm import tqdm
import random
import numpy as np
from env import Environment
from metabandit_srs import MetaBandit
from policy.srs import SRS
from policy.srsch import SRS_CH
import warnings
import time
warnings.simplefilter('ignore', category=RuntimeWarning)

class Simulator:
    def __init__(self, trial, step, K, aleph):
        #上位エージェント・下位エージェント1・下位エージェント2
        self.policy = [SRS_CH(2), SRS(K, aleph), SRS_CH(K)]
        self.trial = trial
        self.step = step
        self.K = K
        self.aleph = aleph
        self.regret = np.zeros(self.step)
        self.regret_tmp = np.zeros(self.step)
        self.meta_count = np.zeros((self.step, 2))
        self.make_folder()

    def run(self):
        for t in tqdm(range(self.trial)):
            self.env = Environment(self.K)
            self.prob = self.env.prob
            self.meta = MetaBandit(self.K, self.policy[0], self.policy[1], self.policy[2])
            self.meta.initialize()
            self.regretV = 0.0
            for s in range(self.step):
                arm = self.meta.select_arm()
                reward = self.env.play(arm)
                self.meta.update(arm, reward)
                self.calc_regret_count(t, s, arm)
        self.calc_rate(t, s)
        self.save_csv()

    def calc_rate(self, t, s):
        for i in range(s+1):
            self.meta_count[i] /= (t+1)*(i+1)

    def calc_regret_count(self, t, s, arm):
        self.regretV += (self.prob.max() - self.prob[arm])
        self.regret[s] += (self.regretV - self.regret[s]) / (t+1)
        self.meta_count[s] += self.meta.lower_count

    def make_folder(self):
        time_now = datetime.now()
        self.results_dir = f'log/{time_now:%Y%m%d%H%M}/'
        os.makedirs(self.results_dir, exist_ok=True)

    def save_csv(self):
        f = open(self.results_dir + 'log.txt', mode='w', encoding='utf-8')
        f.write(f'sim: {self.trial}, step: {self.step}, K: {self.K}, aleph: {self.aleph}\n')
        f.write(f'prob: {self.prob}\n')
        f.write(f'upper: {self.policy[0]}, lower0: {self.policy[1]}, lower1: {self.policy[2]}\n')
        np.savetxt(self.results_dir + 'regret.csv', self.regret, delimiter=",")
        np.savetxt(self.results_dir + 'rate.csv', self.meta_count, delimiter=",")
        f.close()