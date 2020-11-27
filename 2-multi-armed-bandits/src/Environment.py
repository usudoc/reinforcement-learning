import numpy as np
import random


class Environment(object):
    def __init__(self, k, is_stationary=1):
        # self.prob = np.random.rand(k)
        self.k = k
        self.is_stationary = is_stationary
        if is_stationary:
            # 正規分布の平均値
            self.prob = np.array([0.2, -0.8, 1.55, 0.4, 1.3, -1.5, -0.2, -1.0, 0.9, -0.5])
        else:
            self.prob = np.zeros(k)

        self.correct_act = (np.where(self.prob == self.prob.max()))[0]
        # self.correct_act = self.correct_act[0]
        self.max_act_prob = self.prob[self.correct_act[0]]
        # correct_act = np.argmax(self.prob)
        # self.correct_act = correct_act
        # self.max_act_prob = self.prob[correct_act]

        # self.correct_act = 0
        # self.max_act_prob = 0.0

    def initialize(self):
        if self.is_stationary:
            # 正規分布の平均値
            self.prob = np.array([0.2, -0.8, 1.55, 0.4, 1.3, -1.5, -0.2, -1.0, 0.9, -0.5])
        else:
            self.prob = np.zeros(self.k)
        self.correct_act = (np.where(self.prob == self.prob.max()))[0]
        # self.correct_act = self.correct_act[0]
        self.max_act_prob = self.prob[self.correct_act[0]]

    def update(self):
        if self.is_stationary:
            pass
        else:
            for i in range(self.k):
                self.prob[i] += random.gauss(0.0, 0.01)
            correct_act = np.argmax(self.prob)
            self.correct_act = [correct_act]
            self.max_act_prob = self.prob[correct_act]

        for i in range(self.k):
            self.prob[i] += random.gauss(0.0, 0.01)

    def get_max_act_prob(self):
        return self.max_act_prob

    def get_reward(self, selected):
        if self.is_stationary:
            return random.gauss(self.prob[selected], 1.0)
        else:
            return self.prob[selected]
        # if np.random.rand() > self.prob[selected]:
        #     return 1
        # else:
        #     return 0

    def get_correct_act(self, selected):
        # if selected == self.correct_act:
        if selected in self.correct_act:
            return 1
        else:
            return 0

    def get_regret(self, selected):
        return self.max_act_prob - self.prob[selected]
