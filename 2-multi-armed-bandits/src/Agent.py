import numpy as np
import math


class Agent(object):
    def __init__(self, k):
        self.k = k
        self.q = np.zeros(k)
        self.total_count = 0

    def initialize(self):
        self.q = np.zeros(self.k)
        self.total_count = 0

    def update(self):
        self.total_count += 1


def greedy(values):
    max_idx = np.where(values == values.max())
    return np.random.choice(max_idx[0])


def eps_greedy(values, eps=0.1):
    if np.random.rand() > eps:
        return greedy(values)
    else:
        return np.random.randint(len(values))


class ActionValue(Agent):
    ALPHA = 0.1

    def __init__(self, k, eps=0.1, is_const_param=0, bias_q=0.0):
        super().__init__(k)
        self.init_eps = eps
        self.eps = eps
        self.count = np.zeros(k)
        self.sum_reward = np.zeros(k)

        self.is_const_param = is_const_param
        self.bias_q = bias_q

    def initialize(self):
        super().initialize()
        self.q += self.bias_q
        self.eps = self.init_eps
        self.count = np.zeros(self.k)
        self.sum_reward = np.zeros(self.k)

    def update(self, selected, reward):
        super().update()
        self.count[selected] += 1
        self.sum_reward[selected] += reward
        if self.is_const_param:
            self.q[selected] = self.ALPHA * self.sum_reward[selected]
        else:
            self.q[selected] = self.sum_reward[selected] / self.count[selected]

    def select_act(self):
        return eps_greedy(self.q, self.eps)


class SimpleAlgorithm(Agent):
    def __init__(self, k, alpha=0.1, policy="greedy", eps=0.1):
        super().__init__(k)
        self.alpha = alpha
        self.policy = policy
        self.init_eps = eps
        self.eps = eps

    def initialize(self):
        super().initialize()
        self.eps = self.init_eps

    def update(self, selected, reward):
        super().update()
        self.q[selected] += self.alpha * (reward - self.q[selected])

    def select_act(self):
        if self.policy == "greedy":
            return greedy(self.q)
        elif self.policy == "eps_greedy":
            return eps_greedy(self.q, self.eps)


class QLearning(Agent):
    # alpha=learning_rate, gamma=discount_rate
    def __init__(self, k, alpha=0.1, gamma=0.9, policy="greedy", eps=0.1, dec_eps=0.001):
        super().__init__(k)
        self.alpha = alpha
        self.gamma = gamma
        self.policy = policy
        self.init_eps = eps
        self.eps = eps
        self.dec_eps = dec_eps

    def initialize(self):
        super().initialize()
        self.eps = self.init_eps

    def update(self, selected, reward):
        super().update()
        td_error = reward + \
                   self.gamma * np.max(self.q) - self.q[selected]
        self.q[selected] += self.alpha * td_error

    def select_act(self):
        if self.policy == "greedy":
            return greedy(self.q)
        elif self.policy == "eps_greedy":
            self.eps = max(self.eps - self.dec_eps, 0.0)
            return eps_greedy(self.q, self.eps)


class UCB(Agent):
    def __init__(self, k, c=0.1):
        super().__init__(k)
        self.c = c
        self.count = np.zeros(k)
        self.sum_reward = np.zeros(k)

    def initialize(self):
        super().initialize()
        self.count = np.zeros(self.k)
        self.sum_reward = np.zeros(self.k)

    def update(self, selected, reward):
        super().update()
        self.count[selected] += 1
        self.sum_reward[selected] += reward
        # 全ての腕を1回ずつ探索し終わったら価値を更新
        if self.total_count >= self.k:
            for i, q in enumerate(self.q):
                avg_reward = self.sum_reward[i] / self.count[i]
                self.q[i] = avg_reward + self.c * math.sqrt(math.log(self.total_count) / self.count[i])
        # if self.total_count < 20 or self.total_count > 1980:
        #     print(self.q)
        #     print(self.count)

    def select_act(self):
        if self.total_count < self.k:
            return self.total_count
        return greedy(self.q)


class GradientBandit(Agent):
    EXP_RANGE = 709.782712893

    def __init__(self, k, alpha=0.1):
        super().__init__(k)
        self.alpha = alpha
        self.avg_reward = 0
        self.pi = np.ones(k) / k

    def initialize(self):
        super().initialize()
        self.avg_reward = 0
        self.pi = np.ones(self.k) / self.k

    def update(self, selected, reward):
        super().update()
        self.avg_reward += (1.0 / self.total_count) * (reward - self.avg_reward)

        for i in range(self.k):
            if i == selected:
                self.q[i] += self.alpha * (reward - self.avg_reward) * (1 - self.pi[i])
            else:
                self.q[i] -= self.alpha * (reward - self.avg_reward) * self.pi[i]

        exp_q = np.zeros(self.k)
        for i, q in enumerate(self.q):
            if q <= -self.EXP_RANGE:
                exp_q[i] = 1e-15
            elif q >= self.EXP_RANGE:
                exp_q[i] = 1.0 - 1e-15
            else:
                exp_q[i] = np.exp(q)
        self.pi = exp_q / np.sum(exp_q)
        # if self.total_count < 20 or self.total_count > 1980:
            # print("Q")
            # print(self.q)
            # print(self.pi)

    def select_act(self):
        return greedy(self.q)
