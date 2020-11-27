import numpy as np
import matplotlib.pyplot as plt
import Environment as ev
import Agent as ag


ACT_NUM = 10
SIM_NUM = 2000
STEP_NUM = 10000

IS_STATIONARY = 0


def plot_graph(data, labels, file_name, xlabel="Steps", ylabel="Accuracy"):
    plt.figure(figsize=(12, 8))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # each-params
    # param = np.arange(-7, 3, 1)
    # plt.xticks(np.arange(-7, 3, 1), param)
    # plt.ylim([1.0, 1.6])
    for i, graph in enumerate(data):
        plt.plot(graph, label=labels[i], alpha=0.8, linewidth=1.2)
        # each-params
        # plt.plot(param, graph, label=labels[i], alpha=0.8, linewidth=1.2)
    plt.legend(loc="upper left")
    plt.savefig(file_name)
    # plt.show()
    plt.clf()


# 後半の平均報酬をアルゴリズム毎に新たなデータとして取得
def get_second_half_data(agent_num, data):
    # 一定ステップを超えた後の平均報酬をグラフにする
    second_half_reward = np.zeros((agent_num, int(STEP_NUM / 2)))
    each_agent_reward = np.zeros(agent_num)
    # それぞれのエージェントの1000回から最後(2000)までを代入
    second_half_reward = data[0:agent_num, int(STEP_NUM / 2):STEP_NUM + 1]
    for i in range(agent_num):
        # 得られた配列をそれぞれのエージェントごとにすべて合計する
        each_agent_reward[i] = np.sum(second_half_reward[i]) / (int(STEP_NUM / 2))

    return each_agent_reward


# エージェント毎のデータから平均をとってアルゴリズム別(パラメータ違い)に配列に格納
def get_avg_sample_data(algo_num, each_agent_reward):
    # エージェントごとの後半の合計報酬が求まったので、同じアルゴリズム(パラメータ違いで10個)毎に(全4つのアルゴリズム)平均報酬を格納
    reward_graph = np.zeros((algo_num, 10))
    for i in range(algo_num):
        for j in range(10):
            n = int((10*i + j) / 10.0)
            reward_graph[n, j] = each_agent_reward[10 * i + j]
    print(reward_graph)

    return reward_graph


def main():
    env = ev.Environment(ACT_NUM, IS_STATIONARY)
    agent_list = []

    # ステップパラメータによる結果の変化を比較するとき
    agent_list = [ag.ActionValue(k=ACT_NUM, eps=0.1, is_const_param=0),
                  ag.ActionValue(k=ACT_NUM, eps=0.1, is_const_param=1)]

    # それぞれのアルゴリズムでのパラメータ毎の結果の変化を比較するとき(やせた山があるかどうか)
    # for i in range(-7, 3):
    #     agent_list.append(ag.ActionValue(k=ACT_NUM, eps=2**i, is_const_param=0))
    # for i in range(-7, 3):
    #     agent_list.append(ag.GradientBandit(k=ACT_NUM, alpha=2**i))
    # for i in range(-7, 3):
    #     agent_list.append(ag.UCB(k=ACT_NUM, c=2**i))
    # for i in range(-7, 3):
    #     agent_list.append(ag.ActionValue(k=ACT_NUM, eps=0.0, is_const_param=1, bias_q=2**i))
    # for i in range(-7, 3):
    #     agent_list.append(ag.ActionValue(k=ACT_NUM, eps=2**i, is_const_param=1))

    agent_num = len(agent_list)

    accuracy_graph = np.zeros((agent_num, STEP_NUM))
    reward_graph = np.zeros((agent_num, STEP_NUM))
    regret_graph = np.zeros((agent_num, STEP_NUM))

    for agent in agent_list:
        agent.initialize()

    for i, agent in enumerate(agent_list):
        for sim in range(SIM_NUM):
            print(i, " ", sim)
            env.initialize()
            sum_regret = 0
            agent.initialize()
            for step in range(STEP_NUM):
                env.update()
                # print(env.prob)
                selected = agent.select_act()
                reward = env.get_reward(selected)
                #print(reward)
                agent.update(selected, reward)

                accuracy_graph[i, step] += env.get_correct_act(selected)
                reward_graph[i, step] += reward
                sum_regret += env.get_regret(selected)
                regret_graph[i, step] += sum_regret + env.get_regret(selected)

    accuracy_graph /= SIM_NUM
    reward_graph /= SIM_NUM
    regret_graph /= SIM_NUM

    ##### each params #######
    # # データの後半の平均報酬をデータに落とし込む
    # algo_num = int(agent_num / 10)
    # each_agent_reward = get_second_half_data(agent_num, reward_graph)
    # reward_graph = np.zeros((algo_num, 10))
    # reward_graph = get_avg_sample_data(algo_num, each_agent_reward)

    labels = ["sample average(1/n)", "constant step-size(α=0.1)"]
    # labels = ["ε-greedy(1/n)", "gradient bandit", "UCB", "greedy with optimistic initialization α=0.1", "ε-greedy(α=0.1)"]
    file_name = "each_params"
    # plot_graph(reward_graph, labels, file_name, xlabel="params 2^i", ylabel="Average reward over first 100000 steps")

    file_name = "accuracy_action-value_nonstationary_s2000"
    plot_graph(accuracy_graph, labels, file_name, ylabel="Accuracy")

    file_name = "reward_action-value_nonstationry_s2000"
    plot_graph(reward_graph, labels, file_name, ylabel="Average reward")


    # ##### 後半の平均報酬をアルゴリズムごとにデータとして格納 #####
    # # 一定ステップを超えた後の平均報酬をグラフにする
    # second_half_reward = np.zeros((agent_num, int(STEP_NUM / 2)))
    # each_agent_reward = np.zeros(agent_num)
    # # それぞれのエージェントの1000回から最後(2000)までを代入
    # second_half_reward = reward_graph[0:agent_num, int(STEP_NUM / 2):STEP_NUM + 1]
    # # print(reward_temp0)
    # for i in range(agent_num):
    #     # 得られた配列をそれぞれのエージェントごとにすべて合計する
    #     each_agent_reward[i] = np.sum(second_half_reward[i]) / (int(STEP_NUM / 2))
    # # print(reward_temp1)
    # # エージェントごとの後半の合計報酬が求まったので、同じアルゴリズム(10個)毎に(全4つのアルゴリズム)平均報酬を格納
    # reward_graph = np.zeros((4, 10))
    # for i in range(4):
    #     for j in range(10):
    #         n = int((10*i + j) / 10.0)
    #         reward_graph[n, j] = each_agent_reward[10 * i + j]
    # print(reward_graph)

if __name__ == '__main__':
    main()
