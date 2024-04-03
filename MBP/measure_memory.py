import numpy as np
from scipy.stats import rv_discrete
from scipy.special import softmax
import matplotlib.pyplot as plt
from copy import deepcopy
from statistics import mean
import time
from memory_profiler import memory_usage

def rs(state, action, a, b, c, d, r, l):
    stp = rt = None
    if state == a and action == r:
        stp, rt = c, 0
    elif state == a and action == l:
        stp, rt = b, 0
    elif state == b:
        stp, rt = d, rv + np.random.randn()
    return stp, rt


def rbar(state, action, a, b, r, l, rv):
    if state == a and (action == r or action == l):
        return 0
    elif state == b:
        return rv
    else:
        return 0

def update(state, action, Q, a, b, c, d, r, l):
    maxQ = np.zeros(4)
    for s in range(4):
        maxQ[s] = np.max(Q[s])
    if state == a and action == r:
        return maxQ[c]
    elif state == a and action == l:
        return maxQ[b]
    elif state == b:
        return maxQ[d]
    else:
        return 0


def zeroQ(Q):
    for i in range(4):
        Q[i].fill(0.0)

def alphat(alpha0, t):
    return alpha0 / (1 + 0.001 * t)

def softprob(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)

def argmaxR(x):
    p = softmax(1000 * x)
    return np.random.choice(np.arange(len(x)), p=p)

# define states and actions
a, b, c, d = 0, 1, 2, 3
l, r = 0, 1

rv = -0.1
gamma = 0.99
alpha0 = 0.1
eps = 0.1
Exps = 10000
Episodes = 2000

# Initialization
# Q = [np.zeros((2,)), np.zeros((8,)), np.zeros((1,)), np.zeros((1,))]
# zeroQ(Q)

Q = [np.random.uniform(low=0.0001, high=0.001, size=(2,)),
     np.random.uniform(low=0.0001, high=0.001, size=(8,)),
     np.random.uniform(low=0.0001, high=0.001, size=(1,)),
     np.random.uniform(low=0.0001, high=0.001, size=(1,))]

QA = [np.copy(q) for q in Q]
QB = [np.copy(q) for q in Q]

# Get the true solution to the MDP
Qnew = deepcopy(Q)
zeroQ(Qnew)

for _ in range(1000):
    for state in [a, b, c, d]:
        for action in range(len(Q[state])):
            Qnew[state][action] = rbar(state, action, a, b, r, l, rv) + gamma * update(state, action, Q, a, b, c, d, r, l)
    Q = deepcopy(Qnew)

# 这部分突破口
QmaxTrue = Q[a][l]

# Initialize other necessary variables
Qend = np.zeros((Exps, Episodes))
dQend = np.zeros((Exps, Episodes))
sQend = np.zeros((Exps, Episodes))
cQend = np.zeros((Exps, Episodes))


def Q_learning(Q, a, b, c, d, r, l, alpha0, eps, Exps, Episodes):
    Qend = np.zeros((Exps, Episodes))
    actionsQ = []

    for experiment in range(Exps):
        zeroQ(Q)
        ct = [0]
        actionsQ_ex = []
        for episode in range(Episodes):
            st = [a]
            at = [0]  # dummy initialization
            while True:
                ct[0] += 1
                alpha = alphat(alpha0, ct[0])
                if np.random.rand() > eps:  # select action using maxQ
                    at[0] = argmaxR(Q[st[0]])
                else:
                    at[0] = np.random.choice([0, 1])
                actionsQ_ex.append(at[0])
                stp, rt = rs(st[0], at[0], a, b, c, d, r, l)
                Q[st[0]][at[0]] += alpha * (rt + gamma * np.max(Q[stp]) - Q[st[0]][at[0]])
                if stp == d or stp == c:
                    break
                st[0] = stp
            Qend[experiment, episode] = Q[a][l]
        actionsQ.append(actionsQ_ex)

    return Qend, actionsQ


def double_Q_learning(QA, QB, a, b, c, d, r, l, alpha0, eps, Exps, Episodes):
    dQend = np.zeros((Exps, Episodes))
    actionsdQ = []

    for experiment in range(Exps):
        zeroQ(QA)
        zeroQ(QB)
        ct = [0]
        actionsdQ_ex = []
        for episode in range(Episodes):
            st = [a]
            at = [1]
            while True:
                ct[0] += 1
                alpha = alphat(alpha0, ct[0])
                if np.random.rand() > eps:  # select action using maxQ
                    at[0] = argmaxR(QA[st[0]])
                else:
                    at[0] = np.random.choice([0, 1])
                actionsdQ_ex.append(at[0])
                stp, rt = rs(st[0], at[0], a, b, c, d, r, l)
                if np.random.rand() > 0.5:  # update A
                    astar = argmaxR(QA[stp])
                    QA[st[0]][at[0]] += alpha * (rt + gamma * QB[stp][astar] - QA[st[0]][at[0]])
                else:  # update B
                    bstar = argmaxR(QB[stp])
                    QB[st[0]][at[0]] += alpha * (rt + gamma * QA[stp][bstar] - QB[st[0]][at[0]])
                if stp == d or stp == c:
                    break
                st[0] = stp
            dQend[experiment, episode] = QA[a][l]
        actionsdQ.append(actionsdQ_ex)

    return dQend, actionsdQ

def measure_memory(Q_learning, double_Q_learning, QA, QB, a, b, c, d, r, l, alpha0, eps, Exps, Episodes):
    Q_learning_memory = memory_usage((Q_learning, (QA, a, b, c, d, r, l, alpha0, eps, Exps, Episodes)), interval=0.1)
    double_Q_learning_memory = memory_usage((double_Q_learning, (QA, QB, a, b, c, d, r, l, alpha0, eps, Exps, Episodes)), interval=0.1)

    return Q_learning_memory, double_Q_learning_memory

def visualize_memory_usage(Q_learning_memory, double_Q_learning_memory):
    plt.figure(figsize=(10,6))
    plt.plot(Q_learning_memory, label='Q-learning')
    plt.plot(double_Q_learning_memory, label='Double Q-learning')
    plt.ylabel('Memory usage (MB)')
    plt.xlabel('Time (10ms)')
    plt.title('Memory usage comparison between Q-learning and Double Q-learning')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    Q_learning_memory, double_Q_learning_memory = measure_memory(Q_learning, double_Q_learning, QA, QB, a, b, c, d, r, l, alpha0, eps, Exps, Episodes)

    # 打印前1000次的平均内存使用情况
    print("Q-learning average memory usage for 1000 episodes: ", mean(Q_learning_memory[:1000]))
    print("Double Q-learning average memory usage for 1000 episodes: ", mean(double_Q_learning_memory[:1000]))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1000), Q_learning_memory[:1000], label='Q Learning')
    plt.plot(range(1000), double_Q_learning_memory[:1000], label='Double Q Learning')
    plt.xlabel('Time (10ms)')
    plt.ylabel('Memory Usage (MB)')
    plt.legend()
    plt.show()
