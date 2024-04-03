import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

# 定义状态和动作空间
states = np.linspace(-5, 5, 100)
actions = np.arange(10)

# 定义目标值函数Q*(s, a)
def Q_star(s, a):
    return np.where(a < 5, np.sin(s), np.sin(s))  # 确保在每个状态，所有动作的真实值相同

# 采样一些状态，并计算对应的Q值
sample_states = np.array([np.linspace(-5, 5, 20) for _ in actions])  # 对于每个动作，都有独立的样本状态
double_sample_states = np.array([np.random.choice(states, 20, replace=False) for _ in actions])  # 对于每个动作，都有独立的样本状态
sample_Q_values = np.array([Q_star(s, a) for a, s in enumerate(sample_states)])
double_sample_Q_values = np.array([Q_star(s, a) for a, s in enumerate(double_sample_states)])

# 利用多项式逼近
degrees = [6, 6, 9]
fit_func = [Polynomial.fit(s, q, d) for s, q, d in zip(sample_states, sample_Q_values, degrees)]
double_fit_func = [Polynomial.fit(s, q, d) for s, q, d in zip(double_sample_states, double_sample_Q_values, degrees)]

# 对所有状态做出预测
predicted_Q_values = np.array([f(states) for f in fit_func])
double_predicted_Q_values = np.array([f(states) for f in double_fit_func])

# 选择最大的动作值作为最优动作
optimal_actions = np.argmax(predicted_Q_values, axis=0)
double_optimal_actions = np.argmax(double_predicted_Q_values, axis=0)

# 绘制图像
fig, axes = plt.subplots(3, 3, figsize=(15, 10))

for i in range(3):
    # 左列
    axes[i, 0].plot(states, Q_star(states, i), 'purple', label='True Q*')
    axes[i, 0].plot(sample_states[i], sample_Q_values[i], 'go', label='Sampled Q')
    axes[i, 0].plot(states, predicted_Q_values[i], 'g', label='Estimated Q')
    # axes[i, 0].legend()
    axes[i, 0].set_title('True, Sampled, and Estimated Q for action {}'.format(i))

    # 中列
    axes[i, 1].plot(states, Q_star(states, optimal_actions), 'purple', label='True Q*')
    axes[i, 1].plot(states, predicted_Q_values[i], 'g', label='Estimated Q')
    axes[i, 1].plot(states, Q_star(states, np.argmax(predicted_Q_values, axis=0)), 'k--', label='Max Estimated Q')
    # axes[i, 1].legend()
    axes[i, 1].set_title('True, Estimated, and Max Estimated Q for action {}'.format(i))

    # 右列
    axes[i, 2].plot(states, Q_star(states, optimal_actions) - predicted_Q_values[i], 'orange', label='Overestimation')
    axes[i, 2].plot(states, Q_star(states, double_optimal_actions) - double_predicted_Q_values[i], 'blue', label='Double Q-learning Overestimation')
    # axes[i, 2].legend()
    axes[i, 2].set_title('Overestimation and Double Q-learning Overestimation for action {}'.format(i))

plt.tight_layout()
plt.show()
