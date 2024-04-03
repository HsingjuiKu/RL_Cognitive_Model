import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from env import *


def train(cfg, env, agent):
    print(f"Train Start!")
    print(f"环境：{cfg['env_name']}，算法：{cfg['algo_name']}，设备：{cfg['device']}")
    rewards = []  # record rewards for all episodes
    steps = []
    for i_ep in range(cfg["train_eps"]):
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset()  # reset and obtain initial state
        for _ in range(cfg['ep_max_steps']):
            ep_step += 1
            action = agent.sample_action(state)  # sample action
            next_state, reward, done, _ = env.step(action)  # update env and return transitions
            agent.memory.push((state, action, reward, next_state, done))  # save transitions
            state = next_state  # update next state for env
            agent.update()  # update agent
            ep_reward += reward  #
            if done:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            print(f"回合：{i_ep + 1}/{cfg['train_eps']}，奖励：{ep_reward:.2f}，Epislon: {agent.epsilon:.3f}")
    print("完成训练！")
    env.close()
    res_dic = {'episodes': range(len(rewards)), 'rewards': rewards, 'steps': steps}
    return res_dic

def test(cfg, env, agent):
    print("开始测试！")
    print(f"环境：{cfg['env_name']}，算法：{cfg['algo_name']}，设备：{cfg['device']}")
    rewards = []  # record rewards for all episodes
    steps = []
    for i_ep in range(cfg['test_eps']):
        ep_reward = 0# reward per episode
        ep_step = 0
        state = env.reset()  # reset and obtain initial state
        for _ in range(cfg['ep_max_steps']):
            ep_step+=1
            action = agent.predict_action(state)  # predict action
            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_reward += reward
            if done:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        print(f"回合：{i_ep+1}/{cfg['test_eps']}，奖励：{ep_reward:.2f}")
    print("完成测试！")
    env.close()
    return {'episodes':range(len(rewards)),'rewards':rewards,'steps':steps}


def get_args():
    """
    """
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--algo_name',default='DQN_Exercise1',type=str,help="name of algorithm")
    parser.add_argument('--env_name',default='CartPole-v0',type=str,help="name of environment")
    parser.add_argument('--train_eps',default=200,type=int,help="episodes of training") # 训练的回合数
    parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing") # 测试的回合数
    parser.add_argument('--ep_max_steps',default = 100000,type=int,help="steps per episode, much larger value can simulate infinite steps")
    parser.add_argument('--gamma',default=0.99,type=float,help="discounted factor") # 折扣因子
    parser.add_argument('--epsilon_start',default=0.95,type=float,help="initial value of epsilon") #  e-greedy策略中初始epsilon
    parser.add_argument('--epsilon_end',default=0.01,type=float,help="final value of epsilon") # e-greedy策略中的终止epsilon
    parser.add_argument('--epsilon_decay',default=200,type=int,help="decay rate of epsilon") # e-greedy策略中epsilon的衰减率
    parser.add_argument('--memory_capacity',default=200000,type=int) # replay memory的容量
    parser.add_argument('--memory_warmup_size',default=200,type=int) # replay memory的预热容量
    parser.add_argument('--batch_size',default=64,type=int,help="batch size of training") # 训练时每次使用的样本数
    parser.add_argument('--targe_update_fre',default=200,type=int,help="frequency of target network update") # target network更新频率
    parser.add_argument('--seed',default=10,type=int,help="seed")
    parser.add_argument('--lr',default=0.0001,type=float,help="learning rate")
    parser.add_argument('--device',default='cpu',type=str,help="cpu or gpu")
    args = parser.parse_args([])
    args = {**vars(args)}  # type(dict)
    return args

def smooth(data, weight=0.9):
    '''用于平滑曲线，类似于Tensorboard中的smooth

    Args:
        data (List):输入数据
        weight (Float): 平滑权重，处于0-1之间，数值越高说明越平滑，一般取0.9

    Returns:
        smoothed (List): 平滑后的数据
    '''
    last = data[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_rewards(rewards,cfg,path=None,tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{tag}ing curve on {cfg['device']} of {cfg['algo_name']} for {cfg['env_name']}")
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()


# 获取参数
cfg = get_args()
# 训练
env, agent = env_agent_config(cfg)
res_dic = train(cfg, env, agent)

plot_rewards(res_dic['rewards'], cfg, tag="train")
# 测试
res_dic = test(cfg, env, agent)
plot_rewards(res_dic['rewards'], cfg, tag="test")  # 画出结果
