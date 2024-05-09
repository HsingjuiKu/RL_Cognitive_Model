import argparse
import random
import matplotlib.pyplot as plt
import numpy as np

import torch

from Agent import DQNAgent
from replay_buffer import ReplayBuffer
from wrappers import *


def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for Atari games")
    parser.add_argument("--seed", type=int, default=42, help="which seed to use")
    parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4", help="name of the game")
    # parser.add_argument("--env", type=str, default="SpaceInvadersNoFrameskip-v4", help="name of the game")
    parser.add_argument("--replay-buffer-size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--num-steps", type=int, default=int(1e6), help="total number of steps to run the environment for")
    parser.add_argument("--batch-size", type=int, default=32, help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-starts", type=int, default=10000, help="number of steps before learning starts")
    parser.add_argument("--learning-freq", type=int, default=1, help="number of iterations between every optimization step")
    parser.add_argument("--target-update-freq", type=int, default=1000, help="number of iterations between every target network update")
    # parser.add_argument("--use-double-dqn", type=bool, default=True, help="use double deep Q-learning")
    parser.add_argument("--dqn-variant", type=str, default="dqn", help="DQN variant to use, either 'dqn' or 'ddqn'")
    parser.add_argument("--eps-start", type=float, default=1.0, help="e-greedy start threshold")
    parser.add_argument("--eps-end", type=float, default=0.02, help="e-greedy end threshold")
    parser.add_argument("--eps-fraction", type=float, default=0.1, help="fraction of num-steps")
    return parser.parse_args()


def make_env(env_name, seed):
    assert "NoFrameskip" in env_name
    env = gym.make(env_name)
    env.seed(seed)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    return env


class AgentRunner:
    def __init__(self, agent, env, replay_buffer, args, name):
        self.agent = agent
        self.env = env
        self.replay_buffer = replay_buffer
        self.args = args
        self.eps_timesteps = self.args.eps_fraction * float(self.args.num_steps)
        self.state = self.env.reset()
        self.score = 0
        self.name = name
        self.actual_values = []

    def step(self, t):
        fraction = min(1.0, float(t) / self.eps_timesteps)
        eps_threshold = self.args.eps_start + fraction * (self.args.eps_end - self.args.eps_start)
        sample = random.random()
        if sample > eps_threshold:
            action = self.agent.act(np.array(self.state))
        else:
            action = self.env.action_space.sample()

        next_state, reward, done, _ = self.env.step(action)
        self.score += reward
        self.agent.memory.add(self.state, action, reward, next_state, float(done))
        self.state = next_state

        if done:
            # 结束了一个 episode, 记录本次 episode 的奖励到 all_episode_rewards 中
            self.agent.act(np.array(self.state), reward=self.score, done=True)  # Updating the act method with done = True
            self.state = self.env.reset()
            self.score = 0

        if t % int(1e5) == 0:
            print(f"Agent: {self.name}, Step: {t}, Score: {self.score}")

        if t > self.args.learning_starts and t % self.args.learning_freq == 0:
            self.agent.optimise_td_loss()

        if t > self.args.learning_starts and t % self.args.target_update_freq == 0:
            self.agent.update_target_network()

        if t % int(1e5) == 0:
            self.actual_values.append(self.agent.run_policy_and_compute_actual_rewards(self.env))


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Setup first environment, replay buffer and agent
    env1 = make_env(args.env, args.seed)
    replay_buffer1 = ReplayBuffer(args.replay_buffer_size)
    agent1 = DQNAgent(
        env1.observation_space,
        env1.action_space,
        replay_buffer1,
        dqn_variant="dqn",
        lr=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma
    )
    #
    # Setup second environment, replay buffer and agent
    env2 = make_env(args.env, args.seed)
    replay_buffer2 = ReplayBuffer(args.replay_buffer_size)
    agent2 = DQNAgent(
        env2.observation_space,
        env2.action_space,
        replay_buffer2,
        dqn_variant="ddqn",
        lr=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma
    )

    # # Setup third environment, replay buffer and agent for SSDQN
    # env3 = make_env(args.env, args.seed)
    # replay_buffer3 = ReplayBuffer(args.replay_buffer_size)
    # agent3 = DQNAgent(
    #     env3.observation_space,
    #     env3.action_space,
    #     replay_buffer3,
    #     dqn_variant="ssdqn",
    #     lr=args.lr,
    #     batch_size=args.batch_size,
    #     gamma=args.gamma,
    # )
    #
    # # Setup fourth environment, replay buffer and agent for SCDQN
    # env4 = make_env(args.env, args.seed)
    # replay_buffer4 = ReplayBuffer(args.replay_buffer_size)
    # agent4 = DQNAgent(
    #     env4.observation_space,
    #     env4.action_space,
    #     replay_buffer4,
    #     dqn_variant="scdqn",
    #     lr=args.lr,
    #     batch_size=args.batch_size,
    #     gamma=args.gamma,
    # )

    # # Setup fifth environment, replay buffer and agent for CSSDQN
    # env5 = make_env(args.env, args.seed)
    # replay_buffer5 = ReplayBuffer(args.replay_buffer_size)
    # agent5 = DQNAgent(
    #     env5.observation_space,
    #     env5.action_space,
    #     replay_buffer5,
    #     dqn_variant="cssdqn",
    #     lr=args.lr,
    #     batch_size=args.batch_size,
    #     gamma=args.gamma,
    # )

    runner1 = AgentRunner(agent1, env1, replay_buffer1, args, name="DQN")
    runner2 = AgentRunner(agent2, env2, replay_buffer2, args, name="Double DQN")
    # runner3 = AgentRunner(agent3, env3, replay_buffer3, args, name="Smooth DQN(Softmax)")
    # runner4 = AgentRunner(agent4, env4, replay_buffer4, args, name="Smooth DQN(Clipped Max)")
    # runner5 = AgentRunner(agent5, env5, replay_buffer5, args, name="Smooth DQN(Clipped Softmax)")

    for t in range(args.num_steps):
        runner1.step(t)
        runner2.step(t)
        # runner3.step(t)
        # runner4.step(t)
        # runner5.step(t)

    # Get values for plotting for DQN
    predicted_values1 = np.array(runner1.agent.get_predicted_values())
    predicted_values_min1 = np.array(runner1.agent.get_predicted_values_min())
    predicted_values_max1 = np.array(runner1.agent.get_predicted_values_max())
    #
    # Get values for plotting for Double DQN
    predicted_values2 = np.array(runner2.agent.get_predicted_values())
    predicted_values_min2 = np.array(runner2.agent.get_predicted_values_min())
    predicted_values_max2 = np.array(runner2.agent.get_predicted_values_max())

    # # Get values for plotting for SSDQN
    # predicted_values3 = np.array(runner3.agent.get_predicted_values())
    # predicted_values_min3 = np.array(runner3.agent.get_predicted_values_min())
    # predicted_values_max3 = np.array(runner3.agent.get_predicted_values_max())

    # # Get values for plotting for SCDQN
    # predicted_values4 = np.array(runner4.agent.get_predicted_values())
    # predicted_values_min4 = np.array(runner4.agent.get_predicted_values_min())
    # predicted_values_max4 = np.array(runner4.agent.get_predicted_values_max())

    # # Get values for plotting for CSSDQN
    # predicted_values5 = np.array(runner5.agent.get_predicted_values())
    # predicted_values_min5 = np.array(runner5.agent.get_predicted_values_min())
    # predicted_values_max5 = np.array(runner5.agent.get_predicted_values_max())

    reference_length = len(predicted_values1)

    # Calculate mean actual values for each agent
    mean_actual_value1 = np.mean(np.array(runner1.actual_values))
    mean_actual_value2 = np.mean(np.array(runner2.actual_values))
    # mean_actual_value3 = np.mean(np.array(runner3.actual_values))
    # mean_actual_value4 = np.mean(np.array(runner4.actual_values))
    # mean_actual_value5 = np.mean(np.array(runner5.actual_values))

    # Create arrays with the mean values
    mean_actual_values1 = np.full((reference_length,), mean_actual_value1)
    mean_actual_values2 = np.full((reference_length,), mean_actual_value2)
    # mean_actual_values3 = np.full((reference_length,), mean_actual_value3)
    # mean_actual_values4 = np.full((reference_length,), mean_actual_value4)
    # mean_actual_values5 = np.full((reference_length,), mean_actual_value5)

    episode_rewards1 = np.array(runner1.agent.all_episode_rewards)
    episode_rewards2 = np.array(runner2.agent.all_episode_rewards)

    # Plot mean actual values
    plt.plot(mean_actual_values1, label="DQN Mean Actual Value", color='blue', linestyle='--')
    plt.plot(mean_actual_values2, label="Double DQN Mean Actual Value", color='orange', linestyle='--')
    # plt.plot(mean_actual_values3, label="Smooth DQN(Softmax) Mean Actual Value", color='green', linestyle='--')
    # plt.plot(mean_actual_values4, label="Smooth DQN(Clipped Max) Mean Actual Value", color='red', linestyle='--')
    # plt.plot(mean_actual_values5, label="Smooth DQN(Clipped Softmax) Mean Actual Value", color='purple', linestyle='--')

    # Plot the results
    plt.plot(predicted_values1, label="DQN Estimate", color='blue')
    # plt.fill_between(range(len(predicted_values1)), predicted_values_min1, predicted_values_max1, color='blue',
    #                  alpha=.1)
    #
    plt.plot(predicted_values2, label="Double DQN Estimate", color='orange')
    # plt.fill_between(range(len(predicted_values2)), predicted_values_min2, predicted_values_max2, color='orange',
    #                  alpha=.1)

    # # Plot the results for SSDQN
    # plt.plot(predicted_values3, label="Smooth DQN(Softmax) Estimate", color='green')
    # plt.fill_between(range(len(predicted_values3)), predicted_values_min3, predicted_values_max3, color='green',
    #                  alpha=.1)
    #
    # # Plot the results for SCDQN
    # plt.plot(predicted_values4, label="Smooth DQN(Clipped Max) Estimate", color='red')
    # plt.fill_between(range(len(predicted_values4)), predicted_values_min4, predicted_values_max4, color='red', alpha=.1)

    # # Plot the results for SCDQN
    # plt.plot(predicted_values5, label="Smooth DQN(Clipped Softmax) Estimate", color='purple')
    # plt.fill_between(range(len(predicted_values5)), predicted_values_min5, predicted_values_max5, color='purple', alpha=.1)

    plt.legend()
    plt.xlabel("Training Step")
    plt.ylabel("Value Estimate")
    plt.show()



    # Now plot the episode rewards
    # plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards1, label='DQN Episode Rewards')
    plt.plot(episode_rewards2, label='Double DQN Episode Rewards')

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards for DQN and Double DQN')
    plt.legend()
    # plt.tight_layout()
    plt.show()



