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
    # parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4", help="name of the game")
    parser.add_argument("--env", type=str, default="AlienNoFrameskip-v4", help="name of the game")
    # parser.add_argument("--env", type=str, default="SpaceInvadersNoFrameskip-v4", help="name of the game")
    parser.add_argument("--replay-buffer-size", type=int, default=int(1e7), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    # Although named num_steps but run in episode of games
    parser.add_argument("--num-steps", type=int, default=int(2e6), help="total number of steps to run the environment for")
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

def shaded_quantiles(data):
    return np.quantile(data, 0.1), np.quantile(data, 0.9)


class AgentRunner:
    def __init__(self, agent, env, replay_buffer, args, name):
        self.agent = agent
        self.env = env
        self.replay_buffer = replay_buffer
        self.args = args
        self.name = name
        self.q_values_collection = []
        self.rewards_collection = []
        # Plot Collecting
        self.average_q_values = []
        self.min_q_values = []
        self.max_q_values = []
        self.average_reward = []
        self.min_reward = []
        self.max_reward = []
        self.actual_values = []


    def step(self, episode):
        for _ in range(6):  # Run the episode 6 times with different random seeds
            state = self.env.reset()
            episode_q_values = []
            episode_rewards = []
            done = False
            while not done:
                fraction = min(1.0, float(episode) / self.args.eps_fraction)
                eps_threshold = self.args.eps_start + fraction * (self.args.eps_end - self.args.eps_start)
                sample = random.random()
                if sample > eps_threshold:
                    action = self.agent.act(np.array(state))
                else:
                    action = self.env.action_space.sample()

                next_state, reward, done, _ = self.env.step(action)
                episode_q_values.append(self.agent.compute_q_value(np.array(state), action))
                episode_rewards.append(reward)
                self.agent.memory.add(state, action, reward, next_state, float(done))
                state = next_state

            self.q_values_collection.append(episode_q_values)
            self.rewards_collection.append(episode_rewards)

        # Compute statistics based on the 6 runs
        average_q_values = np.mean([np.mean(values) for values in self.q_values_collection])
        min_q_value = np.percentile([np.min(values) for values in self.q_values_collection], 10)
        max_q_value = np.percentile([np.max(values) for values in self.q_values_collection], 90)
        self.average_q_values.append(average_q_values)
        self.max_q_values.append(max_q_value)
        self.min_q_values.append(min_q_value)


        average_reward = np.mean([np.sum(values) for values in self.rewards_collection])
        min_reward = np.percentile([np.min(values) for values in self.rewards_collection], 10)
        max_reward = np.percentile([np.max(values) for values in self.rewards_collection], 90)
        self.average_reward.append(average_reward)
        self.min_reward.append(min_reward)
        self.max_reward.append(max_reward)

        if episode % int(1e3) == 0 and episode != 0:
            # Printing the results
            print(f"Episode {episode}")
            print(
                f"Q-Value stats - Average: {average_q_values}, Min (10 percentile): {min_q_value}, Max (90 percentile): {max_q_value}")
            print(
                f"Reward stats - Average: {average_reward}, Min (10 percentile): {min_reward}, Max (90 percentile): {max_reward}")
        # Continue with learning and updates
        if episode > self.args.learning_starts and episode % self.args.learning_freq == 0:
            self.agent.optimise_td_loss()

        if episode > self.args.learning_starts and episode % self.args.target_update_freq == 0:
            self.agent.update_target_network()

        if episode % int(1e5) == 0:
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

    # Setup third environment, replay buffer and agent for SSDQN
    env3 = make_env(args.env, args.seed)
    replay_buffer3 = ReplayBuffer(args.replay_buffer_size)
    agent3 = DQNAgent(
        env3.observation_space,
        env3.action_space,
        replay_buffer3,
        dqn_variant="ssdqn",
        lr=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
    )

    # Setup fourth environment, replay buffer and agent for SCDQN
    env4 = make_env(args.env, args.seed)
    replay_buffer4 = ReplayBuffer(args.replay_buffer_size)
    agent4 = DQNAgent(
        env4.observation_space,
        env4.action_space,
        replay_buffer4,
        dqn_variant="scdqn",
        lr=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
    )

    # Setup fifth environment, replay buffer and agent for CSSDQN
    env5 = make_env(args.env, args.seed)
    replay_buffer5 = ReplayBuffer(args.replay_buffer_size)
    agent5 = DQNAgent(
        env5.observation_space,
        env5.action_space,
        replay_buffer5,
        dqn_variant="cssdqn",
        lr=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
    )

    runner1 = AgentRunner(agent1, env1, replay_buffer1, args, name="DQN")
    runner2 = AgentRunner(agent2, env2, replay_buffer2, args, name="Double DQN")
    runner3 = AgentRunner(agent3, env3, replay_buffer3, args, name="Smooth DQN(Softmax)")
    runner4 = AgentRunner(agent4, env4, replay_buffer4, args, name="Smooth DQN(Clipped Max)")
    runner5 = AgentRunner(agent5, env5, replay_buffer5, args, name="Smooth DQN(Clipped Softmax)")

    for episode in range(args.num_steps):
        runner1.step(episode)
        runner2.step(episode)
        runner3.step(episode)
        runner4.step(episode)
        runner5.step(episode)

    # Average Q value
    average_q_values1 = np.array(runner1.average_q_values)
    average_q_values1_min = np.array(runner1.min_q_values)
    average_q_values1_max = np.array(runner1.max_q_values)

    average_q_values2 = np.array(runner2.average_q_values)
    average_q_values2_min = np.array(runner2.min_q_values)
    average_q_values2_max = np.array(runner2.max_q_values)

    average_q_values3 = np.array(runner3.average_q_values)
    average_q_values3_min = np.array(runner3.min_q_values)
    average_q_values3_max = np.array(runner3.max_q_values)

    average_q_values4 = np.array(runner4.average_q_values)
    average_q_values4_min = np.array(runner4.min_q_values)
    average_q_values4_max = np.array(runner4.max_q_values)

    average_q_values5 = np.array(runner5.average_q_values)
    average_q_values5_min = np.array(runner5.min_q_values)
    average_q_values5_max = np.array(runner5.max_q_values)


    #Reward
    reward_1 = np.array(runner1.average_reward)
    reward_2 = np.array(runner2.average_reward)
    reward_3 = np.array(runner3.average_reward)
    reward_4 = np.array(runner4.average_reward)
    reward_5 = np.array(runner5.average_reward)

    reward_1_min = np.array(runner1.min_reward)
    reward_2_min = np.array(runner2.min_reward)
    reward_3_min = np.array(runner3.min_reward)
    reward_4_min = np.array(runner4.min_reward)
    reward_5_min = np.array(runner5.min_reward)

    reward_1_max = np.array(runner1.max_reward)
    reward_2_max = np.array(runner2.max_reward)
    reward_3_max = np.array(runner3.max_reward)
    reward_4_max = np.array(runner4.max_reward)
    reward_5_max = np.array(runner5.max_reward)



    reference_length = len(average_q_values1)

    mean_actual_value1 = np.mean(np.array(runner1.actual_values))
    mean_actual_value2 = np.mean(np.array(runner2.actual_values))
    mean_actual_value3 = np.mean(np.array(runner3.actual_values))
    mean_actual_value4 = np.mean(np.array(runner4.actual_values))
    mean_actual_value5 = np.mean(np.array(runner5.actual_values))

    # Create arrays with the mean values
    mean_actual_values1 = np.full((reference_length,), mean_actual_value1)
    mean_actual_values2 = np.full((reference_length,), mean_actual_value2)
    mean_actual_values3 = np.full((reference_length,), mean_actual_value3)
    mean_actual_values4 = np.full((reference_length,), mean_actual_value4)
    mean_actual_values5 = np.full((reference_length,), mean_actual_value5)


    # Plot mean actual values
    plt.plot(mean_actual_values1, label="DQN Mean Actual Value", color='blue', linestyle='--')
    plt.plot(mean_actual_values2, label="Double DQN Mean Actual Value", color='orange', linestyle='--')
    plt.plot(mean_actual_values3, label="Smooth DQN(Softmax) Mean Actual Value", color='green', linestyle='--')
    plt.plot(mean_actual_values4, label="Smooth DQN(Clipped Max) Mean Actual Value", color='red', linestyle='--')
    plt.plot(mean_actual_values5, label="Smooth DQN(Clipped Softmax) Mean Actual Value", color='purple', linestyle='--')

    # Plot the results
    plt.plot(average_q_values1, label="DQN Estimate", color='blue')
    plt.fill_between(range(len(average_q_values1)), average_q_values1_min, average_q_values1_max, color='blue',
                     alpha=.1)
    #
    plt.plot(average_q_values2, label="Double DQN Estimate", color='orange')
    plt.fill_between(range(len(average_q_values2)), average_q_values2_min, average_q_values2_max, color='orange',
                     alpha=.1)

    # Plot the results for SSDQN
    plt.plot(average_q_values3, label="Smooth DQN(Softmax) Estimate", color='green')
    plt.fill_between(range(len(average_q_values3)), average_q_values3_min, average_q_values3_max, color='green',
                     alpha=.1)

    # Plot the results for SCDQN
    plt.plot(average_q_values4, label="Smooth DQN(Clipped Max) Estimate", color='red')
    plt.fill_between(range(len(average_q_values4)), average_q_values4_min, average_q_values4_max, color='red', alpha=.1)

    # Plot the results for SCDQN
    plt.plot(average_q_values5, label="Smooth DQN(Clipped Softmax) Estimate", color='purple')
    plt.fill_between(range(len(average_q_values5)), average_q_values5_min, average_q_values5_max, color='purple', alpha=.1)

    plt.title("Average Q-values over Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Value Estimate")
    plt.legend()
    plt.grid(True)
    plt.show()





    # Now plot the episode rewards
    # plt.figure(figsize=(10, 6))
    plt.plot(reward_1, label='DQN Episode Rewards', color='blue')
    plt.fill_between(range(len(reward_1)), reward_1_min, reward_1_max, color='blue',
                     alpha=.1)
    plt.plot(reward_2, label='Double DQN Episode Rewards', color='red')
    plt.fill_between(range(len(reward_2)), reward_2_min, reward_2_max, color='red',
                     alpha=.1)
    plt.plot(reward_3, label='DQN Episode Rewards', color='blue')
    plt.fill_between(range(len(reward_3)), reward_3_min, reward_3_max, color='blue',
                     alpha=.1)
    plt.plot(reward_4, label='DQN Episode Rewards', color='blue')
    plt.fill_between(range(len(reward_4)), reward_4_min, reward_4_max, color='blue',
                     alpha=.1)
    plt.plot(reward_5, label='DQN Episode Rewards', color='blue')
    plt.fill_between(range(len(reward_5)), reward_5_min, reward_5_max, color='blue',
                     alpha=.1)

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards for DQN and Double DQN')
    plt.legend()
    plt.show()

