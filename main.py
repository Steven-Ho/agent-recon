import gym
import numpy as np 
import tensorflow as tf 
import argparse
from ddqn import DDQN
from buffer import FullReplayMemory
import itertools
import time

parser = argparse.ArgumentParser(description='Duoble DQN Baseline')
parser.add_argument('--scenario', type=str, default="Seaquest-v0", help="environment")
parser.add_argument('--seed', type=int, default=123, help="random seed for env")
parser.add_argument('--num_episodes', type=int, default=40000, help='number of episodes for training')
parser.add_argument('--max_episode_len', type=int, default=100, help='maximum episode length')
parser.add_argument('--gamma', type=float, default=0.95, help='discount factor (default: 0.95)')
parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon-greedy parameter (initial: 0.1)')
parser.add_argument('--buffer_size', type=int, default=1e6, help='maximum size for replay buffer')
parser.add_argument('--update_interval', type=int, default=1, help='update q network for every N steps')
parser.add_argument('--startup_steps', type=int, default=1000, help='initial rollout steps before training')
parser.add_argument('--batch_size', type=int, default=128, help='sample size for training')
args = parser.parse_args()

env = gym.make(args.scenario)
env.seed(args.seed)
np.random.seed(args.seed)

obs_shape_list = env.observation_space.shape
action_shape = env.action_space.n

qnet = DDQN(obs_shape_list, action_shape, args)
memory = FullReplayMemory(args.buffer_size)

total_numsteps = 0
timestep = 0
t_start = time.time()
for i_episode in itertools.count(1):
        
    obs = env.reset()
    obs = obs/255.
    done = False

    episode_reward = 0
    for t in range(args.max_episode_len):
        timestep += 1
        action = qnet.act(np.expand_dims(obs, 0))
        new_obs, reward, done, _ = env.step(action)
        new_obs = new_obs/255.

        memory.push((obs, action, reward, done, new_obs))
        obs = new_obs

        if timestep > args.startup_steps:
            if timestep % args.update_interval == 0:
                obs, action, reward, done, new_obs = memory.sample(args.batch_size)
                qnet.update((obs, action, reward, done, new_obs))

        episode_reward += reward
        if done:
            break

print("End of script.")