import gym
import numpy as np 
import tensorflow as tf 
import argparse
from ddqn import DDQN
import itertools
import time

parser = argparse.ArgumentParser(description='Duoble DQN Baseline')
parser.add_argument('--scenario', type=str, default="Seaquest-v0", help="environment")
parser.add_argument('--seed', type=int, default=123, help="random seed for env")
parser.add_argument('--num_episodes', type=int, default=40000, help='number of episodes for training')
parser.add_argument('--max_episode_len', type=int, default=100, help='maximum episode length')
parser.add_argument('--gamma', type=float, default=0.95, help='discount factor (default: 0.99)')

args = parser.parse_args()

env = gym.make(args.scenario)
env.seed(args.seed)
np.random.seed(args.seed)

obs_shape_list = env.observation_space.shape
action_shape = env.action_space.n

qnet = DDQN(obs_shape_list, action_shape, args)
total_numsteps = 0
updates = 0
t_start = time.time()
for i_episode in itertools.count(1):
        
    state = env.reset()
    state = np.expand_dims(state, 0)/255.

q1, q2 = qnet.forward(state)
q1 = q1.numpy().squeeze()
q2 = q2.numpy().squeeze()
print("End of script.")