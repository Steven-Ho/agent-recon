import gym
import numpy as np 
import tensorflow as tf 
import argparse

parser = argparse.ArgumentParser(description='Duoble DQN Baseline')
parser.add_argument('--scenario', type=str, default="Seaquest-v0", help="environment")
parser.add_argument('--seed', type=int, default=123, help="random seed for env")

args = parser.parse_args()

env = gym.make(args.scenario)
env.seed(args.seed)
np.random.seed(args.seed)

obs_shape_list = env.observation_space.obs_shape_list
action_shape = env.action_space.n
print("End of script.")