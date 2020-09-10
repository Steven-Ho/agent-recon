import gym
import numpy as np 
import cv2
import tensorflow as tf 
import argparse
from ddqn import DDQN, DQN
from predictor import Predictor
from buffer import FullReplayMemory
import itertools
import time
import datetime
# tf.debugging.set_log_device_placement(True)
def make_obs_memory(obs, size=(80, 80)):
    obs_cv = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs_resized = cv2.resize(obs_cv, dsize=size, interpolation=cv2.INTER_CUBIC)
    return obs_resized

def make_obs_network(obs, memory):
    obs = np.expand_dims(obs, axis=-1)
    obs_stack = memory.last_obs()
    obs_stack = np.concatenate([obs_stack, obs], axis=-1)
    obs_stack = np.expand_dims(obs_stack, axis=0)
    return obs_stack

def make_data_predictor(obs, memory, action_shape):
    obs = np.expand_dims(obs, axis=-1)
    obs = np.expand_dims(obs, axis=0)
    obs_stack = memory.last_obs(frames=5)
    obs_stack = np.expand_dims(obs_stack, axis=0)  
    a = memory.last_action()
    a = int(a)
    action = np.zeros(action_shape)
    action[a] = 1. 
    action = np.expand_dims(action, axis=0)
    return obs_stack, obs, action  

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs, ", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
parser = argparse.ArgumentParser(description='Duoble DQN Baseline')
parser.add_argument('--scenario', type=str, default="Centipede-v0", help="environment (default: Seaquest-v0)")
parser.add_argument('--seed', type=int, default=123, help="random seed for env")
parser.add_argument('--num_episodes', type=int, default=40000, help='number of episodes for training')
parser.add_argument('--max_episode_len', type=int, default=5000, help='maximum episode length')
parser.add_argument('--gamma', type=float, default=0.995, help='discount factor (default: 0.95)')
parser.add_argument('--epsilon', type=float, default=0.2, help='epsilon-greedy parameter (initial: 0.5)')
parser.add_argument('--buffer_size', type=int, default=1e6, help='maximum size for replay buffer')
parser.add_argument('--update_interval', type=int, default=10, help='update q network for every N steps')
parser.add_argument('--startup_steps', type=int, default=10000, help='initial rollout steps before training')
parser.add_argument('--batch_size', type=int, default=32, help='sample size for training')
parser.add_argument('--hidden_size', type=int, default=512, help='hidden unit number for network')
parser.add_argument('--lr', type=float, default=0.00025, help='learning rate for q networks')
parser.add_argument('--render', action='store_true', help='render or not')
parser.add_argument('--frames', type=int, default=4, help='N frames for q network')
args = parser.parse_args()

env = gym.make(args.scenario)
env.seed(args.seed)
np.random.seed(args.seed)

obs_shape_list = env.observation_space.shape
action_shape = env.action_space.n

if len(obs_shape_list)>1:
    shapes = [(80, 80), (1,), (1,), (1,), (80, 80)]
    dtypes = [np.uint8, np.uint8, np.float32, np.bool, np.uint8]
    model_type = "CNN"
else:
    shapes = [(obs_shape_list[0],), (1,), (1,), (1,), (obs_shape_list[0],)]
    dtypes = [np.float32, np.uint8, np.float32, np.bool, np.float32]
    model_type = "DNN"
qnet = DDQN(shapes[0]+(args.frames,), action_shape, model_type, args)
pred = Predictor(shapes[0], action_shape, args)
kws = ['obs', 'action', 'reward', 'done', 'new_obs']
memory = FullReplayMemory(args.buffer_size, kws, shapes, dtypes)
writer = tf.summary.create_file_writer("logs/{}_{}".format(args.scenario, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

total_numsteps = 0
timestep = 0
t_start = time.time()
epsilon = args.epsilon

# total_parameters = np.sum([np.prod(v.get_shape().as_list()) for v in qnet.q1.trainable_variables])
with writer.as_default():
    for i_episode in itertools.count(1):
            
        obs = env.reset()
        if model_type == "CNN":
            obs = make_obs_memory(obs)
        done = False
        episode_reward = 0
        for t in range(args.max_episode_len):
            timestep += 1
            
            if timestep % 200000 == 1:
                epsilon = max(epsilon/2., 0.005)
                qnet.set_epsilon(epsilon)
                tf.summary.scalar("parameters/epsilon", epsilon, step=timestep)
                writer.flush()
            obs_net = make_obs_network(obs, memory)
            # if timestep > args.startup_steps:
            #     obs_hist, obs_pres, last_action = make_data_predictor(obs, memory, action_shape)
            #     iu, ic, m = pred.forward(obs_hist, obs_pres, last_action)
            if model_type == "CNN":
                obs_net = obs_net/255.
            action = qnet.act(obs_net)
            new_obs, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            if model_type == "CNN":
                new_obs = make_obs_memory(new_obs)

            memory.push(obs=obs, action=action, reward=reward, done=done, new_obs=new_obs)
            obs = new_obs

            if timestep > args.startup_steps:
                if timestep % args.update_interval == 0:
                    obs_b, action_b, reward_b, done_b, new_obs_b = memory.sample(args.batch_size)
                    lq, qs = qnet.update((obs_b, action_b, reward_b, done_b, new_obs_b))
                    tf.summary.scalar("loss/q", lq, step=timestep)
                    tf.summary.scalar("values/q", qs, step=timestep)

                    loss_all, loss_ap, loss_masked, loss_recon, loss_reg = pred.update((obs_b, action_b, reward_b, done_b, new_obs_b))
                    tf.summary.scalar("loss/all", loss_all, step=timestep)
                    tf.summary.scalar("loss/ap", loss_ap, step=timestep)
                    tf.summary.scalar("loss/masked", loss_masked, step=timestep)
                    tf.summary.scalar("loss/recon", loss_recon, step=timestep)
                    tf.summary.scalar("loss/reg", loss_reg, step=timestep)
                    writer.flush()

            episode_reward += reward
            if done:
                break
        print("Episode reward: {}, current time step: {}".format(episode_reward, timestep))
        tf.summary.scalar("reward/episode_reward", episode_reward, step=i_episode)
        tf.summary.scalar("reward/episode_length", t, step=i_episode)
        writer.flush()
        episode_reward = 0

print("End of script.")