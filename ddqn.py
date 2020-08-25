import tensorflow as tf 
import numpy as np 

class DDQN:
    def __init__(self, obs_shape, action_shape, args):
        self.args = args
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.layers = [
            tf.keras.layers.Conv2D()
        ]