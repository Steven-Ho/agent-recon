import numpy as np 
import tensorflow as tf 

class Predictor:
    def __init__(self, obs_shape, action_shape, args):
        self.args = args
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        conv1 = tf.keras.layers.Conv2D(16, 6, strides=(2, 2), input_shape=obs_shape)
        nonl1 = tf.keras.layers.LeakyReLU()
        drop1 = tf.keras.layers.Dropout(0.3)

        conv2 = tf.keras.layers.Conv2D(32, 6, strides=(2, 2))
        nonl2 = tf.keras.layers.LeakyReLU()
        drop2 = tf.keras.layers.Dropout(0.3)

        conv3 = tf.keras.layers.Conv2D(32, 6, strides=(2, 2))
        nonl3 = tf.keras.layers.LeakyReLU()
        drop3 = tf.keras.layers.Dropout(0.3)

        deco1 = tf.keras.layers.Conv2DTranspose(32, 6, strides=(2, 2), use_bias=False)
        bnor1 = tf.keras.BatchNormalization()
        nonl4 = tf.keras.layers.LeakyReLU()
