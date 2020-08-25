import tensorflow as tf 
import numpy as np 

class DDQN:
    def __init__(self, obs_shape, action_shape, args):
        self.args = args
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        conv1 = tf.keras.layers.Conv2D(16, 11, strides=(3, 3), input_shape=obs_shape)
        pool1 = tf.keras.layers.MaxPooling2D()
        # pool1 output: 33, 25, 16
        conv2 = tf.keras.layers.Conv2D(32, 5, strides=(1, 1), input_shape=(33, 25, 16))
        pool2 = tf.keras.layers.MaxPooling2D()
        # pool2 output: 14, 10, 32
        flat = tf.keras.layers.Flatten()
        fc1 = tf.keras.layers.Dense(128, activation='relu')
        fc2 = tf.keras.layers.Dense(action_shape)
        self.q1 = tf.keras.Sequential([conv1, pool1, conv2, pool2, flat, fc1, fc2])
        self.q2 = tf.keras.models.clone_model(self.q1)
        self.optimizer = tf.keras.optimizers.Adam(0.001)
        self.loss = tf.keras.losses.MeanSquaredError()
        print("Initilized!")

    @tf.function
    def forward(self, obs):
        q_1 = self.q1(obs, training=False)
        q_2 = self.q2(obs, training=False)
        return q_1, q_2

    @tf.function
    def train(self, obs, target):
        with tf.GradientTape() as tape:
            q_1 = self.q1(obs, training=True)
            q_loss_1 = self.loss(target, q_1)

        gradients = tape.gradient(q_loss_1, self.q1.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q1.trainable_variables))