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
        conv2 = tf.keras.layers.Conv2D(32, 5, strides=(1, 1))
        pool2 = tf.keras.layers.MaxPooling2D()
        # pool2 output: 14, 10, 32
        conv3 = tf.keras.layers.Conv2D(32, 3, strides=(1, 1))
        pool3 = tf.keras.layers.MaxPooling2D()
        # pool3 output: 14, 10, 32
        flat = tf.keras.layers.Flatten()
        fc1 = tf.keras.layers.Dense(128, activation='relu')
        fc2 = tf.keras.layers.Dense(action_shape)
        self.q1 = tf.keras.Sequential([conv1, pool1, conv2, pool2, conv3, pool3, flat, fc1, fc2])
        # self.q1 = tf.keras.Sequential([flat, fc1, fc2])
        self.q2 = tf.keras.models.clone_model(self.q1)
        self.optimizer = tf.keras.optimizers.Adam(0.001)
        self.loss = tf.keras.losses.MeanSquaredError()

    @tf.function
    def forward(self, obs, training=False):
        q1 = self.q1(obs, training=training)
        q2 = self.q2(obs, training=training)
        return q1, q2

    @tf.function
    def train(self, obs, target):
        with tf.GradientTape() as tape:
            q1 = self.q1(obs, training=True)
            q_loss_1 = self.loss(target, q1)
            q2 = self.q2(obs, training=True)
            q_loss_2 = self.loss(target, q2)

        gradients = tape.gradient(q_loss_1, self.q1.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q1.trainable_variables))
        gradients = tape.gradient(q_loss_2, self.q2.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q2.trainable_variables))
    
    def act(self, obs, batch_mode=False):
        q1, q2 = self.forward(obs)
        if batch_mode:
            qs = tf.stack([q1, q2])
        else:
            qs = tf.squeeze(tf.stack([q1, q2]), axis=1)
        qmin = tf.math.reduce_min(qs, axis=0)
        if not batch_mode:
            if np.random.random()>self.args.epsilon:
                action = tf.math.argmax(qmin).numpy()
            else:
                action = np.random.randint(0, high=self.action_shape)
            return np.array(action)
        else:
            action = tf.expand_dims(tf.math.argmax(qmin, axis=-1), axis=-1)
            return action

    # @tf.function
    def update(self, samples):
        obs, action, reward, done, new_obs = samples

        q1_next, q2_next = self.forward(new_obs, training=False)
        qs_next = tf.stack([q1_next, q2_next])
        qmin_next = tf.math.reduce_min(qs_next, axis=0)
        done = done.astype(float)
        action = np.expand_dims(action, axis=-1)
        action_next = self.act(new_obs, batch_mode=True)
        qmin_next_a = tf.gather_nd(qmin_next, action_next, batch_dims=1)
        q_target = reward + self.args.gamma * (1 - done) * qmin_next_a

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            q1, q2 = self.forward(obs, training=True)
            q1_a = tf.gather_nd(q1, action, batch_dims=1)
            q2_a = tf.gather_nd(q2, action, batch_dims=1)
            q_loss_1 = self.loss(q_target, q1_a)
            q_loss_2 = self.loss(q_target, q2_a)

        gradients = tape1.gradient(q_loss_1, self.q1.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q1.trainable_variables))
        gradients = tape2.gradient(q_loss_2, self.q2.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q2.trainable_variables))

        return q_loss_1.numpy().tolist(), q_loss_2.numpy().tolist()