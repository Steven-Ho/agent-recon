import tensorflow as tf 
import numpy as np 

class DQN:
    def __init__(self, obs_shape, action_shape, args):
        self.args = args
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.epsilon = args.epsilon
        if len(obs_shape)>1:
            self.model_type = "CNN"
        else:
            self.model_type = "DNN"
        if self.model_type == "CNN":
            conv1 = tf.keras.layers.Conv2D(16, 5, strides=(3, 3), input_shape=obs_shape)
            pool1 = tf.keras.layers.MaxPooling2D()
            conv2 = tf.keras.layers.Conv2D(32, 3, strides=(1, 1))
            pool2 = tf.keras.layers.MaxPooling2D()
            flat = tf.keras.layers.Flatten()
            fc1 = tf.keras.layers.Dense(128, activation='relu')
            fc2 = tf.keras.layers.Dense(action_shape)
            self.q = tf.keras.Sequential([conv1, pool1, conv2, pool2, flat, fc1, fc2])
        else:
            fc1 = tf.keras.layers.Dense(128, activation='relu')
            fc2 = tf.keras.layers.Dense(128, activation='relu')
            fc3 = tf.keras.layers.Dense(action_shape)
            self.q = tf.keras.Sequential([fc1, fc2, fc3])
        self.optimizer = tf.keras.optimizers.Adam(args.lr)
        self.loss = tf.keras.losses.MeanSquaredError()        

    @tf.function
    def forward(self, obs, training=False):
        q = self.q(obs, training=training)
        return q

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def act(self, obs):
        q = self.forward(obs)
        if np.random.random()>self.epsilon:
            action = tf.math.argmax(q, axis=-1).numpy()
        else:
            action = np.random.randint(0, high=self.action_shape)
        return np.array(action)

    @tf.function
    def get_action(self, obs):
        q = self.forward(obs)
        action = tf.math.argmax(q, axis=-1)
        action = tf.expand_dims(action, axis=-1)
        return action, q

    def update(self, samples):
        obs, action, reward, done, new_obs = samples
        obs = obs.astype('float32')
        new_obs = new_obs.astype('float32')

        done = done.astype(float)
        action = np.expand_dims(action, axis=-1).astype(np.int32)
        action_next, q_next = self.get_action(new_obs)
        q_next_a = tf.gather_nd(q_next, action_next, batch_dims=1)
        q_next_a = tf.expand_dims(q_next_a, axis=-1)
        q_target = reward + self.args.gamma * (1 - done) * q_next_a

        with tf.GradientTape() as tape:
            q = self.forward(obs, training=True)
            q_a = tf.gather_nd(q, action, batch_dims=1)
            q_loss = self.loss(q_target, q_a)

        gradients = tape.gradient(q_loss, self.q.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q.trainable_variables))

        return q_loss.numpy().tolist(), tf.math.reduce_mean(q_a).numpy().tolist()

class DDQN:
    def __init__(self, obs_shape, action_shape, args):
        self.args = args
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.epsilon = args.epsilon
        if len(obs_shape)>1:
            self.model_type = "CNN"
        else:
            self.model_type = "DNN"
        if self.model_type == "CNN":
            conv1 = tf.keras.layers.Conv2D(16, 5, strides=(3, 3), input_shape=obs_shape)
            pool1 = tf.keras.layers.MaxPooling2D()
            conv2 = tf.keras.layers.Conv2D(32, 3, strides=(1, 1))
            pool2 = tf.keras.layers.MaxPooling2D()
            flat = tf.keras.layers.Flatten()
            fc1 = tf.keras.layers.Dense(128, activation='relu')
            fc2 = tf.keras.layers.Dense(action_shape)
            self.q1 = tf.keras.Sequential([conv1, pool1, conv2, pool2, flat, fc1, fc2])
        else:
            fc1 = tf.keras.layers.Dense(128, activation='relu')
            fc2 = tf.keras.layers.Dense(128, activation='relu')
            fc3 = tf.keras.layers.Dense(action_shape)
            self.q1 = tf.keras.Sequential([fc1, fc2, fc3])
        self.q2 = tf.keras.models.clone_model(self.q1)
        self.optimizer = tf.keras.optimizers.Adam(args.lr)
        self.loss = tf.keras.losses.MeanSquaredError()

    @tf.function
    def forward(self, obs, training=False):
        q1 = self.q1(obs, training=training)
        q2 = self.q2(obs, training=training)
        return q1, q2

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def act(self, obs):
        q1, q2 = self.forward(obs)
        qs = tf.squeeze(tf.stack([q1, q2]), axis=1)
        qmean = tf.math.reduce_mean(qs, axis=0)
        if np.random.random()>self.epsilon:
            action = tf.math.argmax(qmean, axis=-1).numpy()
        else:
            action = np.random.randint(0, high=self.action_shape)
        return np.array(action)

    @tf.function
    def get_action(self, obs, sel):
        q1, q2 = self.forward(obs)
        qs = tf.stack([q1, q2], axis=1)
        q = tf.gather_nd(qs, sel, batch_dims=1)
        action = tf.math.argmax(q, axis=-1)
        action = tf.expand_dims(action, axis=-1)
        return action, q

    def update(self, samples):
        obs, action, reward, done, new_obs = samples
        obs = obs.astype('float32')
        new_obs = new_obs.astype('float32')

        sel = tf.random.categorical([[0.5, 0.5]], self.args.batch_size)
        sel = tf.reshape(sel, [self.args.batch_size, 1])
        done = done.astype(float)
        action = np.expand_dims(action, axis=-1).astype(np.int32)
        action_next, q_next = self.get_action(new_obs, sel)
        q_next_a = tf.gather_nd(q_next, action_next, batch_dims=1)
        q_next_a = tf.expand_dims(q_next_a, axis=-1)
        q_target = reward + self.args.gamma * (1 - done) * q_next_a

        with tf.GradientTape() as tape:
            q1, q2 = self.forward(obs, training=True)
            qs = tf.stack([q1, q2], axis=1)
            q = tf.gather_nd(qs, 1 - sel, batch_dims=1)
            q_a = tf.gather_nd(q, action, batch_dims=1)
            q_loss = self.loss(q_target, q_a)

        gradients = tape.gradient(q_loss, self.q1.trainable_variables + self.q2.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q1.trainable_variables + self.q2.trainable_variables))

        return q_loss.numpy().tolist()