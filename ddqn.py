import tensorflow as tf 
import numpy as np 

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
        self.q1_target = tf.keras.models.clone_model(self.q1)
        self.q2_target = tf.keras.models.clone_model(self.q1)
        self.optimizer = tf.keras.optimizers.Adam(args.lr)
        self.loss = tf.keras.losses.MeanSquaredError()

    @tf.function
    def forward(self, obs, training=False):
        q1 = self.q1(obs, training=training)
        q2 = self.q2(obs, training=training)
        return q1, q2

    @tf.function
    def forward_t(self, obs, training=False):
        q1 = self.q1_target(obs, training=training)
        q2 = self.q2_target(obs, training=training)
        return q1, q2

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def update_target(self):
        self.q1_target.set_weights(self.q1.get_weights())
        self.q2_target.set_weights(self.q2.get_weights())

    def act(self, obs, batch_mode=False):
        q1, q2 = self.forward_t(obs)
        if batch_mode:
            qs = tf.stack([q1, q2])
        else:
            qs = tf.squeeze(tf.stack([q1, q2]), axis=1)
        qmin = tf.math.reduce_min(qs, axis=0)
        if not batch_mode:
            if np.random.random()>self.epsilon:
                action = tf.math.argmax(qmin).numpy()
            else:
                action = np.random.randint(0, high=self.action_shape)
            return np.array(action)
        else:
            action = tf.expand_dims(tf.math.argmax(qmin, axis=-1), axis=-1)
            return action

    def update(self, samples):
        obs, action, reward, done, new_obs = samples
        obs = obs.astype('float32')
        new_obs = new_obs.astype('float32')

        q1_next, q2_next = self.forward_t(new_obs, training=False)
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