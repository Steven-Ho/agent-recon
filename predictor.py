import numpy as np 
import tensorflow as tf 

class Predictor:
    def __init__(self, obs_shape, action_shape, args):
        self.args = args
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        conv1 = tf.keras.layers.Conv2D(16, 6, strides=(2, 2), padding='same')
        nonl1 = tf.keras.layers.LeakyReLU()

        conv2 = tf.keras.layers.Conv2D(32, 6, strides=(2, 2), padding='same')
        nonl2 = tf.keras.layers.LeakyReLU()

        conv3 = tf.keras.layers.Conv2D(32, 6, strides=(2, 2), padding='same')
        nonl3 = tf.keras.layers.LeakyReLU()

        # Output shape of this conv process is (#, 10, 10, 32)

        deco1 = tf.keras.layers.Conv2DTranspose(32, 6, strides=(2, 2), use_bias=False, padding='same')
        nonl4 = tf.keras.layers.LeakyReLU()

        deco2 = tf.keras.layers.Conv2DTranspose(16, 6, strides=(2, 2), use_bias=False, padding='same')
        nonl5 = tf.keras.layers.LeakyReLU()

        deco3 = tf.keras.layers.Conv2DTranspose(1, 6, strides=(2, 2), use_bias=False, padding='same', activation='sigmoid')
        nonl6 = tf.keras.layers.LeakyReLU()

        deco4 = tf.keras.layers.Conv2DTranspose(32, 6, strides=(2, 2), use_bias=False, padding='same')
        nonl7 = tf.keras.layers.LeakyReLU()

        deco5 = tf.keras.layers.Conv2DTranspose(16, 6, strides=(2, 2), use_bias=False, padding='same')
        nonl8 = tf.keras.layers.LeakyReLU()

        deco6 = tf.keras.layers.Conv2DTranspose(1, 6, strides=(2, 2), use_bias=False, padding='same', activation='sigmoid')
        nonl9 = tf.keras.layers.LeakyReLU()

        # Output shape of this deconv process is (#, 80, 80, 1)

        linr1 = tf.keras.layers.Dense(10*10*8)
        nonl10 = tf.keras.layers.LeakyReLU()

        resh1 = tf.keras.layers.Reshape((10, 10, 8))
        conc1 = tf.keras.layers.Concatenate()

        # The third branch

        conv4 = tf.keras.layers.Conv2D(16, 6, strides=(2, 2), padding='same')
        nonl11 = tf.keras.layers.LeakyReLU()

        conv5 = tf.keras.layers.Conv2D(32, 6, strides=(2, 2), padding='same')
        nonl12 = tf.keras.layers.LeakyReLU()

        conv6 = tf.keras.layers.Conv2D(32, 6, strides=(2, 2), padding='same')
        nonl13 = tf.keras.layers.LeakyReLU()

        deco7 = tf.keras.layers.Conv2DTranspose(32, 6, strides=(2, 2), use_bias=False, padding='same')
        nonl14 = tf.keras.layers.LeakyReLU()

        deco8 = tf.keras.layers.Conv2DTranspose(16, 6, strides=(2, 2), use_bias=False, padding='same')
        nonl15 = tf.keras.layers.LeakyReLU()

        deco9 = tf.keras.layers.Conv2DTranspose(1, 6, strides=(2, 2), use_bias=False, padding='same', activation='sigmoid')
        nonl16 = tf.keras.layers.LeakyReLU()

        # Use functional API to create models
        input1 = tf.keras.Input(shape=obs_shape + (args.frames,))
        x = conv1(input1)
        x = nonl1(x)
        x = conv2(x)
        x = nonl2(x)
        x = conv3(x)
        mid = nonl3(x)
        x = deco1(mid)
        x = nonl4(x)
        x = deco2(x)
        x = nonl5(x)
        output1 = deco3(x)

        input2 = tf.keras.Input(shape=action_shape)
        x = linr1(input2)
        x = nonl10(x)
        x = resh1(x)
        x = conc1([x, mid])
        x = deco4(x)
        x = nonl7(x)
        x = deco5(x)
        x = nonl8(x)
        output2 = deco6(x)

        input3 = tf.keras.Input(shape=obs_shape + (1,))
        x = conv4(x)
        x = nonl11(x)
        x = conv5(x)
        x = nonl12(x)
        x = conv6(x)
        x = nonl13(x)
        x = deco7(x)
        x = nonl14(x)
        x = deco8(x)
        x = nonl15(x)
        output3 = deco9(x)

        self.image = tf.keras.Model(inputs=[input1, input2], outputs=[output1, output2])
        self.mask = tf.keras.Model(inputs=input3, outputs=output3)
        
        conv7 = tf.keras.layers.Conv2D(8, 6, strides=(2,2), padding='same')
        nonl17 = tf.keras.layers.LeakyReLU()

        conv8 = tf.keras.layers.Conv2D(8, 6, strides=(2,2), padding='same')
        nonl18 = tf.keras.layers.LeakyReLU()

        flat = tf.keras.layers.Flatten()
        linr2 = tf.keras.layers.Dense(action_shape)
        smax = tf.keras.layers.Softmax()

        self.action_pred = tf.keras.Sequential([conv7, nonl17, conv8, nonl18, flat, linr2, smax])


    @tf.function
    def forward(self, obs_hist, obs, action):
        iu, ic = self.image([obs_hist, action])
        m = self.mask(obs)
        return iu, ic, m

    @tf.function
    def get_mask(self, obs):
        m = self.mask(obs)
        return m

    @tf.function
    def action_infer(self, masks):
        probs = self.action_pred(masks)
        return probs

    def update(self, samples):
        obs, action, reward, done, new_obs = samples
        obs = obs.astype('float32')
        new_obs = new_obs.astype('float32')
        done = done.astype(float)

        a = np.zeros((self.args.batch_size, self.args.action_shape))
        for i in range(self.args.batch_size):
            a[i, action[i]] = 1.0

        new_obs = new_obs[...,-1]
        new_obs = np.expand_dims(new_obs, axis=-1)
        iu, ic, m = self.forward(obs, new_obs, a)
        old_obs = obs[...,-1]
        old_obs = np.expand_dims(old_obs)
        old_m = self.get_mask(old_obs)

        # train action prediction model

        # train iamge and mask model
        