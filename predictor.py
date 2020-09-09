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

        # Output shape of this conv process is (#, 11, 11, 32)

        deco1 = tf.keras.layers.Conv2DTranspose(32, 6, strides=(2, 2), use_bias=False, padding='same')
        nonl4 = tf.keras.layers.LeakyReLU()

        deco2 = tf.keras.layers.Conv2DTranspose(16, 6, strides=(2, 2), use_bias=False, padding='same')
        nonl5 = tf.keras.layers.LeakyReLU()

        deco3 = tf.keras.layers.Conv2DTranspose(1, 6, strides=(2, 2), use_bias=False, padding='same')
        nonl6 = tf.keras.layers.LeakyReLU()

        deco4 = tf.keras.layers.Conv2DTranspose(32, 6, strides=(2, 2), use_bias=False, padding='same')
        nonl7 = tf.keras.layers.LeakyReLU()

        deco5 = tf.keras.layers.Conv2DTranspose(16, 6, strides=(2, 2), use_bias=False, padding='same')
        nonl8 = tf.keras.layers.LeakyReLU()

        deco6 = tf.keras.layers.Conv2DTranspose(1, 6, strides=(2, 2), use_bias=False, padding='same')
        nonl9 = tf.keras.layers.LeakyReLU()

        # Output shape of this deconv process is (#, 84, 84, 1)

        linr1 = tf.keras.layers.Dense(11*11*8)
        nonl10 = tf.keras.layers.LeakyReLU()

        resh1 = tf.keras.layers.Reshape((11, 11, 8))
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

        deco9 = tf.keras.layers.Conv2DTranspose(1, 6, strides=(2, 2), use_bias=False, padding='same')
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
        x = deco3(x)
        output1 = nonl6(x)

        input2 = tf.keras.Input(shape=action_shape)
        x = linr1(input2)
        x = nonl10(x)
        x = resh1(x)
        x = conc1([x, mid])
        x = deco4(x)
        x = nonl7(x)
        x = deco5(x)
        x = nonl8(x)
        x = deco6(x)
        output2 = nonl9(x)

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
        x = deco9(x)
        output3 = nonl16(x)

        self.model = tf.keras.Model(inputs=[input1, input2, input3], outputs=[output1, output2, output3])