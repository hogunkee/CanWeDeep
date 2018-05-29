import numpy as np
import tensorflow as tf


class GAN:
    def __init__(self, config):
        self.image_size = config.image_size
        self.nchannel = config.nchannel
        self.nz = config.nz
    
        self.X = tf.placeholder(tf.float32, shape=[None, 784])
        self.Z = tf.placeholder(tf.float32, shape=[None, config.nz])

        self.X_fake = self.Generator(self.Z)
        D_fake = self.Discriminator(self.X_fake)
        D_real = self.Discriminator(self.X, True)

        t_vars = tf.trainable_variables()
        D_vars = [v for v in t_vars if 'D' in v.name]
        G_vars = [v for v in t_vars if 'G' in v.name]
        assert len(t_vars) == len(D_vars + G_vars)

        self.D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1-D_fake))
        self.G_loss = -tf.reduce_mean(tf.log(D_fake))

        self.train_D = tf.train.AdamOptimizer(learning_rate = \
                config.learning_rate_D).minimize(self.D_loss, var_list = D_vars)
        self.train_G = tf.train.AdamOptimizer(learning_rate = \
                config.learning_rate_G).minimize(self.G_loss, var_list = G_vars)

    def Discriminator(self, image, reuse=False):
        with tf.variable_scope('Discriminator', reuse = reuse):
            # shape: 28 x 28 x 1
            X_reshaped = tf.reshape(image, [-1,self.image_size,self.image_size,self.nchannel])

            # shape: 14 x 14 x 16 
            init = tf.contrib.layers.xavier_initializer()
            W1 = tf.get_variable('D-W1', [3,3,self.nchannel,16], tf.float32, initializer=init)
            h = tf.nn.conv2d(X_reshaped, W1, strides=[1,2,2,1], padding='SAME')
            h = tf.nn.relu(h)

            # shape: 7 x 7 x 32 
            W2 = tf.get_variable('D-W2', [3, 3, 16, 32], tf.float32, initializer = init)
            h = tf.nn.conv2d(h, W2, strides=[1,2,2,1], padding='SAME')
            h = tf.nn.relu(h)

            h = tf.reduce_mean(h, [1,2])
            h = tf.reshape(h, [-1, 32])

            # shape: 32
            W3 = tf.get_variable('D-W3', [32, 1], tf.float32, initializer = init)
            b3 = tf.get_variable('D-b3', dtype = tf.float32, initializer = tf.zeros([1]))
            D = tf.matmul(h, W3) + b3

        return D 

    def Generator(self, noise):
        init = tf.contrib.layers.xavier_initializer()
        W1 = tf.get_variable('G-W1', [self.nz, (self.image_size//4)**2*32], \
                tf.float32, initializer = init)
        b1 = tf.get_variable('G-b1', dtype = tf.float32, initializer = tf.zeros([7*7*32]))

        h = tf.matmul(noise, W1) + b1
        h = tf.nn.relu(h)

        # shape: 14 x 14 x 16
        h = tf.reshape(h, [-1, self.image_size//4, self.image_size//4, 32])
        W2 = tf.get_variable('G-W2', [3, 3, 16, 32], tf.float32, initializer=init)
        h = tf.nn.conv2d_transpose(h, W2, [-1, self.image_size//2, self.image_size//2, 16], \
                strides=[1,2,2,1], padding='SAME')
        h = tf.nn.relu(h)

        # shape: 28 x 28 x 1
        W3 = tf.get_variable('G-W3', [3, 3, 1, 16], tf.float32, initializer=init)
        h = tf.nn.conv2d_transpose(h, W3, \
                [-1, self.image_size, self.image_size, self.nchannel], \
                strides=[1,2,2,1], padding='SAME')
        X_fake = tf.reshape(tf.nn.tanh(h), [-1, 784])
        return X_fake 

