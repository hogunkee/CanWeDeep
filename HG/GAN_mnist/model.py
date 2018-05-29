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
        self.D_fake = self.Discriminator(self.X_fake)
        self.D_real = self.Discriminator(self.X, True)

        t_vars = tf.trainable_variables()
        D_vars = [v for v in t_vars if 'D' in v.name]
        G_vars = [v for v in t_vars if 'G' in v.name]
        assert len(t_vars) == len(D_vars + G_vars)

        self.D_loss = -tf.reduce_mean(tf.log(self.D_real) + tf.log(1-self.D_fake))
        self.G_loss = -tf.reduce_mean(tf.log(self.D_fake))

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
            h = self.batch_norm(h, 'D-BN1')
            h = tf.nn.relu(h)

            # shape: 7 x 7 x 32 
            W2 = tf.get_variable('D-W2', [3, 3, 16, 32], tf.float32, initializer = init)
            h = tf.nn.conv2d(h, W2, strides=[1,2,2,1], padding='SAME')
            h = self.batch_norm(h, 'D-BN2')
            h = tf.nn.relu(h)

            # shape: 4 x 4 x 64 
            W3 = tf.get_variable('D-W3', [3, 3, 32, 64], tf.float32, initializer = init)
            h = tf.nn.conv2d(h, W3, strides=[1,2,2,1], padding='SAME')
            h = self.batch_norm(h, 'D-BN3')
            h = tf.nn.relu(h)

            h = tf.reduce_mean(h, [1,2])
            h = tf.reshape(h, [-1, 64])

            # shape: 32
            W4 = tf.get_variable('D-W4', [64, 1], tf.float32, initializer = init)
            b4 = tf.get_variable('D-b4', dtype = tf.float32, initializer = tf.zeros([1]))
            D = tf.sigmoid(tf.matmul(h, W4) + b4)

        return D 

    def Generator(self, noise):
        init = tf.contrib.layers.xavier_initializer()
        W1 = tf.get_variable('G-W1', [self.nz, 4*4*64], \
                tf.float32, initializer = init)
        b1 = tf.get_variable('G-b1', dtype = tf.float32, initializer = tf.zeros([4*4*64]))

        # shape: 4 x 4 x 64 
        h = tf.matmul(noise, W1) + b1
        h = tf.nn.relu(h)
        h = tf.reshape(h, [-1, 4, 4, 64])

        # shape: 7 x 7 x 32 
        W2 = tf.get_variable('G-W2', [3, 3, 32, 64], tf.float32, initializer=init)
        h = tf.nn.conv2d_transpose(h, W2, [tf.shape(h)[0], self.image_size//4, \
                self.image_size//4, 32], strides=[1,2,2,1], padding='SAME')
        h = self.batch_norm(h, 'G-BN1')
        h = tf.nn.relu(h)

        # shape: 14 x 14 x 16
        W3 = tf.get_variable('G-W3', [3, 3, 16, 32], tf.float32, initializer=init)
        h = tf.nn.conv2d_transpose(h, W3, [tf.shape(h)[0], self.image_size//2, \
                self.image_size//2, 16], strides=[1,2,2,1], padding='SAME')
        h = self.batch_norm(h, 'G-BN2')
        h = tf.nn.relu(h)

        # shape: 28 x 28 x 1
        W4 = tf.get_variable('G-W4', [3, 3, 1, 16], tf.float32, initializer=init)
        h = tf.nn.conv2d_transpose(h, W4, [tf.shape(h)[0], self.image_size, \
                self.image_size, self.nchannel], strides=[1,2,2,1], padding='SAME')
        h = self.batch_norm(h, 'G-BN3')
        X_fake = tf.reshape(tf.nn.sigmoid(h), [-1, (self.image_size)**2*self.nchannel])
        return X_fake 

    def batch_norm(self, input_layer, scope):
        BN_EPSILON = 1e-5 
        dimension = int(input_layer.shape[3])
        with tf.variable_scope(scope): 
            mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
            beta = tf.get_variable('beta', dimension, tf.float32,
                         initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable('gamma', dimension, tf.float32,
                         initializer=tf.constant_initializer(1.0, tf.float32))
            bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)
         
            return bn_layer

