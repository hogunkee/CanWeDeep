import numpy as np
import tensorflow as tf


class Discriminator:
    def __init__(self, config):
        self.image_size = config.image_size
        self.num_channel = config.nchannel
    
    def __run__(self, input):
        self.X = input
        # shape: 28 x 28 x 1
        X_reshaped = tf.reshape(self.X, [-1, self.images_size, self.image_size, self.nchannel])

        # shape: 14 x 14 x 16 
        init = tf.contrib.layers.xavier_initializer()
        W1 = tf.get_variable('D-W1', [3, 3, self.nchannel, 16], tf.float32, initializer = init)
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
        b3 = tf.get_variable('D-b3', tf.float32, initializer = tf.zeros([1]))
        self.Y_ = tf.matmul(h, W3) + b3

        
    def loss_real(self):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\ 
            self.Y_, tf.ones_like(self.Y_))
            
    def loss_fake(self):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\ 
            self.Y_, tf.zeros_like(self.Y_))


class Generator:
    def __init__(self, config):
        self.image_size = config.image_size
        self.num_channel = config.nchannel
        self.nz = config.nz
    
    def __run__(self, noise):
        self.Z = noise
        # shape: 7 x 7 x 32
        init = tf.contrib.layers.xavier_initializer()
        W1 = tf.get_variable('G-W1', [self.nz, (self.image_size//4)**2*32], \
                tf.float32, initializer = init)
        b1 = tf.get_variable('G-b1', tf.float32, initializer = tf.zeros([7*7*32]))
        h = tf.matmul(self.Z, W1) + b1
        h = tf.nn.relu(h)

        # shape: 14 x 14 x 16
        h = tf.reshape(h, [-1, self.image_size//4, self.image_size//4, 32])
        W2 = tf.get_variable('G-W2', [3, 3, 32, 16], tf.float32, initializer=init)
        h = tf.nn.conv2d_transpose(h, W2, [-1, self.image_size//2, self.image_size//2, 16], \
                strides=[1,2,2,1], padding='SAME')
        h = tf.nn.relu(h)

        # shape: 28 x 28 x 1
        W3 = tf.get_variable('G-W3', [3, 3, 16, 1], tf.float32, initializer=init)
        h = tf.nn.conv2d_transpose(h, W2, \
                [-1, self.image_size, self.image_size, self.num_channel], \
                strides=[1,2,2,1], padding='SAME')
        self.image = tf.reshape(tf.nn.tanh(h), [-1, 784])
        return self.image


config = get_config()

D = Discriminator(sess, config)
G = Generator(sess, config)
optimizer_D = tf.train.AdamOptimizer(learning_rate = config.learning_rate_D)
optimizer_G = tf.train.AdamOptimizer(learning_rate = config.learning_rate_G)

X = tf.placeholder(tf.float32, shape=[None, 784])
Z = tf.placeholder(tf.float32, shape=[None, config.nz])

X_fake = Generator(Z, config)
D_fake = 


t_vars = tf.trainable_variables()
D_vars = [v for v in t_vars if 'D' in v.name]
G_vars = [v for v in t_vars if 'G' in v.name]
assert len(t_vars) == len(D_vars + G_vars)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1-D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))

for epoch in config.num_epochs:
    
