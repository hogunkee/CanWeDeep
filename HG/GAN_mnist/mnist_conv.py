import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from tensorflow.examples.tutorials.mnist import input_data

# data loader
mnist = input_data.read_data_sets("./sample/MNIST_data/")
train_x = mnist.train.images
train_y = mnist.train.labels
print(train_x.shape, train_y.shape)

# hyperparameters
total_epochs = 100
batch_size = 100
learning_rate = 2e-4

# batch normalization
def batch_norm(input_layer, scope, reuse):
    BN_EPSILON = 1e-5 
    dimension = int(input_layer.shape[3])
    with tf.variable_scope(scope): 
        if reuse:
            tf.get_variable_scope().reuse_variables()
        mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
        beta = tf.get_variable('beta', dimension, tf.float32,
                     initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', dimension, tf.float32,
                     initializer=tf.constant_initializer(1.0, tf.float32))
        bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, \
                beta, gamma, BN_EPSILON)

        return bn_layer

# generator
def generator(z, reuse = False):
    with tf.variable_scope('Gen', reuse=reuse) as scope:
        gw1 = tf.get_variable('w1', [3,3,512,128], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))
        gb1 = tf.get_variable('b1', [512], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))

        gw2 = tf.get_variable('w2', [3,3,256,512], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))
        gb2 = tf.get_variable('b2', [256], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))

        gw3 = tf.get_variable('w3', [3,3,128,256], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))
        gb3 = tf.get_variable('b3', [128], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))

        gw4 = tf.get_variable('w4', [3,3,64,128], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))
        gb4 = tf.get_variable('b4', [64], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))

        gw5 = tf.get_variable('w5', [3,3,1,64], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))
        gb5 = tf.get_variable('b5', [1], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))

    z_reshape = tf.reshape(z, [-1, 1, 1, 128])

    h = tf.nn.conv2d_transpose(z_reshape, gw1, [tf.shape(z_reshape)[0], 2, 2, 512], \
            strides=[1,2,2,1], padding='SAME') + gb1
    h = batch_norm(h, 'G-bn1', reuse)
    h = tf.nn.relu(h)

    h = tf.nn.conv2d_transpose(h, gw2, [tf.shape(h)[0], 4, 4, 256], \
            strides=[1,2,2,1], padding='SAME') + gb2
    h = batch_norm(h, 'G-bn2', reuse)
    h = tf.nn.relu(h)

    h = tf.nn.conv2d_transpose(h, gw3, [tf.shape(h)[0], 7, 7, 128], \
            strides=[1,2,2,1], padding='SAME') + gb3
    h = batch_norm(h, 'G-bn3', reuse)
    h = tf.nn.relu(h)

    h = tf.nn.conv2d_transpose(h, gw4, [tf.shape(h)[0], 14, 14, 64], \
            strides=[1,2,2,1], padding='SAME') + gb4
    h = batch_norm(h, 'G-bn4', reuse)
    h = tf.nn.relu(h)

    h = tf.nn.conv2d_transpose(h, gw5, [tf.shape(h)[0], 28, 28, 1], \
            strides=[1,2,2,1], padding='SAME') + gb5
    output = tf.reshape(h, [-1, 784])

    return output

# discriminator
def discriminator(x, reuse=False):
    with tf.variable_scope('Dis', reuse=reuse) as scope:
        dw1 = tf.get_variable('w1', [3,3,1,64], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))
        db1 = tf.get_variable('b1', [64], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))

        dw2 = tf.get_variable('w2', [3,3,64,128], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))
        db2 = tf.get_variable('b2', [128], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))

        dw3 = tf.get_variable('w3', [3,3,128,256], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))
        db3 = tf.get_variable('b3', [256], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))

        dw4 = tf.get_variable('w4', [3,3,256,512], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))
        db4 = tf.get_variable('b4', [512], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))

        dw5 = tf.get_variable('w5', [512, 1], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))
        db5 = tf.get_variable('b5', [1], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))


    x_reshape = tf.reshape(x, [-1, 28, 28, 1])

    h = tf.nn.conv2d(x_reshape, dw1, strides=[1,2,2,1],padding='SAME') + db1
    h = batch_norm(h, 'D-bn1', reuse)
    h = tf.nn.relu(h)

    h = tf.nn.conv2d(h, dw2, strides=[1,2,2,1],padding='SAME') + db2
    h = batch_norm(h, 'D-bn2', reuse)
    h = tf.nn.relu(h)

    h = tf.nn.conv2d(h, dw3, strides=[1,2,2,1],padding='SAME') + db3
    h = batch_norm(h, 'D-bn3', reuse)
    h = tf.nn.relu(h)

    h = tf.nn.conv2d(h, dw4, strides=[1,2,2,1],padding='SAME') + db4
    h = batch_norm(h, 'D-bn4', reuse)
    h = tf.nn.relu(h)

    h = tf.reduce_mean(h, [1,2])
    h = tf.reshape(h, [-1, 512])

    output = tf.nn.sigmoid(tf.matmul(h, dw5) + db5)

    return output

# random noise
def random_noise(batch_size):
    return np.random.normal(size = [batch_size,128])

# Graph
g = tf.Graph()
with g.as_default():
    X = tf.placeholder(tf.float32, [None, 784])
    Z = tf.placeholder(tf.float32, [None, 128])

    fake_x = generator(Z)

    result_fake = discriminator(fake_x)
    result_real = discriminator(X, True)

    D_G_Z = tf.reduce_mean(result_fake)
    D_X = tf.reduce_mean(result_real)

    g_loss = -tf.reduce_mean(tf.log(result_fake))
    d_loss = -tf.reduce_mean(tf.log(result_real) + tf.log(1-result_fake))

    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if 'Gen' in var.name]
    d_vars = [var for var in t_vars if 'Dis' in var.name]

    optimizer = tf.train.AdamOptimizer(learning_rate)
    g_train = optimizer.minimize(g_loss, var_list = g_vars)
    d_train = optimizer.minimize(d_loss, var_list = d_vars)

# Train
with tf.Session(graph = g, config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    sess.run(tf.global_variables_initializer())

    total_batch = int(train_x.shape[0] / batch_size)

    for epoch in range(total_epochs):
        for batch in range(total_batch):
            batch_x = train_x[batch * batch_size: (batch+1) * batch_size]
            batch_y = train_y[batch * batch_size: (batch+1) * batch_size]
            noise = random_noise(batch_size)

            sess.run(g_train, feed_dict = {Z: noise})
            sess.run(d_train, feed_dict = {X: batch_x, Z: noise})

            D_gz, D_x, gl, dl = sess.run([D_G_Z, D_X, g_loss, d_loss], \
                    feed_dict={X: batch_x, Z: noise})

        #if (epoch+1)%20==0 or epoch==1:
        print('\nEpoch: %d/%d' %(epoch, total_epochs))
        print('Generator:', gl)
        print('Discriminator:', dl)
        print('Fake D:', D_gz, '/ Real D:', D_x)
        
        sample_noise = random_noise(10)
        if epoch==0 or (epoch+1)%5 == 0:
            generated = sess.run(fake_x, feed_dict = {Z: sample_noise})

            fig, ax = plt.subplots(1, 10, figsize=(10,1))
            for i in range(10):
                ax[i].set_axis_off()
                ax[i].imshow(np.reshape(generated[i], (28,28)))

            plt.savefig('result/largeconv-%s.png' %str(epoch).zfill(3), bbox_inches='tight')
            plt.close(fig)

    print('Finished!!')
