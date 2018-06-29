import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
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

# generator
def generator(z, reuse = False):
    with tf.variable_scope('Gen', reuse=reuse) as scope:
        gw1 = tf.get_variable('w1', [128,256], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))
        gb1 = tf.get_variable('b1', [256], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))

        gw2 = tf.get_variable('w2', [256,784], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))
        gb2 = tf.get_variable('b2', [784], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))

    hidden = tf.nn.relu(tf.matmul(z, gw1) + gb1)
    output = tf.nn.sigmoid(tf.matmul(hidden, gw2) + gb2)

    return output

# discriminator
def discriminator(z, reuse=False):
    with tf.variable_scope('Dis', reuse=reuse) as scope:
        dw1 = tf.get_variable('w1', [784,256], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))
        db1 = tf.get_variable('b1', [256], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))

        dw2 = tf.get_variable('w2', [256,1], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))
        db2 = tf.get_variable('b2', [1], initializer = \
                tf.random_normal_initializer(mean=0.0, stddev=0.01))

    hidden = tf.nn.relu(tf.matmul(z, dw1) + db1)
    output = tf.nn.sigmoid(tf.matmul(hidden, dw2) + db2)

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
with tf.Session(graph = g) as sess:
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

        if (epoch+1)%20==0 or epoch==1:
            print('\nEpoch: %d/%d' %(epoch, total_epochs))
            print('Generator:', gl)
            print('Discriminator:', dl)
            print('Fake D:', D_gz, '/ Real D:', D_x)
        
        sample_noise = random_noise(10)
        if epoch==0 or (epoch+1)%10 == 0:
            generated = sess.run(fake_x, feed_dict = {Z: sample_noise})

            fig, ax = plt.subplots(1, 10, figsize=(10,1))
            for i in range(10):
                ax[i].set_axis_off()
                ax[i].imshow(np.reshape(generated[i], (28,28)))

            plt.savefig('result/%s.png' %str(epoch).zfill(3), bbox_inches='tight')
            plt.close(fig)

    print('Finished!!')
