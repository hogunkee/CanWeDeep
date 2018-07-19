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

### utility funcions ###
# batch normalization
def batchnorm(input_layer):
    BN_EPSILON = 1e-5 
    dimension = int(input_layer.shape[3])
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE): 
        #if reuse:
        #   tf.get_variable_scope().reuse_variables()
        mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
        beta = tf.get_variable('beta', dimension, tf.float32,
                     initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', dimension, tf.float32,
                     initializer=tf.constant_initializer(1.0, tf.float32))
        bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, \
                beta, gamma, BN_EPSILON)

        return bn_layer

# generate convolution layer
def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), \
            padding="same", kernel_initializer=initializer)

# generate deconvolution layer
def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), \
            padding="same", kernel_initializer=initializer)

# leaky ReLU layer
def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


### Generator ###
def generator(x)
    out_channels = a.out_channels

    x_reshape = tf.reshape(x, [-1, 64, 64, out_channels])
    layers = []

    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, a.ngf)
        layers.append(output)

    layer_specs = [
        a.ngf * 1, # encoder_1: [batch, 64, 64, in_channels] => [batch, 32, 32, ngf]
        a.ngf * 2, # encoder_2: [batch, 32, 32, ngf] => [batch, 16, 16, ngf * 2]
        a.ngf * 4, # encoder_3: [batch, 16, 16, ngf * 2] => [batch, 8, 8, ngf * 4]
        a.ngf * 8, # encoder_4: [batch, 8, 8, ngf * 4] => [batch, 4, 4, ngf * 8]
        a.ngf * 8, # encoder_5: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        a.ngf * 16, # encoder_6: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 16]
    ]

    for i in range(len(layer_specs)):
        out_channels = layer_specs[i]
        with tf.variable_scope("encoder_%d" %(i+1)):
            # first layer dosen't need a relu layer
            if i==0:
                input = x_reshape
            else:
                input = layers[-1]
                input = lrelu(input, 0.2)
            convolved = gen_conv(input, out_channels)
            output = batchnorm(convolved)
            layers.append(output)
            print(output.shape)

    layer_specs = [
        (a.ngf * 16, 0.5),  # decoder_6: [batch, 1, 1, ngf*16] => [batch, 2, 2, ngf*16*2]
        (a.ngf * 8, 0.5),   # decoder_5: [batch, 2, 2, ngf*16*2] => [batch, 4, 4, ngf*8*2]
        (a.ngf * 8, 0.5),   # decoder_4: [batch, 4, 4, ngf*8*2] => [batch, 8, 8, ngf*8*2]
        (a.ngf * 4, 0.0),   # decoder_3: [batch, 8, 8, ngf*8*2] => [batch, 16, 16, ngf*4*2]
        (a.ngf * 2, 0.0),   # decoder_2: [batch, 16, 16, ngf*4*2] => [batch, 32, 32, ngf*2*2]
    ]

    num_encoder_layers = len(layers)
    for i in range(len(layer_specs)):
        out_channels = layer_specs[i][0]
        dropout = layer_specs[i][1]
        skip_layer = num_encoder_layers - i - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            output = gen_deconv(rectified, out_channels)

            # last layer dosen't need a batch normalization
            if i == num_encoder_layers - 1:
                output = tf.tanh(output)
                layers.append(output)
                break

            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)
            layers.append(output)

    return layers[-1]


### Discriminator ###
def discriminator(dis_inputs, dis_targets):
    n_layers = 5
    layers = []

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    input = tf.concat([dis_inputs, dis_targets], axis=3)

    # layer_1: [batch, 64, 64, in_channels * 2] => [batch, 32, 32, ndf]
    # layer_2: [batch, 32, 32, ndf] => [batch, 16, 16, ndf * 2]
    # layer_3: [batch, 16, 16, ndf * 2] => [batch, 8, 8, ndf * 4]
    # layer_4: [batch, 8, 8, ndf * 4] => [batch, 7, 7, ndf * 8]
    # layer_5: [batch, 7, 7, ndf * 8] => [batch, 6, 6, 1]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = a.ndf * (2**i)
            stride = 1 if i >= n_layers - 2 else 2  # last 2 layer here has stride 1
            convolved = discrim_conv(layers[-1], out_channels, stride=stride)

            if i==n_layers -1:
                output = tf.sigmoid(convolved)
                layers.append(output)
                break

            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    return layers[-1]


### Model ###
class Model(object):
    def __init__(self, config):
        self.out_channels = config.out_channels

        EPS = 1e-12 # prevent 1/0 to be INFINITY
        out_channels = config.out_channels
        with tf.variable_scope("generator"):
            outputs = create_generator(inputs, out_channels)

        # create two copies of discriminator, one for real pairs and one for fake pairs
        # they share the same underlying variables
        with tf.name_scope("real_discriminator"):
            with tf.variable_scope("discriminator"):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_real = create_discriminator(inputs, targets)

        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("discriminator", reuse=True):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_fake = create_discriminator(inputs, outputs)

        with tf.name_scope("discriminator_loss"):
            # minimizing -tf.log will try to get inputs to 1
            # predict_real => 1 # predict_fake => 0
            discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

        with tf.name_scope("generator_loss"):
            # predict_fake => 1
            # abs(targets - outputs) => 0
            gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
            gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
            gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

        with tf.name_scope("discriminator_train"):
            discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
            discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

        with tf.name_scope("generator_train"):
            with tf.control_dependencies([discrim_train]):
                gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
                gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
                gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

        global_step = tf.train.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step+1)

        self.predict_real = predict_real
        self.predict_fake = predict_fake
        self.discrim_loss = ema.average(discrim_loss)
        self.discrim_grads_and_vars = discrim_grads_and_vars
        self.gen_loss_GAN = ema.average(gen_loss_GAN)
        self.gen_loss_L1 = ema.average(gen_loss_L1)
        self.gen_grads_and_vars = gen_grads_and_vars
        self.outputs = outputs
        self.train = tf.group(update_losses, incr_global_step, gen_train)

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
