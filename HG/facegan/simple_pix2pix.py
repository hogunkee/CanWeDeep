import os
import cv2
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument("--out_channels", default=3, help="output channel")
parser.add_argument("--data_dir", default="data", help="directory of image data")
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=1000.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
parser.add_argument("--ngf", type=int, default=16, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=16, help="number of discriminator filters in first conv layer")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--total_epochs", type=int, default=200, help="num epochs")

a = parser.parse_args()

### data loader ###
def data_load(data_dir):
    before = []
    after = []
    for file in sorted(os.listdir(os.path.join(a.data_dir, 'before'))):
        img = np.array(cv2.imread(os.path.join(a.data_dir, 'before', file)), dtype=np.uint8)
        if img.shape == (64,64,3):
            before.append(img)
    for file in sorted(os.listdir(os.path.join(a.data_dir, 'after'))):
        img = np.array(cv2.imread(os.path.join(a.data_dir, 'after', file)), dtype=np.uint8)
        if img.shape == (64,64,3):
            after.append(img)

    return before, after 

### utility funcions ###
def cv2color(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

def f(x):
    if x<0:
        return 0
    elif x>1:
        return 1
    return x

# batch normalization
def batchnorm(input_layer):
    BN_EPSILON = 1e-5 
    dimension = int(input_layer.shape[3])
    with tf.variable_scope('bn', reuse=tf.AUTO_REUSE): 
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

# generater convolution layer
def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), \
            padding="same", kernel_initializer=initializer)

# generater deconvolution layer
def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), \
            padding="same", kernel_initializer=initializer)

# discrminator convolution layer
def discrim_conv(batch_input, out_channels, stride):
    #padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(stride, stride), padding="same", kernel_initializer=tf.random_normal_initializer(0, 0.02))

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
def generator(gen_inputs):
    out_channels = a.out_channels

    layers = []

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
                input = gen_inputs
            else:
                input = layers[-1]
                input = lrelu(input, 0.2)
            convolved = gen_conv(input, out_channels)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (a.ngf * 16, 0.5), # decoder_6: [batch, 1, 1, ngf*16] => [batch, 2, 2, ngf*16*2]
        (a.ngf * 8, 0.5),  # decoder_5: [batch, 2, 2, ngf*16*2] => [batch, 4, 4, ngf*8*2]
        (a.ngf * 8, 0.5),  # decoder_4: [batch, 4, 4, ngf*8*2] => [batch, 8, 8, ngf*8*2]
        (a.ngf * 4, 0.0),  # decoder_3: [batch, 8, 8, ngf*8*2] => [batch, 16, 16, ngf*4*2]
        (a.ngf * 2, 0.0),  # decoder_2: [batch, 16, 16, ngf*4*2] => [batch, 32, 32, ngf*2*2]
        (a.out_channels * 1, 0.0),  # decoder_1: [batch, 32, 32, ngf*2*2] => [batch, 64, 64, 3]
    ]

    num_encoder_layers = len(layers)
    for i in range(len(layer_specs)):
        out_channels = layer_specs[i][0]
        dropout = layer_specs[i][1]
        skip_layer = num_encoder_layers - i - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if i == 0:
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
            if i==0:
                input = dis_inputs
            else:
                input = layers[-1]

            convolved = discrim_conv(input, out_channels, stride=stride)

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
        self.__make__()

    def __make__(self):
        EPS = 1e-12 # prevent 1/0 to be INFINITY
        out_channels = self.out_channels

        self.inputs = inputs = tf.placeholder(tf.float32, [None, 64, 64, out_channels])
        self.targets = targets = tf.placeholder(tf.float32, [None, 64, 64, out_channels])

        with tf.variable_scope("generator"):
            outputs = generator(inputs)

        # create two copies of discriminator, one for real pairs and one for fake pairs
        # they share the same underlying variables
        with tf.name_scope("real_discriminator"):
            with tf.variable_scope("discriminator"):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_real = discriminator(inputs, targets)

        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("discriminator", reuse=True):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_fake = discriminator(inputs, outputs)

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
        self.D_loss = discrim_loss
        #self.D_loss = ema.average(discrim_loss)
        self.D_grads_and_vars = discrim_grads_and_vars
        self.G_loss_GAN = gen_loss_GAN
        self.G_loss_L1 = gen_loss_L1
        #self.G_loss_GAN = ema.average(gen_loss_GAN)
        #self.G_loss_L1 = ema.average(gen_loss_L1)
        self.G_grads_and_vars = gen_grads_and_vars
        self.outputs = outputs
        self.train = tf.group(update_losses, incr_global_step, gen_train)

## MAIN ##
def main():
    g = tf.Graph()
    with g.as_default():
        model = Model(a)

        input_list, target_list = data_load(a.data_dir)
        test_x = input_list[-5:]
        test_y = target_list[-5:]

        fetches = {}
        fetches['predict_real'] = model.predict_real
        fetches['predict_fake'] = model.predict_fake
        fetches['D_loss'] = model.D_loss
        fetches['G_loss_GAN'] = model.G_loss_GAN
        fetches['G_loss_L1'] = model.G_loss_L1
        fetches['output'] = model.outputs
        fetches['train'] = model.train

        # Train
        sess_config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        with tf.Session(graph = g, config =sess_config) as sess:
            sess.run(tf.global_variables_initializer())

            total_batch = int(len(input_list) / a.batch_size)

            for epoch in range(a.total_epochs):
                for batch in range(total_batch):
                    batch_x = input_list[batch * a.batch_size: (batch+1) * a.batch_size]
                    batch_y = target_list[batch * a.batch_size: (batch+1) * a.batch_size]

                    result = sess.run(fetches, feed_dict = \
                            {model.inputs: batch_x, model.targets: batch_y})

                #if (epoch+1)%20==0 or epoch==1:
                print('\nEpoch: %d/%d' %(epoch, a.total_epochs))
                print('G GAN loss: %3.f   /   G L1 loss: %.3f'%(result['G_loss_GAN'],\
                    result['G_loss_L1']))
                print('D loss: %.3f'%result['D_loss'])
                #print('Real D: %.3f   /   Fake D: %.3f'%(result['predict_real'],\
                #   result['predict_fake']))

                if True:#epoch==0 or (epoch+1)%5 == 0:
                    generated = sess.run(fetches['output'], feed_dict = {model.inputs: test_x})

                    fig, ax = plt.subplots(3, 5, figsize=(5,3))
                    for i in range(len(generated)):
                        ax[0][i].set_axis_off()
                        ax[1][i].set_axis_off()
                        ax[2][i].set_axis_off()
                        ax[0][i].imshow(cv2color(np.reshape(test_x[i], (64,64,3))))
                        #print(np.vectorize(f)(generated[i])[32,32])
                        print(generated[i].shape)
                        ax[1][i].imshow(cv2color(np.reshape(np.vectorize(f)(generated[i]), (64,64,3))))
                        ax[2][i].imshow(cv2color(np.reshape(test_y[i], (64,64,3))))

                    plt.savefig('result/sample-%s.png' %str(epoch).zfill(3), bbox_inches='tight')
                    plt.close(fig)

            print('Finished!!')

if __name__=='__main__':
    main()
