from tensorflow.examples.tutorials.mnist import input_data
import cv2
import os
from config import get_config
#from data_loader import *
from model import *


def main(config):
    if config.outf is None:
        config.outf = 'result'
    os.system('mkdir {0}'.format(config.outf))

    if config.dataset=='mnist':
        mnist = input_data.read_data_sets('./sample/MNIST_data', one_hot=False)
    else:
        pass

    model = GAN(config)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    test_Z = np.random.normal(size=[config.test_size, config.nz])

    for iter in range(config.niter):
        print(config.batch_size)
        input_X = mnist.train.next_batch(config.batch_size)[0]
        input_Z = np.random.normal(size=[config.batch_size, config.nz])
        print('X:',input_X.shape)
        print('Z:',input_Z.shape)

        fetches = {
                'D_loss': model.D_loss,
                'G_loss': model.G_loss,
                'D_train_step': model.train_D,
                'G_train_step': model.train_G
                }

        vals = sess.run(fetches, feed_dict={ model.X: input_X, model.Z: input_Z })
        D_loss = vals['D_loss']
        G_loss = vals['G_loss']

        if (iter+1)%config.print_step==0:
            print("[%d/%d] steps. D_loss: %.3f, G_loss: %.3f" \
                    %(iter+1, config.niter, D_loss, G_loss))

        gen_images = sess.run(model.X_fake, feed_dict={model.Z: test_Z})
        gen_images = np.reshape(gen_images, [-1, 28, 28, 1])

        img = gen_images[0]
        for i in range(1, config.test_size):
            img = np.concatenate((img, gen_images[i]), axis=1)
        cv2.imwrite('config.outf/iter%d.png' %iter, img)

    print('training fished!')

if __name__ == '__main__':
    config = get_config()
    main(config)
