import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_loader import *

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):

            self.X = tf.placeholder(tf.float32, [None, 100, 100, 3])
            self.Y = tf.placeholder(tf.float32, [None, 10])
            self.keep_prob = tf.placeholder(tf.float32)

            W1 = tf.Variable(tf.random_normal([3,3,3,32], stddev=0.01))
            L1 = tf.nn.conv2d(self.X, W1, strides=[1,1,1,1], padding='SAME')
            print(L1.shape)
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            L1 = tf.nn.dropout(L1, keep_prob = self.keep_prob)

            W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
            L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
            print(L2.shape)
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            L2 = tf.nn.dropout(L2, keep_prob = self.keep_prob)

            W3 = tf.Variable(tf.random_normal([3,3,64,128], stddev=0.01))
            L3 = tf.nn.conv2d(L2, W3, strides=[1,1,1,1], padding='SAME')
            print(L3.shape)
            L3 = tf.nn.relu(L3)
            L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            print(L3.shape)
            L3 = tf.nn.dropout(L3, keep_prob = self.keep_prob)
            L3 = tf.reshape(L3, [-1,128*13*13])

            W4 = tf.get_variable('W4', shape=[128*13*13, 625],
                    initializer = tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([625]))
            L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
            L4 = tf.nn.dropout(L4, keep_prob = self.keep_prob)

            W5 = tf.get_variable('W5', shape=[625, 10],
                    initializer = tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([10]))
            hypothesis = tf.matmul(L4, W5) + b5

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=self.Y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
            correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(self.Y, 1))
            self.logits = tf.argmax(hypothesis, 1)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, keep_prob=1.0):
        return self.sess.run(self.logits,
                feed_dict={self.X: x_test, self.keep_prob: keep_prob})

    def get_accuracy(self, x_test, y_test, keep_prob=1.0):
        return self.sess.run(self.accuracy,
                feed_dict = {self.X: x_test, self.Y: y_test, self.keep_prob: keep_prob})

    def train(self, x_data, y_data, keep_prob=0.7):
        return self.sess.run([self.cost, self.accuracy, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.keep_prob: keep_prob})

learning_rate = 0.001
training_epochs = 15
batch_size = 20

x_train, y_train, x_test, y_test = data_load(DATAPATH)

sess = tf.Session()
m1 = Model(sess, 'm1')

sess.run(tf.global_variables_initializer())

print('Learning Started!')
for epoch in range(training_epochs):
    avg_cost = 0.0
    avg_accur = 0.0
    total_batch = int(len(x_train) // batch_size)

    for i in range(total_batch):
        batch_xs = x_train[batch_size * i: batch_size * (i+1)]
        batch_ys = y_train[batch_size * i: batch_size * (i+1)]
        cost, accur, _ = m1.train(batch_xs, batch_ys)
        avg_accur += accur / total_batch
        avg_cost += cost / total_batch
    print('Epoch:', '%04d' %(epoch + 1), 'cost = %.9f, accuracy = %.3f' %(avg_cost, avg_accur))

print('Learning Finished!')

accur = m1.get_accuracy(x_test, y_test)
print('Accuracy:', accur)
