# -*- coding: utf-8 -*-
"""car regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iJxAlBDC0i0lRsrSa5ColZYC8Cva3sAc
"""

import tensorflow as tf
import csv
import numpy as np
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(name=fn, length=len(uploaded[fn])))

xy = np.genfromtxt('car1.csv', delimiter=',')

for i in range(7):
  min = xy[:,i]
  mean = np.mean(xy[:,i])
  std = np.std(xy[:,i])
  xy[:,i]=(xy[:,i]-mean)/std
print(xy)


x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# placeholders for a tensor that will be always fed.
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([7, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())

   feed = {X: x_data, Y: y_data}
   for step in range(10001):
       sess.run(train, feed_dict=feed)
       if step % 200 == 0:
           print(step, sess.run(cost, feed_dict=feed))

   # Accuracy report
   h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict=feed)
   print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
