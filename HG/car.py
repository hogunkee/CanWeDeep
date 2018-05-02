import tensorflow as tf
import numpy as np

xy = np.genfromtxt('car.csv', delimiter=',', dtype=np.float32)
xy_trans = np.transpose(xy)
new_xy = []
for col in xy_trans[:-1]:
    col_mean = np.mean(col)
    col_std = np.std(col)
    new_xy.append((col - col_mean) / col_std)
new_xy.append(xy_trans[-1])

xy = np.transpose(new_xy)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 7])
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

       accur = sess.run(accuracy, feed_dict=feed)
       print("Accuracy: ", accur)

   # Accuracy report
   h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict=feed)
   print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
