import numpy as np
import tensorflow as tf

TB_SUMMARY_DIR = './sample/'
x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype = np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype = np.float32)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
with tf.variable_scope('layer1') as scope:
    W1 = tf.get_variable("W", shape = [2,2])
    b1 = tf.Variable(tf.random_normal([2]), name='b')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    tf.summary.histogram("X", X)
    tf.summary.histogram("weights", W1)
    tf.summary.histogram("bias", b1)
    tf.summary.histogram("layer", layer1)

with tf.variable_scope('layer2') as scope:
    W2 = tf.get_variable("W", shape = [2,1])
    b2 = tf.Variable(tf.random_normal([1]), name='b')
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

cost = - tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

tf.summary.scalar("loss", cost)
summary = tf.summary.merge_all()

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(TB_SUMMARY_DIR)
    writer.add_graph(sess.graph)

    for step in range(10001):
        s, _ = sess.run([summary, train], feed_dict = {X: x_data, Y: y_data})
        writer.add_summary(s, global_step = step)

        if step%100 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run([W1, W2]))

    
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)
