import tensorflow as tf
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

data_x=[]
data_y=[]
for i in range(10):
    dirname = 'C:/Users/JisungKim/Desktop/Sign-Language-Digits-Dataset/Dataset/%d'%i
    for filename in os.listdir(dirname):
        im = Image.open(dirname+'/'+filename)
        lst=list(im.getdata(),one_hot=True)
        data_x.append(lst)
        data_y.append(i)


learning_rate = 0.001
training_epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32, [None, 10000, 3])
X_img = tf.reshape(X, [-1, 100, 100, 3])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2_flat = tf.reshape(L2, [-1, 25 * 25 * 64])

W3 = tf.get_variable("W3", shape=[25 * 25 * 64, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L2_flat, W3) + b

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
print('Learning started. It takes sometime.')
X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=0.1, random_state=42)
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(data_x) / batch_size)

    for i in range(total_batch):
        batch_xs = X_train[batch_size * i: batch_size * (i+1)]
        batch_ys = Y_train[batch_size * i: batch_size * (i+1)]
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: X_test, Y: Y_test}))

# Get one and predict
r = random.randint(0, len(Y_train) - 1)
print("Label: ", sess.run(tf.argmax(Y_test[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(logits, 1), feed_dict={X: X_test[r:r + 1]}))
