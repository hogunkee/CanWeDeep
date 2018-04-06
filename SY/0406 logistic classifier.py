
# coding: utf-8

# In[22]:


import tensorflow as tf
x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

x = tf.placeholder(tf.float32, shape=[None,2])
y = tf.placeholder(tf.float32, shape = [None,1])

w = tf.Variable(tf.random_normal([2,1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

hypothesis = tf.sigmoid(tf.matmul(x,w)+b)
cost = -tf.reduce_mean(y* tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate= 0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))


# In[23]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        cost_val, _ = sess.run([cost,train], feed_dict={x: x_data, y:y_data })
        if step % 200 ==0:
            print(step,cost_val)
    
    h,c,a = sess.run([hypothesis, predicted, accuracy],feed_dict={x: x_data,
                                                                y: y_data})
    print("\nhypothesis",h,"\nCorrec(Y)",c,"\nAccuary:",a)


# In[24]:


#당료병을 예측하는 data! 
import numpy as np
xy = np.loattxt('data-03-diabetes.csv',delimiter=',',dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

x = tf.placeholder(tf.float32, shape=[None,8])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([8,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x*w)+b)
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*(tf.log(1-hypothesis)))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01.minimize(cost))

predicted = tf.cast(hypothesis> 0.5, dtype=tf.float32)
accuarcy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtyple=tf.float32))

with tf.Session as sess:
    sess.run(tf.global_variables_initializer())
    
    feed = {x: x_data, y:y_data}
    
    for step in range(10001):
            sess.run(train,feed_dict=feed)
            if  step % 200 == 0:
                    print(step,sess.run(cost,feed_dict=feed))
            
    h,c,a = sess.run([hypothesis,predicted,accuarcy], feed_dict=feed)
    

            


