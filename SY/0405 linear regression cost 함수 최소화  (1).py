
# coding: utf-8

# In[2]:


import tensorflow as tf
import matplotlib.pyplot as plt


# In[3]:


X = [1,2,3]
Y = [1.2,3]


# In[4]:


W = tf.placeholder(tf.float32)
hypothesis = X * W


# In[5]:


cost = tf.reduce_mean(tf.square(hypothesis - Y))


# In[6]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[7]:


w_val = []
cost_cal = []
for i in range(-30, 50):
    feed_w = i * 0.1 #0.1 간격으로 움직이겠다 
    curr_cost, curr_w = sess.run([cost,W],feed_dict={W: feed_w})
    w_val.append(curr_w)
    cost_val.append(curr_cost)
    
plt.plot(w_val,cost_val)
plt.show()


# In[8]:


learningrate = 0.1
gradient = tf.reduce_mean((W * X-Y)* X)
descent = W - learningrate *gradient
update = W.assign(descent)

#실행부분 sess.run에다가 update를 넣어줌


# In[11]:


#gradient descent magic을 사용
X = [1,2,3]
Y = [1,2,3]

W = tf.Variable(5.0)
hypothesis = X * W
cost = tf.reduce_mean(tf.square(hypothesis-Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer()) #한번에 넣어줄게요!

for step in range(100):
    print(step,sess.run(W))
    sess.run(train)
    
    
#출력상황 처음엔 5.0줬으니까 5.0인데 1.0으로 쫙내려감 


# In[12]:


#gradient descent magic을 사용
X = [1,2,3]
Y = [1,2,3]

W = tf.Variable(-3.0)
hypothesis = X * W
cost = tf.reduce_mean(tf.square(hypothesis-Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer()) #한번에 넣어줄게요!

for step in range(100):
    print(step,sess.run(W))
    sess.run(train)
    
    
#출력상황 처음엔 -3.0줬으니까  1.0으로 쫙내려감  

