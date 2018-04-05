
# coding: utf-8

# In[3]:


import tensorflow as tf

x_train = [1,2,3]
y_train = [1,2,3]

w = tf.Variable(tf.random_normal([1]), name='height')  #1 차원 rank입니다
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis= x_train*w+b
cost = tf.reduce_mean(tf.square(hypothesis-y_train))



# In[7]:


optimizer =tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost) #cost를 그라디언트 방법을 사용해서 minimize하는것 


# In[9]:


sess = tf.Session()
sess.run(tf.global_variables_initializer()) #한번에 사용한  모든 variable 을 넣어줌 

for step in range(2001): # 2000번 실행 
    sess.run(train)
    if step %20 == 0: # 학습이 되는지 보고싶어서  
        print(step, sess.run(cost), sess.run(w), sess.run(b))
 # 당연히 cost는 작아져야되고, w는 1에 가까워지고 b는 0에 가까워짐        


# In[15]:


x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

for step in range(2001): # 2000번 실행 
    cost_val, w_val, b_val, _ = sess.run([cost,w,b,train],
        feed_dict={x:[1,2,3], y:[1,2,3]})
    #변수를 받긴 받아야되는데 안쓸거니까 이름을 _ 로 지어준 거임 

    
    if step %20 == 0: # 학습이 되는지 보고싶어서  
        print(step, sess.run(cost), sess.run(w), sess.run(b))

        
#testing 해볼때는 
print(sess.run(hypothesis, feed_dict = {x: [5]}))
#이런식으로 넣어줘서 값을 확인해서 잘 되어있는지 확인 가능! 

