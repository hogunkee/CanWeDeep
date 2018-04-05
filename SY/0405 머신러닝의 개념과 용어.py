
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


hello = tf.constant("Hello, Tensorflow")
sess = tf.Session()
print(sess.run(hello)) #b는 바이트 스트림이여서 걱정없어도됨


# In[4]:


node1 = tf.constant(3.0,tf.float32) #float32는 데이터 타입 
node2 = tf.constant(4.0) 
node3 = tf.add(node1,node2)

print("node1:",node1,"node2",node2)
print("node3")


# In[10]:


print("sess.run(node1,node2):",sess.run([node1,node2]))
print("node3: ",sess.run(node3))


# In[23]:


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b 

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1,3], b: [2,4]}))


# In[24]:


s=1 #0차원
[1. , 2., 3]  # 1차원 shape는 3개 
[[1.,2.,3.], [4.,5.,6.]] #2차원 rank 안에 요소는 3개니까 3차원 shape 

