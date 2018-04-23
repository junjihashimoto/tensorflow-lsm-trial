import numpy as np
import tensorflow as tf
import random


sample_x = np.arange(0,10,0.1)
sample_y = np.frompyfunc(lambda t: t*2 + 3,1,1)(sample_x)
sample_ty = sample_y
rand_y = np.random.rand(100)
sample_y = sample_y + rand_y
a = tf.get_variable('a',shape=[],dtype=tf.float64)
print(a.shape)
b = tf.get_variable('b',shape=[],dtype=tf.float64)

x = tf.placeholder(tf.float64,shape=[None],name='x')
y = tf.placeholder(tf.float64,shape=[None],name='y')
fy = a*x + b
assert(fy.dtype==tf.float64)

loss = tf.reduce_mean(tf.square(y - fy))
init =tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep = 3)

with tf.Session() as sess:
  sess.run(init)
  a.assign(0)
  b.assign(0)
  print("a,b:")
  print(sess.run(a))
  print(sess.run(b))
  print(sess.run(fy, feed_dict={x:np.array([0,1,2])}))
  yy = np.frompyfunc(lambda t:t*2+3,1,1)(np.array([0,1,2]))
  print("loss:")
  print(sess.run(loss, feed_dict={x:np.array([0,1,2]),y: yy}))
  step = 0.005
  for i in range(0,100):
    train_step = tf.train.GradientDescentOptimizer(step).minimize(loss)
  step = 0.003
  for i in range(0,10):
    train_step = tf.train.GradientDescentOptimizer(step).minimize(loss)
  step = 0.001
  for i in range(0,10):
    train_step = tf.train.GradientDescentOptimizer(step).minimize(loss)
  step = 0.001
  for i in range(0,100):
    print(f"step:{step}")
    train_step = tf.train.GradientDescentOptimizer(step).minimize(loss)
    sess.run(train_step, feed_dict={x:sample_x,y:sample_y})
    print(f"a,b:{sess.run(a)},{sess.run(b)}")
    print(f"loss:{sess.run(loss, feed_dict={x:np.array([0,1,2]),y: yy})}")
    step = 0.95 * step
    if i % 25 == 24 : 
      saver.save(sess, 'ckpt/my_model')
