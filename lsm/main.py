import numpy as np
import tensorflow as tf
import random

global_step = tf.Variable(0, name='global_step', trainable=False)

sample_x = np.arange(0,10,0.1)
sample_y = np.frompyfunc(lambda t: t*2 + 3,1,1)(sample_x)
sample_ty = sample_y
rand_y = np.random.rand(100)
sample_y = sample_y + rand_y
a = tf.Variable(2,name='a',dtype=tf.float64)
b = tf.Variable(3,name='b',dtype=tf.float64)

x = tf.placeholder(tf.float64,shape=[None],name='x')
y = tf.placeholder(tf.float64,shape=[None],name='y')
fy = a*x + b
assert(fy.dtype==tf.float64)

loss = tf.reduce_mean(tf.square(y - fy))

#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss, global_step=global_step)
train_step = tf.train.MomentumOptimizer(0.01,0.9).minimize(loss, global_step=global_step)

init =tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep = 3)

tf.summary.scalar("loss", loss)
summary_op = tf.summary.merge_all()

graph = tf.get_default_graph()
summary_writer = tf.summary.FileWriter('logs/', graph)


with tf.Session() as sess:
  sess.run(init)
  loss_v = sess.run(loss, feed_dict={x:sample_x,y:sample_y})
  print(f"loss:{loss_v}")
  for i in range(0,100):
    (_,loss_v,a_v,b_v) = sess.run((train_step,loss,a,b), feed_dict={x:sample_x,y:sample_y})
    print(f"a,b:{a_v},{b_v}")
    print(f"loss:{loss_v}")
    if i % 10 == 9 :
      summary_str = sess.run(summary_op, feed_dict={x:sample_x,y:sample_y})
      summary_writer.add_summary(summary_str, i)
      saver.save(sess, 'ckpt/my_model')

summary_writer.close()

