# Day_03_07_restore_model.py
# Day_03_06_save_model.py
import tensorflow as tf

xx = [1, 2, 3]
yy = [1, 2, 3]

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable(tf.random_uniform([1], -1, 1))
b = tf.Variable(tf.random_uniform([1], -1, 1))

hypothesis = w * x + b
cost = tf.reduce_mean(tf.square(hypothesis - y))
learning_rate = tf.Variable(0.1)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

sess = tf.Session()

saver = tf.train.Saver()
latest = tf.train.latest_checkpoint("Model")

if latest:
    saver.restore(sess, latest)
else:
    pass

print(sess.run(hypothesis, feed_dict={x: 7}))

sess.close()

















