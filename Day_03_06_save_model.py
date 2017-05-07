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

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2001):
    sess.run(train, feed_dict={x: xx, y: yy})

    if i % 20 == 0:
        print(i, sess.run(cost, feed_dict={x: xx, y: yy}))
        saver.save(sess, 'Model/basic', global_step=i)

# saver.save(sess, 'Model/model')

sess.close()








