# Day_03_01_LogisticClassification.py
import tensorflow as tf
import numpy as np

def not_used():
    x = [[1., 1., 1., 1., 1., 1.],
         [2., 3., 3., 5., 7., 2.],
         [1., 2., 5., 5., 5., 5.]]
    y = np.array([0, 0, 0, 1, 1, 1])
    # print(1 - y)
    # y - y

    w = tf.Variable(tf.random_uniform([1, 3], -1, 1))

    z = tf.matmul(w, x)
    hypothesis = tf.div(1., 1. + tf.exp(-z))                # math.e ** -z
    cost = -tf.reduce_mean(   y *tf.log(  hypothesis) +
                           (1-y)*tf.log(1-hypothesis))
    learning_rate = tf.Variable(0.1)

    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        sess.run(train)

        if i%20 == 0:
            print(i, sess.run(cost), sess.run(w))
            # print(i, sess.run([cost, w]))

    sess.close()


# 문제
# x1이 (4, 6), x2가 (4, 3)일 때의 결과를 예측해 보세요.
xx = [[1., 1., 1., 1., 1., 1.],
      [2., 3., 3., 5., 7., 2.],
      [1., 2., 5., 5., 5., 5.]]
yy = np.array([0, 0, 0, 1, 1, 1])
# print(1 - y)
# y - y

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable(tf.random_uniform([1, 3], -1, 1))

z = tf.matmul(w, x)
hypothesis = tf.div(1., 1. + tf.exp(-z))                # math.e ** -z
cost = -tf.reduce_mean(   y *tf.log(  hypothesis) +
                       (1-y)*tf.log(1-hypothesis))
learning_rate = tf.Variable(0.1)

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2001):
    sess.run(train, feed_dict={x: xx, y: yy})

    if i%20 == 0:
        print(i, sess.run(cost, feed_dict={x: xx, y: yy}), sess.run(w))
        # print(i, sess.run([cost, w]))

a1 = sess.run(hypothesis, feed_dict={x: [[1], [4], [4]]})
a2 = sess.run(hypothesis, feed_dict={x: [[1], [6], [3]]})
a3 = sess.run(hypothesis, feed_dict={x: [[1, 1], [4, 6], [4, 3]]})

print(a1)
print(a2)
print(a3)

print(a1 > 0.5)
print(a2 > 0.5)
print(a3 > 0.5)

sess.close()
