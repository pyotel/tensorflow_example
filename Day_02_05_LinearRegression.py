# Day_02_05_LinearRegression.py
import tensorflow as tf


def not_used_1():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.Variable(tf.random_uniform([1], -1, 1))
    b = tf.Variable(tf.random_uniform([1], -1, 1))

    hypothesis = w*x + b
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    learning_rate = tf.Variable(0.1)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        sess.run(train)

        if i%20 == 0:
            print(i, sess.run(cost))

    ww = sess.run(w)
    bb = sess.run(b)
    print(ww*7 + bb)

    sess.close()

    # 문제
    # x가 7일 때의 y 값을 예측해 보세요.


def not_used_2():
    # 문제
    # place holder 버전으로 수정하세요.
    # 그리고, x가 5일 때, 7일 때, 5와 7일 때의 값을 예측해 보세요.
    xx = [1, 2, 3]
    yy = [1, 2, 3]

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_uniform([1], -1, 1))
    b = tf.Variable(tf.random_uniform([1], -1, 1))

    hypothesis = w*x + b
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    learning_rate = tf.Variable(0.1)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        sess.run(train, feed_dict={x: xx, y: yy})

        if i%20 == 0:
            print(i, sess.run(cost, feed_dict={x: xx, y: yy}))

    print(sess.run(hypothesis, feed_dict={x: 5}))
    print(sess.run(hypothesis, feed_dict={x: 7}))
    print(sess.run(hypothesis, feed_dict={x: [5, 7]}))

    sess.close()


import numpy as np

xy = np.loadtxt('Data/xy.txt',
                skiprows=1, delimiter=',', unpack=True)
print(xy)
print(type(xy))
print(xy[0], xy[-1])
print(xy[:2])
print(xy[:-1])

# 문제
# xy.txt 파일로부터 값을 읽어오는 코드로 수정해 보세요.
xx = xy[0]
yy = xy[1]

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
sess.run(tf.global_variables_initializer())

for i in range(2001):
    sess.run(train, feed_dict={x: xx, y: yy})

    if i % 20 == 0:
        print(i, sess.run(cost, feed_dict={x: xx, y: yy}))

print(sess.run(hypothesis, feed_dict={x: 5}))
print(sess.run(hypothesis, feed_dict={x: 7}))
print(sess.run(hypothesis, feed_dict={x: [5, 7]}))

sess.close()
