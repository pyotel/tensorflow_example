# Day_02_07_MultiLinear.py
import tensorflow as tf


def not_used_1():
    x1 = [1, 0, 3, 0, 5]
    x2 = [0, 2, 0, 4, 0]
    y  = [1, 2, 3, 4, 5]

    w1 = tf.Variable(tf.random_uniform([1], -1, 1))
    w2 = tf.Variable(tf.random_uniform([1], -1, 1))
    b = tf.Variable(tf.random_uniform([1], -1, 1))

    hypothesis = w1*x1 + w2*x2 + b
    cost = tf.reduce_mean((hypothesis-y)**2)
    learning_rate = tf.Variable(0.1)

    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        sess.run(train)

        if i%20 == 0:
            print(sess.run(cost), sess.run(w1), sess.run(w2))

    sess.close()


def not_used_2():
    x = [[1., 0., 3., 0., 5.],
         [0., 2., 0., 4., 0.]]
    y = [1, 2, 3, 4, 5]

    w = tf.Variable(tf.random_uniform([1, 2], -1, 1))
    b = tf.Variable(tf.random_uniform([1], -1, 1))

    hypothesis = tf.matmul(w, x) + b            # (1, 2) x (2, 5) = (1, 5)
    cost = tf.reduce_mean((hypothesis-y)**2)
    learning_rate = tf.Variable(0.1)

    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        sess.run(train)

        if i%20 == 0:
            print(sess.run(cost), sess.run(w))

    sess.close()


def not_used_3():
    # 문제
    # bias를 없애보세요.
    x = [[1., 1., 1., 1., 1.],
         [1., 0., 3., 0., 5.],
         [0., 2., 0., 4., 0.]]
    y = [1, 2, 3, 4, 5]

    w = tf.Variable(tf.random_uniform([1, 3], -1, 1))

    hypothesis = tf.matmul(w, x)            # (1, 3) x (3, 5) = (1, 5)
    cost = tf.reduce_mean((hypothesis-y)**2)
    learning_rate = tf.Variable(0.1)

    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        sess.run(train)

        if i%20 == 0:
            print(sess.run(cost), sess.run(w))

    sess.close()


# 문제
# xxy.txt 파일을 만들어서 데이터를 직접 입력하세요.
# 아래 코드를 파일로부터 읽어오는 place holder 버전으로 수정합니다.
import numpy as np
# xxy = np.loadtxt('Data/xxy.txt',
#                  delimiter=',', unpack=True)
# print(xxy.shape)
#
# xx = xxy[:-1]
# yy = xxy[-1]

xxy = np.loadtxt('Data/xxy2.txt',
                 delimiter=',', unpack=True, dtype=np.float32)
print(xxy.shape)
print(np.ones([5]))
print(xxy.dtype)

# xx = [np.ones(len(xxy[0])), xxy[0], xxy[1]]
# xx = [np.ones(xxy.shape[1]), xxy[0], xxy[1]]
# xx = np.vstack((np.ones(xxy.shape[1]), xxy[:-1]))
xx = [np.ones(xxy.shape[1])]
xx.extend(xxy[:-1])
yy = xxy[-1]

xx = np.array(xx)
# print(len(xx), len(xx[0]))

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable(tf.random_uniform([1, 3], -1, 1))

hypothesis = tf.matmul(w, x)            # (1, 3) x (3, 5) = (1, 5)
cost = tf.reduce_mean((hypothesis-y)**2)
learning_rate = tf.Variable(0.1)

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

feed = {x: xx, y: yy}
for i in range(2001):
    sess.run(train, feed_dict=feed)

    if i%20 == 0:
        print(sess.run(cost, feed_dict=feed), sess.run(w))

sess.close()

