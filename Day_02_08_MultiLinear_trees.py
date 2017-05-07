# # Day_02_08_MultiLinear_trees.py
import tensorflow as tf
import numpy as np


def not_used():
    # 문제
    # Girth와 Height를 x로 하고
    # Volume을 y로 하는 모델을 구축해 보세요.
    # x1이 8, x2가 10일 때의 y를 예측해 보세요.
    # x1이 (8, 11), x2가 (10, 7)일 때의 y를 예측해 보세요.
    trees = np.loadtxt('Data/trees.csv',
                       delimiter=',', unpack=True, skiprows=1)
    print(trees.shape)

    xx = [np.ones(trees.shape[1]), trees[0], trees[1]]
    yy = trees[-1]

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_uniform([1, 3], -1, 1))

    hypothesis = tf.matmul(w, x)            # (1, 3) x (3, 31) = (1, 31)
    cost = tf.reduce_mean((hypothesis-y)**2)
    learning_rate = tf.Variable(0.00015)

    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    feed = {x: xx, y: yy}
    for i in range(2001):
        sess.run(train, feed_dict=feed)

        if i%20 == 0:
            print(sess.run(cost, feed_dict=feed), sess.run(w))

    # (1, 3) x (3, 1) = (1, 1)
    print(sess.run(hypothesis, feed_dict={x: [[1], [8], [10]]}))
    # (1, 3) x (3, 2) = (1, 2)
    print(sess.run(hypothesis, feed_dict={x: [[1, 1], [8, 11], [10, 7]]}))

    sess.close()


# 문제
# "Girth","Height","Volume"에 대해서 동작하는 모델링 함수를 만드세요.
# "Girth", "Height" --> "Volume"
# "Volume", "Girth" --> "Height"
# "Height", "Volume" --> "Girth"

def show_trees(x1, x2, label, learning):
    xx = [np.ones(len(x1)), x1, x2]
    yy = label

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_uniform([1, 3], -1, 1))

    hypothesis = tf.matmul(w, x)            # (1, 3) x (3, 31) = (1, 31)
    cost = tf.reduce_mean((hypothesis-y)**2)
    learning_rate = tf.Variable(learning)

    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    feed = {x: xx, y: yy}
    for i in range(2001):
        sess.run(train, feed_dict=feed)

        if i%20 == 0:
            print(sess.run(cost, feed_dict=feed), sess.run(w))

    sess.close()


# trees = np.loadtxt('Data/trees.csv',
#                    delimiter=',', unpack=True, skiprows=1)
# print(trees.shape)

girth, height, volume = np.loadtxt('Data/trees.csv',
                                   delimiter=',', unpack=True, skiprows=1)
print(girth.shape)

# show_trees(girth, height, volume, 0.00015)
# show_trees(volume, girth, height, 0.0007)
# show_trees(height, volume, girth, 0.0001)

import matplotlib.pyplot as plt

# plt.plot(girth, height, 'ro')
# plt.plot(height, volume, 'g^')
# plt.plot(volume, girth, 'b>')
# plt.show()

# plt.subplot(221)
# plt.plot(girth, height, 'ro')
# plt.subplot(222)
# plt.plot(height, volume, 'g^')
# plt.subplot(224)
# plt.plot(volume, girth, 'b>')
# plt.show()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(girth, height, volume, 'ro')

plt.xlabel('girth')
plt.ylabel('height')
ax.set_zlabel('volume')
plt.show()
