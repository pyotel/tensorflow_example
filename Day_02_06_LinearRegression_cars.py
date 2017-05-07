# Day_02_06_LinearRegression_cars.py
import tensorflow as tf
import numpy as np

# 문제
# cars.csv 파일에 대해 모델을 구축하세요.
# 속도가 30, 50일 때의 제동거리를 예측해 보세요.
cars = np.loadtxt('Data/cars.csv',
                  delimiter=',', unpack=True)
print(cars)
print(len(cars), len(cars[0]))
print(cars.shape)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable(tf.random_uniform([1], -1, 1))
b = tf.Variable(tf.random_uniform([1], -1, 1))

hypothesis = w * x + b
cost = tf.reduce_mean(tf.square(hypothesis - y))
learning_rate = tf.Variable(0.0035)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2001):
    sess.run(train, feed_dict={x: cars[0], y: cars[1]})

    if i % 20 == 0:
        print(i, sess.run(cost, feed_dict={x: cars[0], y: cars[1]}))

print(sess.run(hypothesis, feed_dict={x: 30}))
print(sess.run(hypothesis, feed_dict={x: 50}))
print(sess.run(hypothesis, feed_dict={x: [30, 50]}))

y1 = sess.run(hypothesis, feed_dict={x:  0})
y2 = sess.run(hypothesis, feed_dict={x: 30})

sess.close()

import matplotlib.pyplot as plt

plt.plot(cars[0], cars[1], 'ro')
plt.plot([0, 30], [0, y2], 'g')
plt.plot([0, 30], [y1, y2], 'r')
plt.show()
