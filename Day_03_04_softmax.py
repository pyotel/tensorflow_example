# Day_03_04_softmax.py
import tensorflow as tf
import numpy as np

def not_used():
    xy = np.loadtxt('Data/softmax.txt',
                    unpack=True, dtype=np.float32)
    # print(xy)

    x = xy[:-3].transpose()     # (8, 3)
    y = xy[-3:].transpose()     # (8, 3)

    # print(x.shape, y.shape)

    w = tf.Variable(tf.zeros([3, 3]))

    #  (8, 3) x (3, 3) = (8, 3)
    z = tf.matmul(x, w)
    hypothesis = tf.nn.softmax(z)
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis)))
    learning_rate = tf.Variable(0.035)

    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        sess.run(train)
        if i%20 == 0:
            print(i, sess.run(cost))

    sess.close()


# 문제
# x1이 (11, 7), x2가 (3, 4)일 때의 결과를 예측해 보세요.
xy = np.loadtxt('Data/softmax.txt',
                unpack=True, dtype=np.float32)

xx = xy[:-3].transpose()     # (8, 3)
yy = xy[-3:].transpose()     # (8, 3)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable(tf.zeros([3, 3]))

#  (8, 3) x (3, 3) = (8, 3)
z = tf.matmul(x, w)
hypothesis = tf.nn.softmax(z)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis)))
learning_rate = tf.Variable(0.035)

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2001):
    sess.run(train, feed_dict={x: xx, y: yy})
    if i%20 == 0:
        print(i, sess.run(cost, feed_dict={x: xx, y: yy}))

# x1이 (11, 7), x2가 (3, 4)일 때의 결과를 예측해 보세요.
a1 = sess.run(hypothesis, feed_dict={x: [[1, 11, 3]]})
a2 = sess.run(hypothesis, feed_dict={x: [[1, 7, 4]]})
a3 = sess.run(hypothesis, feed_dict={x: [[1, 11, 3], [1, 7, 4]]})

print(a1)
print(a2)
print(a3)

print(np.sum(a1))
print(np.sum(a2))
print(np.sum(a3), np.sum(a3, 1))

yhat1 = sess.run(tf.argmax(a1, 1))
yhat2 = sess.run(tf.argmax(a2, 1))
yhat3 = sess.run(tf.argmax(a3, 1))

print(yhat1)
print(yhat2)
print(yhat3)

nomial = ['A', 'B', 'C']
print(nomial[yhat1[0]])
print(nomial[yhat2[0]])
print(nomial[yhat3[0]], nomial[yhat3[1]])

nomial = np.array(nomial)
print(nomial[yhat1])
print(nomial[yhat2])
print(nomial[yhat3])

sess.close()
