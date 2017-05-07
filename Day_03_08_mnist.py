# Day_03_08_mnist.py
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist', one_hot=True)
print(mnist)
print(mnist.train)

learning_rate = 0.01
epoches = 15
batch_size = 100

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# ----------------------------------------------- #

# [1]
# w = tf.Variable(tf.zeros([784, 10]))        # 28x28 = 784
# b = tf.Variable(tf.zeros([10]))
#
# activation = tf.nn.softmax(tf.matmul(x, w) + b)
# cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation), reduction_indices=1))

# [2]
# w1 = tf.Variable(tf.random_normal([784, 256]))
# w2 = tf.Variable(tf.random_normal([256, 256]))
# w3 = tf.Variable(tf.random_normal([256,  10]))
#
# b1 = tf.Variable(tf.random_normal([256]))
# b2 = tf.Variable(tf.random_normal([256]))
# b3 = tf.Variable(tf.random_normal([ 10]))
#
# a1 = tf.add(tf.matmul(x, w1), b1)
# r1 = tf.nn.relu(a1)
# a2 = tf.add(tf.matmul(r1, w2), b2)
# r2 = tf.nn.relu(a2)
# a3 = tf.add(tf.matmul(r2, w3), b3)
#
# activation = a3
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=activation,
#                                                               labels=y))

# [3]
# w1 = tf.get_variable('w1', shape=[784, 256],
#                      initializer=tf.contrib.layers.xavier_initializer())
# w2 = tf.get_variable('w2', shape=[256, 256],
#                    initializer=tf.contrib.layers.xavier_initializer())
# w3 = tf.get_variable('w3', shape=[256, 10],
#                      initializer=tf.contrib.layers.xavier_initializer())
#
# b1 = tf.Variable(tf.zeros(256))
# b2 = tf.Variable(tf.zeros(256))
# b3 = tf.Variable(tf.zeros( 10))
#
# a1 = tf.add(tf.matmul(x, w1), b1)
# r1 = tf.nn.relu(a1)
# a2 = tf.add(tf.matmul(r1, w2), b2)
# r2 = tf.nn.relu(a2)
# a3 = tf.add(tf.matmul(r2, w3), b3)
#
# activation = a3
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=activation,
#                                                               labels=y))

# [4]
w1 = tf.get_variable('w1', shape=[784, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
w2 = tf.get_variable('w2', shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
w3 = tf.get_variable('w3', shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
w4 = tf.get_variable('w4', shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
w5 = tf.get_variable('w5', shape=[256, 10],
                     initializer=tf.contrib.layers.xavier_initializer())

b1 = tf.Variable(tf.zeros(256))
b2 = tf.Variable(tf.zeros(256))
b3 = tf.Variable(tf.zeros(256))
b4 = tf.Variable(tf.zeros(256))
b5 = tf.Variable(tf.zeros( 10))

dropout_rate = tf.placeholder(tf.float32)

r1 = tf.nn.relu(tf.add(tf.matmul( x, w1), b1))
d1 = tf.nn.dropout(r1, dropout_rate)
r2 = tf.nn.relu(tf.add(tf.matmul(d1, w2), b2))
d2 = tf.nn.dropout(r2, dropout_rate)
r3 = tf.nn.relu(tf.add(tf.matmul(d2, w3), b3))
d3 = tf.nn.dropout(r3, dropout_rate)
r4 = tf.nn.relu(tf.add(tf.matmul(d3, w4), b4))
d4 = tf.nn.dropout(r4, dropout_rate)

activation = tf.add(tf.matmul(d4, w5), b5)
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=activation, labels=y))

# ----------------------------------------------- #

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(epoches):
    avg_cost = 0
    total_batch = mnist.train.num_examples // batch_size    # 55,000 // 100 = 550

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # [1], [2], [3]
        # _, c = sess.run([optimizer, cost],
        #                 feed_dict={x: batch_xs, y: batch_ys})

        # [4]
        _, c = sess.run([optimizer, cost],
                        feed_dict={x: batch_xs, y: batch_ys,
                                   dropout_rate: 0.7})

        avg_cost += c / total_batch

    print('{:2} : {}'.format(epoch+1, avg_cost))

# -------------------------------- #

# import random
# r = random.randrange(mnist.test.num_examples)
# print('label :', sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
# print('prediction :', sess.run(tf.argmax(activation, 1),
#                                feed_dict={x: mnist.test.images[r:r+1]}))
#
# import matplotlib.pyplot as plt
# plt.imshow(mnist.test.images[r:r+1].reshape(28, 28),
#            cmap='Greys', interpolation='nearest')
# plt.show()

pred = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

# [1], [2], [3]
# print('accuracy :', sess.run(accuracy,
#                              feed_dict={x: mnist.test.images,
#                                         y: mnist.test.labels}))

# [4]
print('accuracy :',
      sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y: mnist.test.labels,
                                    dropout_rate: 1.0}))

sess.close()

# [1]
#  1 : 1.1821389803561293
#  2 : 0.6647238938374954
#  3 : 0.5526436823606488
#  4 : 0.4985456136139961
#  5 : 0.46543053643269966
#  6 : 0.44254436457699026
#  7 : 0.42549742454832234
#  8 : 0.41215200077403646
#  9 : 0.40135953681035513
# 10 : 0.3923696910522207
# 11 : 0.3846402803334323
# 12 : 0.37820207788185606
# 13 : 0.37235379059206347
# 14 : 0.3672827160629358
# 15 : 0.3627079488472503
# accuracy : 0.9088

# [2]
#  1 : 61.33217712358992
#  2 : 11.433859104026439
#  3 : 6.917257260463459
#  4 : 4.921961528916097
#  5 : 3.789511958821251
#  6 : 3.0267911279066064
#  7 : 2.4918515302118625
#  8 : 2.1083051426828567
#  9 : 1.794113747268165
# 10 : 1.5480191197482316
# 11 : 1.3570140808470486
# 12 : 1.1831571741266684
# 13 : 1.0490608508673775
# 14 : 0.9343943890710367
# 15 : 0.8368978861067636
# accuracy : 0.9199

# [3]
#  1 : 1.134528270932762
#  2 : 0.44547787403518513
#  3 : 0.3578770895708691
#  4 : 0.3186962825059895
#  5 : 0.2929688273099337
#  6 : 0.27370355839079097
#  7 : 0.25751960177313205
#  8 : 0.24336733040484526
#  9 : 0.23127515286884537
# 10 : 0.22029034357179286
# 11 : 0.21048718642104758
# 12 : 0.2012074931101367
# 13 : 0.1928012588891116
# 14 : 0.18507756715471071
# 15 : 0.17776776144450374
# accuracy : 0.949

# [4]
#  1 : 1.957496085600418
#  2 : 0.9128862186995417
#  3 : 0.6170013435862275
#  4 : 0.5019520586187187
#  5 : 0.4341651827096939
#  6 : 0.3910884460806846
#  7 : 0.3538022097403352
#  8 : 0.3248296009952376
#  9 : 0.30535439633510336
# 10 : 0.28452775500037486
# 11 : 0.2672169365937061
# 12 : 0.2552733839235523
# 13 : 0.24299048651348462
# 14 : 0.23057059263641205
# 15 : 0.222714341987263
# accuracy : 0.956
