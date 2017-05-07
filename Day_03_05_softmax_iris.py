# Day_03_05_softmax_iris.py
import tensorflow as tf
import numpy as np
import csv

# 문제
# iris.csv 파일을 소프트맥스에서 사용할 수 있는 형태로 화면에 출력해 보세요.
# 1.,4.6,3.2,1.4,0.2,1,0,0

def make_iris_softmax():
    f = open('Data/iris.csv', 'r', encoding='utf-8')
    f.readline()

    iris = []
    for row in csv.reader(f):

        line =[1.]
        for item in row[1:-1]:
            line.append(float(item))

        if row[-1] == 'setosa':
            line += [1, 0, 0]
        elif row[-1] == 'versicolor':
            line += [0, 1, 0]
        else:
            line += [0, 0, 1]

        iris.append(line)

    f.close()
    # print(*iris, sep='\n')

    f = open('Data/iris_softmax.csv', 'w',
             encoding='utf-8', newline='')
    csv.writer(f).writerows(iris)
    f.close()


# make_iris_softmax()


# 문제
# 120개 train_set, 30개의 test_set을 반환하는 함수를 만들어 주세요.
#       (120, 8)         (30, 8)
import random
def get_iris_softmax():
    iris = np.loadtxt('Data/iris_softmax.csv',
                      delimiter=',', dtype=np.float32)

    # train_set = iris[:40] + iris[50:90] + iris[100:140]
    # train_set = np.array(list(iris[:40]) + list(iris[50:90]) + list(iris[100:140]))

    train_set = np.vstack((iris[:40], iris[50:90], iris[100:140]))
    test_set = np.vstack((iris[40:50], iris[90:100], iris[140:]))

    # random.shuffle(iris)
    #
    # train_set = iris[:120]
    # test_set = iris[-30:]

    # random.shuffle(iris[:50])
    # random.shuffle(iris[50:100])
    # random.shuffle(iris[100])
    #
    # train_set = np.vstack((iris[:40], iris[50:90], iris[100:140]))
    # test_set = np.vstack((iris[40:50], iris[90:100], iris[140:]))

    return train_set, test_set


# 문제
# iris_softmax 데이터를 예측해 보세요.
train_set, test_set = get_iris_softmax()
print(train_set.shape, test_set.shape)

# xx = train_set.transpose()[:5].transpose()     # (120, 5)
xx = train_set[:,:5]
yy = train_set[:,5:]

print(xx.shape, yy.shape)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable(tf.zeros([5, 3]))

#  (120, 5) x (5, 3) = (120, 3)
z = tf.matmul(x, w)
hypothesis = tf.nn.softmax(z)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis)))
learning_rate = tf.Variable(0.0015)

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2001):
    sess.run(train, feed_dict={x: xx, y: yy})
    if i%20 == 0:
        print(i, sess.run(cost, feed_dict={x: xx, y: yy}))

test_x = test_set[:,:5]
test_y = test_set[:,5:]

y1 = sess.run(hypothesis, feed_dict={x: test_x})
print(y1)
print(y1.shape)

y2 = sess.run(tf.argmax(y1, 1))
t2 = sess.run(tf.argmax(test_y, 1))

print(y2)
print(t2)

print(y2 == t2)
print(np.mean(y2 == t2))

nomial = np.array(['setosa', 'versicolor', 'virginica'])
print(nomial[y2])

sess.close()
