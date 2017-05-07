# Day_03_02_Logistic_iris.py
import tensorflow as tf
import numpy as np
import csv

# iris = np.loadtxt('Data/iris.csv',
#                   delimiter=',',
#                   skiprows=1)
# print(iris)


def get_iris(species_true, species_false):
    f = open('Data/iris.csv', 'r', encoding='utf-8')

    # skip header.
    f.readline()

    iris = []
    for row in csv.reader(f):
        # print(row)
        #
        # for item in row[1:-1]:
        #     print(float(item), end=' ')
        # print()

        if species_true != row[-1] and species_false != row[-1]:
            continue

        line =[1.]
        for item in row[1:-1]:
            line.append(float(item))

        line.append(int(row[-1] == species_true))
        # if row[-1] == species_true:
        #     line.append(1)
        # else:
        #     line.append(0)
        # line.append(row[-1])

        # print(line)
        iris.append(line)

    f.close()
    return np.array(iris, dtype=np.float32).transpose()


def basic_usage():
    # iris = get_iris('setosa', 'versicolor')
    # iris = get_iris('versicolor', 'virginica')
    iris = get_iris('virginica', 'setosa')
    # print(*iris, sep='\n')
    print(iris.shape)

    # 문제
    # 로지스틱 클래서피케이션 코드에 연결해 보세요.
    x = iris[:-1]
    y = iris[-1]

    w = tf.Variable(tf.random_uniform([1, len(x)], -1, 1))

    # (1, 5) x (5, 100) = (1, 100)
    z = tf.matmul(w, x)
    hypothesis = tf.div(1., 1. + tf.exp(-z))  # math.e ** -z
    cost = -tf.reduce_mean(y * tf.log(hypothesis) +
                           (1 - y) * tf.log(1 - hypothesis))
    learning_rate = tf.Variable(0.1)

    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        sess.run(train)

        if i % 20 == 0:
            print(i, sess.run(cost))

    sess.close()


# 100개 : 70개(학습), 30개(검사)
# 문제
# get_iris 함수를 학습과 검사 데이터를 반환하도록 수정해 보세요.
# get_iris_real 이름을 사용합니다.
# 학습 데이터로 학습하고, 검사 데이터로 예측해 보세요.
def get_iris_real(species_true, species_false):
    f = open('Data/iris.csv', 'r', encoding='utf-8')

    f.readline()

    iris = []
    for row in csv.reader(f):
        if species_true != row[-1] and species_false != row[-1]:
            continue

        line =[1.]
        for item in row[1:-1]:
            line.append(float(item))

        line.append(int(row[-1] == species_true))
        iris.append(line)

    f.close()
    # print(*iris, sep='\n')

    # ------------------------------- #

    train = iris[:35] + iris[-35:]
    test = iris[35:-35]

    return np.array(train).transpose(), \
           np.array(test).transpose()


def show_accuracy(species_true, species_false):
    train_set, test_set = get_iris_real(species_true, species_false)
    # print(train_set.shape, test_set.shape)

    xx = train_set[:-1]
    yy = train_set[-1]

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_uniform([1, len(xx)], -1, 1))

    # (1, 5) x (5, 100) = (1, 100)
    z = tf.matmul(w, x)
    hypothesis = tf.div(1., 1. + tf.exp(-z))  # math.e ** -z
    cost = -tf.reduce_mean(y * tf.log(hypothesis) +
                           (1 - y) * tf.log(1 - hypothesis))
    learning_rate = tf.Variable(0.1)

    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        sess.run(train, feed_dict={x: xx, y: yy})

        # if i % 20 == 0:
        #     print(i, sess.run(cost, feed_dict={x: xx, y: yy}))

    yhat = sess.run(hypothesis, feed_dict={x: test_set[:-1]})
    # print(yhat)
    # print(yhat > 0.5)
    # print(test_set[-1])
    print((yhat>0.5) == test_set[-1])
    print(np.mean((yhat>0.5) == test_set[-1]))

    sess.close()


show_accuracy('setosa', 'versicolor')
show_accuracy('versicolor', 'virginica')
show_accuracy('virginica', 'setosa')
