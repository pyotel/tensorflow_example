# Day_02_01_placeholder.py
import tensorflow as tf


def not_used():
    a = tf.placeholder(tf.int32)
    b = tf.placeholder(tf.int32)
    c = tf.constant(6)

    add1 = tf.add(a, b)
    add2 = tf.add(a, c)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print(sess.run(add1, feed_dict={a: 3, b: 4}))
    print(sess.run(add1, feed_dict={a: 2, b: 3}))
    print(sess.run(add2, feed_dict={a: 2}))

    sess.close()


# 문제
# 구구단의 특정 단을 출력하는 함수를 만드세요.
# 텐서플로우의 placeholder 사용.
# multiply.
def dan99(dan):
    left = tf.placeholder(tf.int32)
    rite = tf.placeholder(tf.int32)

    calc = tf.multiply(left, rite)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1, 10):
        result = sess.run(calc, feed_dict={left: dan, rite: i})
        print('{} x {} = {}'.format(dan, i, result))
        # print(dan, 'x',  i, '=', result)

    sess.close()


def dan99_adv(dan):
    left = tf.constant(dan)
    rite = tf.placeholder(tf.int32)

    calc = tf.multiply(left, rite)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1, 10):
        result = sess.run(calc, feed_dict={rite: i})
        print('{} x {} = {:2}'.format(dan, i, result))

    sess.close()


# dan99(7)
dan99_adv(7)
